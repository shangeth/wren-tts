import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from config import Config
from model import TTSModel

logger = logging.getLogger(__name__)


def _make_logger(log_dir: str, logger_type: str, config: Config):
    if logger_type == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter
        return _TBLogger(SummaryWriter(log_dir=log_dir))
    elif logger_type == "wandb":
        import wandb
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=asdict(config),
        )
        return _WandbLogger()
    else:
        raise ValueError(f"Unknown logger: {logger_type!r}. Choose 'tensorboard' or 'wandb'.")


class _TBLogger:
    def __init__(self, writer):
        self._w = writer

    def log(self, losses, step, prefix):
        for k, v in losses.items():
            self._w.add_scalar(f"{prefix}/{k}", v, step)

    def log_audio(self, tag, audio, step, sample_rate):
        self._w.add_audio(tag, audio, step, sample_rate=sample_rate)

    def close(self):
        self._w.close()


class _WandbLogger:
    def log(self, losses, step, prefix):
        import wandb
        wandb.log({f"{prefix}/{k}": v for k, v in losses.items()}, step=step)

    def log_audio(self, tag, audio, step, sample_rate):
        import wandb
        audio_np = audio.squeeze().cpu().float().numpy()
        wandb.log({tag: wandb.Audio(audio_np, sample_rate=sample_rate)}, step=step)

    def close(self):
        import wandb
        wandb.finish()


class Trainer:
    def __init__(
        self,
        model: TTSModel,
        train_loader,
        val_loader,
        config: Config,
        test_loader=None,
        tokenizer=None,
        mimi_codec=None,
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.config       = config
        self.tokenizer    = tokenizer
        self.mimi_codec   = mimi_codec

        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=tuple(config.betas),
            weight_decay=config.weight_decay,
        )

        # Steps per epoch (approximate — used to compute total steps if needed).
        # lr_decay_steps is a FLOOR, not a cap: the cosine schedule spans the full run
        # when epochs × steps_per_epoch exceeds it (which is the usual case).
        steps_per_epoch = max(1, len(train_loader) // config.grad_accum_steps)
        total_steps = max(config.lr_decay_steps, steps_per_epoch * config.epochs)
        logger.info(
            f"LR schedule: cosine decay from {config.lr:.2e} to 0 over {total_steps} steps "
            f"(warmup={config.lr_warmup_steps}, steps/epoch≈{steps_per_epoch}, epochs={config.epochs}, "
            f"lr_decay_steps floor={config.lr_decay_steps})"
        )

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=total_steps,
        )

        self._device_type = torch.device(config.device).type
        # BFloat16 has the same exponent range as float32 — no loss scaling needed.
        # GradScaler only works with float16; use it only when bf16 is unavailable.
        _use_bf16 = (self._device_type == "cuda" and torch.cuda.is_bf16_supported())
        self._amp_dtype = torch.bfloat16 if _use_bf16 else torch.float16
        _scaler_enabled = config.use_amp and not _use_bf16
        self.scaler = GradScaler(self._device_type, enabled=_scaler_enabled)

        self.start_epoch   = 0
        self.global_step   = 0
        self.best_val_loss = float("inf")
        self._log_samples  = None  # fixed val examples for audio logging, built lazily

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot the resolved config (YAML + CLI overrides merged) next to
        # checkpoints so the experiment is self-describing.
        config.save(str(self.checkpoint_dir / "config.yaml"))

        self._run_logger = None
        if getattr(config, "logger", None):
            self._run_logger = _make_logger(config.log_dir, config.logger, config)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.checkpoint_dir / "train.log"),
            ],
        )

        if config.resume_from:
            self.load_checkpoint(config.resume_from)

        # Compile AFTER loading weights so state_dict keys stay untouched during load.
        if getattr(config, "compile_model", False):
            self.model = torch.compile(self.model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self):
        logger.info(f"Starting training from epoch {self.start_epoch} for {self.config.epochs} epochs")
        for epoch in range(self.start_epoch, self.config.epochs):
            train_losses = self._train_epoch(epoch)
            val_losses   = self._val_epoch(epoch)

            self._log(train_losses, self.global_step, prefix="epoch/train")
            self._log(val_losses,   self.global_step, prefix="epoch/val")

            self._log_audio_samples(self.global_step)

            is_best = val_losses["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses["loss"]
            self._save_checkpoint(epoch, is_best=is_best)

            logger.info(
                f"Epoch {epoch:04d} | "
                f"train_loss={train_losses['loss']:.4f} "
                f"val_loss={val_losses['loss']:.4f}"
                + (" [BEST]" if is_best else "")
            )

        if self._run_logger:
            self._run_logger.close()

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.start_epoch   = ckpt["epoch"] + 1
        self.global_step   = ckpt["global_step"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info(f"Resumed from checkpoint: {path} (epoch {ckpt['epoch']})")

    # ------------------------------------------------------------------
    # Train / val loops
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        # Discard any partial accumulation left over from the previous epoch's tail.
        self.optimizer.zero_grad()
        accum       = _LossAccumulator()
        accum_steps = self.config.grad_accum_steps
        pbar = tqdm(self.train_loader, desc=f"Train {epoch:04d}", leave=False, dynamic_ncols=True)

        for batch_idx, batch in enumerate(pbar):
            is_update = (batch_idx + 1) % accum_steps == 0
            losses    = self._train_step(batch, accum_steps=accum_steps, do_update=is_update)
            accum.update(losses)

            if is_update:
                self.global_step += 1
                pbar.set_postfix({"loss": f"{losses['loss']:.4f}"})
                self._log(losses, self.global_step, prefix="step/train")

                if self.global_step % self.config.log_audio_every == 0:
                    self._log_audio_samples(self.global_step)

        return accum.mean()

    def _val_epoch(self, epoch: int) -> dict:
        self.model.eval()
        accum = _LossAccumulator()
        eos_correct_total, eos_total_total = 0, 0
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Val   {epoch:04d}", leave=False, dynamic_ncols=True)
            for batch in pbar:
                losses = self._val_step(batch)
                eos_correct_total += losses.pop("eos_correct", 0)
                eos_total_total   += losses.pop("eos_total",   0)
                accum.update(losses)
                pbar.set_postfix({"loss": f"{losses['loss']:.4f}"})
        result = accum.mean()
        if eos_total_total > 0:
            result["eos_accuracy"] = eos_correct_total / eos_total_total
            logger.info(f"Val EOS accuracy: {result['eos_accuracy']:.3f}  ({eos_correct_total}/{eos_total_total})")
        return result

    # ------------------------------------------------------------------
    # Single step logic
    # ------------------------------------------------------------------

    def _train_step(self, batch: dict, accum_steps: int = 1, do_update: bool = True) -> dict:
        input_ids      = batch["input_ids"].to(self.device)
        audio_codes    = batch["audio_codes"].to(self.device)
        audio_mask     = batch["audio_mask"].to(self.device)
        labels         = batch["labels"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        with autocast(self._device_type, dtype=self._amp_dtype, enabled=self.config.use_amp):
            loss, loss_dict = self.model(
                input_ids, audio_codes, audio_mask, labels, attention_mask
            )

        if torch.isfinite(loss):
            self.scaler.scale(loss / accum_steps).backward()
            if do_update:
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                loss_dict["grad_norm"] = float(grad_norm)
                loss_dict["lr"]        = self.scheduler.get_last_lr()[0]
        else:
            logger.warning(f"step {self.global_step}: non-finite loss, skipping update")
            if do_update:
                self.optimizer.zero_grad()

        return loss_dict

    def _val_step(self, batch: dict) -> dict:
        input_ids      = batch["input_ids"].to(self.device)
        audio_codes    = batch["audio_codes"].to(self.device)
        audio_mask     = batch["audio_mask"].to(self.device)
        labels         = batch["labels"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        with autocast(self._device_type, dtype=self._amp_dtype, enabled=self.config.use_amp):
            _, loss_dict = self.model(
                input_ids, audio_codes, audio_mask, labels, attention_mask
            )

        # EOS accuracy: at shifted cb0 positions where label == AUDIO_EOS, how often do we predict EOS?
        AUDIO_EOS = self.model.AUDIO_EOS
        target_cb0 = labels[:, 1:, 0]                          # [B, L-1]
        eos_mask   = target_cb0 == AUDIO_EOS                    # [B, L-1]
        if eos_mask.any():
            with torch.no_grad():
                hidden = self.model.llm.model(
                    inputs_embeds=self.model._build_inputs_embeds(input_ids, audio_codes, audio_mask),
                    attention_mask=attention_mask,
                    use_cache=False,
                ).last_hidden_state
                pred_hidden = hidden[:, :-1, :]
                h_cb0  = pred_hidden[eos_mask]                  # [N, H]
                logits = self.model.audio_heads[0](h_cb0.float())
                preds  = logits.argmax(-1)
                correct = (preds == AUDIO_EOS).sum().item()
            loss_dict["eos_correct"] = correct
            loss_dict["eos_total"]   = int(eos_mask.sum().item())
        else:
            loss_dict["eos_correct"] = 0
            loss_dict["eos_total"]   = 0

        return loss_dict

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        # Unwrap torch.compile so keys don't get the `_orig_mod.` prefix on disk.
        model_to_save = getattr(self.model, "_orig_mod", self.model)
        state = {
            "epoch":         epoch,
            "global_step":   self.global_step,
            "best_val_loss": self.best_val_loss,
            "model":         model_to_save.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "scheduler":     self.scheduler.state_dict(),
            "scaler":        self.scaler.state_dict(),
            "config":        asdict(self.config),
        }
        torch.save(state, self.checkpoint_dir / "last.pt")
        torch.save(state, self.checkpoint_dir / f"epoch_{epoch:04d}.pt")
        if is_best:
            torch.save(state, self.checkpoint_dir / "best.pt")
        self._cleanup_old_checkpoints(keep=self.config.keep_last_n)

    def _cleanup_old_checkpoints(self, keep: int = 3):
        ckpts = sorted(self.checkpoint_dir.glob("epoch_*.pt"))
        for old in ckpts[:-keep]:
            old.unlink()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, losses: dict, step: int, prefix: str):
        if self._run_logger:
            self._run_logger.log(losses, step, prefix)

    def _prepare_log_samples(self, n: int = 2) -> list:
        """
        Pick n fixed val examples for consistent audio logging across all steps.
        Works with HFMimiDataset: uses dataset.indices + dataset.ds for row access.
        """
        dataset            = self.val_loader.dataset
        reference_start_id = self.tokenizer.convert_tokens_to_ids("<|reference_start|>")
        reference_end_id   = self.tokenizer.convert_tokens_to_ids("<|reference_end|>")
        total              = len(dataset)
        local_indices      = [int(i * total / n) for i in range(n)]

        samples = []
        for local_idx in local_indices:
            row_idx = dataset.indices[local_idx]
            row     = dataset.ds[row_idx]
            text    = row["text"]

            ref_codes = None
            if self.config.multispeaker:
                ref_local_idx = None
                if dataset.has_speakers:
                    speaker_id = row["speaker_id"]
                    candidates = [i for i in dataset._speaker_to_indices.get(speaker_id, [])
                                  if i != local_idx]
                    ref_local_idx = candidates[0] if candidates else None
                else:
                    ref_local_idx = (local_idx + 1) % total

                if ref_local_idx is not None:
                    ref_row  = dataset.ds[dataset.indices[ref_local_idx]]
                    rc       = torch.tensor(ref_row["codes"], dtype=torch.long)
                    T_ref    = min(rc.shape[1], self.config.max_ref_frames)
                    ref_codes = rc[: self.config.k_codebooks, :T_ref]

            tag = f"audio/sample_{local_idx:04d}"
            samples.append(dict(
                text               = text,
                ref_codes          = ref_codes,
                reference_start_id = reference_start_id if ref_codes is not None else None,
                reference_end_id   = reference_end_id   if ref_codes is not None else None,
                tag                = tag,
            ))
            logger.info(f"Log sample {local_idx}: '{text[:60]}'")
        return samples

    def _log_audio_samples(self, step: int):
        """Generate audio for the fixed val samples and log to TensorBoard/wandb."""
        if self._run_logger is None or self.mimi_codec is None or self.tokenizer is None:
            return

        if self._log_samples is None:
            try:
                self._log_samples = self._prepare_log_samples(n=2)
            except Exception:
                logger.exception("Failed to prepare audio log samples; audio logging disabled")
                return

        self.model.eval()
        try:
            for sample in self._log_samples:
                codes = self.model.generate(
                    sample["text"], self.tokenizer,
                    max_audio_frames=self.config.max_audio_frames,
                    min_audio_frames=20,
                    temperature=0.8, top_k=50, top_p=0.9,
                    ref_codes=sample["ref_codes"],
                    reference_start_id=sample["reference_start_id"],
                    reference_end_id=sample["reference_end_id"],
                )  # [k, n_frames]

                if codes.shape[1] == 0:
                    logger.warning(f"Audio log step {step}: EOS immediately for '{sample['text'][:40]}'")
                    continue

                wav = self.mimi_codec.decode(codes)  # [1, T]
                self._run_logger.log_audio(sample["tag"], wav, step, sample_rate=24000)
                logger.info(
                    f"Logged audio step {step}: '{sample['text'][:60]}' "
                    f"({codes.shape[1]} frames, {wav.shape[-1]/24000:.1f}s)"
                )
        except Exception:
            logger.exception(f"Audio logging failed at step {step}")
        finally:
            self.model.train()


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

class _LossAccumulator:
    def __init__(self):
        self._sums   = {}
        self._counts = {}

    def update(self, losses: dict):
        for k, v in losses.items():
            self._sums[k]   = self._sums.get(k, 0.0) + v
            self._counts[k] = self._counts.get(k, 0) + 1

    def mean(self) -> dict:
        return {k: self._sums[k] / self._counts[k] for k in self._sums}
