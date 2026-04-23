"""
Evaluate a Wren-TTS checkpoint on a held-out test set.

For each test utterance:
  - Decode the cached Mimi codes → reference waveform (same-speaker, for conditioning + SECS)
  - Pick another random speaker's utterance → different-speaker reference (for EER negatives)
  - Generate audio from (target_text, same-speaker reference)
  - Transcribe generated audio via Whisper → WER / CER vs target text
  - UTMOS on generated audio
  - SECS(gen, same-speaker ref)     → positive score for EER
  - SECS(gen, different-speaker ref) → negative score for EER

Usage:
  python evaluate.py --checkpoint checkpoints/librispeech/best.pt
  python evaluate.py --checkpoint ... --n_samples 500 --output_json eval.json
  python evaluate.py --checkpoint ... --whisper_model openai/whisper-large-v3
"""

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoTokenizer

from config import Config
from model import TTSModel
from mimi import MimiCodec
from metrics import WhisperASR, WER, CER, UTMOS, SECS, EER


def load_model(checkpoint_path: str, device: torch.device):
    """Load a TTSModel from a local training checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "config" not in ckpt:
        raise ValueError(f"Checkpoint has no 'config' key: {checkpoint_path}")
    saved = ckpt["config"]
    cfg   = Config(**{k: v for k, v in saved.items() if k in Config.__dataclass_fields__})
    print(f"Checkpoint config: llm={cfg.llm_name}  k={cfg.k_codebooks}  epoch={ckpt.get('epoch', '?')}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_special_tokens({"additional_special_tokens": [
        "<|audio_sep|>", "<|audio_eos|>", "<|audio_start|>", "<|audio_end|>",
    ]})

    model = TTSModel(cfg, tokenizer).to(device).eval()
    model.load_state_dict(ckpt["model"])
    return model, tokenizer, cfg


def sample_eval_set(cfg: Config, split: str, n_samples: int, seed: int):
    """Load the requested HF split and sample n_samples rows (deterministic)."""
    from datasets import load_dataset

    ds = load_dataset(cfg.hf_dataset, split=split)
    n = len(ds)
    if n_samples >= n:
        return ds
    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(n), n_samples))
    return ds.select(idxs)


@torch.no_grad()
def run_evaluation(
    model:           TTSModel,
    tokenizer,
    cfg:             Config,
    mimi:            MimiCodec,
    eval_ds,
    device:          torch.device,
    max_audio_frames: int,
    min_audio_frames: int,
    temperature:     float,
    top_k:           int,
    top_p:           float,
    max_ref_frames:  int,
    whisper_model:   str,
    seed:            int,
    amp_dtype:       Optional[torch.dtype] = None,
):
    asr    = WhisperASR(model_name=whisper_model, device=str(device))
    wer    = WER()
    cer    = CER()
    utmos  = UTMOS(device=str(device))
    secs   = SECS(device=str(device))
    eer    = EER()

    audio_sep_id   = tokenizer.convert_tokens_to_ids("<|audio_sep|>")
    audio_start_id = tokenizer.convert_tokens_to_ids("<|audio_start|>")
    audio_end_id   = tokenizer.convert_tokens_to_ids("<|audio_end|>")

    has_speakers = "speaker_id" in eval_ds.column_names
    rng = random.Random(seed)

    # Autocast speeds up autoregressive generation ~2–3× with no quality impact.
    from contextlib import nullcontext
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None and device.type == "cuda"
        else nullcontext()
    )

    per_sample = []
    for i, ex in enumerate(tqdm(eval_ds, desc="Evaluating")):
        target_text = ex["text"]

        # --- Same-speaker reference (also feeds the model's conditioning) ---
        ref_codes_full = torch.tensor(ex["codes"], dtype=torch.long)[:cfg.k_codebooks]
        ref_codes_cond = ref_codes_full[:, :max_ref_frames]
        ref_wav = mimi.decode(ref_codes_full)   # [1, T] @ 24 kHz

        # --- Different-speaker reference for EER negatives ---
        neg_ref_wav = None
        if has_speakers:
            candidates = [j for j in range(len(eval_ds))
                          if eval_ds[j]["speaker_id"] != ex["speaker_id"]]
            if candidates:
                neg = eval_ds[rng.choice(candidates)]
                neg_codes = torch.tensor(neg["codes"], dtype=torch.long)[:cfg.k_codebooks]
                neg_ref_wav = mimi.decode(neg_codes)

        # --- Generate ---
        with amp_ctx:
            codes = model.generate(
                text             = target_text,
                tokenizer        = tokenizer,
                max_audio_frames = max_audio_frames,
                min_audio_frames = min_audio_frames,
                temperature      = temperature,
                top_k            = top_k,
                top_p            = top_p,
                ref_codes        = ref_codes_cond,
                audio_start_id   = audio_start_id,
                audio_end_id     = audio_end_id,
            )
        if codes.shape[1] == 0:
            # model produced nothing — record a dummy sample and skip
            per_sample.append({"id": ex.get("id"), "n_frames": 0, "skipped": "empty_generation"})
            continue
        gen_wav = mimi.decode(codes)  # [1, T] @ 24 kHz

        # --- Metrics ---
        hyp        = asr.transcribe(gen_wav, 24000)
        wer_s      = wer.update(target_text, hyp)
        cer_s      = cer.update(target_text, hyp)
        utmos_s    = utmos.update(gen_wav, 24000)
        secs_same  = secs.update(gen_wav, 24000, ref_wav, 24000)
        eer.update(secs_same, is_same_speaker=True)

        secs_diff: Optional[float] = None
        if neg_ref_wav is not None:
            # separate SECS object would double-store; use the shared one but
            # don't aggregate the negative in its mean — keep negatives only for EER.
            secs_diff = secs.score(gen_wav, 24000, neg_ref_wav, 24000)
            eer.update(secs_diff, is_same_speaker=False)

        per_sample.append({
            "id":           ex.get("id"),
            "target":       target_text,
            "hypothesis":   hyp,
            "n_frames":     int(codes.shape[1]),
            "wer":          wer_s,
            "cer":          cer_s,
            "utmos":        utmos_s,
            "secs_same":    secs_same,
            "secs_diff":    secs_diff,
        })

    summary = {
        "wer":   wer.compute(),
        "cer":   cer.compute(),
        "utmos": utmos.compute(),
        "secs":  secs.compute(),
        "eer":   eer.compute(),
        "n_evaluated": sum(1 for s in per_sample if "skipped" not in s),
        "n_skipped":   sum(1 for s in per_sample if "skipped" in s),
    }
    return summary, per_sample


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Wren-TTS checkpoint")
    parser.add_argument("--checkpoint",        required=True, help="Path to .pt checkpoint")
    parser.add_argument("--hf_dataset",        default=None,  help="Override cfg.hf_dataset")
    parser.add_argument("--test_split",        default="test_clean", help="HF split to evaluate on")
    parser.add_argument("--n_samples",         type=int,   default=200)
    parser.add_argument("--seed",              type=int,   default=0)
    parser.add_argument("--max_audio_frames",  type=int,   default=300)
    parser.add_argument("--min_audio_frames",  type=int,   default=2)
    parser.add_argument("--max_ref_frames",    type=int,   default=150)
    parser.add_argument("--temperature",       type=float, default=0.8)
    parser.add_argument("--top_k",             type=int,   default=50)
    parser.add_argument("--top_p",             type=float, default=0.9)
    parser.add_argument("--whisper_model",     default="openai/whisper-base")
    parser.add_argument("--amp_dtype",         default="bf16", choices=["bf16", "fp16", "none"],
                        help="Mixed-precision for Wren generation (GPU only). Default bf16 for ~2–3× speedup.")
    parser.add_argument("--output_json",       default=None, help="Write summary + per-sample results here")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer, cfg = load_model(args.checkpoint, device)

    # CLI override for the dataset if user wants to eval on a different corpus
    if args.hf_dataset:
        cfg.hf_dataset = args.hf_dataset

    print(f"Loading eval set: {cfg.hf_dataset} split={args.test_split}, {args.n_samples} samples")
    eval_ds = sample_eval_set(cfg, args.test_split, args.n_samples, args.seed)
    print(f"Eval set size: {len(eval_ds)}")

    mimi = MimiCodec(
        model_name  = cfg.mimi_model_name,
        device      = str(device),
        k_codebooks = cfg.k_codebooks,
    )

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[args.amp_dtype]

    summary, per_sample = run_evaluation(
        model            = model,
        tokenizer        = tokenizer,
        cfg              = cfg,
        mimi             = mimi,
        eval_ds          = eval_ds,
        device           = device,
        max_audio_frames = args.max_audio_frames,
        min_audio_frames = args.min_audio_frames,
        temperature      = args.temperature,
        top_k            = args.top_k,
        top_p            = args.top_p,
        max_ref_frames   = args.max_ref_frames,
        whisper_model    = args.whisper_model,
        seed             = args.seed,
        amp_dtype        = amp_dtype,
    )

    # --- Pretty-print summary ---
    print()
    print("=" * 60)
    print(f"Evaluated {summary['n_evaluated']} / {summary['n_evaluated'] + summary['n_skipped']} samples")
    print("-" * 60)
    print(f"  WER       : {summary['wer']:.4f}")
    print(f"  CER       : {summary['cer']:.4f}")
    print(f"  UTMOS     : {summary['utmos']['mean']:.3f} ± {summary['utmos']['std']:.3f}  (n={summary['utmos']['n']})")
    print(f"  SECS      : {summary['secs']['mean']:.3f} ± {summary['secs']['std']:.3f}  (n={summary['secs']['n']})")
    print(f"  EER       : {summary['eer']['eer']:.4f}  (pos={summary['eer']['n_pos']}, neg={summary['eer']['n_neg']}, "
          f"thr={summary['eer']['threshold']:.3f})")
    print("=" * 60)

    if args.output_json:
        out = {"checkpoint": args.checkpoint, "args": vars(args),
               "summary": summary, "per_sample": per_sample}
        Path(args.output_json).write_text(json.dumps(out, indent=2, default=str))
        print(f"Wrote: {args.output_json}")


if __name__ == "__main__":
    main()
