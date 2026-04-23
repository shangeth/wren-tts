"""
Training data for Wren-TTS.

Reads Mimi-encoded codes + transcripts from a single HuggingFace dataset repo
(`cfg.hf_dataset`, with `cfg.hf_splits`), concatenates the requested splits, and
deterministically partitions off the tail as a val split.

For custom dataset mixes (e.g. combining LJSpeech with a slice of LibriSpeech),
build the `hf_dataset` argument to `HFMimiDataset` however you want right here.
"""

import logging
import random
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import Config

logger = logging.getLogger(__name__)


def _build_sequence(
    text_ids:       List[int],
    tgt_codes:      torch.LongTensor,   # [k, T_frames]
    audio_sep_id:   int,
    k:              int,
    codebook_size:  int,
    ref_codes:      Optional[torch.LongTensor] = None,  # [k, T_ref] or None
    audio_start_id: Optional[int] = None,
    audio_end_id:   Optional[int] = None,
) -> dict:
    """
    Build one training sequence.

    Single-speaker:  [ text... | <audio_sep> | tgt_codes... | eos_sentinel ]
    Multispeaker:    [ <audio_start> | ref_codes... | <audio_end> | text... | <audio_sep> | tgt_codes... | eos_sentinel ]

    Labels: -100 everywhere except tgt_codes positions and the eos_sentinel (which
    supervises cb0 → AUDIO_EOS). Reference codes are context-only (labels=-100).
    """
    AUDIO_EOS = codebook_size  # = 2048

    T = tgt_codes.shape[1]
    tgt_audio_tokens = tgt_codes.T.reshape(-1)        # [T*k]
    n_tgt = len(tgt_audio_tokens)
    tgt_cb_indices = torch.arange(n_tgt) % k

    parts_ids:    List[torch.Tensor] = []
    parts_amask:  List[torch.Tensor] = []
    parts_cb:     List[torch.Tensor] = []
    parts_labels: List[torch.Tensor] = []

    # Optional reference block
    if ref_codes is not None and audio_start_id is not None and audio_end_id is not None:
        T_ref = ref_codes.shape[1]
        ref_tokens = ref_codes.T.reshape(-1)           # [T_ref*k]
        n_ref = len(ref_tokens)
        ref_cb_indices = torch.arange(n_ref) % k

        parts_ids.append(torch.tensor([audio_start_id], dtype=torch.long))
        parts_amask.append(torch.zeros(1, dtype=torch.bool))
        parts_cb.append(torch.zeros(1, dtype=torch.long))
        parts_labels.append(torch.full((1,), -100, dtype=torch.long))

        parts_ids.append(ref_tokens)
        parts_amask.append(torch.ones(n_ref, dtype=torch.bool))
        parts_cb.append(ref_cb_indices)
        parts_labels.append(torch.full((n_ref,), -100, dtype=torch.long))

        parts_ids.append(torch.tensor([audio_end_id], dtype=torch.long))
        parts_amask.append(torch.zeros(1, dtype=torch.bool))
        parts_cb.append(torch.zeros(1, dtype=torch.long))
        parts_labels.append(torch.full((1,), -100, dtype=torch.long))

    # Text tokens
    text_tensor = torch.tensor(text_ids, dtype=torch.long)
    parts_ids.append(text_tensor)
    parts_amask.append(torch.zeros(len(text_ids), dtype=torch.bool))
    parts_cb.append(torch.zeros(len(text_ids), dtype=torch.long))
    parts_labels.append(torch.full((len(text_ids),), -100, dtype=torch.long))

    # <audio_sep>
    parts_ids.append(torch.tensor([audio_sep_id], dtype=torch.long))
    parts_amask.append(torch.zeros(1, dtype=torch.bool))
    parts_cb.append(torch.zeros(1, dtype=torch.long))
    parts_labels.append(torch.full((1,), -100, dtype=torch.long))

    # Target audio tokens
    parts_ids.append(tgt_audio_tokens)
    parts_amask.append(torch.ones(n_tgt, dtype=torch.bool))
    parts_cb.append(tgt_cb_indices)
    parts_labels.append(tgt_audio_tokens.clone())

    # EOS sentinel (input value irrelevant; label = AUDIO_EOS for cb0 head)
    parts_ids.append(torch.tensor([0], dtype=torch.long))
    parts_amask.append(torch.ones(1, dtype=torch.bool))
    parts_cb.append(torch.zeros(1, dtype=torch.long))
    parts_labels.append(torch.tensor([AUDIO_EOS], dtype=torch.long))

    input_ids      = torch.cat(parts_ids)
    audio_mask     = torch.cat(parts_amask)
    cb_indices     = torch.cat(parts_cb)
    labels         = torch.cat(parts_labels)
    attention_mask = torch.ones(len(input_ids), dtype=torch.long)

    return {
        "input_ids":      input_ids,
        "audio_mask":     audio_mask,
        "cb_indices":     cb_indices,
        "labels":         labels,
        "attention_mask": attention_mask,
    }


class HFMimiDataset(Dataset):
    """
    TTS dataset over a HuggingFace `datasets.Dataset` whose rows have:
      id, text, codes [k_extracted, n_frames], n_frames, k_codebooks,
      and (optionally) speaker_id.
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        audio_sep_id:   int,
        cfg:            Config,
        audio_start_id: Optional[int] = None,
        audio_end_id:   Optional[int] = None,
    ):
        self.ds             = hf_dataset
        self.tokenizer      = tokenizer
        self.audio_sep_id   = audio_sep_id
        self.audio_start_id = audio_start_id
        self.audio_end_id   = audio_end_id
        self.cfg            = cfg
        self.k              = cfg.k_codebooks
        self.multispeaker   = cfg.multispeaker
        self.has_speakers   = "speaker_id" in hf_dataset.column_names

        import numpy as np

        # --- Vectorized n_frames filter (no Python loop) ---
        n_frames_arr = np.array(hf_dataset["n_frames"])
        frames_ok    = n_frames_arr <= cfg.max_audio_frames

        # --- Batch tokenize in chunks (100-200x faster than per-row loop) ---
        texts = hf_dataset["text"]
        CHUNK = 2000
        text_lengths = np.zeros(len(texts), dtype=np.int32)
        for start in tqdm(range(0, len(texts), CHUNK), desc="Tokenizing dataset"):
            batch   = texts[start : start + CHUNK]
            lengths = tokenizer(batch, add_special_tokens=False, return_length=True)["length"]
            text_lengths[start : start + CHUNK] = lengths
        text_ok = text_lengths <= cfg.max_text_tokens

        kept_mask = frames_ok & text_ok
        kept_rows = list(np.where(kept_mask)[0])

        # --- Speaker index (only over kept rows, fast) ---
        speaker_to_indices: dict = {}
        if self.has_speakers:
            speakers_col = hf_dataset["speaker_id"]
            for local_idx, row_idx in enumerate(kept_rows):
                speaker_to_indices.setdefault(speakers_col[row_idx], []).append(local_idx)

        self.indices             = kept_rows
        self._speaker_to_indices = speaker_to_indices

        n_dropped = len(hf_dataset) - len(self.indices)
        if n_dropped:
            logger.warning(
                f"HF dataset: dropped {n_dropped}/{len(hf_dataset)} examples "
                f"(max_text_tokens={cfg.max_text_tokens} / max_audio_frames={cfg.max_audio_frames})."
            )
        logger.info(f"HF dataset: {len(self.indices)} aligned examples.")

    def __len__(self) -> int:
        return len(self.indices)

    def _codes_tensor(self, row_idx: int) -> torch.LongTensor:
        return torch.tensor(self.ds[row_idx]["codes"], dtype=torch.long)

    def __getitem__(self, idx: int) -> dict:
        row_idx = self.indices[idx]
        ex      = self.ds[row_idx]

        text = ex["text"]
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)

        codes     = torch.tensor(ex["codes"], dtype=torch.long)       # [k_extracted, n_frames]
        tgt_codes = codes[: self.k, : self.cfg.max_audio_frames]

        ref_codes = None
        if self.multispeaker and self.audio_start_id is not None:
            ref_idx: Optional[int] = None
            if self.has_speakers:
                candidates = [i for i in self._speaker_to_indices.get(ex["speaker_id"], []) if i != idx]
                if candidates:
                    ref_idx = random.choice(candidates)
            else:
                if len(self.indices) > 1:
                    ref_idx = random.choice([i for i in range(len(self.indices)) if i != idx])

            if ref_idx is not None:
                rc    = self._codes_tensor(self.indices[ref_idx])
                T_ref = min(rc.shape[1], self.cfg.max_ref_frames)
                ref_codes = rc[: self.k, :T_ref]

        return _build_sequence(
            text_ids       = text_ids,
            tgt_codes      = tgt_codes,
            audio_sep_id   = self.audio_sep_id,
            k              = self.k,
            codebook_size  = self.cfg.codebook_size,
            ref_codes      = ref_codes,
            audio_start_id = self.audio_start_id,
            audio_end_id   = self.audio_end_id,
        )


def _load_hf_split(
    split: str,
    cfg:   Config,
):
    """Load and combine all datasets from cfg.hf_datasets, return the train|val partition.

    Each entry in hf_datasets/hf_splits/hf_weights is one dataset source:
      - hf_splits[i]  : comma-sep HF split names (concatenated within the same repo)
      - hf_weights[i] : fraction in (0, 1] — 1.0 = full dataset, 0.2 = 20% (fixed at load time)
    """
    from datasets import load_dataset, concatenate_datasets

    # Pad hf_weights to len(hf_datasets) in case user omits trailing 1.0s
    weights = list(cfg.hf_weights) + [1.0] * max(0, len(cfg.hf_datasets) - len(cfg.hf_weights))

    sources = []
    for repo, splits_str, weight in zip(cfg.hf_datasets, cfg.hf_splits, weights):
        split_names = [s.strip() for s in splits_str.split(",")]
        parts = [load_dataset(repo, split=s) for s in split_names]
        ds    = concatenate_datasets(parts) if len(parts) > 1 else parts[0]

        if weight < 1.0:
            n = max(1, int(len(ds) * weight))
            ds = ds.shuffle(seed=42).select(range(n))
            logger.info(f"  {repo} [{splits_str}]: using {n}/{len(ds)} rows (weight={weight})")
        else:
            logger.info(f"  {repo} [{splits_str}]: using all {len(ds)} rows")

        sources.append(ds)

    combined = concatenate_datasets(sources) if len(sources) > 1 else sources[0]

    use_explicit_val = bool(cfg.hf_val_datasets)

    if split == "train":
        if use_explicit_val:
            # Explicit val datasets defined — train gets all its data.
            return combined
        # Fallback: carve val_fraction off the tail.
        n = len(combined)
        n_val = max(1, int(n * cfg.val_fraction))
        return combined.select(range(n - n_val))

    if split == "val":
        if use_explicit_val:
            val_parts = []
            for repo, splits_str in zip(cfg.hf_val_datasets, cfg.hf_val_splits):
                split_names = [s.strip() for s in splits_str.split(",")]
                parts = [load_dataset(repo, split=s) for s in split_names]
                val_parts.append(concatenate_datasets(parts) if len(parts) > 1 else parts[0])
                logger.info(f"  val: {repo} [{splits_str}]: {val_parts[-1].num_rows} rows")
            return concatenate_datasets(val_parts) if len(val_parts) > 1 else val_parts[0]
        # Fallback: tail fraction of combined train data.
        n = len(combined)
        n_val = max(1, int(n * cfg.val_fraction))
        return combined.select(range(n - n_val, n))

    raise ValueError(f"Unknown split: {split!r}")


def collate_fn(batch: List[dict]) -> dict:
    """Dynamic padding to the longest sequence in the batch."""
    max_len = max(b["input_ids"].shape[0] for b in batch)

    input_ids_list      = []
    audio_mask_list     = []
    cb_indices_list     = []
    labels_list         = []
    attention_mask_list = []

    for b in batch:
        L   = b["input_ids"].shape[0]
        pad = max_len - L

        input_ids_list.append(F.pad(b["input_ids"],          (0, pad), value=0))
        audio_mask_list.append(F.pad(b["audio_mask"].long(), (0, pad), value=0).bool())
        cb_indices_list.append(F.pad(b["cb_indices"],        (0, pad), value=0))
        labels_list.append(F.pad(b["labels"],                (0, pad), value=-100))
        attention_mask_list.append(F.pad(b["attention_mask"],(0, pad), value=0))

    return {
        "input_ids":      torch.stack(input_ids_list),
        "audio_mask":     torch.stack(audio_mask_list),
        "cb_indices":     torch.stack(cb_indices_list),
        "labels":         torch.stack(labels_list),
        "attention_mask": torch.stack(attention_mask_list),
    }


def get_dataloader(
    split:          str,
    tokenizer,
    audio_sep_id:   int,
    cfg:            Config,
    shuffle:        bool = True,
    audio_start_id: Optional[int] = None,
    audio_end_id:   Optional[int] = None,
) -> DataLoader:
    hf_ds = _load_hf_split(split, cfg)
    dataset = HFMimiDataset(
        hf_ds,
        tokenizer      = tokenizer,
        audio_sep_id   = audio_sep_id,
        cfg            = cfg,
        audio_start_id = audio_start_id,
        audio_end_id   = audio_end_id,
    )
    return DataLoader(
        dataset,
        batch_size         = cfg.batch_size,
        shuffle            = shuffle,
        num_workers        = cfg.num_workers,
        pin_memory         = cfg.pin_memory,
        prefetch_factor    = cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers = cfg.num_workers > 0,
        collate_fn         = collate_fn,
    )


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from config import Config

    cfg = Config(batch_size=2, num_workers=0, max_audio_frames=300)
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "<|audio_sep|>", "<|audio_eos|>",
            "<|audio_start|>", "<|audio_end|>",
        ]
    })
    audio_sep_id   = tokenizer.convert_tokens_to_ids("<|audio_sep|>")
    audio_start_id = tokenizer.convert_tokens_to_ids("<|audio_start|>")
    audio_end_id   = tokenizer.convert_tokens_to_ids("<|audio_end|>")

    loader = get_dataloader(
        "train", tokenizer, audio_sep_id, cfg, shuffle=False,
        audio_start_id=audio_start_id, audio_end_id=audio_end_id,
    )
    batch = next(iter(loader))

    print(f"input_ids:      {batch['input_ids'].shape}")
    print(f"audio_mask:     {batch['audio_mask'].shape}  (True count: {batch['audio_mask'].sum().item()})")
    print(f"cb_indices:     {batch['cb_indices'].shape}")
    print(f"labels:         {batch['labels'].shape}  (non-100 count: {(batch['labels'] >= 0).sum().item()})")
    print(f"attention_mask: {batch['attention_mask'].shape}")
    valid_labels = batch["labels"][batch["labels"] >= 0]
    print(f"label range: [{valid_labels.min().item()}, {valid_labels.max().item()}]  (expected [0, 2048])")
