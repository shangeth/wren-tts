"""
Training data for Wren-TTS — delay-pattern layout.

Reads Mimi-encoded codes + transcripts from HuggingFace dataset repos, concatenates
the requested splits, and deterministically partitions off the tail as a val split.

Sequence layout (single-speaker):
  [ text... | <audio_start> | tgt_delayed ]

Sequence layout (multispeaker):
  [ <reference_start> | ref_delayed | <reference_end> | text... | <audio_start> | tgt_delayed ]

Audio blocks are laid out in MusicGen-style delay pattern: at step s, codebook q holds
frame s-q (or PAD at the leading/trailing edges). Total audio steps per block = T + k - 1.
cb0's AUDIO_EOS label is placed at step T of the target block (one past the last real frame).

Per-sample output tensors:
  input_ids     [L]    int64  — text/special-token IDs; 0 at audio positions (unused by embed)
  audio_codes   [L,k]  int64  — per-codebook input; AUDIO_PAD at non-audio and invalid delay edges
  audio_mask    [L]    bool   — True at audio steps (ref or target)
  labels        [L,k]  int64  — per-codebook target; -100 at text/ref/invalid; AUDIO_EOS for cb0 at step T
  attention_mask[L]    int64  — 1 everywhere, 0 at batch padding
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


def apply_delay(codes: torch.LongTensor, k: int, pad: int) -> torch.LongTensor:
    """
    Apply MusicGen-style delay to a [k, T] code matrix.

    Returns [k, T + k - 1] where delayed[q, q:q+T] = codes[q, :] and other positions are `pad`.
    At step s, codebook q holds frame s-q if 0 <= s-q < T, else pad.
    """
    assert codes.dim() == 2 and codes.shape[0] == k, f"expected [k={k}, T], got {tuple(codes.shape)}"
    T = codes.shape[1]
    L = T + k - 1
    delayed = torch.full((k, L), pad, dtype=torch.long)
    for q in range(k):
        delayed[q, q:q + T] = codes[q]
    return delayed


def undelay(delayed: torch.LongTensor, k: int, T: int) -> torch.LongTensor:
    """
    Inverse of apply_delay. Given a [k, T + k - 1] delayed matrix (or longer), recover [k, T].
    """
    assert delayed.dim() == 2 and delayed.shape[0] == k
    assert delayed.shape[1] >= T + k - 1, f"need at least {T + k - 1} steps, got {delayed.shape[1]}"
    out = torch.empty((k, T), dtype=delayed.dtype)
    for q in range(k):
        out[q] = delayed[q, q:q + T]
    return out


def _build_sequence(
    text_ids:          List[int],
    tgt_codes:         torch.LongTensor,   # [k, T_tgt]
    audio_start_id:    int,                # <|audio_start|> — text→target marker
    k:                 int,
    codebook_size:     int,
    ref_codes:         Optional[torch.LongTensor] = None,  # [k, T_ref] or None
    reference_start_id: Optional[int] = None,
    reference_end_id:   Optional[int] = None,
) -> dict:
    """
    Build one training sequence in delay-pattern layout.

    Labels:
      - Text + special tokens: -100
      - Reference audio steps: -100 (context-only)
      - Target audio steps: actual codes at valid positions, -100 at leading/trailing PAD edges
      - cb0 at step T (first post-target step within the target delay window) = AUDIO_EOS
    """
    AUDIO_PAD = codebook_size       # = 2048 (input-side; also the PAD-label in labels tensor)
    AUDIO_EOS = codebook_size       # = 2048 (cb0 output-class meaning "stop")

    def _text_part(ids_list: List[int]):
        """Text-mode segment: audio_codes=PAD, audio_mask=False, labels=-100."""
        n = len(ids_list)
        return dict(
            input_ids      = torch.tensor(ids_list, dtype=torch.long),
            audio_codes    = torch.full((n, k), AUDIO_PAD, dtype=torch.long),
            audio_mask     = torch.zeros(n, dtype=torch.bool),
            labels         = torch.full((n, k), -100, dtype=torch.long),
        )

    def _audio_part(codes: torch.LongTensor, supervise: bool, eos_at_T: bool):
        """
        Audio-mode delayed segment.

        Args:
            codes:     [k, T] real codes
            supervise: True for target block (labels = codes at valid positions),
                       False for reference block (all labels = -100)
            eos_at_T:  if True, cb0's label at step T (first post-target step within
                       the delay window) is AUDIO_EOS. Only True for the target block.
        """
        T = codes.shape[1]
        L = T + k - 1
        # Input-side codes with PAD at edges
        delayed = apply_delay(codes, k, AUDIO_PAD)  # [k, L]

        # Labels: default -100
        lab = torch.full((k, L), -100, dtype=torch.long)
        if supervise:
            for q in range(k):
                lab[q, q:q + T] = codes[q]
            if eos_at_T and k >= 2:
                # cb0 emits AUDIO_EOS at step T (one past last real cb0 frame).
                # Column T exists in the delay window iff k >= 2 (since L = T + k - 1 > T).
                # cb0's input at this step is PAD (frame T has no real data); its label is EOS.
                lab[0, T] = AUDIO_EOS

        return dict(
            input_ids   = torch.zeros(L, dtype=torch.long),  # unused at audio steps
            audio_codes = delayed.T.contiguous(),            # [L, k]
            audio_mask  = torch.ones(L, dtype=torch.bool),
            labels      = lab.T.contiguous(),                # [L, k]
        )

    parts: List[dict] = []

    # --- Optional reference block ---
    if ref_codes is not None and reference_start_id is not None and reference_end_id is not None:
        parts.append(_text_part([reference_start_id]))
        parts.append(_audio_part(ref_codes, supervise=False, eos_at_T=False))
        parts.append(_text_part([reference_end_id]))

    # --- Text ---
    parts.append(_text_part(list(text_ids)))

    # --- <audio_start> (text→target marker) ---
    parts.append(_text_part([audio_start_id]))

    # --- Target audio block (delay, supervised, EOS at step T) ---
    parts.append(_audio_part(tgt_codes, supervise=True, eos_at_T=True))

    input_ids   = torch.cat([p["input_ids"]   for p in parts], dim=0)
    audio_codes = torch.cat([p["audio_codes"] for p in parts], dim=0)
    audio_mask  = torch.cat([p["audio_mask"]  for p in parts], dim=0)
    labels      = torch.cat([p["labels"]      for p in parts], dim=0)
    attention_mask = torch.ones(input_ids.shape[0], dtype=torch.long)

    return {
        "input_ids":      input_ids,
        "audio_codes":    audio_codes,
        "audio_mask":     audio_mask,
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
        audio_start_id:     int,
        cfg:                Config,
        reference_start_id: Optional[int] = None,
        reference_end_id:   Optional[int] = None,
    ):
        self.ds                 = hf_dataset
        self.tokenizer          = tokenizer
        self.audio_start_id     = audio_start_id
        self.reference_start_id = reference_start_id
        self.reference_end_id   = reference_end_id
        self.cfg                = cfg
        self.k                  = cfg.k_codebooks
        self.multispeaker       = cfg.multispeaker
        self.has_speakers       = "speaker_id" in hf_dataset.column_names

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
        if self.multispeaker and self.reference_start_id is not None:
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
            text_ids           = text_ids,
            tgt_codes          = tgt_codes,
            audio_start_id     = self.audio_start_id,
            k                  = self.k,
            codebook_size      = self.cfg.codebook_size,
            ref_codes          = ref_codes,
            reference_start_id = self.reference_start_id,
            reference_end_id   = self.reference_end_id,
        )


_CANONICAL_COLUMNS = {"id", "text", "speaker_id", "codes", "n_frames", "k_codebooks"}


def _normalize_schema(ds, repo_id: str):
    """
    Make a heterogeneous mix of mimi-codes datasets concat-compatible:
      - Single-speaker sets (ljspeech, jenny) have no `speaker_id` — add one
        constant tag (= dataset name) so concat schemas line up and the multispeaker
        ref-picker treats them as one speaker.
      - Cast `speaker_id` to string — VCTK/LibriTTS store it as int32, single-speaker
        synthetic tags are strings; concat needs the column type to match.
      - Drop any columns outside the canonical schema (e.g. VCTK's `accent`).
    """
    from datasets import Value
    if "speaker_id" not in ds.column_names:
        tag = repo_id.split("/")[-1].replace("-mimi-codes", "")
        ds = ds.add_column("speaker_id", [tag] * len(ds))
        logger.info(f"  {repo_id}: added synthetic speaker_id={tag!r} (single-speaker set)")
    if ds.features["speaker_id"].dtype != "string":
        ds = ds.cast_column("speaker_id", Value("string"))
    extra = [c for c in ds.column_names if c not in _CANONICAL_COLUMNS]
    if extra:
        ds = ds.remove_columns(extra)
        logger.info(f"  {repo_id}: dropped columns {extra}")
    return ds


def _load_hf_split(
    split: str,
    cfg:   Config,
):
    """Load and combine all datasets from cfg.hf_datasets, return the train|val partition.

    Per-dataset config (parallel lists, indexed by i):
      - hf_splits[i]   : comma-sep HF split names (concatenated within the same repo)
      - hf_weights[i]  : fraction in (0, 1].
                         1.0 → use every row every epoch.
                         <1.0 → per-epoch stratified-by-speaker subsample of that fraction
                                (different rows seen each epoch; many speakers ≈ stratified,
                                 single speaker ≈ random row-level subsample).

    Subsampling is NOT applied here — it happens per-epoch via EpochStratifiedSampler
    in get_dataloader. We return the full combined dataset plus per-source row ranges.

    Returns:
        combined:    HuggingFace Dataset with all rows from all sources concatenated
        source_meta: list of (start_row, end_row, weight) per source (in concat order)
    """
    from datasets import load_dataset, concatenate_datasets

    n_ds = len(cfg.hf_datasets)
    weights = list(cfg.hf_weights) + [1.0] * max(0, n_ds - len(cfg.hf_weights))

    loaded = []
    for repo, splits_str, weight in zip(cfg.hf_datasets, cfg.hf_splits, weights):
        split_names = [s.strip() for s in splits_str.split(",")]
        parts = [load_dataset(repo, split=s) for s in split_names]
        ds    = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        ds    = _normalize_schema(ds, repo)
        loaded.append((repo, splits_str, ds, weight))
        logger.info(f"  {repo} [{splits_str}]: {len(ds)} rows  (weight={weight})")

    combined    = concatenate_datasets([d for _, _, d, _ in loaded]) if len(loaded) > 1 else loaded[0][2]
    source_meta = []
    cursor = 0
    for _, _, d, w in loaded:
        source_meta.append((cursor, cursor + len(d), w))
        cursor += len(d)

    use_explicit_val = bool(cfg.hf_val_datasets)

    if split == "train":
        if use_explicit_val:
            return combined, source_meta
        # Fallback: carve val_fraction off the tail of combined train data.
        n = len(combined)
        n_val = max(1, int(n * cfg.val_fraction))
        train_ds = combined.select(range(n - n_val))
        # Truncate source_meta to the train range (drop any sources beyond train cutoff).
        truncated_meta = []
        for s, e, w in source_meta:
            if s >= n - n_val:
                break
            truncated_meta.append((s, min(e, n - n_val), w))
        return train_ds, truncated_meta

    if split == "val":
        if use_explicit_val:
            val_parts = []
            for repo, splits_str in zip(cfg.hf_val_datasets, cfg.hf_val_splits):
                split_names = [s.strip() for s in splits_str.split(",")]
                parts = [load_dataset(repo, split=s) for s in split_names]
                val_ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
                val_ds = _normalize_schema(val_ds, repo)
                val_parts.append(val_ds)
                logger.info(f"  val: {repo} [{splits_str}]: {val_ds.num_rows} rows")
            val_combined = concatenate_datasets(val_parts) if len(val_parts) > 1 else val_parts[0]
            # Val never subsamples — single "source" with weight 1.0
            return val_combined, [(0, len(val_combined), 1.0)]
        # Fallback: tail fraction of combined train data.
        n = len(combined)
        n_val = max(1, int(n * cfg.val_fraction))
        val_ds = combined.select(range(n - n_val, n))
        return val_ds, [(0, len(val_ds), 1.0)]

    raise ValueError(f"Unknown split: {split!r}")


# ----------------------------------------------------------------------
# Per-epoch stratified sampler
# ----------------------------------------------------------------------

class EpochStratifiedSampler(torch.utils.data.Sampler):
    """
    Multi-source, per-epoch stratified-by-speaker sampler.

    For each source in `source_meta`, picks `weight` fraction of rows with stratification
    by `speaker_id` (≥1 row per speaker, picked fresh each epoch). Sources with weight=1.0
    contribute every row each epoch.

    Indices yielded are LOCAL to the dataset (`HFMimiDataset`) — i.e. into `dataset.indices`.
    """
    def __init__(
        self,
        dataset:        "HFMimiDataset",
        source_meta:    List,            # list of (start_row, end_row, weight)
        shuffle:        bool = True,
        base_seed:      int  = 0,
    ):
        self.dataset    = dataset
        self.shuffle    = shuffle
        self.base_seed  = base_seed
        self.epoch      = 0

        # Build per-source, per-speaker LOCAL-index mapping.
        # local_idx i ↔ raw row dataset.indices[i]; we need source assignment + speaker for that row.
        # `dataset.ds["speaker_id"]` reads the column for ALL raw rows once (fast for our sizes).
        speaker_col = dataset.ds["speaker_id"]
        source_of_row = [None] * (max(end for _, end, _ in source_meta) if source_meta else 0)
        for src_i, (s, e, _) in enumerate(source_meta):
            for r in range(s, e):
                source_of_row[r] = src_i

        # per_source_buckets[src_i] = {speaker_id: [local_idx, local_idx, ...]}
        self.per_source_buckets: List[dict] = [dict() for _ in source_meta]
        self.weights: List[float] = [w for _, _, w in source_meta]

        for local_idx, raw_row in enumerate(dataset.indices):
            src_i = source_of_row[raw_row]
            if src_i is None:
                continue  # row not in any tracked source (shouldn't happen)
            sp = speaker_col[raw_row]
            self.per_source_buckets[src_i].setdefault(sp, []).append(local_idx)

        # Approximate length once (per-epoch length will fluctuate by ±1 per speaker due to rounding).
        self._length = sum(self._target_per_source(i) for i in range(len(source_meta)))

        # Log mix summary
        for src_i, (s, e, w) in enumerate(source_meta):
            n_kept = sum(len(v) for v in self.per_source_buckets[src_i].values())
            n_speakers = len(self.per_source_buckets[src_i])
            target = self._target_per_source(src_i)
            tag = "all" if w >= 1.0 else f"{w:.0%}/spk"
            logger.info(f"  sampler src{src_i}: rows[{s}:{e}] kept={n_kept} speakers={n_speakers} "
                        f"target={target}/epoch ({tag})")

    def _target_per_source(self, src_i: int) -> int:
        w = self.weights[src_i]
        buckets = self.per_source_buckets[src_i]
        if w >= 1.0:
            return sum(len(v) for v in buckets.values())
        return sum(max(1, int(round(len(v) * w))) for v in buckets.values())

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.base_seed * 100003 + self.epoch)
        all_indices: list = []
        for src_i, w in enumerate(self.weights):
            buckets = self.per_source_buckets[src_i]
            if w >= 1.0:
                for v in buckets.values():
                    all_indices.extend(v)
            else:
                for v in buckets.values():
                    n = max(1, int(round(len(v) * w)))
                    if n >= len(v):
                        all_indices.extend(v)
                    else:
                        all_indices.extend(rng.sample(v, n))
        if self.shuffle:
            rng.shuffle(all_indices)
        return iter(all_indices)

    def __len__(self):
        return self._length


def make_collate_fn(codebook_size: int):
    """Build a collator closed over `codebook_size`, which is the AUDIO_PAD index."""
    AUDIO_PAD = codebook_size

    def collate_fn(batch: List[dict]) -> dict:
        max_len = max(b["input_ids"].shape[0] for b in batch)
        k       = batch[0]["audio_codes"].shape[1]

        input_ids_list      = []
        audio_codes_list    = []
        audio_mask_list     = []
        labels_list         = []
        attention_mask_list = []

        for b in batch:
            L   = b["input_ids"].shape[0]
            pad = max_len - L

            input_ids_list.append(F.pad(b["input_ids"], (0, pad), value=0))

            if pad > 0:
                pad_codes  = torch.full((pad, k), AUDIO_PAD, dtype=torch.long)
                pad_labels = torch.full((pad, k), -100,      dtype=torch.long)
                audio_codes_list.append(torch.cat([b["audio_codes"], pad_codes],  dim=0))
                labels_list.append(     torch.cat([b["labels"],      pad_labels], dim=0))
            else:
                audio_codes_list.append(b["audio_codes"])
                labels_list.append(     b["labels"])

            audio_mask_list.append(F.pad(b["audio_mask"].long(), (0, pad), value=0).bool())
            attention_mask_list.append(F.pad(b["attention_mask"], (0, pad), value=0))

        return {
            "input_ids":      torch.stack(input_ids_list),
            "audio_codes":    torch.stack(audio_codes_list),
            "audio_mask":     torch.stack(audio_mask_list),
            "labels":         torch.stack(labels_list),
            "attention_mask": torch.stack(attention_mask_list),
        }

    return collate_fn


def get_dataloader(
    split:              str,
    tokenizer,
    audio_start_id:     int,
    cfg:                Config,
    shuffle:            bool = True,
    reference_start_id: Optional[int] = None,
    reference_end_id:   Optional[int] = None,
):
    """
    Returns (dataloader, sampler_or_None). The sampler is non-None when any source has
    weight<1.0 — in that case the trainer should call sampler.set_epoch(e) at the start
    of each epoch to refresh the per-epoch stratified subsample.
    """
    hf_ds, source_meta = _load_hf_split(split, cfg)
    dataset = HFMimiDataset(
        hf_ds,
        tokenizer          = tokenizer,
        audio_start_id     = audio_start_id,
        cfg                = cfg,
        reference_start_id = reference_start_id,
        reference_end_id   = reference_end_id,
    )

    needs_sampler = any(w < 1.0 for _, _, w in source_meta)
    if needs_sampler:
        sampler = EpochStratifiedSampler(dataset, source_meta, shuffle=shuffle)
        loader = DataLoader(
            dataset,
            batch_size         = cfg.batch_size,
            sampler            = sampler,
            num_workers        = cfg.num_workers,
            pin_memory         = cfg.pin_memory,
            prefetch_factor    = cfg.prefetch_factor if cfg.num_workers > 0 else None,
            persistent_workers = cfg.num_workers > 0,
            collate_fn         = make_collate_fn(cfg.codebook_size),
        )
        return loader, sampler

    loader = DataLoader(
        dataset,
        batch_size         = cfg.batch_size,
        shuffle            = shuffle,
        num_workers        = cfg.num_workers,
        pin_memory         = cfg.pin_memory,
        prefetch_factor    = cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers = cfg.num_workers > 0,
        collate_fn         = make_collate_fn(cfg.codebook_size),
    )
    return loader, None


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from config import Config

    cfg = Config(batch_size=2, num_workers=0, max_audio_frames=300)
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "<|audio_start|>", "<|reference_start|>", "<|reference_end|>",
        ]
    })
    audio_start_id     = tokenizer.convert_tokens_to_ids("<|audio_start|>")
    reference_start_id = tokenizer.convert_tokens_to_ids("<|reference_start|>")
    reference_end_id   = tokenizer.convert_tokens_to_ids("<|reference_end|>")

    loader, _sampler = get_dataloader(
        "train", tokenizer, audio_start_id, cfg, shuffle=False,
        reference_start_id=reference_start_id, reference_end_id=reference_end_id,
    )
    batch = next(iter(loader))

    print(f"input_ids:      {batch['input_ids'].shape}")
    print(f"audio_codes:    {batch['audio_codes'].shape}  (PAD count: {(batch['audio_codes'] == cfg.codebook_size).sum().item()})")
    print(f"audio_mask:     {batch['audio_mask'].shape}  (True count: {batch['audio_mask'].sum().item()})")
    print(f"labels:         {batch['labels'].shape}  (supervised count: {(batch['labels'] >= 0).sum().item()})")
    print(f"attention_mask: {batch['attention_mask'].shape}")
    valid_labels = batch["labels"][batch["labels"] >= 0]
    print(f"label range: [{valid_labels.min().item()}, {valid_labels.max().item()}]  (expected [0, 2048])")
