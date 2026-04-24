# Wren-TTS

Text-to-speech model for the [Wren](https://github.com/shangeth/wren-tts) series of
small (<3B) multimodal speech LLMs. Generates [Kyutai Mimi](https://huggingface.co/kyutai/mimi)
neural-codec tokens from text with an autoregressive LLM backbone, then decodes to
24 kHz waveform.

```
text → tokenizer → LLM backbone → k Mimi-code heads → Mimi decoder → 24 kHz audio
```

## Released

**Model:** [shangeth/Wren-TTS-360M-v1](https://huggingface.co/shangeth/Wren-TTS-360M-v1) —
SmolLM2-360M backbone, trained on LibriTTS-R train-clean-{100,360} + train-other-500.
Multispeaker — requires a reference audio clip at inference for voice conditioning.

**In progress:** v1.1 — fine-tune of v1 with broader speaker coverage (VCTK, Jenny added; LibriTTS-R/LJSpeech replayed at 10%/epoch).

## Quickstart — inference

```bash
pip install torch torchaudio transformers datasets
```

> A reference audio clip is **required**. The model is multispeaker-only.

```python
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor

model_id  = "shangeth/Wren-TTS-360M-v1"
device    = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model     = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()

# Reference voice (one LibriSpeech test-clean clip; swap for your own .wav)
sample  = next(iter(load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True)))
ref_wav = torch.from_numpy(np.asarray(sample["audio"]["array"], dtype=np.float32)).unsqueeze(0)
ref_sr  = sample["audio"]["sampling_rate"]
ref_codes = model.encode_audio(ref_wav, ref_sr)[:, :150]

inputs = processor("Hello world, how are you today?")
inputs = {k: v.to(device) for k, v in inputs.items()}

waveform = model.generate(
    **inputs,
    ref_codes=ref_codes,
    max_audio_frames=200, min_audio_frames=2,
    temperature=0.8, top_k=50, top_p=0.9,
    output_audio=True,
)
processor.save_audio(waveform, "out.wav")
```

Sampling tips + more usage: see the [model card](https://huggingface.co/shangeth/Wren-TTS-360M-v1).

## Architecture

- **Backbone:** any HF causal LM (default [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M))
- **Audio tokenizer:** Mimi @ 24 kHz, 12.5 fps, 2048-entry codebooks
- **Codebooks used:** all 8 Mimi codebooks (`k_codebooks=8`)
- **Layout:** MusicGen-style **delay pattern** — at each step, k summed codebook input embeddings → k parallel heads. Codebook q at frame f lives at step `s = f + q`. Sequence length is `T + k − 1` instead of `T × k`.
- **Per-codebook input tables:** `Embedding(2049, hidden)` — extra row = `AUDIO_PAD` for sequence edges
- **Per-codebook output heads:** `Linear(hidden, 2048)` for cb1..cb7. cb0 gets `Linear(hidden, 2049)` with the extra class = `AUDIO_EOS` (stop token)
- **Speaker conditioning (required):** prepend `<|reference_start|> ref_codes <|reference_end|>` to the prompt

## Training

```bash
git clone https://github.com/shangeth/wren-tts
cd wren-tts
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login           # only needed for private datasets / pushing
```

Training streams Mimi-encoded codes from a HuggingFace dataset repo — no local
`data/` or extraction step required. Parquet files are cached under
`~/.cache/huggingface/` on first call.

Recommended path: launch from a YAML in `experiments/`:

```bash
# v1: from-scratch on LibriTTS-R (~22 h on A100-40GB)
python train.py --config experiments/wren-tts-360m-v1.yaml

# v1.1: fine-tune of v1 — adds VCTK + Jenny, replays LibriTTS-R/LJSpeech at 10%/epoch
python train.py --config experiments/wren-tts-360m-v1.1.yaml
```

Key flags (from `Config` defaults):

| Flag | Default | Notes |
|---|---|---|
| `--hf_datasets` | `[shangeth/librispeech-mimi-codes]` | parallel list of HF dataset repos |
| `--hf_splits` | `[train_clean_100,train_clean_360]` | parallel list of comma-sep splits per dataset |
| `--hf_weights` | `[1.0]` | per-dataset fraction. `<1.0` → per-epoch stratified-by-speaker subsample (resampled fresh each epoch) |
| `--llm_name` | `HuggingFaceTB/SmolLM2-360M` | any causal LM |
| `--k_codebooks` | `8` | Mimi codebooks used (delay pattern makes k=8 tractable) |
| `--batch_size` | `4` | per-step batch (bump to 16 on A100-40GB) |
| `--grad_accum_steps` | `4` | effective batch = batch_size × this |
| `--lr` | `1e-4` | peak LR; `3e-5` for fine-tune |
| `--epochs` | `50` | |
| `--multispeaker` | `true` | prepend a reference-audio block during training |
| `--eos_loss_weight` | `1.0` | bump to 50–100 to fix EOS underlearning |
| `--resume_from` | `None` | path to a `.pt` checkpoint |
| `--reset_optimizer` | `false` | with `resume_from`: load only model weights, reset optimizer/scheduler/step (for fine-tune) |

See `python train.py --help` for everything. YAML configs via `--config path.yaml`.

### Experiments

Drop a YAML into [`experiments/`](experiments/) per run:

```bash
python train.py --config experiments/ljspeech.yaml
python train.py --config experiments/ljspeech.yaml --batch_size 4    # CLI still overrides
```

Each run dumps the fully-resolved config (YAML + CLI merged) as `config.yaml`
into `cfg.checkpoint_dir` at startup, so checkpoints are self-describing:

```
checkpoints/ljspeech/
  config.yaml     ← snapshot of what ran
  train.log
  best.pt / last.pt / epoch_*.pt
```

The same config is embedded inside every `.pt` under the `"config"` key for
resume-time use.

For custom dataset mixes (combining LJSpeech + LibriSpeech, interleaving a new
corpus, etc.), edit [`dataset.py`](dataset.py) directly — the `_load_hf_split`
function is the single entry point to the HF dataset loader.

## Evaluation

Automatic metrics over a held-out set: **WER**, **CER**, **UTMOS**, **SECS**, **EER**.

```bash
pip install jiwer scikit-learn   # evaluation-only deps (UTMOS pulls via torch.hub)

python evaluate.py \
  --checkpoint checkpoints/librispeech/best.pt \
  --test_split test_clean \
  --n_samples 200 \
  --max_audio_frames 200 \
  --output_json results.json
```

Prints a one-line summary:

```
WER   : 0.1xxx
CER   : 0.0xxx
UTMOS : 3.xx ± 0.xx
SECS  : 0.xx ± 0.xx
EER   : 0.0xxx  (pos=200, neg=200, thr=0.xxx)
```

Per-sample results (including the Whisper hypothesis vs target text, useful for
eyeballing hallucination) are written to `--output_json`.

### Useful flags

| Flag | Default | Notes |
|---|---|---|
| `--whisper_model` | `openai/whisper-base` | Fast for iteration. Use `openai/whisper-large-v3` for release numbers. |
| `--n_samples` | `200` | n<50 has high variance with temperature sampling — don't draw conclusions from small runs. |
| `--max_audio_frames` | `300` | Lower to `200` for `test_clean` (~99% of utterances fit) — cuts hallucinated-tail cost. |
| `--amp_dtype` | `bf16` | bf16 autocast on CUDA. Helps on large-batch decoding; for single-sample autoregressive it can actually be *slower* due to dtype-juggling overhead. Try `none` (pure fp32) if bf16 doesn't help on your GPU. |
| `--temperature`, `--top_k`, `--top_p` | 0.8 / 50 / 0.9 | Match the sampling used in `inference.py`. |

### Timing

Rough per-sample cost on a single GPU: **10–15 s** (dominated by Wren's autoregressive generation at ~300 tokens/sample). Plus ~20 s of one-time model loads (Whisper, WavLM-SV, UTMOS). Budget for **n=200**: ~30–50 minutes end to end.

### Caveats

- **Reference audio for SECS is Mimi-decoded** from the cached codes (self-contained, no extra dataset download needed). Codec loss is ~constant across samples, so *relative* SECS across model versions is comparable. *Absolute* SECS is biased high because both generated and reference audio carry the same codec fingerprint.
- **EER negatives** are random different-speaker utterances from the same split. With n<100 pairs, EER has substantial variance.
- **Reference conditioning** masks most hallucination in evaluation. Since the model requires a reference at inference anyway, raw (unconditioned) WER/CER aren't meaningful targets.

Metric implementations live in [`metrics.py`](metrics.py) as reusable classes — import them directly for custom eval loops.

## Inference (CLI)

`--ref_audio` is effectively required — the model is trained multispeaker-only and
ref-less output quality is poor.

```bash
python inference.py \
  --checkpoint checkpoints/best.pt \
  --text "Hello world." \
  --ref_audio reference.wav \
  --out_dir out/
```

## Publishing to HuggingFace

```bash
export HF_TOKEN=hf_...
python hf/push.py \
  --repo_id shangeth/Wren-TTS-360M-v1 \
  --checkpoint checkpoints/best.pt \
  --private
```

`hf/push.py` converts a training checkpoint into a transformers-compatible layout:
`model.safetensors` + `config.json` (with `auto_map`) + tokenizer + `processor_config.json`
+ the three `trust_remote_code` files in [hf/](hf/).

## Repository layout

```
.
├── config.py          dataclass config, YAML + argparse
├── dataset.py         HF-Datasets-backed TTS dataset + dataloader
├── model.py           TTSModel (LLM + audio embeds/heads + generate)
├── trainer.py         training loop, checkpointing, logging
├── train.py           entry point
├── inference.py       text → speech CLI
├── evaluate.py        run WER/CER/UTMOS/SECS/EER on a held-out split
├── metrics.py         metric classes (WER, CER, UTMOS, SECS, EER)
├── mimi.py            MimiCodec wrapper (inference-time decode only)
├── experiments/       per-run YAML configs (see experiments/README.md)
└── hf/                HuggingFace model publishing
    ├── push.py        checkpoint → HF repo
    ├── MODEL_CARD.md  uploaded as README.md on the Hub
    ├── configuration_wren.py
    ├── modeling_wren.py
    └── processing_wren.py
```

## Related

- [wren-datasets](https://github.com/shangeth/wren-datasets) — Mimi-code
  extraction + publishing for LJSpeech and LibriSpeech.

## Known issues

- **EOS hallucination:** occasionally generates plausible speech *past* the input
  text. Mitigations at inference: raise `eos_bias` (e.g. 2–6), lower `max_audio_frames`,
  lower `temperature`. Reduced (not eliminated) in v1 by `eos_loss_weight=50` during training.
- **cb0 overfits earlier than cb3–cb7** — coarse semantic codebook is over-pressured
  in v1 (cb0 weight 2×). Addressed in v1.1 with uniform per-codebook weights.
- **English only**, audiobook-style prosody inherited from LibriTTS-R / LJSpeech.

## Citation

```bibtex
@misc{wren2026,
  title  = {Wren: A Family of Small Open-Weight Models for Unified Speech-Text Modelling},
  author = {Shangeth Rajaa},
  year   = {2026},
  url    = {https://github.com/shangeth/wren}
}
```

Please also cite the training corpora you use (LJSpeech, LibriSpeech, etc.).

## License

Apache-2.0. See [LICENSE](LICENSE).
