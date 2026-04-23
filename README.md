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
SmolLM2-360M backbone, trained on LibriSpeech train-clean-{100,360}.

**Coming next:** v1.1 (EOS-reweighting fine-tune + LJSpeech voice), v2 (retrained on LibriTTS-R).

## Quickstart — inference

```bash
pip install torch torchaudio transformers
```

```python
import torch
from transformers import AutoModel, AutoProcessor

model_id  = "shangeth/Wren-TTS-360M-v1"
device    = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model     = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()

inputs = processor("Hello world, how are you today?")
inputs = {k: v.to(device) for k, v in inputs.items()}

waveform = model.generate(
    **inputs,
    max_audio_frames=200, min_audio_frames=2,
    temperature=0.8, top_k=50, top_p=0.9,
    output_audio=True,
)
processor.save_audio(waveform, "out.wav")
```

Voice cloning + sampling tips: see the [model card](https://huggingface.co/shangeth/Wren-TTS-360M-v1).

## Architecture

- **Backbone:** any HF causal LM (default [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M))
- **Audio tokenizer:** Mimi @ 24 kHz, 12.5 fps, 2048-entry codebooks
- **Codebooks used:** `k` per training config (default 3 of 8 extracted)
- **Interleaved layout:** `[ text | <audio_sep> | cb0_f0 cb1_f0 cb2_f0 | cb0_f1 … | AUDIO_EOS ]`
- **Per-codebook heads:** `Linear(hidden, 2048)`. `cb0` has one extra class (`AUDIO_EOS`).
- **Optional voice cloning:** prepend `<audio_start> ref_codes <audio_end>` to the prompt.

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

```bash
# Default: LibriSpeech train-clean-{100,360}
python train.py

# LJSpeech (single speaker)
python train.py --hf_dataset shangeth/ljspeech-mimi-codes --hf_splits train

# Custom LibriSpeech split mix
python train.py --hf_splits train_clean_100,train_clean_360,train_other_500
```

Common flags:

| Flag | Default | Notes |
|---|---|---|
| `--hf_dataset` | `shangeth/librispeech-mimi-codes` | any dataset with Wren's schema |
| `--hf_splits` | `train_clean_100,train_clean_360` | comma-sep list of HF splits to concat |
| `--lowercase_text` | true | LibriSpeech is pre-lowercased; turn off for mixed-case corpora |
| `--llm_name` | `HuggingFaceTB/SmolLM2-360M` | any causal LM |
| `--k_codebooks` | `3` | audio codebooks used during training (datasets ship 8) |
| `--batch_size` | `8` | per-step batch |
| `--grad_accum_steps` | `4` | effective batch = 32 |
| `--lr` | `1e-4` | peak LR |
| `--epochs` | `50` | |
| `--use_lora` | false | freeze backbone, train adapters + audio heads |
| `--resume_from` | `None` | path to a `.pt` checkpoint |

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
- **v1 hallucination** is largely suppressed under reference conditioning. To measure the raw (unconditioned) hallucination, you'd need to disable the reference block — not currently a flag.

Metric implementations live in [`metrics.py`](metrics.py) as reusable classes — import them directly for custom eval loops.

## Inference (CLI)

```bash
python inference.py \
  --checkpoint checkpoints/best.pt \
  --text "Hello world." \
  --out_dir out/

# Voice cloning
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

- **EOS hallucination (v1):** occasionally generates plausible speech *past* the
  input text. Caused by class imbalance in the `AUDIO_EOS` supervision. Mitigations
  at inference: raise `eos_bias`, lower `max_audio_frames`, lower `temperature`.
  Proper fix (cb0 cross-entropy reweighting) lands in v1.1.
- **English only**, audiobook-style prosody inherited from LibriSpeech.

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
