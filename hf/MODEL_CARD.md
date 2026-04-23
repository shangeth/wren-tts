---
license: apache-2.0
language:
- en
library_name: pytorch
tags:
- text-to-speech
- tts
- audio
- speech-synthesis
- wren
- mimi
- smollm2
- neural-codec
pipeline_tag: text-to-speech
datasets:
- openslr/librispeech_asr
---

# Wren-TTS-360M (v1)

**Wren** is a series of small (<3B) multimodal speech LLMs covering TTS, ASR, and
speech-language modelling. This is the first public checkpoint: **Wren-TTS-360M-v1**,
a text-to-speech model that generates [Kyutai Mimi](https://huggingface.co/kyutai/mimi)
neural-codec tokens from text and decodes them to 24 kHz waveform with the Mimi decoder.
The autoregressive backbone is [HuggingFaceTB/SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M).

An early research checkpoint — useful for experimentation, not production.

## Architecture

```
text ──► SmolLM2-360M ──► k audio-code heads ──► Mimi decoder ──► 24 kHz waveform
```

- **Backbone:** SmolLM2-360M (causal LM, embeddings shared with text input)
- **Audio tokenizer:** Mimi (`kyutai/mimi`), 12.5 fps, 2048-entry codebooks
- **Codebooks used:** 3 (of Mimi's 8 extractable)
- **Interleaved layout:** `[ text | <audio_sep> | cb0_f0 cb1_f0 cb2_f0 | cb0_f1 ... ]`
- **Per-codebook heads:** `Linear(hidden, 2048)`. `cb0` has one extra output (index 2048)
  used as the `AUDIO_EOS` stop token.
- **Optional speaker conditioning:** prepend `<audio_start> ref_codes <audio_end>` before the
  text prompt for zero-shot voice cloning from a short reference clip.

## Training data

- [LibriSpeech](https://www.openslr.org/12) `train-clean-100` + `train-clean-360` — ~460 h, multi-speaker, English audiobooks
- Trained for 5 epochs with effective batch size 32

Text is lowercased before tokenization. This v1 checkpoint was trained on LibriSpeech
only; LJSpeech was not included in this run.

## Usage

```bash
pip install torch torchaudio transformers
```

### Text-to-speech

```python
import torch
from transformers import AutoModel, AutoProcessor

model_id = "shangeth/Wren-TTS-360M-v1"
device   = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model     = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()

inputs = processor("Hello world, how are you today?")
inputs = {k: v.to(device) for k, v in inputs.items()}

waveform = model.generate(
    **inputs,
    max_audio_frames=200,
    min_audio_frames=2,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    output_audio=True,
)
processor.save_audio(waveform, "out.wav")
```

### Zero-shot voice cloning

```python
import torchaudio

ref_wav, sr = torchaudio.load("reference.wav")
ref_codes   = model.encode_audio(ref_wav, sr)[:, :150]   # cap at ~12 s of reference

waveform = model.generate(
    **inputs,
    ref_codes=ref_codes,
    output_audio=True,
    max_audio_frames=200,
    min_audio_frames=2,
    temperature=0.8, top_k=50, top_p=0.9,
)
processor.save_audio(waveform, "cloned.wav")
```

## Sampling tips

Defaults are `temperature=0.8`, `top_k=50`, `top_p=0.9`, `max_audio_frames=200` (~16 s).
If you hear the model generating extra words past the prompt (hallucination), try:

- Lower `temperature` (e.g. 0.6) and `top_p` (e.g. 0.8)
- Raise `eos_bias` (e.g. 2.0–6.0) to make the model more eager to stop
- Lower `max_audio_frames` to roughly `12 * len(text)`
- Set `min_audio_frames=1` for very short prompts

## Limitations & known issues

- **Hallucinated continuations:** the model sometimes generates plausible speech beyond
  the input text. This is a class-imbalance artifact of the `AUDIO_EOS` supervision
  (one EOS target per ~100–300 audio frames) and is being addressed in training.
- **English only.**
- **Limited expressiveness:** read-style prosody inherited from LibriSpeech audiobook data.
- **Small backbone (360M)** + modest training data — quality is below production TTS systems.
- **Val loss began to overfit** around epoch 5; v2 will add early stopping, LJSpeech,
  and EOS-class re-weighting.

## The Wren series

Wren is a family of compact (<3B parameter) multimodal speech LLMs — small enough to run
on a single consumer GPU, designed for open research on unified speech understanding and
synthesis. Planned siblings:

- **Wren-TTS** — text → speech (this release)
- **Wren-ASR** — speech → text
- **Wren-LM** — speech-language modelling / dialog
- **Wren-Omni** — unified ASR + TTS + LM in one checkpoint

All Wren models share the same design principles: small backbone LLM + neural audio codec,
open weights, simple PyTorch checkpoints, reproducible training recipes.

## Repository contents

| File | Purpose |
|---|---|
| `model.safetensors` | Model weights |
| `config.json` | `WrenConfig` (with `auto_map` for `trust_remote_code`) |
| `tokenizer.json` + friends | SmolLM2 tokenizer with Wren's 4 special tokens added |
| `processor_config.json` | `WrenProcessor` auto_map |
| `configuration_wren.py` | `WrenConfig(PretrainedConfig)` |
| `modeling_wren.py` | `WrenForTTS(PreTrainedModel)` — loads Mimi codec lazily on first generate |
| `processing_wren.py` | `WrenProcessor(ProcessorMixin)` — tokenize + `save_audio` |
| `README.md` | This model card |

## Citation

```bibtex
@misc{wren2026,
  title  = {Wren: A Family of Small Open-Weight Models for Unified Speech-Text Modelling},
  author = {Shangeth Rajaa},
  year   = {2026},
  url    = {https://github.com/shangeth/wren}
}

@inproceedings{panayotov2015librispeech,
  title     = {Librispeech: an ASR corpus based on public domain audio books},
  author    = {Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle = {ICASSP},
  year      = {2015}
}
```

## License

Apache-2.0 for the checkpoint weights and code in this repo.
Upstream components carry their own licenses — review before redistribution.
