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
- mythicinfinity/libritts_r
- keithito/lj_speech
---

# Wren-TTS-360M (v1)

**Wren** is a series of small (<3B) multimodal speech LLMs covering TTS, ASR, and
speech-language modelling. **Wren-TTS-360M-v1** generates
[Kyutai Mimi](https://huggingface.co/kyutai/mimi) neural-codec tokens from text
using a [HuggingFaceTB/SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)
backbone, then decodes to 24 kHz waveform with the Mimi decoder.

An open research checkpoint — useful for experimentation, not production.

## Architecture

```
text ──► SmolLM2-360M ──► k=8 parallel Mimi heads ──► Mimi decoder ──► 24 kHz
```

- **Backbone:** SmolLM2-360M (causal LM; text + audio share the same backbone)
- **Audio tokenizer:** Mimi (`kyutai/mimi`), 12.5 fps, 2048-entry codebooks
- **Codebooks used:** all 8 Mimi codebooks
- **Layout:** **MusicGen-style delay pattern** — at each step, k summed codebook
  embeddings go in, k parallel heads predict k tokens out. Codebook q at frame f
  lives at step `s = f + q`, so same-frame RVQ conditioning is preserved via the delay.
- **Per-codebook input tables:** `Embedding(2049, hidden)` — extra row = `AUDIO_PAD` at
  sequence edges.
- **Per-codebook output heads:** `Linear(hidden, 2048)` for cb1..cb7.
  cb0 gets `Linear(hidden, 2049)` with the extra class = `AUDIO_EOS` (stop token).
- **Speaker conditioning (required):** prepend `<|reference_start|> ref_codes <|reference_end|>`
  to the prompt; `ref_codes` is the Mimi encoding of a short reference clip. The model was
  trained multispeaker-only and expects a reference at inference — without one, output quality
  is poor.

## Training data

- [LibriTTS-R](https://huggingface.co/datasets/mythicinfinity/libritts_r)
  `train-clean-{100,360}` + `train-other-500` — ~960 h multi-speaker English
- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) — ~24 h single speaker

Text casing and punctuation are preserved. Pass text naturally — do not pre-lowercase.

## Usage

```bash
pip install torch torchaudio transformers datasets
```

> **A reference audio clip is required.** The model was trained multispeaker-only; without
> `ref_codes` it produces poor output. Any 3–12 s English speech clip works as the voice reference.

```python
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor

model_id = "shangeth/Wren-TTS-360M-v1"
device   = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model     = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()

# Grab one LibriSpeech test-clean clip (~3.5s, not in training) as the reference voice.
# Swap this block for `torchaudio.load("your_reference.wav")` to use your own clip.
sample  = next(iter(load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True)))
ref_wav = torch.from_numpy(np.asarray(sample["audio"]["array"], dtype=np.float32)).unsqueeze(0)
ref_sr  = sample["audio"]["sampling_rate"]
ref_codes = model.encode_audio(ref_wav, ref_sr)[:, :150]   # cap at ~12s; encode_audio resamples to 24 kHz

# Tokenize the target text and generate speech in the reference voice
inputs = processor("Hello world, how are you today?")
inputs = {k: v.to(device) for k, v in inputs.items()}

waveform = model.generate(
    **inputs,
    ref_codes=ref_codes,
    max_audio_frames=200,
    min_audio_frames=2,
    temperature=0.8, top_k=50, top_p=0.9,
    output_audio=True,
)
processor.save_audio(waveform, "out.wav")
```

## Sampling tips

Defaults: `temperature=0.8`, `top_k=50`, `top_p=0.9`, `max_audio_frames=200` (~16 s).
If you hear the model generating extra speech past the intended text (hallucination):

- Raise `eos_bias` — e.g. 2.0–6.0 — to make the model more eager to stop
- Lower `temperature` (0.6) and `top_p` (0.8)
- Set `max_audio_frames` ≈ `12 * len(text_in_chars)`
- Set `min_audio_frames=1` for very short prompts

## Why delay pattern

Mimi uses **residual vector quantization (RVQ)**: cb0 is semantic, cb1..cb7 encode
successive residuals. cb_q is only meaningful given cb0..cb_{q-1}, so same-frame
conditioning matters.

A flat interleaved layout (`cb0_f0, cb1_f0, ..., cb0_f1, ...`) preserves that
conditioning best but balloons sequence length by `k×` and forces `k` autoregressive
LLM calls per frame. The delay pattern keeps RVQ conditioning (cb_q at frame f is
predicted from a hidden state that has already attended over cb0..cb_{q-1} of the
same frame) while cutting sequence length to `T + k - 1` and LLM calls to **one per
step** — enabling all 8 Mimi codebooks without blowing up context.

## Limitations & known issues

- **Hallucinated continuations**: occasionally generates plausible speech past the
  input text. Mitigate with `eos_bias` at inference.
- **English only.**
- **Audiobook-style prosody** inherited from LibriTTS-R; not as expressive as modern
  conversational TTS.
- **Small backbone (360M)** — quality is below frontier TTS systems.
- cb0 begins to overfit earlier than cb3–cb7; the released checkpoint is the
  best-epoch point (by overall val loss) from the full training run.

## The Wren series

Wren is a family of compact (<3B parameter) multimodal speech LLMs — small enough
to run on a single consumer GPU, designed for open research on unified speech
understanding and synthesis. Planned siblings:

- **Wren-TTS** — text → speech (this release)
- **Wren-ASR** — speech → text
- **Wren-LM** — speech-language modelling / dialog
- **Wren-Omni** — unified ASR + TTS + LM in one checkpoint

All Wren models share the same design principles: small backbone LLM + neural
audio codec, open weights, simple PyTorch checkpoints, reproducible training recipes.

## Repository contents

| File | Purpose |
|---|---|
| `model.safetensors` | Model weights |
| `config.json` | `WrenConfig` (with `auto_map` for `trust_remote_code`) |
| `tokenizer.json` + friends | SmolLM2 tokenizer with Wren's 3 special tokens added |
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

@inproceedings{koizumi2023libritts,
  title     = {LibriTTS-R: A Restored Multi-Speaker Text-to-Speech Corpus},
  author    = {Koizumi, Yuma and Zen, Heiga and Karita, Shigeki and Ding, Yifan
               and Yatabe, Kohei and Morioka, Nobuyuki and Bacchiani, Michiel and
               Zhang, Yu and Han, Wei and Bapna, Ankur},
  booktitle = {Interspeech},
  year      = {2023}
}
```

## License

Apache-2.0 for the checkpoint weights and code in this repo.
Upstream components carry their own licenses — review before redistribution.
