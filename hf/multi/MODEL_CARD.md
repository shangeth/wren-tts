---
license: apache-2.0
language:
- en
- de
- fr
- es
- nl
- it
- pl
- pt
library_name: pytorch
tags:
- text-to-speech
- tts
- audio
- speech-synthesis
- multilingual
- wren
- mimi
- qwen2.5
- neural-codec
pipeline_tag: text-to-speech
datasets:
- shangeth/mls-mimi-codes
- shangeth/libritts-r-mimi-codes
- shangeth/vctk-mimi-codes
- shangeth/jenny-mimi-codes
- shangeth/ljspeech-mimi-codes
- facebook/multilingual_librispeech
- mythicinfinity/libritts_r
- keithito/lj_speech
- CSTR-Edinburgh/vctk
- reach-vb/jenny_tts_dataset
---

# Wren-TTS-0.5B-v1-multi

**Multilingual** speech LLM in the Wren series. Generates
[Kyutai Mimi](https://huggingface.co/kyutai/mimi) neural-codec tokens from text using a
[Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) backbone, then decodes to
24 kHz waveform with the Mimi decoder.

Supports **8 languages**: English, German, French, Spanish, Dutch, Italian, Polish, Portuguese.

For an English-only sibling on the same architecture, see
[shangeth/Wren-TTS-0.5B-v1](https://huggingface.co/shangeth/Wren-TTS-0.5B-v1) (when published).

## Links

- **Training & inference code:** [github.com/shangeth/wren-tts](https://github.com/shangeth/wren-tts)
- **Wren research project:** [github.com/shangeth/wren](https://github.com/shangeth/wren)
- **Dataset extraction (Mimi codes):** [github.com/shangeth/wren-datasets](https://github.com/shangeth/wren-datasets)
- **Demo Space:** [huggingface.co/spaces/shangeth/wren-tts-multi-demo](https://huggingface.co/spaces/shangeth/wren-tts-multi-demo)

## Why Qwen2.5-0.5B for multilingual?

The earlier [Wren-TTS-360M-v1](https://huggingface.co/shangeth/Wren-TTS-360M-v1) used
SmolLM2-360M, which was pretrained almost entirely on English text and tokenizes
non-Latin / diacritic-heavy languages poorly (per-character explosion). Qwen2.5-0.5B
ships a 151k-token multilingual vocabulary with efficient tokens for all 7 MLS languages.

Architecture is otherwise identical to the 360M v1 — same delay-pattern Mimi prediction,
same 8 codebooks, same multispeaker reference conditioning.

## Architecture

```
text ──► Qwen2.5-0.5B ──► k=8 Mimi heads (delay pattern) ──► Mimi decoder ──► 24 kHz
```

Within each step the k heads run **in parallel** (one forward, k logits), but across
steps they sit on a **delay** so cb_q at frame f is emitted at step `s = f + q`. The
delay preserves RVQ conditioning (cb_q sees cb_0..cb_{q-1} of the same frame in the
hidden state) while cutting sequence length to `T + k - 1` and LLM calls to **one
forward per step** — instead of k forwards per step in a flat interleaved layout.

- **Backbone:** Qwen2.5-0.5B (causal LM; transformer body ~358M params, multilingual 151k-token vocab)
- **Audio tokenizer:** Mimi (`kyutai/mimi`), 12.5 fps, 2048-entry codebooks
- **Codebooks used:** all 8 Mimi codebooks
- **Per-codebook input tables:** `Embedding(2049, hidden)` per cb (extra row = `AUDIO_PAD` for delay edges)
- **Per-codebook output heads:** `Linear(hidden, 2048)` for cb1..cb7. cb0 is `Linear(hidden, 2049)` — the extra class is `AUDIO_EOS` (stop token).
- **Speaker conditioning (required):** prepend `<|reference_start|> ref_codes <|reference_end|>`
  to the prompt; `ref_codes` is the Mimi encoding of a short reference clip in the
  target voice. Trained multispeaker-only — output quality is poor without a reference.
- **Cross-lingual voice transfer:** the reference voice and the target text language
  do not have to match. Take an English speaker's voice and synthesize French in
  that voice; quality varies (English voices generally transfer cleanly because
  English is mixed throughout multilingual training).

## Training data (~1.87M utterances total)

**English (~433k)**
- **VCTK** — 109 speakers, multiple English accents (~44k)
- **Jenny** — single high-quality voice (~21k)
- **LibriTTS-R** `train-clean-{100,360}` + `train-other-500` (~355k)
- **LJSpeech** — single speaker (~13k)

**Multilingual via [MLS](https://huggingface.co/datasets/facebook/multilingual_librispeech) (~1.45M)**
| Language | utterances |
|---|---|
| German (de)     | 469,942 |
| Dutch (nl)      | 374,287 |
| French (fr)     | 258,213 |
| Spanish (es)    | 220,701 |
| Italian (it)    |  59,623 |
| Portuguese (pt) |  37,533 |
| Polish (pl)     |  25,043 |

Mimi codes are pre-extracted and published as
[shangeth/mls-mimi-codes](https://huggingface.co/datasets/shangeth/mls-mimi-codes) +
the per-corpus mimi-codes datasets — this avoids redundant 6M-utterance encoding
during training.

Single-pass from-scratch training (no two-stage pretrain → fine-tune). Held-out validation
combines LibriTTS-R `dev_clean` + MLS `dev` (all 7 langs) + 5% per single-speaker English source.
All weights set to 1.0 (every row, every epoch, no subsampling). Trained on a single A100-40GB.

Text casing and punctuation are preserved. Pass text naturally — do not pre-lowercase.

## Usage

```bash
pip install torch torchaudio transformers datasets
```

> **A reference audio clip is required.** The model was trained multispeaker-only;
> without `ref_codes` it produces poor output. Any 3–12 s speech clip in any of the
> supported languages works as the voice reference.

```python
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor

model_id = "shangeth/Wren-TTS-0.5B-v1-multi"
device   = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model     = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()

# Reference voice — any short clip in any of the 8 supported languages.
sample  = next(iter(load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True)))
ref_wav = torch.from_numpy(np.asarray(sample["audio"]["array"], dtype=np.float32)).unsqueeze(0)
ref_sr  = sample["audio"]["sampling_rate"]
ref_codes = model.encode_audio(ref_wav, ref_sr)[:, :150]   # cap at ~12s

# Synthesize in any supported language — pass the text in that language naturally.
texts = [
    "Hello world, how are you today?",                       # en
    "Hallo Welt, wie geht es dir heute?",                    # de
    "Bonjour le monde, comment ça va aujourd'hui ?",         # fr
    "Hola mundo, ¿cómo estás hoy?",                          # es
    "Hallo wereld, hoe gaat het vandaag?",                   # nl
    "Ciao mondo, come stai oggi?",                           # it
    "Witaj świecie, jak się masz dzisiaj?",                  # pl
    "Olá mundo, como estás hoje?",                           # pt
]
for i, text in enumerate(texts):
    inputs = processor(text)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    waveform = model.generate(
        **inputs,
        ref_codes=ref_codes,
        max_audio_frames=300,
        min_audio_frames=20,
        temperature=0.8, top_k=50, top_p=0.9,
        output_audio=True,
    )
    processor.save_audio(waveform, f"out_{i}.wav")
```

## Sampling tips

Defaults: `temperature=0.8`, `top_k=50`, `top_p=0.9`, `max_audio_frames=300` (~24 s).
If the model hallucinates extra speech past the input text:

- Raise `eos_bias` — e.g. 2.0–6.0 — to make the model more eager to stop
- Lower `temperature` (0.6) and `top_p` (0.8)
- Set `max_audio_frames` ≈ `12 * len(text_in_chars)`
- Set `min_audio_frames=1` for very short prompts

## Limitations & known issues

- **Hallucinated continuations**: occasionally generates plausible speech past the
  input text. Mitigate with `eos_bias` at inference.
- **Language coverage:** only the 8 trained languages. Anything else produces
  noise / accented gibberish.
- **Per-language quality varies with data volume:** German / Dutch / French are
  strongest (largest training shares); Polish / Portuguese / Italian have less
  training data and may sound less natural.
- **Audiobook-style prosody** inherited from MLS / LibriTTS-R (both LibriVox-derived);
  not as expressive as conversational TTS.
- **Speaker-id collisions across MLS languages:** MLS uses per-language integer
  speaker IDs that may overlap numerically across languages. The training-time
  multispeaker reference picker bucketed colliding IDs together, so very occasionally
  the model saw a cross-language reference during training. In practice this had
  negligible effect on quality.
- **0.5B backbone** — quality is below frontier multilingual TTS systems.

## The Wren series

Wren is a family of compact (<3B parameter) multimodal speech LLMs — small enough
to run on a single consumer GPU, designed for open research on unified speech
understanding and synthesis. Planned siblings:

- **Wren-TTS** — text → speech (this release; English + multilingual variants)
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
| `tokenizer.json` + friends | Qwen2.5 tokenizer with Wren's 3 special tokens added |
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
```

## License

Apache-2.0 for the checkpoint weights and code in this repo.
Upstream components carry their own licenses — review before redistribution.
