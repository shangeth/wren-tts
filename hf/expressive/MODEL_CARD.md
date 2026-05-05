---
license: cc-by-nc-4.0
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
- expressive
- style-tags
- multilingual
- wren
- mimi
- qwen2.5
- neural-codec
pipeline_tag: text-to-speech
base_model: shangeth/Wren-TTS-0.5B-multi
datasets:
- shangeth/expresso-mimi-codes-tagged
- shangeth/mls-mimi-codes
- shangeth/libritts-r-mimi-codes
- shangeth/vctk-mimi-codes
- shangeth/jenny-mimi-codes
- shangeth/ljspeech-mimi-codes
- ylacombe/expresso
- facebook/multilingual_librispeech
- mythicinfinity/libritts_r
- keithito/lj_speech
- CSTR-Edinburgh/vctk
- reach-vb/jenny_tts_dataset
---

# Wren-TTS-0.5B-multi-expressive

**Expressive** multilingual speech LLM in the Wren series. Fine-tuned from
[shangeth/Wren-TTS-0.5B-multi](https://huggingface.co/shangeth/Wren-TTS-0.5B-multi)
on the style-tagged [Expresso](https://huggingface.co/datasets/ylacombe/expresso) dataset
to add **23 emotion / delivery style tags** while retaining the 8-language
voice-cloning ability of the base model.

Generates [Kyutai Mimi](https://huggingface.co/kyutai/mimi) neural-codec tokens from
text using a [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) backbone,
then decodes to 24 kHz waveform with the Mimi decoder.

Supports the same **8 languages** as the base: English, German, French, Spanish,
Dutch, Italian, Polish, Portuguese — but style tags were only seen with English
text during fine-tuning.

## Style tags

Prepend any of these tags to the input text to condition the speaking style:

```
<angry>       <articulated>  <awe>         <bored>       <breath>
<calm>        <confused>     <desire>      <disgusted>   <enunciated>
<fast>        <fearful>      <happy>       <laugh>       <laughing>
<narration>   <projected>    <reduced>     <sad>         <sarcastic>
<sleepy>      <sympathetic>  <whisper>
```

Example:
```
<happy> Welcome to the show!
<whisper> I'll tell you a secret.
<sad> This is the saddest day of my life.
```

## Links

- **Base model:** [shangeth/Wren-TTS-0.5B-multi](https://huggingface.co/shangeth/Wren-TTS-0.5B-multi)
- **Demo Space:** [shangeth/Wren-TTS-0.5B-multi-expressive](https://huggingface.co/spaces/shangeth/Wren-TTS-0.5B-multi-expressive)
- **Training & inference code:** [github.com/shangeth/wren-tts](https://github.com/shangeth/wren-tts)
- **Wren research project:** [github.com/shangeth/wren](https://github.com/shangeth/wren)

## Architecture

Identical to the multi base — same delay-pattern Mimi prediction, same 8 codebooks,
same multispeaker reference conditioning. Only the LLM weights are fine-tuned.

```
[<style_tag>] text ──► Qwen2.5-0.5B ──► k=8 Mimi heads (delay) ──► Mimi decoder ──► 24 kHz
```

See the [base model card](https://huggingface.co/shangeth/Wren-TTS-0.5B-multi) for
the full architectural detail.

## Fine-tuning recipe

Per-epoch data mix (~52k rows ≈ 30 min on a single A100-40GB):

| dataset | weight | rows / epoch | role |
|---|---|---|---|
| [shangeth/expresso-mimi-codes-tagged](https://huggingface.co/datasets/shangeth/expresso-mimi-codes-tagged) | 1.0 | ~26,000 | the new domain (style tags) |
| [shangeth/mls-mimi-codes](https://huggingface.co/datasets/shangeth/mls-mimi-codes) | 0.004 | ~24,000 | multilingual replay |
| [shangeth/libritts-r-mimi-codes](https://huggingface.co/datasets/shangeth/libritts-r-mimi-codes) | 0.005 | ~2,400 | English speaker diversity |
| [shangeth/vctk-mimi-codes](https://huggingface.co/datasets/shangeth/vctk-mimi-codes) | 0.05 | ~2,200 | accent diversity |
| shangeth/jenny-mimi-codes | 0.0001 | ~2 | retention |
| shangeth/ljspeech-mimi-codes | 0.0002 | ~3 | retention |

- LR: 1e-5 with 2k warmup steps, cosine decay to 10k
- Effective batch size: 24
- Optimizer: AdamW (β=(0.9, 0.95)), wd=0.01, grad-clip 1.0
- AMP: enabled (autocast)
- Early stopped at **epoch 3 / 20** (val patience=2; best val_loss=3.826)

Replay weights tuned for **balanced per-voice exposure** rather than per-row volume,
so single-speaker datasets (Jenny, LJSpeech) don't overexpose their one voice.

## Usage

```bash
pip install torch torchaudio transformers datasets
```

> **A reference audio clip is required.** The model was trained multispeaker-only;
> without `ref_codes` it produces poor output.

```python
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor

model_id = "shangeth/Wren-TTS-0.5B-multi-expressive"
device   = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model     = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()

# Reference voice
sample  = next(iter(load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True)))
ref_wav = torch.from_numpy(np.asarray(sample["audio"]["array"], dtype=np.float32)).unsqueeze(0)
ref_sr  = sample["audio"]["sampling_rate"]
ref_codes = model.encode_audio(ref_wav, ref_sr)[:, :150]

# Style-tagged synthesis
texts = [
    "<happy> Welcome to the show, everybody!",
    "<sad> This is the saddest day of my life.",
    "<whisper> I will tell you a secret.",
    "<sarcastic> Oh sure, that worked out perfectly.",
    "<sleepy> I'm so tired, I can barely keep my eyes open.",
]
for i, text in enumerate(texts):
    inputs = processor(text)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    waveform = model.generate(
        **inputs,
        ref_codes=ref_codes,
        max_audio_frames=300, min_audio_frames=20,
        temperature=0.8, top_k=50, top_p=0.9,
        output_audio=True,
    )
    processor.save_audio(waveform, f"out_{i}.wav")
```

## Limitations & known issues

- **Limited expressive training data.** Expresso is a single English domain (~37 h),
  so style following is **decent, not perfect**. Quality varies across tags;
  high-frequency tags (`<happy>`, `<sad>`, `<whisper>`) work most reliably.
- **Style tags are English-only.** During fine-tune, tags only co-occurred with
  English text; behaviour with multilingual prompts is undefined and may degrade
  language quality.
- **Inherited from base:** hallucinated continuations, audiobook-style prosody on
  untagged English, varying per-language quality (German/Dutch/French strongest).
- **0.5B backbone** — quality is below frontier expressive TTS systems.

## License

**CC-BY-NC-4.0** — non-commercial use only. Inherited from the
[Expresso](https://huggingface.co/datasets/ylacombe/expresso) fine-tune data.
The base model and upstream components carry their own licenses; review before
redistribution.

## Citation

```bibtex
@misc{wren2026,
  title  = {Wren: A Family of Small Open-Weight Models for Unified Speech-Text Modelling},
  author = {Shangeth Rajaa},
  year   = {2026},
  url    = {https://github.com/shangeth/wren}
}
```
