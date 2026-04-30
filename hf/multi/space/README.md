---
title: Wren-TTS-0.5B-multi
emoji: 🐦
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
python_version: "3.11"
pinned: false
license: apache-2.0
short_description: Multilingual voice-cloning TTS — 8 languages
models:
  - shangeth/Wren-TTS-0.5B-multi
---

# Wren-TTS-0.5B-multi — Gradio demo

Multilingual voice-cloning text-to-speech across **8 languages** —
English, German, French, Spanish, Dutch, Italian, Polish, Portuguese.

A small (<3B) multimodal speech LLM that generates
[Kyutai Mimi](https://huggingface.co/kyutai/mimi) neural-codec tokens from text,
using a [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) backbone,
then decodes to 24 kHz waveform.

The model is multispeaker-only — a reference audio clip (3–8 s of any speaker
in any of the 8 supported languages) is required to condition the voice. Three
LibriTTS-R test-clean English clips (held out from training) are bundled so you
can demo cross-lingual voice cloning out of the box (English voice → text in any
of the 8 languages).

- **Model:** [shangeth/Wren-TTS-0.5B-multi](https://huggingface.co/shangeth/Wren-TTS-0.5B-multi)
- **Code:** [github.com/shangeth/wren-tts](https://github.com/shangeth/wren-tts)

## Expected latency on the free CPU tier

| Phase | Time |
|---|---|
| Cold start (first Space visit after idle) | ~30–60 s — model download (~1.1 GB) + load |
| Per utterance (≤ 5 s of output) | ~30–90 s |

For faster inference, upgrade the Space hardware to a GPU tier.

## Local development

```bash
# Inside hf/multi/space/:
python fetch_samples.py         # one-time, pulls reference clips into samples/
pip install -r requirements.txt
python app.py                   # launches on http://127.0.0.1:7860
```

## Deploying

```bash
huggingface-cli login
python hf/multi/push_space.py                       # pushes to default Space repo
python hf/multi/push_space.py --space_id <user>/<name> --private
```

## License

Apache-2.0. Upstream model, codec, and backbone carry their own licenses — see the
model card.
