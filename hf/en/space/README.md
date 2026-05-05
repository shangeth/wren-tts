---
title: Wren-TTS-360M-en
emoji: 🐦
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
python_version: "3.11"
pinned: false
license: apache-2.0
short_description: Voice-cloning TTS — Mimi codec + SmolLM2-360M (English)
models:
  - shangeth/Wren-TTS-360M-en
---

# Wren-TTS-360M-en — Gradio demo

Voice-cloning text-to-speech (English). Small (<3B) multimodal speech LLM that
generates [Kyutai Mimi](https://huggingface.co/kyutai/mimi) neural-codec tokens
from text, using a [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)
backbone, then decodes to 24 kHz waveform.

The model is multispeaker-only — a reference audio clip (3–8 s of any English speaker)
is required to condition the voice. Three LibriTTS-R test-clean clips (held out from
training) are bundled with this Space so you can try it without uploading anything.

- **Model:** [shangeth/Wren-TTS-360M-en](https://huggingface.co/shangeth/Wren-TTS-360M-en)
- **Code:** [github.com/shangeth/wren-tts](https://github.com/shangeth/wren-tts)

## Expected latency on the free CPU tier

| Phase | Time |
|---|---|
| Cold start (first Space visit after idle) | ~30–45 s — model download + load |
| Per utterance (≤ 5 s of output) | ~30–90 s |

For faster inference, upgrade the Space hardware to a GPU tier.

## Local development

```bash
# Inside hf/en/space/:
python fetch_samples.py         # one-time, pulls 3 LibriSpeech test-clean clips into samples/
pip install -r requirements.txt
python app.py                   # launches on http://127.0.0.1:7860
```

## Deploying

```bash
huggingface-cli login
python hf/en/push_space.py                         # pushes to default Space repo
python hf/en/push_space.py --space_id <user>/<name> --private
```

## License

Apache-2.0. Upstream model, codec, and backbone carry their own licenses — see the
model card.
