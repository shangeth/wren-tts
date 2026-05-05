---
title: Wren-TTS-0.5B-multi-expressive
emoji: 🎭
colorFrom: pink
colorTo: yellow
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
python_version: "3.11"
pinned: false
license: cc-by-nc-4.0
short_description: Expressive multilingual voice-cloning TTS — 23 style tags
models:
  - shangeth/Wren-TTS-0.5B-multi-expressive
---

# Wren-TTS-0.5B-multi-expressive — Gradio demo

Expressive multilingual voice-cloning text-to-speech with **23 style tags**
(e.g. `<happy>`, `<sad>`, `<whisper>`, `<confused>`, `<sarcastic>`, `<sleepy>`).

Fine-tuned from [`shangeth/Wren-TTS-0.5B-multi`](https://huggingface.co/shangeth/Wren-TTS-0.5B-multi)
on the [Expresso](https://huggingface.co/datasets/ylacombe/expresso) dataset
with multilingual replay (MLS, LibriTTS-R, VCTK, Jenny, LJSpeech) to retain
the 8-language voice-cloning ability of the base model.

Pick a style tag from the dropdown, enter text, choose a reference voice, and
generate. The reference voice and target text language do **not** have to match
(cross-lingual voice cloning).

- **Model:** [shangeth/Wren-TTS-0.5B-multi-expressive](https://huggingface.co/shangeth/Wren-TTS-0.5B-multi-expressive)
- **Code:** [github.com/shangeth/wren-tts](https://github.com/shangeth/wren-tts)

## Expected latency on the free CPU tier

| Phase | Time |
|---|---|
| Cold start (first Space visit after idle) | ~30–60 s — model download (~1.1 GB) + load |
| Per utterance (≤ 5 s of output) | ~30–90 s |

For faster inference, upgrade the Space hardware to a GPU tier.

## Limitations

- Trained on limited expressive data (Expresso, ~37 h, single English domain) —
  style following is **decent, not perfect**. Long utterances may drift out of
  the requested style.
- Style tags were only seen alongside English text in fine-tuning; behaviour
  with other languages is undefined.

## Local development

```bash
# Inside hf/expressive/space/:
python fetch_samples.py         # one-time, pulls reference clips into samples/
pip install -r requirements.txt
python app.py                   # launches on http://127.0.0.1:7860
```

## Deploying

```bash
huggingface-cli login
python hf/expressive/push_space.py                       # pushes to default Space repo
python hf/expressive/push_space.py --space_id <user>/<name> --private
```

## License

CC-BY-NC-4.0 (inherited from Expresso fine-tune data — non-commercial use only).
Upstream model, codec, and backbone carry their own licenses — see the model card.
