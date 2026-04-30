"""
Gradio demo for shangeth/Wren-TTS-0.5B-multi.

Voice-cloning multilingual TTS across 8 languages (en, de, fr, es, nl, it, pl, pt):
user supplies text + a reference voice (bundled sample or upload), model generates
24 kHz speech in that voice — and the reference voice and target text language do
NOT have to match (cross-lingual voice cloning).

Runs on CPU (HF Space CPU-basic tier). Expect ~30–90 s per short utterance;
model load on cold start takes ~30–60 s (~1.1 GB checkpoint).
"""
import os
import tempfile
from pathlib import Path

# --- Monkey-patch workaround for gradio_client bool-schema bug ---
# Several gradio_client.utils functions do `"x" in schema` (`get_type`, `get_desc`),
# which crashes on bool schemas (JSON Schema's `additionalProperties: true`).
# Intercept the recursive entry point so non-dict schemas short-circuit to "Any".
# Must run BEFORE gradio is imported (which in turn imports gradio_client).
import gradio_client.utils as _gc_utils  # noqa: E402
_orig_json_to_py = _gc_utils._json_schema_to_python_type
def _safe_json_to_py(schema, defs=None):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_json_to_py(schema, defs)
_gc_utils._json_schema_to_python_type = _safe_json_to_py

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor


def _load_wav(path: str):
    """Read a WAV via soundfile and return (tensor [1, T] float32, sr)."""
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if arr.ndim == 2:                             # [T, C] — mix to mono
        arr = arr.mean(axis=1)
    return torch.from_numpy(arr).unsqueeze(0), sr


def _save_wav(path: str, wav_tensor: torch.Tensor, sr: int):
    """Save a [1, T] or [T] torch tensor as WAV via soundfile."""
    arr = wav_tensor.detach().cpu().float().numpy()
    if arr.ndim == 2:
        arr = arr.squeeze(0)                      # [1, T] -> [T]
    sf.write(str(path), arr, sr)

MODEL_ID = "shangeth/Wren-TTS-0.5B-multi"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
SR_OUT   = 24000

# ----- Load model (cold start) -----
print(f"Loading {MODEL_ID} on {DEVICE} ...")
# force_download=True bypasses any stale transformers_modules cache — safe to keep.
# Adds ~30s to cold start but guarantees we always get the current remote code + weights.
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, force_download=True)
model     = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, force_download=True).to(DEVICE).eval()
print("Model loaded.")

# ----- Bundled reference clips — pre-encoded once at startup -----
SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
BUNDLED_LABELS = {
    "Sample A (LibriTTS-R test-clean)": "ref_a.wav",
    "Sample B (LibriTTS-R test-clean)": "ref_b.wav",
    "Sample C (LibriTTS-R test-clean)": "ref_c.wav",
}

SAMPLE_CACHE: dict = {}
for label, fn in BUNDLED_LABELS.items():
    path = SAMPLES_DIR / fn
    if not path.exists():
        print(f"  WARN bundled sample missing: {path}")
        continue
    wav, sr = _load_wav(str(path))
    with torch.no_grad():
        codes = model.encode_audio(wav, sr)[:, :150]
    SAMPLE_CACHE[label] = codes
    print(f"  cached {label}: codes {tuple(codes.shape)}")

VOICE_CHOICES = list(SAMPLE_CACHE.keys()) + ["Upload my own"]


# ----- Generation -----

def synthesize(text, voice_label, uploaded_audio,
               temperature, top_k, top_p, eos_bias, max_frames, min_frames):
    if not text or not text.strip():
        return None, "⚠️ Please enter some text."

    print(f"[synth] voice_label={voice_label!r}  uploaded_audio={uploaded_audio!r}")

    # Resolve reference audio → Mimi codes
    if voice_label == "Upload my own":
        if uploaded_audio is None:
            return None, "⚠️ Please upload a reference .wav, or pick a bundled sample."
        try:
            wav, sr = _load_wav(uploaded_audio)
            print(f"[synth] upload loaded: shape={tuple(wav.shape)} sr={sr} "
                  f"dur={wav.shape[-1]/sr:.2f}s "
                  f"rms={wav.pow(2).mean().sqrt().item():.4f}")
            # Cap to ~10 s to limit encoding cost on CPU
            max_samples = int(sr * 10)
            wav = wav[:, :max_samples]
            with torch.no_grad():
                ref_codes = model.encode_audio(wav, sr)[:, :150]
        except Exception as e:
            return None, f"⚠️ Could not read reference audio: {e}"
    else:
        ref_codes = SAMPLE_CACHE.get(voice_label)
        if ref_codes is None:
            return None, f"⚠️ Bundled sample not found: {voice_label}"

    # Log reference fingerprint so user can confirm different uploads produce different codes.
    cb0_uniq = ref_codes[0].unique().numel() if ref_codes.numel() else 0
    print(f"[synth] ref_codes: shape={tuple(ref_codes.shape)} "
          f"cb0 first 10={ref_codes[0, :10].tolist()} "
          f"cb0 unique={cb0_uniq}")

    # Tokenize text
    inputs = processor(text)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        waveform = model.generate(
            **inputs,
            ref_codes=ref_codes.to(DEVICE),
            max_audio_frames=int(max_frames),
            min_audio_frames=int(min_frames),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            eos_bias=float(eos_bias),
            output_audio=True,
        )

    if waveform.numel() == 0:
        return None, "⚠️ Model produced no audio (EOS fired immediately). Try lowering eos_bias."

    dur = waveform.shape[-1] / SR_OUT
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    _save_wav(tmp.name, waveform, SR_OUT)
    return tmp.name, f"✅ Generated {dur:.2f}s"


# ----- UI -----

DESCRIPTION = f"""
# 🐦 Wren-TTS-0.5B-multi

Multilingual voice-cloning text-to-speech across **8 languages** —
en · de · fr · es · nl · it · pl · pt. A small (<3B) multimodal speech LLM
[Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) backbone +
[Kyutai Mimi](https://huggingface.co/kyutai/mimi) neural codec.

**1.** Enter text in any of the 8 languages · **2.** Pick a voice reference
(any speaker, any language) · **3.** Generate

Running on **{DEVICE.upper()}**. On the free CPU-basic tier, expect **~30–90 s per short utterance**.

- [Model](https://huggingface.co/shangeth/Wren-TTS-0.5B-multi) · [Code](https://github.com/shangeth/wren-tts)
"""

with gr.Blocks(title="Wren-TTS-0.5B-multi", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=3):
            text_in = gr.Textbox(
                label="Text to synthesize",
                lines=3,
                max_lines=6,
                placeholder="Hello world, how are you today?",
                value="Hello world, how are you today?",
            )
            voice = gr.Radio(
                choices=VOICE_CHOICES,
                value=VOICE_CHOICES[0] if VOICE_CHOICES else None,
                label="Voice reference",
                info="Pick a bundled sample, or upload a short clip (3–8 s) of any speaker. "
                     "Cross-lingual voice cloning supported — the reference language does not have to match the target text.",
            )
            _first_bundled_path = None
            if VOICE_CHOICES and VOICE_CHOICES[0] != "Upload my own":
                _first_bundled_path = str(SAMPLES_DIR / BUNDLED_LABELS[VOICE_CHOICES[0]])
            preview = gr.Audio(
                label="Preview selected reference",
                type="filepath",
                value=_first_bundled_path,
                interactive=False,
                visible=_first_bundled_path is not None,
            )
            upload = gr.Audio(
                label="Upload reference audio",
                type="filepath",
                visible=False,
            )

            with gr.Accordion("Advanced sampling", open=False):
                temperature = gr.Slider(0.1, 1.5, value=0.2, step=0.05, label="temperature")
                top_k       = gr.Slider(0, 200, value=50, step=1, label="top_k (0 = disable)")
                top_p       = gr.Slider(0.05, 1.0, value=0.9, step=0.05, label="top_p")
                eos_bias    = gr.Slider(
                    0.0, 10.0, value=2.0, step=0.5,
                    label="eos_bias",
                    info="Additive bias on EOS logit. Raise (2–6) if output runs past the text; lower if it cuts off mid-word.",
                )
                max_frames  = gr.Slider(
                    30, 300, value=150, step=10,
                    label="max_audio_frames",
                    info="12.5 fps → 150 = ~12 s cap. Lower to cap generation cost on CPU.",
                )
                min_frames  = gr.Slider(
                    1, 50, value=10, step=1, label="min_audio_frames",
                    info="Suppress EOS for this many steps to avoid immediate stop.",
                )

            go = gr.Button("🎙️ Generate", variant="primary", size="lg")

        with gr.Column(scale=2):
            out_audio = gr.Audio(label="Generated speech", type="filepath", autoplay=False)
            status    = gr.Textbox(label="Status", interactive=False, lines=2)

    # Show upload widget for "Upload my own"; otherwise update the preview with the selected sample.
    def _on_voice_change(v):
        if v == "Upload my own":
            return gr.update(visible=False), gr.update(visible=True)  # preview hidden, upload shown
        fn = BUNDLED_LABELS.get(v)
        path = str(SAMPLES_DIR / fn) if fn else None
        return gr.update(value=path, visible=(path is not None)), gr.update(visible=False)
    voice.change(_on_voice_change, inputs=voice, outputs=[preview, upload])

    go.click(
        synthesize,
        inputs=[text_in, voice, upload, temperature, top_k, top_p, eos_bias, max_frames, min_frames],
        outputs=[out_audio, status],
        concurrency_limit=1,
    )

    if VOICE_CHOICES and VOICE_CHOICES[0] != "Upload my own":
        v0 = VOICE_CHOICES[0]
        v1 = VOICE_CHOICES[1] if len(VOICE_CHOICES) > 2 else v0
        v2 = VOICE_CHOICES[2] if len(VOICE_CHOICES) > 3 else v0
        gr.Examples(
            label="Click an example to pre-fill (multilingual)",
            examples=[
                # English
                ["Hello world, how are you today?",                                v0],
                ["The quick brown fox jumps over the lazy dog.",                  v1],
                # German
                ["Hallo Welt, wie geht es dir heute?",                            v0],
                ["Der schnelle braune Fuchs springt über den faulen Hund.",       v1],
                # French
                ["Bonjour le monde, comment ça va aujourd'hui ?",                 v2],
                ["Le vif renard brun saute par-dessus le chien paresseux.",       v0],
                # Spanish
                ["Hola mundo, ¿cómo estás hoy?",                                  v1],
                # Dutch
                ["Hallo wereld, hoe gaat het vandaag?",                           v2],
                # Italian
                ["Ciao mondo, come stai oggi?",                                   v0],
                # Portuguese
                ["Olá mundo, como estás hoje?",                                   v1],
                # Polish
                ["Witaj świecie, jak się masz dzisiaj?",                          v2],
            ],
            inputs=[text_in, voice],
        )


if __name__ == "__main__":
    # show_api=False disables the /info endpoint. gradio_client<1.6 has a bug where
    # `get_type(schema)` crashes on bool schemas (additionalProperties: true), which
    # affects page loads. Disabling the API docs sidesteps the whole code path.
    demo.queue(max_size=8).launch(show_api=False)
