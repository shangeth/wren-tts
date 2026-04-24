"""
Gradio demo for shangeth/Wren-TTS-360M-v1.

Voice-cloning TTS: user supplies text + a reference voice (bundled sample or upload),
model generates 24 kHz speech in that voice.

Runs on CPU (HF Space CPU-basic tier). Expect ~30–90 s per short utterance;
model load on cold start takes ~30 s.
"""
import os
import tempfile
from pathlib import Path

import gradio as gr
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor

MODEL_ID = "shangeth/Wren-TTS-360M-v1"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
SR_OUT   = 24000

# ----- Load model (cold start) -----
print(f"Loading {MODEL_ID} on {DEVICE} ...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model     = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE).eval()
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
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # stereo → mono
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
            wav, sr = torchaudio.load(uploaded_audio)
            print(f"[synth] upload loaded: shape={tuple(wav.shape)} sr={sr} "
                  f"dur={wav.shape[-1]/sr:.2f}s "
                  f"rms={wav.pow(2).mean().sqrt().item():.4f}")
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
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
    torchaudio.save(tmp.name, waveform.cpu(), SR_OUT)
    return tmp.name, f"✅ Generated {dur:.2f}s"


# ----- UI -----

DESCRIPTION = f"""
# 🐦 Wren-TTS-360M v1

Voice-cloning text-to-speech. A small (<3B) multimodal speech LLM —
[SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) backbone +
[Kyutai Mimi](https://huggingface.co/kyutai/mimi) neural codec.

**1.** Enter text · **2.** Pick a voice reference · **3.** Generate

Running on **{DEVICE.upper()}**. On the free CPU-basic tier, expect **~30–90 s per short utterance**.

- [Model](https://huggingface.co/shangeth/Wren-TTS-360M-v1) · [Code](https://github.com/shangeth/wren-tts)
"""

with gr.Blocks(title="Wren-TTS-360M v1", theme=gr.themes.Soft()) as demo:
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
                info="Pick a bundled sample, or upload a short clip (3–8 s) of any English speaker.",
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
        gr.Examples(
            label="Click an example to pre-fill",
            examples=[
                ["Hello world, how are you today?",                  VOICE_CHOICES[0]],
                ["The quick brown fox jumps over the lazy dog.",    VOICE_CHOICES[0]],
                ["A journey of a thousand miles begins with a single step.", VOICE_CHOICES[1] if len(VOICE_CHOICES) > 2 else VOICE_CHOICES[0]],
            ],
            inputs=[text_in, voice],
        )


if __name__ == "__main__":
    demo.queue(max_size=8).launch()
