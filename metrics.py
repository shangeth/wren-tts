"""
Automatic TTS evaluation metrics for Wren.

Each metric is a class with a consistent interface:
    m = Metric()
    m.update(...)         # per-sample, returns the sample's score (or None if N/A)
    m.update(...)
    m.compute()           # corpus-level aggregate
    m.reset()

Heavyweight models (Whisper, UTMOS, WavLM-SV) are lazy-loaded on first update().
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as T


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _ensure_1d(audio: torch.Tensor) -> torch.Tensor:
    """Any shape → [T] mono."""
    if audio.dim() == 2:
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=False)
        else:
            audio = audio.squeeze(0)
    return audio


def _resample(audio: torch.Tensor, src_sr: int, tgt_sr: int) -> torch.Tensor:
    return audio if src_sr == tgt_sr else T.Resample(src_sr, tgt_sr)(audio)


# -----------------------------------------------------------------------------
# Whisper ASR (shared by WER, CER)
# -----------------------------------------------------------------------------

class WhisperASR:
    """Thin wrapper around transformers' ASR pipeline for TTS evaluation.

    Uses `pipeline("automatic-speech-recognition", ...)` instead of raw model.generate()
    because recent transformers versions require forced_decoder_ids / generation_config
    handling that the pipeline takes care of internally.
    """

    def __init__(self, model_name: str = "openai/whisper-base", device: str = "cuda"):
        self.model_name = model_name
        self.device     = device
        self._pipe      = None

    def _load(self):
        if self._pipe is None:
            from transformers import pipeline
            device_idx = 0 if str(self.device).startswith("cuda") else -1
            self._pipe = pipeline(
                task     = "automatic-speech-recognition",
                model    = self.model_name,
                device   = device_idx,
            )

    @torch.no_grad()
    def transcribe(self, audio: torch.Tensor, sample_rate: int, language: str = "en") -> str:
        self._load()
        x = _resample(_ensure_1d(audio), sample_rate, 16000).cpu().numpy()
        out = self._pipe(
            x,
            generate_kwargs={"language": language, "task": "transcribe"},
        )
        return out["text"].strip()


# -----------------------------------------------------------------------------
# Intelligibility: WER, CER
# -----------------------------------------------------------------------------

def _text_transforms():
    import jiwer
    return jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ])


def _char_transforms():
    import jiwer
    return jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfChars(),
    ])


class WER:
    """Word Error Rate via jiwer, computed at corpus level on normalized text."""

    def __init__(self):
        self._refs: List[str] = []
        self._hyps: List[str] = []

    def update(self, reference: str, hypothesis: str) -> float:
        self._refs.append(reference)
        self._hyps.append(hypothesis)
        return self._single(reference, hypothesis)

    def _single(self, reference: str, hypothesis: str) -> float:
        import jiwer
        return float(jiwer.wer(
            [reference], [hypothesis],
            reference_transform=_text_transforms(),
            hypothesis_transform=_text_transforms(),
        ))

    def compute(self) -> float:
        if not self._refs:
            return float("nan")
        import jiwer
        return float(jiwer.wer(
            self._refs, self._hyps,
            reference_transform=_text_transforms(),
            hypothesis_transform=_text_transforms(),
        ))

    def reset(self):
        self._refs.clear()
        self._hyps.clear()


class CER:
    """Character Error Rate via jiwer, computed at corpus level on normalized text."""

    def __init__(self):
        self._refs: List[str] = []
        self._hyps: List[str] = []

    def update(self, reference: str, hypothesis: str) -> float:
        self._refs.append(reference)
        self._hyps.append(hypothesis)
        return self._single(reference, hypothesis)

    def _single(self, reference: str, hypothesis: str) -> float:
        import jiwer
        return float(jiwer.cer(
            [reference], [hypothesis],
            reference_transform=_char_transforms(),
            hypothesis_transform=_char_transforms(),
        ))

    def compute(self) -> float:
        if not self._refs:
            return float("nan")
        import jiwer
        return float(jiwer.cer(
            self._refs, self._hyps,
            reference_transform=_char_transforms(),
            hypothesis_transform=_char_transforms(),
        ))

    def reset(self):
        self._refs.clear()
        self._hyps.clear()


# -----------------------------------------------------------------------------
# Naturalness: UTMOS (automatic MOS predictor)
# -----------------------------------------------------------------------------

class UTMOS:
    """UTMOS (automatic MOS predictor) via `tarepan/SpeechMOS` on torch.hub.

    Expects 16 kHz audio. Returns scores in ~[1, 5]. Treat ±0.1 as noise; only
    gaps >~0.3 are meaningful.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
        self._scores: List[float] = []

    def _load(self):
        if self._model is None:
            self._model = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0",
                "utmos22_strong",
                trust_repo=True,
            ).to(self.device).eval()

    @torch.no_grad()
    def update(self, audio: torch.Tensor, sample_rate: int) -> float:
        self._load()
        x     = _resample(_ensure_1d(audio), sample_rate, 16000).unsqueeze(0).to(self.device)
        score = self._model(x, 16000)
        val   = float(score.squeeze().item())
        self._scores.append(val)
        return val

    def compute(self) -> dict:
        if not self._scores:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        s = np.asarray(self._scores)
        return {"mean": float(s.mean()), "std": float(s.std()), "n": int(s.size)}

    def reset(self):
        self._scores.clear()


# -----------------------------------------------------------------------------
# Speaker similarity: SECS (WavLM-SV cosine)
# -----------------------------------------------------------------------------

class SECS:
    """Speaker Embedding Cosine Similarity via WavLM-SV.

    Stores per-sample scores. For voice-cloning evaluation, `update(gen, ref)`
    where `ref` is the conditioning reference audio. Values are in [-1, 1];
    typical floor for different speakers ~0.3, ceiling for same speaker ~0.75.
    """

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus-sv",
        device:     str = "cuda",
    ):
        self.model_name = model_name
        self.device     = device
        self._extractor = None
        self._model     = None
        self._scores: List[float] = []

    def _load(self):
        if self._model is None:
            from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
            self._extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
            self._model     = AutoModelForAudioXVector.from_pretrained(self.model_name).to(self.device).eval()

    @torch.no_grad()
    def embed(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        self._load()
        x   = _resample(_ensure_1d(audio), sample_rate, 16000).numpy()
        in_ = self._extractor(x, sampling_rate=16000, return_tensors="pt", padding=True).to(self.device)
        out = self._model(**in_)
        return F.normalize(out.embeddings, dim=-1)   # [1, D]

    def score(
        self,
        audio_a: torch.Tensor, sr_a: int,
        audio_b: torch.Tensor, sr_b: int,
    ) -> float:
        ea = self.embed(audio_a, sr_a)
        eb = self.embed(audio_b, sr_b)
        return float((ea * eb).sum(dim=-1).item())

    def update(
        self,
        gen_audio: torch.Tensor, sr_gen: int,
        ref_audio: torch.Tensor, sr_ref: int,
    ) -> float:
        s = self.score(gen_audio, sr_gen, ref_audio, sr_ref)
        self._scores.append(s)
        return s

    def compute(self) -> dict:
        if not self._scores:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        s = np.asarray(self._scores)
        return {"mean": float(s.mean()), "std": float(s.std()), "n": int(s.size)}

    def reset(self):
        self._scores.clear()


# -----------------------------------------------------------------------------
# Equal Error Rate (aggregate over same/different-speaker SECS scores)
# -----------------------------------------------------------------------------

class EER:
    """Equal Error Rate for speaker verification.

    Feed SECS scores with labels: positive = same speaker, negative = different
    speaker. At the threshold where FAR = FRR, EER = that shared error rate.
    Lower is better; ~0.01 is state-of-the-art for full SV models.
    """

    def __init__(self):
        self.positives: List[float] = []
        self.negatives: List[float] = []

    def update(self, score: float, is_same_speaker: bool):
        (self.positives if is_same_speaker else self.negatives).append(score)

    def compute(self) -> dict:
        if not self.positives or not self.negatives:
            return {"eer": float("nan"), "threshold": float("nan"),
                    "n_pos": len(self.positives), "n_neg": len(self.negatives)}

        from sklearn.metrics import roc_curve
        y_true  = np.concatenate([np.ones(len(self.positives)),
                                  np.zeros(len(self.negatives))])
        y_score = np.concatenate([np.asarray(self.positives),
                                  np.asarray(self.negatives)])
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        fnr = 1.0 - tpr
        idx = int(np.nanargmin(np.abs(fpr - fnr)))
        eer = float((fpr[idx] + fnr[idx]) / 2)
        return {
            "eer":       eer,
            "threshold": float(thresholds[idx]),
            "n_pos":     len(self.positives),
            "n_neg":     len(self.negatives),
        }

    def reset(self):
        self.positives.clear()
        self.negatives.clear()
