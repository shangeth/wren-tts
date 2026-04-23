"""
Thin wrapper around the Kyutai Mimi codec for inference-time encode/decode.

Training does not use this — the dataset ships pre-extracted codes. This is only
needed by inference.py (for decoding model output → waveform) and for optional
voice-cloning reference encoding.

For offline corpus-wide extraction, see the `wren-datasets` repo.
"""

from typing import Dict

import torch
import torchaudio.transforms as T
from transformers import MimiModel


class MimiCodec:
    def __init__(
        self,
        model_name:  str = "kyutai/mimi",
        device:      str = "cuda",
        k_codebooks: int = 3,
    ):
        self.k         = k_codebooks
        self.device    = torch.device(device)
        self.target_sr = 24000

        self.model = MimiModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._resamplers: Dict[int, T.Resample] = {}

    def _resample(self, waveform: torch.Tensor, src_sr: int) -> torch.Tensor:
        if src_sr == self.target_sr:
            return waveform
        if src_sr not in self._resamplers:
            self._resamplers[src_sr] = T.Resample(src_sr, self.target_sr)
        return self._resamplers[src_sr](waveform)

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor, src_sample_rate: int) -> torch.LongTensor:
        """waveform: [1, T] or [T] float32 at src_sample_rate → LongTensor [k, n_frames]."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform = self._resample(waveform, src_sample_rate)
        x   = waveform.unsqueeze(0).to(self.device)  # [1, 1, T]
        out = self.model.encode(x, num_quantizers=self.k)
        return out.audio_codes[0].cpu()

    @torch.no_grad()
    def decode(self, codes: torch.LongTensor) -> torch.Tensor:
        """codes: LongTensor [k, n_frames] → Tensor [1, T] float32 at 24 kHz."""
        codes_b = codes.unsqueeze(0).to(self.device)
        out     = self.model.decode(codes_b)
        return out.audio_values[0].cpu()
