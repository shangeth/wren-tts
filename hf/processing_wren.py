"""
Wren processor: text tokenization + audio saving.

Text is lowercased to match training distribution. The `<|audio_sep|>` separator is
always appended so `model.generate(**processor(text))` "just works".
"""

from typing import List, Union

import torch
from transformers.processing_utils import ProcessorMixin


class WrenProcessor(ProcessorMixin):
    attributes        = ["tokenizer"]
    tokenizer_class   = "AutoTokenizer"

    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer)
        self.audio_sep_id   = tokenizer.convert_tokens_to_ids("<|audio_sep|>")
        self.audio_start_id = tokenizer.convert_tokens_to_ids("<|audio_start|>")
        self.audio_end_id   = tokenizer.convert_tokens_to_ids("<|audio_end|>")

    def __call__(
        self,
        text:           Union[str, List[str]],
        return_tensors: str = "pt",
        **kwargs,
    ):
        # Training data was lowercased; match that distribution.
        text = text.lower() if isinstance(text, str) else [t.lower() for t in text]

        enc = self.tokenizer(
            text,
            add_special_tokens = False,
            return_tensors     = return_tensors,
            **kwargs,
        )
        ids = enc["input_ids"]
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)

        # Append <|audio_sep|> as the final prompt token
        sep = torch.full(
            (ids.shape[0], 1),
            self.audio_sep_id,
            dtype=ids.dtype,
            device=ids.device,
        )
        ids = torch.cat([ids, sep], dim=1)
        return {"input_ids": ids}

    def save_audio(
        self,
        waveform:       torch.Tensor,
        path:           str,
        sampling_rate:  int = 24000,
    ) -> None:
        """Save a [1, T] or [T] waveform to disk at the given sample rate."""
        import torchaudio
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        torchaudio.save(path, waveform.cpu(), sampling_rate)
