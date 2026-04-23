"""
Wren processor: text tokenization + audio saving.

Text casing is preserved as-is. Pass text naturally ("Hello, World!") — the model
is trained on mixed-case data (LJSpeech mixed-case, LibriSpeech lowercase).
The text→target-audio boundary token is always appended so
`model.generate(**processor(text))` "just works".

For delay-pattern models (new), the boundary token is `<|audio_start|>`.
For legacy flat v1 models, it's `<|audio_sep|>` — we fall back to that name
if the tokenizer doesn't know `<|audio_start|>`.
"""

from typing import List, Union

import torch
from transformers.processing_utils import ProcessorMixin


def _lookup_id(tokenizer, *names):
    for n in names:
        tid = tokenizer.convert_tokens_to_ids(n)
        if tid is not None and tid != tokenizer.unk_token_id:
            return tid
    return None


class WrenProcessor(ProcessorMixin):
    attributes        = ["tokenizer"]
    tokenizer_class   = "AutoTokenizer"

    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer)
        # New-name first, legacy name fallback — handles both delay-pattern and v1 models.
        self.audio_start_id     = _lookup_id(tokenizer, "<|audio_start|>", "<|audio_sep|>")
        self.reference_start_id = _lookup_id(tokenizer, "<|reference_start|>", "<|audio_start|>")
        self.reference_end_id   = _lookup_id(tokenizer, "<|reference_end|>", "<|audio_end|>")

    def __call__(
        self,
        text:           Union[str, List[str]],
        return_tensors: str = "pt",
        **kwargs,
    ):
        enc = self.tokenizer(
            text,
            add_special_tokens = False,
            return_tensors     = return_tensors,
            **kwargs,
        )
        ids = enc["input_ids"]
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)

        # Append the text→target-audio boundary token as the final prompt token.
        sep = torch.full(
            (ids.shape[0], 1),
            self.audio_start_id,
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
