"""Wren TTS configuration — transformers-compatible."""
from transformers import PretrainedConfig


class WrenConfig(PretrainedConfig):
    model_type = "wren"

    def __init__(
        self,
        llm_name:           str = "HuggingFaceTB/SmolLM2-360M",
        mimi_model_name:    str = "kyutai/mimi",
        k_codebooks:        int = 3,
        codebook_size:      int = 2048,
        vocab_size:         int = 49160,
        audio_sep_id:       int = None,
        audio_eos_token_id: int = None,
        audio_start_id:     int = None,
        audio_end_id:       int = None,
        sampling_rate:      int = 24000,
        **kwargs,
    ):
        self.llm_name           = llm_name
        self.mimi_model_name    = mimi_model_name
        self.k_codebooks        = k_codebooks
        self.codebook_size      = codebook_size
        self.vocab_size         = vocab_size
        self.audio_sep_id       = audio_sep_id
        self.audio_eos_token_id = audio_eos_token_id
        self.audio_start_id     = audio_start_id
        self.audio_end_id       = audio_end_id
        self.sampling_rate      = sampling_rate
        super().__init__(**kwargs)
