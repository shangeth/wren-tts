"""Wren TTS configuration — transformers-compatible."""
from transformers import PretrainedConfig


class WrenConfig(PretrainedConfig):
    model_type = "wren"

    def __init__(
        self,
        llm_name:           str = "HuggingFaceTB/SmolLM2-360M",
        mimi_model_name:    str = "kyutai/mimi",
        k_codebooks:        int = 8,
        codebook_size:      int = 2048,
        vocab_size:         int = 49160,
        # Special-token IDs (tokenized positions in the text vocab)
        audio_start_id:     int = None,   # <|audio_start|> — text→target-audio boundary
        reference_start_id: int = None,   # <|reference_start|> — opens speaker-reference block
        reference_end_id:   int = None,   # <|reference_end|> — closes speaker-reference block
        sampling_rate:      int = 24000,
        **kwargs,
    ):
        self.llm_name           = llm_name
        self.mimi_model_name    = mimi_model_name
        self.k_codebooks        = k_codebooks
        self.codebook_size      = codebook_size
        self.vocab_size         = vocab_size
        self.audio_start_id     = audio_start_id
        self.reference_start_id = reference_start_id
        self.reference_end_id   = reference_end_id
        self.sampling_rate      = sampling_rate
        super().__init__(**kwargs)
