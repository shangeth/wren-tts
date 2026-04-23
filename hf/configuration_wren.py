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
        pattern:            str = None,      # "delay" (new) | "flat" (legacy v1); inferred if None
        # Delay-pattern token IDs (new models)
        audio_start_id:     int = None,      # <|audio_start|> — text→target boundary (delay)
                                             #                   OR ref-start under legacy flat
        reference_start_id: int = None,      # <|reference_start|> — delay only
        reference_end_id:   int = None,      # <|reference_end|>   — delay only
        # Legacy flat-pattern token IDs (v1 models on the Hub)
        audio_sep_id:       int = None,      # legacy <|audio_sep|> — flat text→audio boundary
        audio_end_id:       int = None,      # legacy <|audio_end|>  — flat ref-end
        audio_eos_token_id: int = None,      # legacy bookkeeping, unused at runtime
        sampling_rate:      int = 24000,
        **kwargs,
    ):
        self.llm_name           = llm_name
        self.mimi_model_name    = mimi_model_name
        self.k_codebooks        = k_codebooks
        self.codebook_size      = codebook_size
        self.vocab_size         = vocab_size
        # Infer pattern from presence of legacy fields when not explicit.
        if pattern is None:
            pattern = "flat" if audio_sep_id is not None else "delay"
        self.pattern            = pattern
        self.audio_start_id     = audio_start_id
        self.reference_start_id = reference_start_id
        self.reference_end_id   = reference_end_id
        self.audio_sep_id       = audio_sep_id
        self.audio_end_id       = audio_end_id
        self.audio_eos_token_id = audio_eos_token_id
        self.sampling_rate      = sampling_rate
        super().__init__(**kwargs)
