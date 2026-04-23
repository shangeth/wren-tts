import argparse
import yaml
import torch
from dataclasses import dataclass, field, asdict
from typing import Optional, List


@dataclass
class Config:
    # --- LLM backbone ---
    llm_name:              str   = "HuggingFaceTB/SmolLM2-360M"
    use_lora:              bool  = False
    lora_r:                int   = 16
    lora_alpha:            int   = 32
    lora_target_modules:   List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # --- Mimi codec ---
    mimi_model_name:       str   = "kyutai/mimi"
    k_codebooks:           int   = 3          # codebooks to USE during training (HF datasets ship 8)
    codebook_size:         int   = 2048       # Mimi codebook vocab size (fixed)

    # --- Data (HuggingFace datasets only) ---
    # Parallel lists — index i defines one dataset source:
    #   hf_datasets[i] : HF repo id
    #   hf_splits[i]   : comma-sep split names from that repo (e.g. "train_clean_100,train_clean_360")
    #   hf_weights[i]  : fraction of that dataset to use, in (0, 1]. 1.0 = full, 0.2 = 20%.
    #                    Useful for replay buffers when adding new data (avoid catastrophic forgetting).
    hf_datasets:           List[str]   = field(default_factory=lambda: ["shangeth/librispeech-mimi-codes"])
    hf_splits:             List[str]   = field(default_factory=lambda: ["train_clean_100,train_clean_360"])
    hf_weights:            List[float] = field(default_factory=lambda: [1.0])
    # Optional explicit val sources. If non-empty, used as-is for validation.
    # If empty, falls back to val_fraction tail of the combined training data.
    hf_val_datasets:       List[str]   = field(default_factory=list)
    hf_val_splits:         List[str]   = field(default_factory=list)
    multispeaker:          bool        = True   # prepend a reference-audio block at training time

    # EOS class reweighting — compensates for the 1:~200 class imbalance of AUDIO_EOS
    # vs normal codes in cb0's training targets. Raise if model hallucinates past prompts.
    eos_loss_weight:       float       = 1.0    # set to 50–100 to fix EOS underlearning
    val_fraction:          float     = 0.01
    max_text_tokens:       int       = 200
    max_audio_frames:      int       = 300    # 10 s at 12.5 fps
    max_ref_frames:        int       = 150
    batch_size:            int       = 4
    grad_accum_steps:      int       = 4      # effective batch = 32
    num_workers:           int       = 4
    pin_memory:            bool      = True
    prefetch_factor:       int       = 2

    # --- Optimizer ---
    lr:                    float = 1e-4
    weight_decay:          float = 0.01
    betas:                 List[float] = field(default_factory=lambda: [0.9, 0.95])
    lr_warmup_steps:       int   = 500
    lr_decay_steps:        int   = 20000

    # --- Training ---
    epochs:                int   = 50
    grad_clip:             float = 1.0
    use_amp:               bool  = True
    device:                str   = "cuda" if torch.cuda.is_available() else "cpu"
    compile_model:         bool  = False
    # Per-codebook loss weights (cb0..cb_{k-1}). Truncated / padded to k via last value.
    # Normalized as a weighted average, so [1, 0.5, 0.5] and [2, 1, 1] are equivalent.
    cb_loss_weights:       List[float] = field(default_factory=lambda: [1.0, 0.5, 0.5])

    # --- Checkpointing ---
    checkpoint_dir:        str   = "checkpoints"
    keep_last_n:           int   = 3
    resume_from:           Optional[str] = None

    # --- Logging ---
    logger:                Optional[str] = "tensorboard"   # "tensorboard" | "wandb" | null
    log_dir:               str   = "runs"
    log_audio_every:       int   = 1000
    wandb_project:         str   = "tts"
    wandb_run_name:        Optional[str] = None

    def __post_init__(self):
        # YAML parses bare scientific notation (e.g. 1e-4) as a string, not a float.
        # Coerce every annotated-float field so YAML quirks never silently break training.
        import typing
        for name, hint in typing.get_type_hints(self.__class__).items():
            if hint is float:
                val = getattr(self, name)
                if not isinstance(val, float):
                    object.__setattr__(self, name, float(val))

    def save(self, path: str):
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Wren-TTS training config")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    defaults = Config()
    for name, f in Config.__dataclass_fields__.items():
        val = getattr(defaults, name)
        if isinstance(val, bool):
            parser.add_argument(f"--{name}", default=None, action=argparse.BooleanOptionalAction)
        elif isinstance(val, (list, dict)):
            parser.add_argument(f"--{name}", type=lambda s: s.split(","), default=None)
        else:
            parser.add_argument(f"--{name}", type=type(val) if val is not None else str, default=None)

    args = parser.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    for name in Config.__dataclass_fields__:
        cli_val = getattr(args, name)
        if cli_val is not None:
            setattr(cfg, name, cli_val)

    return cfg
