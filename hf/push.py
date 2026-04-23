"""
Push a Wren checkpoint to the Hugging Face Hub in transformers-compatible format.

Converts the legacy training checkpoint (weights + optimizer + scheduler + cfg) into
a `AutoModel.from_pretrained` / `AutoProcessor.from_pretrained` layout:
  - model.safetensors            (weights only)
  - config.json                  (WrenConfig with auto_map)
  - tokenizer.json / *.txt       (tokenizer with our 4 special tokens)
  - processor_config.json        (auto_map to WrenProcessor)
  - configuration_wren.py        (remote code)
  - modeling_wren.py             (remote code)
  - processing_wren.py           (remote code)
  - README.md                    (model card)

Usage:
  huggingface-cli login
  python hf/push.py --repo_id shangeth/Wren-TTS-360M-v1 --checkpoint checkpoints/best.pt
  python hf/push.py --repo_id shangeth/Wren-TTS-360M-v1 --checkpoint checkpoints/best.pt --private

Dry-run (build the staging directory locally without uploading):
  python hf/push.py --repo_id shangeth/Wren-TTS-360M-v1 --local_dir ./staged
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoTokenizer


HF_DIR   = Path(__file__).resolve().parent
TTS_ROOT = HF_DIR.parent

# The trust_remote_code files are siblings in hf/
sys.path.insert(0, str(HF_DIR))
from configuration_wren import WrenConfig              # noqa: E402
from modeling_wren     import WrenForTTS               # noqa: E402


REMOTE_CODE_FILES = ["configuration_wren.py", "modeling_wren.py", "processing_wren.py"]

AUTO_MAP_MODEL = {
    "AutoConfig": "configuration_wren.WrenConfig",
    "AutoModel":  "modeling_wren.WrenForTTS",
}
AUTO_MAP_PROCESSOR = {
    "AutoProcessor": "processing_wren.WrenProcessor",
}


def _infer_resized_vocab(state_dict: dict) -> int:
    """The training code resizes embeddings with pad_to_multiple_of=8, so we take the
    actual shape from the checkpoint rather than re-deriving it from tokenizer length."""
    for key, val in state_dict.items():
        if key.endswith("embed_tokens.weight"):
            return val.shape[0]
    raise ValueError("Could not find embed_tokens.weight in checkpoint")


def convert_checkpoint_to_hf_repo(checkpoint_path: Path, staging: Path) -> None:
    """Convert a legacy training checkpoint into a transformers-style repo layout."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt       = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_cfg  = ckpt["config"]
    state_dict = ckpt["model"]

    # --- Tokenizer: base + our 4 special tokens ---
    tokenizer = AutoTokenizer.from_pretrained(saved_cfg["llm_name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_special_tokens({"additional_special_tokens": [
        "<|audio_sep|>", "<|audio_eos|>", "<|audio_start|>", "<|audio_end|>",
    ]})

    # --- Config ---
    vocab_size = _infer_resized_vocab(state_dict)
    cfg = WrenConfig(
        llm_name           = saved_cfg["llm_name"],
        mimi_model_name    = saved_cfg["mimi_model_name"],
        k_codebooks        = saved_cfg["k_codebooks"],
        codebook_size      = saved_cfg["codebook_size"],
        vocab_size         = vocab_size,
        audio_sep_id       = tokenizer.convert_tokens_to_ids("<|audio_sep|>"),
        audio_eos_token_id = tokenizer.convert_tokens_to_ids("<|audio_eos|>"),
        audio_start_id     = tokenizer.convert_tokens_to_ids("<|audio_start|>"),
        audio_end_id       = tokenizer.convert_tokens_to_ids("<|audio_end|>"),
    )
    cfg.auto_map = AUTO_MAP_MODEL
    print(f"Config built: vocab_size={vocab_size}, k={cfg.k_codebooks}")

    # --- Build model shell, load weights ---
    print("Building WrenForTTS from config (no pretrained backbone download) ...")
    model   = WrenForTTS(cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"  Unexpected keys ignored ({len(unexpected)}): {unexpected[:5]}...")
    if missing:
        print(f"  Missing keys (left at init) ({len(missing)}): {missing[:5]}...")

    # --- Write model (config.json + model.safetensors) ---
    print(f"Saving model to {staging} ...")
    model.save_pretrained(staging, safe_serialization=True)

    # --- Write tokenizer ---
    tokenizer.save_pretrained(staging)

    # --- Write processor_config.json with auto_map to WrenProcessor ---
    (staging / "processor_config.json").write_text(json.dumps({
        "processor_class": "WrenProcessor",
        "auto_map":        AUTO_MAP_PROCESSOR,
    }, indent=2))

    # --- Copy remote-code files from hf/ ---
    for name in REMOTE_CODE_FILES:
        shutil.copy2(HF_DIR / name, staging / name)
    print(f"Copied remote code: {REMOTE_CODE_FILES}")

    # --- Copy model card (hf/MODEL_CARD.md → README.md on the Hub) ---
    card = HF_DIR / "MODEL_CARD.md"
    if card.exists():
        shutil.copy2(card, staging / "README.md")
        print("Copied MODEL_CARD.md → README.md")

    print(f"Staging ready: {staging}")
    print(f"  Files: {sorted(p.name for p in staging.iterdir())}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id",        required=True, help="<user_or_org>/<model_name>")
    parser.add_argument("--checkpoint",     default="checkpoints/best.pt")
    parser.add_argument("--private",        action="store_true")
    parser.add_argument("--commit_message", default="Upload Wren checkpoint")
    parser.add_argument("--token",          default=None,
                        help="HF token; falls back to `huggingface-cli login` / HF_TOKEN env")
    parser.add_argument("--local_dir",      default=None,
                        help="If set, build the repo layout here and skip uploading.")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    if args.local_dir:
        staging = Path(args.local_dir).resolve()
        staging.mkdir(parents=True, exist_ok=True)
        convert_checkpoint_to_hf_repo(checkpoint_path, staging)
        print(f"\nDry run complete — inspect {staging} before pushing.")
        return

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=args.token)
    create_repo(args.repo_id, private=args.private, exist_ok=True, token=args.token)
    print(f"Repo ready: {args.repo_id} (private={args.private})")

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        convert_checkpoint_to_hf_repo(checkpoint_path, staging)
        print(f"\nUploading → https://huggingface.co/{args.repo_id}")
        api.upload_folder(
            folder_path    = str(staging),
            repo_id        = args.repo_id,
            commit_message = args.commit_message,
        )
        print("Done.")


if __name__ == "__main__":
    main()
