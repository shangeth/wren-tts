"""
Push the Wren-TTS-0.5B-multi-expressive Gradio Space to the Hugging Face Hub.

Uploads the full hf/expressive/space/ folder (app.py, README.md, requirements.txt,
fetch_samples.py, samples/) to a HF Space repo. Mirrors the model push.py
flow but with repo_type="space".

Prerequisite — bundled sample wavs must exist in space/samples/:
  cd hf/expressive/space
  python fetch_samples.py     # one-time

Usage:
  huggingface-cli login
  python hf/expressive/push_space.py                                # default Space repo
  python hf/expressive/push_space.py --space_id <user>/<name> --private
"""

import argparse
from pathlib import Path

HF_DIR    = Path(__file__).resolve().parent
SPACE_DIR = HF_DIR / "space"

DEFAULT_SPACE_ID = "shangeth/Wren-TTS-0.5B-multi-expressive"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space_id",       default=DEFAULT_SPACE_ID,
                        help="<user_or_org>/<space_name>  (default: %(default)s)")
    parser.add_argument("--private",        action="store_true")
    parser.add_argument("--commit_message", default="Update Wren-TTS-0.5B-multi-expressive Gradio demo")
    parser.add_argument("--token",          default=None,
                        help="HF token; falls back to `huggingface-cli login` / HF_TOKEN env")
    args = parser.parse_args()

    if not SPACE_DIR.exists():
        raise FileNotFoundError(f"Space folder missing: {SPACE_DIR}")

    samples_dir = SPACE_DIR / "samples"
    if not samples_dir.exists() or not any(samples_dir.glob("*.wav")):
        raise FileNotFoundError(
            f"No reference wavs in {samples_dir}. Run `python fetch_samples.py` first."
        )

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=args.token)

    # Spaces require space_sdk on creation; harmless if the repo already exists.
    create_repo(
        args.space_id,
        repo_type  = "space",
        space_sdk  = "gradio",
        private    = args.private,
        exist_ok   = True,
        token      = args.token,
    )
    print(f"Space ready: https://huggingface.co/spaces/{args.space_id} (private={args.private})")

    # Excludes that sneak into the folder during local dev — keep the repo clean.
    ignore = ["__pycache__/*", "*.pyc", ".DS_Store", ".gradio/*"]

    print(f"Uploading {SPACE_DIR} → spaces/{args.space_id}")
    api.upload_folder(
        folder_path     = str(SPACE_DIR),
        repo_id         = args.space_id,
        repo_type       = "space",
        commit_message  = args.commit_message,
        ignore_patterns = ignore,
    )
    print("Done.")
    print(f"\nSpace URL: https://huggingface.co/spaces/{args.space_id}")


if __name__ == "__main__":
    main()
