"""
TTS inference: text → speech.

Usage:
  python inference.py --checkpoint checkpoints/best.pt --text "Hello world"
  python inference.py --checkpoint checkpoints/best.pt --text_file prompts.txt
  python inference.py --checkpoint checkpoints/best.pt --text "Hi" --temperature 0.8 --top_k 50
"""

import argparse
import time
from pathlib import Path

import torch
import torchaudio
from transformers import AutoTokenizer

from config import Config
from model import TTSModel
from mimi import MimiCodec


def load_model_and_tokenizer(
    checkpoint_path: str,
    user_cfg: "Config | None",
    device: torch.device,
):
    """
    Load a checkpoint. Config precedence:
      1. explicit user_cfg (from --config)
      2. config saved inside the checkpoint
      3. Config() defaults
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if user_cfg is not None:
        cfg = user_cfg
    elif "config" in ckpt:
        saved = ckpt["config"]
        cfg   = Config(**{k: v for k, v in saved.items() if k in Config.__dataclass_fields__})
        print(f"Using config saved in checkpoint (llm={cfg.llm_name}, k={cfg.k_codebooks})")
    else:
        cfg = Config()
        print("WARNING: no config in checkpoint and no --config given; using defaults "
              "— model architecture may not match.")

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "<|audio_sep|>", "<|audio_eos|>",
            "<|audio_start|>", "<|audio_end|>",
        ]
    })

    model = TTSModel(cfg, tokenizer).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded: {checkpoint_path}  ({params:.1f}M params, epoch {ckpt.get('epoch', '?')})")
    return model, tokenizer, cfg


@torch.no_grad()
def text_to_speech(
    text: str,
    model: TTSModel,
    tokenizer,
    mimi_codec: MimiCodec,
    max_audio_frames: int = 200,
    min_audio_frames: int = 20,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    ref_audio_path: "str | None" = None,
    max_ref_frames: int = 150,
) -> torch.Tensor:
    """
    Full TTS pipeline: text → codes → waveform.

    Args:
        ref_audio_path: optional path to a reference .wav for speaker conditioning.
    Returns:
        [1, T] float32 waveform at 24kHz
    """
    # Training data is lowercased (LJSpeech is mixed-case, LibriSpeech is ALL UPPER);
    # match that distribution at inference.
    text = text.lower()

    ref_codes      = None
    audio_start_id = None
    audio_end_id   = None

    if ref_audio_path is not None:
        import torchaudio as _ta
        audio_start_id = tokenizer.convert_tokens_to_ids("<|audio_start|>")
        audio_end_id   = tokenizer.convert_tokens_to_ids("<|audio_end|>")
        wav, sr = _ta.load(ref_audio_path)
        ref_codes = mimi_codec.encode(wav, sr)                        # [k, T_ref]
        T_ref     = min(ref_codes.shape[1], max_ref_frames)
        ref_codes = ref_codes[:, :T_ref]

    codes = model.generate(
        text=text,
        tokenizer=tokenizer,
        max_audio_frames=max_audio_frames,
        min_audio_frames=min_audio_frames,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        ref_codes=ref_codes,
        audio_start_id=audio_start_id,
        audio_end_id=audio_end_id,
    )  # [k, n_frames]
    waveform = mimi_codec.decode(codes)  # [1, T]
    return waveform


def main():
    parser = argparse.ArgumentParser(description="TTS inference")
    parser.add_argument("--checkpoint",       required=True,          help="Path to .pt checkpoint")
    parser.add_argument("--config",           default=None,           help="Optional YAML config path")
    parser.add_argument("--text",             default=None,           help="Input text to synthesize")
    parser.add_argument("--text_file",        default=None,           help="File with one text per line")
    parser.add_argument("--ref_audio",        default=None,           help="Reference .wav for speaker voice cloning")
    parser.add_argument("--out_dir",          default="inference_out")
    parser.add_argument("--max_audio_frames", type=int,   default=200)
    parser.add_argument("--min_audio_frames", type=int,   default=20)
    parser.add_argument("--max_ref_frames",   type=int,   default=150)
    parser.add_argument("--temperature",      type=float, default=0.8)
    parser.add_argument("--top_k",            type=int,   default=50)
    parser.add_argument("--top_p",            type=float, default=0.9)
    args = parser.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_cfg  = Config.load(args.config) if args.config else None

    model, tokenizer, cfg = load_model_and_tokenizer(args.checkpoint, user_cfg, device)
    mimi_codec = MimiCodec(
        model_name=cfg.mimi_model_name,
        device=str(device),
        k_codebooks=cfg.k_codebooks,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts: list = []
    if args.text:
        texts.append(args.text)
    if args.text_file:
        texts.extend(Path(args.text_file).read_text().strip().splitlines())

    if not texts:
        parser.error("Provide --text or --text_file")

    if args.ref_audio:
        print(f"Speaker reference: {args.ref_audio}")

    for i, text in enumerate(texts):
        print(f"\n[{i + 1}/{len(texts)}] {text[:80]}")
        t0 = time.time()

        waveform = text_to_speech(
            text=text,
            model=model,
            tokenizer=tokenizer,
            mimi_codec=mimi_codec,
            max_audio_frames=args.max_audio_frames,
            min_audio_frames=args.min_audio_frames,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            ref_audio_path=args.ref_audio,
            max_ref_frames=args.max_ref_frames,
        )

        dt       = time.time() - t0
        duration = waveform.shape[-1] / 24000
        rtf      = dt / max(duration, 1e-6)
        print(f"  Duration: {duration:.2f}s  |  RTF: {rtf:.2f}x  |  Time: {dt:.2f}s")

        out_path = out_dir / f"sample_{i:03d}.wav"
        torchaudio.save(str(out_path), waveform.cpu(), 24000)
        print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()


