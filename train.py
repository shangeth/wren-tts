"""
TTS training entry point.

Reads Mimi-encoded codes from a HuggingFace dataset repo (cfg.hf_dataset). Defaults
to LibriSpeech train-clean-{100,360}. Set --hf_dataset / --hf_splits to switch, or
edit dataset.py for custom mixes.

  python train.py
  python train.py --hf_dataset shangeth/ljspeech-mimi-codes --hf_splits train
  python train.py --config my_config.yaml
  python train.py --llm_name HuggingFaceTB/SmolLM2-360M --k_codebooks 4 --batch_size 4
"""

from transformers import AutoTokenizer

from config import parse_args
from dataset import get_dataloader
from model import TTSModel
from mimi import MimiCodec
from trainer import Trainer


def main():
    cfg = parse_args()

    print("=" * 60)
    print("TTS Training")
    print(f"  llm_name:         {cfg.llm_name}")
    print(f"  dataset:          {cfg.dataset}  (multispeaker={cfg.multispeaker})")
    print(f"  k_codebooks:      {cfg.k_codebooks}")
    print(f"  batch_size:       {cfg.batch_size}  (effective: {cfg.batch_size * cfg.grad_accum_steps})")
    print(f"  max_audio_frames: {cfg.max_audio_frames}  ({cfg.max_audio_frames / 12.5:.1f}s at 12.5fps)")
    print(f"  device:           {cfg.device}")
    print(f"  checkpoint_dir:   {cfg.checkpoint_dir}")
    print("=" * 60)

    # Tokenizer + audio special tokens
    # <|audio_sep|>:   text→audio boundary
    # <|audio_eos|>:   bookkeeping only; EOS signalled by cb0 head predicting index 2048
    # <|audio_start|>: opens reference audio block (multi-speaker)
    # <|audio_end|>:   closes reference audio block (multi-speaker)
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "<|audio_sep|>", "<|audio_eos|>",
            "<|audio_start|>", "<|audio_end|>",
        ]
    })
    audio_sep_id   = tokenizer.convert_tokens_to_ids("<|audio_sep|>")
    audio_start_id = tokenizer.convert_tokens_to_ids("<|audio_start|>")
    audio_end_id   = tokenizer.convert_tokens_to_ids("<|audio_end|>")

    # Model
    model = TTSModel(cfg, tokenizer)
    total     = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total params:     {total:.2f}M")
    print(f"Trainable params: {trainable:.2f}M")

    # Mimi codec (for audio logging in trainer; frozen)
    mimi_codec = MimiCodec(
        model_name=cfg.mimi_model_name,
        device=cfg.device,
        k_codebooks=cfg.k_codebooks,
    )

    # Dataloaders
    train_loader = get_dataloader(
        "train", tokenizer, audio_sep_id, cfg, shuffle=True,
        audio_start_id=audio_start_id, audio_end_id=audio_end_id,
    )
    val_loader = get_dataloader(
        "val", tokenizer, audio_sep_id, cfg, shuffle=False,
        audio_start_id=audio_start_id, audio_end_id=audio_end_id,
    )

    print(f"Train examples: {len(train_loader.dataset)}")
    print(f"Val examples:   {len(val_loader.dataset)}")

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        tokenizer=tokenizer,
        mimi_codec=mimi_codec,
    )
    trainer.train()


if __name__ == "__main__":
    main()
