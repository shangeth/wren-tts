"""
One-time fetcher for bundled reference clips.

Grabs 3 short LibriTTS-R test-clean clips from distinct speakers and writes them
to `samples/` next to app.py. LibriTTS-R is natively 24 kHz (matches Mimi) and
test.clean is held out from v1 training. Run once locally before pushing the Space.

  python fetch_samples.py
"""
from pathlib import Path

import numpy as np
import torch
import torchaudio
from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parent / "samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LABELS = ["ref_a.wav", "ref_b.wav", "ref_c.wav"]


def main():
    print("Streaming mythicinfinity/libritts_r (clean, test.clean) ...")
    ds = load_dataset(
        "mythicinfinity/libritts_r", "clean",
        split="test.clean", streaming=True,
    )

    seen_speakers = set()
    saved = 0
    for row in ds:
        if saved >= len(TARGET_LABELS):
            break
        sp  = row["speaker_id"]
        arr = np.asarray(row["audio"]["array"], dtype=np.float32)
        sr  = row["audio"]["sampling_rate"]
        dur = arr.shape[0] / sr
        if sp in seen_speakers or dur < 3.0 or dur > 7.0:
            continue
        seen_speakers.add(sp)
        wav = torch.from_numpy(arr).unsqueeze(0)
        out_path = OUT_DIR / TARGET_LABELS[saved]
        torchaudio.save(str(out_path), wav, sr)
        text = row.get("text_normalized") or row.get("text_original") or ""
        print(f"  {out_path.name}: speaker {sp}, {dur:.2f}s @ {sr} Hz, text: {text[:60]!r}")
        saved += 1

    if saved < len(TARGET_LABELS):
        raise RuntimeError(f"Only found {saved}/{len(TARGET_LABELS)} clips in range 3–7s")
    print(f"Saved {saved} reference clips to {OUT_DIR}/")


if __name__ == "__main__":
    main()
