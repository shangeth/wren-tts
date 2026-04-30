# `experiments/`

One YAML per training run. Each file is a diff from `base.yaml` — only set the
fields you want to override. Keeps your history of runs self-describing and
re-runnable.

## Workflow

1. **Pick or copy a template.** `base.yaml` is the full default config (all
   `Config` fields). A variation file only needs to override what changes.
2. **Run with `--config`:**
   ```bash
   python train.py --config experiments/ljspeech.yaml
   ```
   CLI flags on top of `--config` still work and win:
   ```bash
   python train.py --config experiments/ljspeech.yaml --batch_size 4
   ```
3. **Checkpoint traceability.** Every training run dumps the fully-resolved
   config (YAML + CLI overrides merged) as `config.yaml` into
   `cfg.checkpoint_dir` at startup. So you can always read the exact recipe a
   checkpoint was trained with:
   ```
   checkpoints/ljspeech/
     config.yaml     ← snapshot of what ran
     train.log
     best.pt
     last.pt
     epoch_0005.pt
   ```
   The same config is also embedded inside each `.pt` under the `"config"` key
   for resume-time use.

## Files

| File | Purpose |
|---|---|
| `base.yaml` | Full default config — reference for every available knob. Regenerated from `Config()` defaults; don't edit this, copy it. |
| `librispeech.yaml` | Early recipe: SmolLM2-360M + LibriSpeech train-clean-{100,360}, k=3, multispeaker. |
| `ljspeech.yaml` | Single-speaker LJSpeech run (mixed-case, no ref block). |
| `wren-tts-360m-v1.yaml` | **English v1** released as `shangeth/Wren-TTS-360M-v1`: SmolLM2-360M, LibriTTS-R focus. |
| `wren-tts-360m-v1.1.yaml` | Fine-tune of v1 — adds VCTK + Jenny, replays LibriTTS-R/LJSpeech at 10%/epoch. |
| `wren-tts-360m-v2.yaml` | **English v2**: SmolLM2-360M from-scratch on the full English mix (VCTK + Jenny + LibriTTS-R + LJSpeech). |
| `wren-tts-qwen-v2-multilingual.yaml` | **Multilingual v1** released as `shangeth/Wren-TTS-0.5B-multi`: Qwen2.5-0.5B + English mix + 7-lang MLS (~1.87M utterances). |
| `test.yaml` | Smoke test — all 5 datasets sliced to ~16 rows each, ~12 train + ~4 val steps. Verifies model load + tokenizer + multilingual data path + wandb + audio logging in ~3-5 min. |

## Regenerating `base.yaml`

If you add fields to `Config`:

```bash
python -c "from config import Config; Config().save('experiments/base.yaml')"
```

## Per-experiment output directories

Each variation yaml should set its own `checkpoint_dir` and `log_dir` so runs
don't stomp each other:

```yaml
checkpoint_dir: checkpoints/<experiment_name>
log_dir:        runs/<experiment_name>
wandb_run_name: <experiment_name>
```
