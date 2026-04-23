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
| `librispeech.yaml` | The v1 recipe: SmolLM2-360M + LibriSpeech train-clean-{100,360}, k=3, multispeaker. |
| `ljspeech.yaml` | Single-speaker LJSpeech run (mixed-case, no ref block). |

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
