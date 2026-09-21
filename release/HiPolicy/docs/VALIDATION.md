# Validation record and Linux GPU handoff

## Completed locally

Validation was performed on macOS with an isolated Python 3.9 CPU environment.
No simulation assets, SAPIEN runtime or CUDA device were available.

- Python syntax and active YAML/Hydra configuration composition pass.
- CPU tests cover all four conversion routes, preserved RGB channel order and
  observation/action time alignment, no-overwrite conversion, failed-conversion
  cleanup/retry and rejection of incomplete stores, entropy thresholds,
  zero variance, first-candidate execution and the global arm-shared scale.
- Actual DP/DP3 inference methods, with a lightweight test denoiser, pass the
  single-sample and **100-candidate parallel batch** interface tests for both
  benchmarks, including two input observations without mixing their identities.
- Synthetic legacy-format checkpoint tests pass strict parameter loading, EMA
  selection, DDP/compile prefix removal, DP3 DDIM-to-DDPM inference configuration,
  and rejection of missing neural parameters. These fixtures use compact models;
  they are not the authors' trained checkpoints.
- Actual trainer checkpoint I/O passes relocation checks on all four routes:
  the selected output directory is preserved while model weights, AdamW state
  and epoch/step counters are restored. Incompatible explicit resume
  configurations are rejected before loading training state.
- Full-size architecture comparisons against the original checkout pass for
  **V1 DP, V1 DP3, V2 DP and V2 DP3**: state keys/shapes, buffers and module types
  match. The large networks use the meta device; RGB shape probing uses a CPU
  encoder because torchvision's normalization checks actual constants.
- **Full-size numerical CPU comparisons pass for all four routes.** Fixed-input
  actions and loss are exactly equal (`rtol=0`, `atol=0`); every gradient hash
  and every state hash after one AdamW update also matches. The checks use each
  original scheduler/optimizer, batch size two, synthetic observations/actions,
  identity normalizers and fixed random seeds. They do not test trained task
  performance, the approved new sampler defaults, or full-size parallel EG.
  Worker import checks guard against accidentally using the same policy tree.
- **64 original executable bodies** match their source fingerprints: architecture,
  encoders, normalization, losses and all four training loops. Checkpoint I/O
  blocks are excluded from the training-loop fingerprints because their paths
  were deliberately changed.
- Editable installation and the installed CLI work from outside the repository.
- Release-source scans cover private machine paths and the removed service
  endpoint. Original source hashes and Git status confirm the original trees
  were not edited by the release preparation.

Key local package versions: PyTorch 2.4.1, torchvision 0.19.1, NumPy 1.23.5,
Hydra 1.2.0, OmegaConf 2.2.3, diffusers 0.11.1 and Zarr 2.12.0. The local macOS
test environment uses numcodecs 0.12.1 and numba 0.60.0 wheels; these are **not**
substituted into the Linux release requirements. The CPU checks are not a
validation of the complete Linux dependency set.

## Run the local checks

From the release root, in an installed policy environment:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/check_computation.py
```

`docs/computation_manifest.json` fingerprints original executable AST bodies;
comments, whitespace and docstrings do not affect these checks.
`docs/source_manifest.json` records retained source-file provenance and hashes.
`docs/validation_summary.json` records the four CPU comparison results without
machine-specific paths. To repeat one full-size CPU comparison:

```bash
python scripts/verify_parity.py --original-root /path/to/original/checkout \
  --benchmark robotwin_v1 --policy dp --mode numerical --device cpu \
  --output outputs/parity/cpu-v1-dp
```

Run the remaining benchmark/policy combinations separately; DP3's full model and
AdamW state require several GB of memory. The local run used a 16 GiB Mac.

## Numerical comparison on Linux GPU

Keep the original two source trees in a separate directory. Run each command in
the matching benchmark environment. The script uses isolated subprocesses so
the original and release packages cannot shadow each other.

```bash
ORIGINAL_ROOT=/path/to/original/checkout
bash scripts/verify_gpu.sh "$ORIGINAL_ROOT" robotwin_v1 dp  outputs/parity/v1-dp
bash scripts/verify_gpu.sh "$ORIGINAL_ROOT" robotwin_v1 dp3 outputs/parity/v1-dp3
bash scripts/verify_gpu.sh "$ORIGINAL_ROOT" robotwin_v2 dp  outputs/parity/v2-dp
bash scripts/verify_gpu.sh "$ORIGINAL_ROOT" robotwin_v2 dp3 outputs/parity/v2-dp3
```

The numerical stage compares fixed-input actions, loss, all gradient hashes and
one AdamW update, using each original scheduler/optimizer configuration. It uses
the full neural model with a synthetic input batch of two and identity
normalizers. It intentionally separates packaging equivalence from the approved
new training defaults and EG execution. Exact differences fail the check; inspect
them before considering numerical tolerances. Full training, EMA/scheduler
evolution and multi-GPU behavior still require real-data runs.

Check each available original trained checkpoint:

```bash
python scripts/check_checkpoint.py --benchmark robotwin_v1 --checkpoint /path/to/v1-dp.ckpt
python scripts/check_checkpoint.py --benchmark robotwin_v1 --checkpoint /path/to/v1-dp3.ckpt
python scripts/check_checkpoint.py --benchmark robotwin_v2 --checkpoint /path/to/v2-dp.ckpt
python scripts/check_checkpoint.py --benchmark robotwin_v2 --checkpoint /path/to/v2-dp3.ckpt
```

This strictly loads the saved architecture/normalizer and runs synthetic inference
with EG off and on. It does not measure manipulation success.

## Simulation and training smoke test

For each of the four README routes, render, collect and convert a small dataset
in a **separate smoke-test data/output root**. Verify the selected task setting
and point-cloud modality. Then run a short training invocation:

```bash
python scripts/hipolicy.py train "${COMMON[@]}" \
  --set training.num_epochs=1 --set training.checkpoint_every=1 \
  --set training.max_train_steps=1 --set training.max_val_steps=1
```

Use the generated `1.ckpt` with `evaluate --eval-episodes 1`, both with and
without `--entropy-guidance`. `COMMON` is the Bash argument array in the README;
point it to the smoke-test roots and the number of demonstrations actually
collected. Check videos, resolved configuration, checkpoint loading and episode
completion. Restore the published defaults for paper runs.

Before claiming reproduction, run the intended task matrix with 100 demonstrations
and 100 evaluation episodes, recording task setting, seed, checkpoint, EG mode,
success counts, resolved configuration, environment, asset revisions and GPU.
No success-rate values are asserted by this release preparation.

## Still awaiting external inputs or execution

- Original full environment exports, V1's `pytorch3d_simplified` source, and
  original V2 PyTorch3D/cuRobo revisions.
- Author experiment checkpoints and real raw/processed dataset comparisons.
- Linux installation, CUDA numerical comparison, renderer/collection/training/
  evaluation smoke tests, multi-GPU checks and paper-scale reproduction.

The supplied V2 `task_config/` is included. Its default ALOHA route is 14D.
The historical 16D configuration remains reference-only; no algorithm changes
were made to accommodate it.
