# HiPolicy

**HiPolicy: Hierarchical Multi-Frequency Action Chunking for Policy Learning**

This repository contains the simulation implementation of HiPolicy for ECCV 2026,
with **DP + HiPolicy** (RGB observations) and **DP3 + HiPolicy** (point clouds) on
RoboTwin V1 and V2. [Project page](https://hipolicy.github.io/).

The release preserves the original neural architectures, temporal sampling,
normalizers, benchmark-specific training loops, and checkpoint parameter names.
The approved training defaults and entropy-guided execution are documented below.
Real-robot deployment, datasets, pretrained checkpoints, and the paper PDF are not
bundled. Existing trusted checkpoints can be supplied directly for evaluation.

## Repository layout

```text
policy/diffusion_policy/       # Shared DP package; original import paths
policy/diffusion_policy_3d/    # Shared DP3 package; original import paths
policy/hipolicy_compat/        # Original V1/V2 trainers and dataset implementations
benchmarks/robotwin_v1/        # V1 simulation, tasks, collection and conversion
benchmarks/robotwin_v2/        # V2 simulation, tasks, descriptions and assets setup
configs/                      # Four active 14D profiles; historical references
execution/                    # CLI, checkpoint loading and entropy selection
scripts/                      # Installation helpers and verification tools
requirements/                 # Separate benchmark dependency sets
tests/                        # CPU checks and synthetic data fixtures
```

DP retains explicit benchmark branches where the supplied implementations differ,
including ImageNet buffers and loss return values. DP3 shares its unchanged
network. Run one benchmark per process and use a separate environment for each.

## Environment setup

Use Linux, Python 3.8, an NVIDIA GPU, a compatible CUDA 12.1 toolchain, Vulkan,
and FFmpeg for simulation. The original setup requires sufficiently recent
drivers for SAPIEN ray tracing; first verify `nvidia-smi` and `vulkaninfo`.
CPU tests do not require SAPIEN or downloaded simulation assets.

```bash
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools ffmpeg unzip
conda create -n hipolicy-v1 python=3.8
conda activate hipolicy-v1
cd /path/to/HiPolicy
python -m pip install 'pip>=22,<25'
python -m pip install -r requirements/robotwin_v1.txt
python -m pip install -e .
python scripts/patch_sim_dependencies.py --benchmark robotwin_v1
```

For V2, create `hipolicy-v2` with Python 3.8 and repeat with
`requirements/robotwin_v2.txt` and `--benchmark robotwin_v2`.
V1 uses `mplib==0.1.1`; V2 uses `mplib==0.2.1`. Do not combine them. The patch
helper backs up the dependency files before applying the original RoboTwin edits.

Install the point-cloud extension in the appropriate environment:

```bash
# V1: restore the pytorch3d_simplified source used by your original environment.
python -m pip install -e /path/to/pytorch3d_simplified

# V2: the supplied installation used the upstream stable reference.
python -m pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'
git clone https://github.com/NVlabs/curobo.git benchmarks/robotwin_v2/envs/curobo
python -m pip install -e benchmarks/robotwin_v2/envs/curobo --no-build-isolation
```

**Environment provenance:** the supplied checkout does not include V1's
`pytorch3d_simplified` source, the original complete environment exports, or pinned
V2 PyTorch3D/cuRobo revisions. Recover these for an exact paper-environment rebuild;
do not silently substitute a different point sampler. The original V2 commands
above are retained as setup guidance, not a verified dependency lock. Record the
resolved revisions and environment with `scripts/record_environment.py`.

Zarr 2 and a compatible numcodecs version are constrained because the converters
use their original APIs. Torchvision is paired with PyTorch 2.4.1. Additional
constraints for NumPy, OpenCV and MoviePy keep the older runtime APIs available;
they do not change network code. The new release has not yet passed Linux GPU
simulation or full paper reproduction; see [validation status](docs/VALIDATION.md).

## Download simulation assets

```bash
python scripts/download_assets.py --benchmark robotwin_v1
python scripts/download_assets.py --benchmark robotwin_v2
```

These use the original `ZanxinChen/RoboTwin_asset` and
`TianxingChen/RoboTwin2.0` Hugging Face datasets. V2 also resolves embodiment
configuration paths after extraction. Add `--revision DATASET_COMMIT` to pin an
asset snapshot. Assets and generated absolute-path configurations are ignored by
Git. Existing extracted files are preserved; use a fresh benchmark assets directory
when intentionally changing revisions.

Check rendering in the selected environment:

```bash
python scripts/hipolicy.py render --benchmark robotwin_v1 --policy dp --task block_hammer_beat
python scripts/hipolicy.py render --benchmark robotwin_v2 --policy dp3 --task place_bread_basket
```

## Collect, convert and train: all four routes

Set the release root and select one row from this table. Activate the matching
environment first. `--policy` selects both the observation modality and the
corresponding training implementation.

| Route | `BENCHMARK` | `POLICY` | Example `TASK` | Environment |
| --- | --- | --- | --- | --- |
| V1 DP + HiPolicy | `robotwin_v1` | `dp` | `block_hammer_beat` | `hipolicy-v1` |
| V1 DP3 + HiPolicy | `robotwin_v1` | `dp3` | `block_hammer_beat` | `hipolicy-v1` |
| V2 DP + HiPolicy | `robotwin_v2` | `dp` | `place_bread_basket` | `hipolicy-v2` |
| V2 DP3 + HiPolicy | `robotwin_v2` | `dp3` | `place_bread_basket` | `hipolicy-v2` |

```bash
HIPOLICY_ROOT=/path/to/HiPolicy
BENCHMARK=robotwin_v1
POLICY=dp
TASK=block_hammer_beat
DATA_ROOT="$HIPOLICY_ROOT/data"
OUTPUT_ROOT="$HIPOLICY_ROOT/outputs"

COMMON=(--benchmark "$BENCHMARK" --policy "$POLICY" --task "$TASK"
        --camera D435 --task-config demo_randomized --episodes 100 --seed 0
        --gpus 0 --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT")

python "$HIPOLICY_ROOT/scripts/hipolicy.py" collect "${COMMON[@]}"
python "$HIPOLICY_ROOT/scripts/hipolicy.py" convert "${COMMON[@]}"
python "$HIPOLICY_ROOT/scripts/hipolicy.py" train   "${COMMON[@]}"
```

The commands use Bash array syntax. They work from any directory. Use
`--dry-run` to inspect resolved paths without importing the simulator.
For V2, `demo_clean` and `demo_randomized` are separate task settings; select the
setting matching the experiment being reproduced. V1 uses its per-task YAML and
ignores `--task-config`. Task identifiers differ between benchmarks.
`--seed` controls training/evaluation; collection retains the original candidate
seed search and saved demonstration seed list.

The CLI explicitly collects **100 demonstrations**, overriding the supplied V2
YAML default of 50. DP consumes `head_cam` RGB and 14D joint state. DP3 enables
the existing point-cloud producer for collection and evaluation; the original
crop, 1,024-point sampling, and XYZ-only policy input remain unchanged. Other
collection modalities already enabled by the task YAML are retained. V2's bundled
instruction generation remains in the workflow to preserve its RNG behavior;
ordinary collection/evaluation need no language-model API key.

Raw and processed data are separated by benchmark and policy:

```text
data/<benchmark>/<policy>/raw/<task>_<camera>_pkl/          # V1
data/<benchmark>/<policy>/raw/<task>/<task-config>/data/   # V2
data/<benchmark>/<policy>/processed/<task>_<setting>_100.zarr
outputs/<benchmark>/<policy>/<task>/<setting>/demos100/seed0/checkpoints/<epoch>.ckpt
```

Here `<setting>` is `D435` for V1 or the task configuration for V2. Conversion
refuses to overwrite an existing dataset. It stages output in a temporary sibling
directory, checks episode boundaries and array shapes, and publishes the dataset
only after success. A failed conversion can be retried without deleting raw data.
V1 pairs observations and actions from
the same recorded frame; V2 pairs observations/state at `t` with actions at
`t+1`, as in the supplied converters. RGB channel decoding is also preserved.
Do not interchange processed datasets between versions.

## Training configuration

| Setting | DP + HiPolicy | DP3 + HiPolicy |
| --- | --- | --- |
| Epochs | 600 | 3,000 |
| Batch per process | 128 | 128 |
| AdamW learning rate | `1e-4` | `1e-4` |
| AdamW betas / weight decay | `[0.9, 0.999]` / `1e-6` | `[0.9, 0.999]` / `1e-6` |
| Scheduler / inference steps | DDPM / 100 | DDPM / 100 |
| Prediction target | `epsilon` | **`sample`**, preserved from implementation |
| Diffusion-step embedding | 128 | **64**, preserved for checkpoint compatibility |
| Action / state dimensions | 14 / 14 | 14 / 14 |

Each scale receives **3 observations and predicts 8 actions**. The dataset exposes
a 7-frame observation window and 29-step trajectory so the policy can sample
long/mid/short observations at `[0,3,6]`, `[2,4,6]`, `[4,5,6]`, and actions at
`[7,10,...,28]`, `[7,9,...,21]`, `[7,8,...,14]`. The three predicted chunks are
concatenated in **long, mid, short** order. The larger window dimensions are not
the number of frames or actions used by one scale.

The approved release defaults change DP3's DDIM/10 sampler to DDPM/100 while
preserving its `sample` target. The paper's summarized epsilon/128 settings do not
override DP3's checkpoint architecture. EMA, warmup, conditioning, losses,
clipping and trainer-specific behavior remain original. V1 DP keeps EMA off and
warmup 8,000; the other profiles retain EMA on and warmup 500.

Explicit experimental overrides use `--set key=value`, for example
`--set training.num_epochs=2`. Every run saves `resolved_config.yaml`.
Resume a compatible run with `--resume /path/to/epoch.ckpt`; a differing policy
configuration is rejected before loading training state. The V1 DP trainer also
retains its original automatic `latest.ckpt` resume behavior. The explicitly
selected output root takes precedence over paths saved on the previous machine;
model/optimizer state and epoch/step counters retain their original loading rules.

Only V1 DP has the supplied multi-GPU trainer: use `--gpus 0,1`. Its existing
per-process batch, gradient accumulation and scheduler stepping are preserved;
the nominal effective batch is `128 × world_size × gradient_accumulate_every`.
Use one GPU for the other three routes. Changing GPU count or effective batch
is a new experiment, not an equivalent paper run.

## Evaluate existing or newly trained checkpoints

Set `CHECKPOINT` to an actual file; no directory-name guessing or weight
conversion is required. Use the benchmark and variant that produced it.

```bash
CHECKPOINT=/path/to/epoch.ckpt

# Default: one sample, execute the short eight-step chunk.
python "$HIPOLICY_ROOT/scripts/hipolicy.py" evaluate "${COMMON[@]}" \
  --checkpoint "$CHECKPOINT" --eval-episodes 100

# Entropy guidance: sample 100 candidates together in the model batch.
python "$HIPOLICY_ROOT/scripts/hipolicy.py" evaluate "${COMMON[@]}" \
  --checkpoint "$CHECKPOINT" --eval-episodes 100 --entropy-guidance
```

The loader instantiates the saved architecture, selects the saved EMA/model
weights, restores its normalizer, and checks state keys strictly. It handles
DDP/compile wrapper prefixes and only the known legacy V1 ImageNet constant
buffers. Inference applies DDPM/100 without rewriting the checkpoint or changing
its prediction target. Load only trusted pickle-based checkpoints.

With entropy guidance enabled, the original V2 batch expansion convention is used
for both policies; the attention layout is unchanged. For the 100 **unnormalized**
candidate chunks, variance across candidates uses the original `torch.var`
sample correction. Compute `0.5 * log(2*pi*e*variance + 1e-6)`, then average over
all three scales, eight steps, and all 14 action dimensions. Both arms use the
same scale:

| Global entropy | Executed chunk from **candidate 0** |
| --- | --- |
| `H < -6` | Short: indices `16:24` |
| `-6 <= H < -5.5` | Mid: indices `8:16` |
| `H >= -5.5` | Long: indices `0:8` |

Candidates are not averaged. No interval-dependent action interpolation is added
by the selector. Simulator control/interpolation and observation updates remain
benchmark-specific. Parallel EG requires additional GPU memory. Results, invocation
arguments and videos are written below the run's `evaluation/eg_on` or `eg_off`
directory. Evaluation retains original valid-seed filtering and starts from
`100000 * (1 + seed)`.

## Validation and troubleshooting

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/check_computation.py
python scripts/record_environment.py --output outputs/environment.json
```

Tests cover configuration composition, path independence, original computation
fingerprints, four data-conversion routes, entropy boundaries, first-candidate
selection, batched sampling interfaces, and relocated checkpoint resumption.
Full-size CPU comparisons against the original source pass for all four routes:
fixed-input actions, loss, gradient hashes and one AdamW update match exactly,
using each original scheduler and optimizer configuration. Linux GPU parity and simulation
commands are in [VALIDATION.md](docs/VALIDATION.md). CPU tests and syntax checks
do not establish task success rates or paper reproduction.

- **Missing assets/embodiment YAML:** download the matching benchmark assets;
  rerun V2's path resolver after relocating them.
- **No point cloud:** recollect V2 DP3 data through the DP3 route; RGB-only raw
  episodes cannot supply the missing modality.
- **Dataset already exists:** select another data root or deliberately move the
  previous dataset before converting. Outputs are never silently replaced.
- **Checkpoint key/shape mismatch:** check the benchmark and saved configuration.
  Do not use `strict=False`, change layer widths, or reinterpret 16D checkpoints.
- **Renderer/Vulkan or extension failure:** verify the active conda environment,
  CUDA/Vulkan drivers and original extension sources before running experiments.
- **Out of GPU memory:** check the selected profile and GPU capacity. Reducing
  batch size or the EG candidate count changes the experiment; the released EG
  route keeps 100 candidates.

Logs default to offline mode. Do not commit credentials, generated data, local
environment exports, weights, downloaded assets, or generated absolute paths.
The 16D historical configuration is documented under `configs/reference/` and
is not an active CLI route. See [release changes](docs/CHANGES.md) and
[third-party notices](NOTICE.md) for provenance.
