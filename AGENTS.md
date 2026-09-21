# Repository Guidelines

## Project Structure & Module Organization

This repository contains two Python robotics simulation and policy trees: `RoboTwin-V1/` and `RoboTwin-V2/`. Keep changes scoped to the intended version.

- `envs/`: manipulation tasks, base environments, and shared utilities.
- `policy/`: diffusion policies; V1 uses `Diffusion-Policy/` and `3D-Diffusion-Policy/`, while V2 uses `DP/` and `DP3/`.
- `script/`: data conversion, collection, evaluation, and rendering utilities.
- V1 `task_config/`: task and camera YAML configurations; `files/`: documentation images.
- V2 `assets/`, `description/`, and `code_gen/`: asset support, task/object descriptions, and task generation.
- V1 `test_*.py` and each version’s `script/test_render.py`: standalone diagnostics.

## Build, Test, and Development Commands

Follow `RoboTwin-V1/INSTALLATION.md` or the setup guidance linked from `RoboTwin-V2/README.md`. Simulation requires compatible GPUs, drivers, dependencies, and downloaded assets. There is no repository-wide build command.

Run commands from the indicated directory:

- V1 `policy/Diffusion-Policy/`: `pip install -e .` installs the policy package for development.
- V1 root: `bash run_task.sh block_hammer_beat 0` collects demonstrations on GPU 0.
- V1 `policy/Diffusion-Policy/`: `bash train.sh block_hammer_beat D435 100 0 0` trains with 100 demonstrations and seed 0. Configure its hard-coded `/DATA/...` paths first.
- V2 root: `bash collect_data.sh beat_block_hammer demo_randomized 0` collects demonstrations after restoring local configuration and assets. This checkout omits `task_config/` and the wrapper’s `script/.update_path.sh` helper.

## Coding Style & Naming Conventions

Use four-space Python indentation and match surrounding formatting. Use `snake_case` for functions, variables, and task modules. Preserve task class names matching their modules: V2 loads them dynamically with `getattr`. Keep YAML keys consistent with their consumers. No shared formatter or linter configuration is present.

## Testing Guidelines

Tests are standalone Python diagnostics; no centralized test framework or coverage threshold is configured. From V1, run `python test_ddp_quick.py`; from either version, run `python script/test_render.py` for rendering checks. Inspect local paths and GPU requirements first. Name new diagnostics `test_*.py`. For policy changes, record task, seed, configuration, checkpoint, and evaluation results.

## Commit & Pull Request Guidelines

History contains only `init`, so no established commit convention exists. Use concise imperative subjects naming the affected version or component. PRs should describe behavior changes, reference relevant issues, and report validation commands and hardware limitations. Include images or videos for visual simulation changes. Keep generated datasets, checkpoints, credentials, and machine-specific paths out of commits.
