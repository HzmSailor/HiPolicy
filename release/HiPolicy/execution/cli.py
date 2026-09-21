"""Portable CLI for the four HiPolicy simulation routes."""
import argparse
import importlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def identifier(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise argparse.ArgumentTypeError("Use letters, digits, underscores or hyphens")
    return value


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def parser():
    p = argparse.ArgumentParser(description="HiPolicy simulation collection, conversion, training and evaluation")
    p.add_argument("command", choices=("collect", "convert", "train", "evaluate", "render"))
    p.add_argument("--benchmark", required=True, choices=("robotwin_v1", "robotwin_v2"))
    p.add_argument("--policy", required=True, choices=("dp", "dp3"))
    p.add_argument("--task", required=True, type=identifier)
    p.add_argument("--task-config", default="demo_clean", type=identifier)
    p.add_argument("--camera", default="D435", type=identifier)
    p.add_argument("--episodes", default=100, type=positive, help="Number of demonstrations")
    p.add_argument("--eval-episodes", default=100, type=positive)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--gpus", default="0", help="Visible GPU IDs; comma-separated IDs enable original V1 DP DDP")
    p.add_argument("--master-port", default=29500, type=int)
    p.add_argument("--data-root", default=str(ROOT / "data"), type=Path)
    p.add_argument("--output-root", default=str(ROOT / "outputs"), type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--resume", type=Path, help="Resume a training checkpoint without changing its architecture")
    p.add_argument("--entropy-guidance", action="store_true", help="Use 100 parallel candidates; default: one short chunk")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Explicit Hydra training override (recorded)")
    p.add_argument("--dry-run", action="store_true", help="Show paths without importing torch or a simulator")
    return p


def paths(args):
    data = args.data_root.expanduser().resolve() / args.benchmark / args.policy
    setting = args.camera if args.benchmark == "robotwin_v1" else args.task_config
    raw = data / "raw"
    source = raw / (f"{args.task}_{args.camera}_pkl" if args.benchmark == "robotwin_v1" else f"{args.task}/{args.task_config}")
    dataset = data / "processed" / f"{args.task}_{setting}_{args.episodes}.zarr"
    run = args.output_root.expanduser().resolve() / args.benchmark / args.policy / args.task / setting / f"demos{args.episodes}" / f"seed{args.seed}"
    return {"benchmark_root": ROOT / "benchmarks" / args.benchmark, "raw_root": raw,
            "raw_dataset": source, "dataset": dataset, "run": run, "checkpoints": run / "checkpoints"}


def prepare(args, p):
    os.environ["HIPOLICY_BENCHMARK"] = args.benchmark
    os.environ["HIPOLICY_POLICY"] = args.policy
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.update(HIPOLICY_TASK=args.task, HIPOLICY_CAMERA=args.camera,
                      HIPOLICY_EPISODES=str(args.episodes), HIPOLICY_RAW_ROOT=str(p["raw_root"]),
                      HIPOLICY_RUN_DIR=str(p["run"]), HIPOLICY_VIDEO_ROOT=str(p["run"] / "evaluation/videos"),
                      HIPOLICY_CONVERT_INPUT=str(p["raw_dataset"]), HIPOLICY_CONVERT_OUTPUT=str(p["dataset"]))


def training_config(args, p):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import yaml
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(config_dir=str(ROOT / "configs" / args.benchmark), version_base="1.2"):
        cfg = compose(config_name=args.policy, overrides=args.set)
    OmegaConf.set_struct(cfg, False)
    cfg.task_name = args.task
    cfg.task.name = args.task
    cfg.task.dataset.zarr_path = str(p["dataset"])
    cfg.training.seed = args.seed
    cfg.head_camera_type = args.camera
    cfg.expert_data_num = args.episodes
    if "setting" in cfg:
        cfg.setting = args.task_config
    if "task_name" in cfg.task.dataset:
        cfg.task.dataset.task_name = args.task
    if args.policy == "dp":
        camera_file = p["benchmark_root"] / "task_config/_camera_config.yml"
        camera = yaml.safe_load(camera_file.read_text())[args.camera]
        for meta in cfg.task.shape_meta.obs.values():
            if meta.get("type") == "rgb":
                meta.shape = [3, camera["h"], camera["w"]]
    OmegaConf.resolve(cfg)
    return cfg


def train(args, p):
    from execution.checkpoint import load_training_payload
    from omegaconf import OmegaConf
    cfg = training_config(args, p)
    if not p["dataset"].is_dir():
        raise FileNotFoundError(f"Convert demonstrations first: {p['dataset']}")
    payload = load_training_payload(args.resume, cfg) if args.resume else None
    module = importlib.import_module(f"hipolicy_compat.{args.benchmark}.{args.policy}_workspace")
    cls = module.RobotWorkspace if args.policy == "dp" else module.TrainDP3Workspace
    workspace = cls(cfg, output_dir=str(p["run"]))
    if payload is not None:
        workspace.load_payload(payload, include_keys=["global_step", "epoch"])
        cfg.training.resume = False
    OmegaConf.save(cfg, p["run"] / "resolved_config.yaml")
    workspace.run()


def run_script(path, argv):
    previous = sys.argv
    try:
        sys.argv = [str(path), *map(str, argv)]
        runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = previous


def convert(args, p):
    from execution.data import conversion_output

    if args.benchmark == "robotwin_v1":
        source = p["benchmark_root"] / f"script/pkl2zarr_{args.policy}.py"
        setting = args.camera
        raw_files = [p["raw_dataset"] / f"episode{index}/0.pkl" for index in range(args.episodes)]
    else:
        source = p["benchmark_root"] / f"script/process_{args.policy}.py"
        setting = args.task_config
        raw_files = [p["raw_dataset"] / f"data/episode{index}.hdf5" for index in range(args.episodes)]
    for raw_file in raw_files:
        if not raw_file.is_file():
            raise FileNotFoundError(f"Missing raw episode: {raw_file}")
    with conversion_output(p["dataset"], args.policy, args.episodes):
        run_script(source, [args.task, setting, args.episodes])


def evaluate(args, p):
    from datetime import datetime
    if args.checkpoint is None or not args.checkpoint.is_file():
        raise FileNotFoundError("Supply an existing --checkpoint file")
    result = p["run"] / "evaluation" / ("eg_on" if args.entropy_guidance else "eg_off") / datetime.now().strftime("%Y%m%d-%H%M%S")
    result.mkdir(parents=True, exist_ok=False)
    (result / "invocation.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    os.environ["HIPOLICY_VIDEO_ROOT"] = str(result / "videos")
    if args.benchmark == "robotwin_v1":
        from types import SimpleNamespace
        module = importlib.import_module("eval_policy_dp")
        module.main(SimpleNamespace(task_name=args.task, head_camera_type=args.camera,
            checkpoint_num=args.checkpoint.stem, seed=args.seed, ensemble=False, no_hierarchical_cond=False,
            expert_data_num=args.episodes, checkpoint=str(args.checkpoint), variant=args.policy,
            entropy_guidance=args.entropy_guidance, eval_episodes=args.eval_episodes, result_dir=str(result)))
    else:
        module = importlib.import_module("eval_policy")
        module.main(dict(task_name=args.task, task_config=args.task_config, ckpt_setting=args.task_config,
            policy_name="execution.adapter", variant=args.policy, checkpoint=str(args.checkpoint),
            head_camera_type=args.camera,
            instruction_type="unseen", seed=args.seed, entropy_guidance=args.entropy_guidance,
            eval_episodes=args.eval_episodes, result_dir=str(result)))


def main(argv=None):
    args = parser().parse_args(argv)
    if args.checkpoint:
        args.checkpoint = args.checkpoint.expanduser().resolve()
    if args.resume:
        args.resume = args.resume.expanduser().resolve()
    if not re.fullmatch(r"\d+(,\d+)*", args.gpus):
        raise ValueError("--gpus must be a comma-separated list of GPU indices")
    if "," in args.gpus and (args.command != "train" or args.benchmark != "robotwin_v1" or args.policy != "dp"):
        raise ValueError("Only the original robotwin_v1 DP trainer supports multiple GPUs")
    p = paths(args)
    if args.dry_run:
        print(json.dumps({"command": args.command, "benchmark": args.benchmark, "policy": args.policy,
                          "entropy_guidance": args.entropy_guidance, **p}, default=str, indent=2))
        return
    prepare(args, p)
    if "," in args.gpus and "RANK" not in os.environ:
        command = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(len(args.gpus.split(','))),
                   "--master_port", str(args.master_port), str(ROOT / "scripts/hipolicy.py"), *(argv or sys.argv[1:])]
        subprocess.run(command, check=True)
        return
    p["run"].mkdir(parents=True, exist_ok=True)
    # Training imports shared policies; simulation imports exactly one envs tree.
    if args.command == "train":
        train(args, p)
        return
    os.chdir(p["benchmark_root"])
    sys.path.insert(0, str(p["benchmark_root"]))
    sys.path.insert(0, str(p["benchmark_root"] / "script"))
    if args.command in ("collect", "evaluate", "render"):
        from test_render import Sapien_TEST
        # Collection scripts already call this diagnostic in their __main__.
        if args.command != "collect":
            Sapien_TEST()
    if args.command == "collect":
        if args.benchmark == "robotwin_v1":
            run_script(p["benchmark_root"] / "script/run_task.py", [])
        else:
            run_script(p["benchmark_root"] / "script/collect_data.py", [args.task, args.task_config])
    elif args.command == "convert":
        convert(args, p)
    elif args.command == "evaluate":
        evaluate(args, p)
