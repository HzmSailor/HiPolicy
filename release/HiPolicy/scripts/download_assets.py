#!/usr/bin/env python3
"""Download the original RoboTwin asset sets into their benchmark trees."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True, choices=("robotwin_v1", "robotwin_v2"))
    p.add_argument("--revision", help="Optional Hugging Face dataset commit for a pinned snapshot")
    args = p.parse_args()
    from huggingface_hub import snapshot_download
    benchmark = ROOT / "benchmarks" / args.benchmark
    if args.benchmark == "robotwin_v1":
        repo, files, dest = "ZanxinChen/RoboTwin_asset", ["aloha_urdf.zip", "main_models.zip"], benchmark
    else:
        repo, files, dest = "TianxingChen/RoboTwin2.0", ["background_texture.zip", "embodiments.zip", "objects.zip"], benchmark / "assets"
    snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=files,
                      local_dir=str(dest), revision=args.revision)
    for name in files:
        # -n preserves existing local files. Archives remain available for hashes.
        subprocess.run(["unzip", "-n", str(dest / name), "-d", str(dest)], check=True)
    if args.benchmark == "robotwin_v2":
        subprocess.run([sys.executable, str(benchmark / "script/update_embodiment_config_path.py")],
                       cwd=benchmark, check=True)


if __name__ == "__main__":
    main()
