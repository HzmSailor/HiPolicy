#!/usr/bin/env python3
"""Apply the dependency edits specified by the original RoboTwin installation."""
import argparse
import importlib.util
from pathlib import Path
import re
import shutil


def replace(path, old, new):
    text = path.read_text()
    if old not in text:
        if new in text:
            print(f"Already patched: {path.name}")
            return
        raise RuntimeError(f"Unexpected dependency source: {path}; inspect its version")
    backup = path.with_suffix(path.suffix + ".hipolicy-original")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text.replace(old, new))
    print(f"Patched {path.name}; backup: {backup.name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True, choices=("robotwin_v1", "robotwin_v2"))
    args = p.parse_args()
    mplib = Path(importlib.util.find_spec("mplib").origin).parent / "planner.py"
    if args.benchmark == "robotwin_v1":
        replace(mplib, "            convex=True,", "            # convex=True,")
    replace(mplib, "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:",
            "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:")
    if args.benchmark == "robotwin_v2":
        loader = Path(importlib.util.find_spec("sapien").origin).parent / "wrapper/urdf_loader.py"
        # Exactly the original V2 installer edit: add UTF-8, do not alter parsing.
        content = loader.read_text()
        patched = re.sub(r'("r")(\))( as)', r'\1, encoding="utf-8") as', content)
        if content != patched:
            backup = loader.with_suffix('.py.hipolicy-original')
            if not backup.exists():
                shutil.copy2(loader, backup)
            loader.write_text(patched)
        print("SAPIEN UTF-8 patch checked")


if __name__ == "__main__":
    main()
