#!/usr/bin/env python3
"""Run this file from any working directory, with the selected Python environment."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "policy"))

from execution.cli import main

if __name__ == "__main__":
    main()
