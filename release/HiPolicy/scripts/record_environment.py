#!/usr/bin/env python3
"""Record local reproducibility metadata without exporting environment secrets."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def command(args):
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=30)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    record = {"python": sys.version, "platform": platform.platform(),
              "pip_freeze": command([sys.executable, '-m', 'pip', 'freeze']),
              "gpu": command(['nvidia-smi']), "cuda": command(['nvcc', '--version'])}
    record['extensions'] = {}
    for relative in ('third_party/pytorch3d_simplified', 'benchmarks/robotwin_v2/envs/curobo'):
        path = ROOT / relative
        if path.exists():
            record['extensions'][relative] = command(['git', '-C', str(path), 'rev-parse', 'HEAD'])
    record['asset_archives'] = {}
    for archive in (ROOT / 'benchmarks').rglob('*.zip'):
        digest = hashlib.sha256()
        with archive.open('rb') as source:
            for chunk in iter(lambda: source.read(1024*1024), b''):
                digest.update(chunk)
        record['asset_archives'][str(archive.relative_to(ROOT))] = digest.hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    print(f'Local environment record: {args.output}')


if __name__ == '__main__':
    main()
