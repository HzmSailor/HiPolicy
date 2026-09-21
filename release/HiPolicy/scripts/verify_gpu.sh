#!/usr/bin/env bash
# Run in the matching installed benchmark environment. All outputs are local.
set -euo pipefail
RELEASE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 4 ]]; then
  echo "Usage: $0 ORIGINAL_ROOT robotwin_v1|robotwin_v2 dp|dp3 OUTPUT_DIR" >&2
  exit 2
fi
python "$RELEASE_ROOT/scripts/verify_parity.py" --original-root "$1" \
  --benchmark "$2" --policy "$3" --mode structure --output "$4/structure"
python "$RELEASE_ROOT/scripts/verify_parity.py" --original-root "$1" \
  --benchmark "$2" --policy "$3" --mode numerical --output "$4/numerical"
python "$RELEASE_ROOT/scripts/record_environment.py" --output "$4/environment.json"
