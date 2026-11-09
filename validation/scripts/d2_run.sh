#!/usr/bin/env bash
# D2: run the 5-year GPU set, one run after the other (about 22 min).
# Usage: FRING_RUN=<path to fring-run> bash validation/scripts/d2_run.sh [out_dir]
# Needs a fring-run that accepts domain.kind = "arc" on the GPU.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${FRING_RUN:-$ROOT/target/release/fring-run}"
OUT="${1:-$ROOT/runs/d2}"
mkdir -p "$OUT"
while read -r name; do
  [ -d "$OUT/$name" ] && { echo "skip $name"; continue; }
  "$BIN" --config "$ROOT/validation/reference/d2/configs/$name.toml" --out "$OUT/$name" --quiet || { echo "FAILED $name"; exit 1; }
  echo "done $name $(date +%H:%M:%S)"
done < "$ROOT/validation/reference/d2/configs/list.txt"
echo ALL_DONE
