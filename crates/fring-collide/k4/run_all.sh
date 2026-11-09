#!/bin/sh
# Usage: run_all.sh <config-name>...   (run from the repository root; 2 jobs at a time, 3 threads each)
printf '%s\n' "$@" | xargs -P2 -I{} sh -c 'RAYON_NUM_THREADS=3 runs/k4/bin/fring-run --config crates/fring-collide/k4/configs/{}.toml --out runs/k4/{} --quiet > runs/logs/k4-{}.log 2>&1; echo "exit $?" >> runs/logs/k4-{}.log'
