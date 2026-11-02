#!/bin/zsh
# Runs every config named in a list file, one after the other (one GPU run at a time).
# usage (from the repository root): docs/results/configs/ecc/run_list.sh <list.txt> <tag>
# Raw runs go to /Users/adampain/Projects/fring/runs/<name>, logs to runs/logs/.
# A failed run is recorded (name, exit code, seconds) and the loop goes on (A0 decision).
RUNS=/Users/adampain/Projects/fring/runs
mkdir -p $RUNS/logs
STATUS=$RUNS/logs/ecc-status-$2.csv
[ -f $STATUS ] || echo "name,exit,seconds" > $STATUS
for n in $(cat $1); do
  t0=$(date +%s)
  ./target/release/fring-run --config docs/results/configs/ecc/$n.toml --out $RUNS/$n --quiet > $RUNS/logs/$n.log 2>&1
  rc=$?
  echo "$n,$rc,$(( $(date +%s) - t0 ))" >> $STATUS
done
