#!/bin/bash
# 10 seeds x 3 settings; same perturbed states in each setting.
cd "$(dirname "$0")/../../.."   # repository root; output in runs/e1-real
H4=$(python3 -c "print(356.2537/4)")
for s in 1 2 3 4 5 6 7 8 9 10; do
  for set in h substep h4; do
    case $set in
      h) A="--variants plain";;
      substep) A="--variants substep";;
      h4) A="--variants plain --l2-step $H4 --l2-stride 4";;
    esac
    echo "seed $s setting $set"
    target/release/fring-e1 layer2 $A --l2-perturb-km 1e-6 --l2-seed $s --out runs/e1-real/s$s/$set 2>&1 | grep "^layer2"
  done
done
echo FINISHED
