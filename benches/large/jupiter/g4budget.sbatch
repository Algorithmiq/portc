#!/bin/bash
#SBATCH --account=e-dev-2026d11-370
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=portc-g4budget
#SBATCH --output=%x-%j.out
# m4000 on 4 GPUs at the largest GPU budget that runs: tries BUDGETS (GB) from the
# largest down, then repeats the first that completes. No output saved, no distances.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/common.sh"
B="uv run python benches/large/bench_large.py --debug run $STACKS/m4000-$DT --chi-out 2000 \
  --devices 4 --scratch-dir $SPILL --reserve --host-memory ${HOST_MEMORY:-520GB} \
  --results $N/raw/g4budget_m4000$S.jsonl"
for gb in ${BUDGETS:-90 85 80 75 70}; do
  if run g4budget-m4000-${gb}gb-r1$S $B --gpu-memory ${gb}GB; then
    run g4budget-m4000-${gb}gb-r2$S $B --gpu-memory ${gb}GB
    exit 0
  fi
done
exit 1
