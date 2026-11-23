#!/bin/bash
#SBATCH --account=e-dev-2026d11-370
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
#SBATCH --job-name=portc-capped
#SBATCH --output=%x-%j.out
# m4000 on GPUS with the GPU and host budgets of the 4-GPU point, twice each, for a
# scaling curve under one budget. No output saved, no distances.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/common.sh"
GPU_MEMORY=${GPU_MEMORY:-60GB}
B="uv run python benches/large/bench_large.py --debug run $STACKS/m4000-$DT --chi-out 2000 \
  --scratch-dir $SPILL --reserve --gpu-memory $GPU_MEMORY --host-memory ${HOST_MEMORY:-520GB} \
  --results $N/raw/capped_m4000$S.jsonl"
for rep in 1 2; do
  for g in ${GPUS:-1 2}; do
    run capped-m4000-$GPU_MEMORY-r$rep-g$g$S $B --devices $g
  done
done
