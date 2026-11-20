#!/bin/bash
#SBATCH --account=e-dev-2026d11-370
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00
#SBATCH --job-name=portc-step4
#SBATCH --output=%x-%j.out
# Strong scaling on 1, 2 and 4 GPUs, twice per size, then traced runs.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/common.sh"
stage m4000-$DT m1000-$DT
B="uv run python benches/large/bench_large.py --debug scaling"
for rep in ${REPS:-1 2}; do
  run step4-m4000-r$rep$S $B "$SHM/m4000-$DT" --chi-out 2000 --devices 1 2 4 \
    --scratch-dir "$SPILL" --results "$N/raw/step4_m4000$S.jsonl"
  run step4-m1000-r$rep$S $B "$SHM/m1000-$DT" --chi-out 500 --devices 1 2 4 \
    --scratch-dir "$SPILL" --results "$N/raw/step4_m1000$S.jsonl"
done
for g in ${TRACES:-2 4}; do
  run step4-trace-g$g-m4000$S uv run python "$N/trace.py" "$SHM/m4000-$DT" \
    --chi-out 2000 --devices $g --scratch-dir "$SPILL" --out "$N/raw/trace_g${g}_m4000$S.json"
done
