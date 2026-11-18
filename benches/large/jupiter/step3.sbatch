#!/bin/bash
#SBATCH --account=e-dev-2026d11-370
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --job-name=portc-step3
#SBATCH --output=%x-%j.out
# Traced 1-GPU run at the reference size, then the scaling prediction.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/common.sh"
stage m4000-c64
run step3-trace-g1-m4000 uv run python "$N/trace.py" "$SHM/m4000-c64" \
  --chi-out 2000 --devices 1 --scratch-dir "$SPILL" --out "$N/raw/trace_g1_m4000.json"
uv run python "$N/predict.py" "$N/raw/probe.json" "$N/raw/trace_g1_m4000.json" \
  > "$N/raw/prediction.json"
