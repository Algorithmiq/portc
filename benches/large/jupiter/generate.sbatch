#!/bin/bash
#SBATCH --account=e-dev-2026d11-370
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --job-name=portc-generate
#SBATCH --output=%x-%j.out
# The two complex64 stacks, on the project scratch.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/common.sh"
B="uv run python benches/large/bench_large.py generate"
run generate-m4000-c64 $B "$STACKS/m4000-c64" --bond-m 4000 --dtype complex64
run generate-m1000-c64 $B "$STACKS/m1000-c64" --bond-m 1000 --dtype complex64
