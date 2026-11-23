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
# STAGE=0 memory-maps the stacks from the project file system instead of /dev/shm,
# leaving that memory to the sweep; stack reads then show up as site stalls in the logs.
if [ "${STAGE:-1}" = 1 ]; then
  stage m4000-$DT m1000-$DT
  D=$SHM
else
  D=$STACKS
fi
# One process per GPU count, so no run inherits the host pages of another: on the
# GH200 those pages land in GPU memory once the CPU memory is full.
B="uv run python benches/large/bench_large.py --debug"
# G4_GPU_MEMORY caps the GPU budget of the m4000 runs on 4 GPUs. complex128 m4000
# overflows the CPU memory into GPU memory, and with every GPU busy that overflow has
# to fit beside the sweep.
budget() {
  if [ "$1" = 4000 ] && [ "$2" = 4 ] && [ -n "${G4_GPU_MEMORY:-}" ]; then
    echo "--gpu-memory $G4_GPU_MEMORY"
  fi
}
for rep in ${REPS:-1 2}; do
  for m in 4000:2000 1000:500; do
    size=${m%:*} chi=${m#*:}
    for g in 1 2 4; do
      run step4-m$size-r$rep-g$g$S $B run "$D/m$size-$DT" --chi-out $chi --devices $g \
        --scratch-dir "$SPILL" --results "$N/raw/step4_m$size$S.jsonl" --save "$SPILL/out-m$size-g$g.npz" \
        $(budget $size $g)
    done
    for g in 2 4; do
      run step4-m$size-r$rep-distance-g$g$S $B distance "$SPILL/out-m$size-g1.npz" \
        "$SPILL/out-m$size-g$g.npz" --results "$N/raw/step4_m$size$S-distance.jsonl"
    done
    rm -f "$SPILL"/out-m$size-g*.npz
  done
done
for g in ${TRACES:-2 4}; do
  run step4-trace-g$g-m4000$S uv run python "$N/trace.py" "$D/m4000-$DT" \
    --chi-out 2000 --devices $g --scratch-dir "$SPILL" --out "$N/raw/trace_g${g}_m4000$S.json" \
    $(budget 4000 $g)
done
