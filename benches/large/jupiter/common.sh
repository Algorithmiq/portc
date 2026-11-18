# Sourced by the sbatch scripts: paths, strict FP32 math, and `run`, which logs one
# Submit from this directory: sbatch probe.sbatch
# timed command with its date and commit.
W=/e/scratch/e-dev-2026d11-370/materia1
REPO=$W/portc
N=$REPO/benches/large/jupiter
STACKS=$W/stacks
SHM=/dev/shm/portc
SPILL=$W/spill
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=$W/.uv-cache
export CUPY_TF32=0 NVIDIA_TF32_OVERRIDE=0
cd "$REPO"
mkdir -p "$N/logs" "$N/raw" "$N/node" "$SPILL"

run() {
  local log=$N/logs/$1.log
  shift
  { echo "# $(date -Is) $(hostname) job $SLURM_JOB_ID: $*"; echo "# git $(git rev-parse HEAD)"; } > "$log"
  /usr/bin/time -v "$@" >> "$log" 2>&1
}

# Copy stacks to node memory, untimed, so the sweep reads them as on the RTX node.
stage() {
  mkdir -p "$SHM"
  for s in "$@"; do cp -r "$STACKS/$s" "$SHM/"; done
}
