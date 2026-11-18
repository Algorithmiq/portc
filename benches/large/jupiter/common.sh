# Sourced by the sbatch scripts: paths, strict FP32 math, and `run`, which logs one
# Submit from this directory: sbatch probe.sbatch
# timed command with its date and commit.
W=/e/project1/e-dev-2026d11-370/materia1
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
  # git is not on the compute nodes' PATH; the branch ref is a loose file.
  { echo "# $(date -Is) $(hostname) job $SLURM_JOB_ID: $*"; echo "# git $(cat ".git/$(sed 's/^ref: //' .git/HEAD)")"; } > "$log"
  echo "$(date -Is) start $log"
  { time "$@"; } >> "$log" 2>&1
  echo "$(date -Is) end $log"
}

# Copy stacks to node memory, untimed, so the sweep reads them as on the RTX node.
stage() {
  mkdir -p "$SHM"
  for s in "$@"; do cp -r "$STACKS/$s" "$SHM/"; done
}
