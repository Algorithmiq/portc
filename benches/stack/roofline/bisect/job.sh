#!/usr/bin/env bash
#SBATCH --job-name=bisect
#SBATCH --account=EUHPC_D30_139
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=benches/stack/roofline/bisect/logs/%x_%j.out

# Bisect the single-GPU slowdown of `bench_depth.py evolve`:
#   sbatch -J bisect-a job.sh <commit>...   (from the repository root)
# Each commit runs its own bench script from a worktree in $WT/<commit>, with
# PYTHONPATH=<worktree>/src and the repository's venv, twice: the first run is
# the warm-up, the second is the timing kept (<commit>.md).
# Mode `profile <commit>...` runs cProfile on n=50, chi=64, k=1 instead.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
ROOT=$PWD
OUT=$ROOT/benches/stack/roofline/bisect
WT=${WT:-$(realpath ../bisect-wt)}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_PLACES=cores OMP_PROC_BIND=spread
export TMPDIR=$CINECA_SCRATCH/tmp && mkdir -p "$TMPDIR"
export CUPY_CACHE_DIR=$ROOT/.cupy-cache
export LD_LIBRARY_PATH=$ROOT/.cuda-compat/usr/local/cuda-13.4/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_VISIBLE_DEVICES=0
source "$ROOT/.venv/bin/activate"
echo "job $SLURM_JOB_ID on $(hostname)"

ARGS=(evolve --n-sites 50 --chis 64 --chis 512 --steps-per-sweep 1 --steps-per-sweep 4
  --reference-chi 64 --quimb-max-chi 0)
MODE=time
if [[ $1 == profile ]]; then MODE=profile; shift; fi

for c in "$@"; do
  cd "$WT/$c" || { echo "no worktree $c"; continue; }
  export PYTHONPATH=$WT/$c/src
  echo "=== $c $(git log -1 --format='%h %s' | cut -c1-70)"
  if [[ $MODE == time ]]; then
    for run in warmup timed; do
      out=$OUT/$c.md
      [[ $run == warmup ]] && out=$TMPDIR/bisect_${c}_warmup.md
      echo "+ $run"
      /usr/bin/time -f "elapsed %e s" python benches/stack/bench_depth.py "${ARGS[@]}" \
        --output "$out" 2>&1 | grep -E "reference|done:|elapsed|Error|Traceback"
    done
  else
    python "$OUT/profile_k1.py" --out "$OUT/cprofile_$c.txt"
  fi
done
