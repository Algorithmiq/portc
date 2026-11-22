#!/usr/bin/env bash
#SBATCH --job-name=roofline
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
#SBATCH --output=benches/stack/roofline/logs/%x_%j.out

# One stage of the roofline study per job: sbatch -J <name> job.sh <stage>.
# Submit from the repository root. One A100 of an exclusive node.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
OUT=benches/stack/roofline
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_PLACES=cores OMP_PROC_BIND=spread
export TMPDIR=$CINECA_SCRATCH/tmp && mkdir -p "$TMPDIR"
export CUPY_CACHE_DIR=$PWD/.cupy-cache
export LD_LIBRARY_PATH=$PWD/.cuda-compat/usr/local/cuda-13.4/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_VISIBLE_DEVICES=0
source .venv/bin/activate
echo "job $SLURM_JOB_ID on $(hostname), stage $1, commit $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

run() { echo "+ $*"; /usr/bin/time -f "elapsed %e s" python "$@"; }

case $1 in
  n50)
    run $OUT/ceilings.py --output $OUT/ceilings.json
    run $OUT/profile_quench.py --n-sites 10 --chis 16 --ks 1 --ks 2 --steps 4 \
      --out-dir "$TMPDIR/roofline-smoke-$SLURM_JOB_ID"
    run benches/stack/bench_depth.py evolve --n-sites 50 --quimb-max-chi 0 \
      --output $OUT/quench_n50.md
    run benches/stack/bench_depth.py evolve --n-sites 50 --quimb-max-chi 0 \
      --output $OUT/quench_n50_repeat.md
    ;;
  n100)
    run benches/stack/bench_depth.py evolve --n-sites 100 --quimb-max-chi 0 \
      --output $OUT/quench_n100.md
    ;;
  quimb)
    run benches/stack/bench_depth.py evolve --n-sites 50 --chis 64 --quimb-max-chi 64 \
      --output $OUT/quench_quimb.md
    ;;
  profile)
    shift
    run $OUT/profile_quench.py --out-dir "$OUT/parts/$SLURM_JOB_ID" "$@"
    ;;
  nsight)
    # Optional cross-check, each tool capped so that a hang cannot eat the slot.
    module load nvhpc/25.11
    P=$OUT/nsight && mkdir -p $P "$TMPDIR/nsight"
    for k in 1 4; do
      timeout 240 nsys profile --capture-range=cudaProfilerApi \
        --capture-range-end=stop -t cuda --force-overwrite true \
        -o "$TMPDIR/nsight/sweep_chi512_k$k" \
        python $OUT/one_sweep.py --chi 512 --k $k \
        && nsys stats -q --force-export true \
          --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
          --format csv -o $P/nsys_chi512_k$k "$TMPDIR/nsight/sweep_chi512_k$k.nsys-rep" \
        || echo "nsys k=$k failed with status $?"
    done
    M=dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum
    M=$M,sm__sass_thread_inst_executed_op_dfma_pred_on.sum
    M=$M,sm__sass_thread_inst_executed_op_dmul_pred_on.sum
    M=$M,sm__sass_thread_inst_executed_op_dadd_pred_on.sum
    M=$M,sm__inst_executed_pipe_tensor_op_dmma.sum
    for k in 1 4; do
      timeout 240 ncu --profile-from-start off --metrics $M --csv \
        --log-file $P/ncu_site_chi512_k$k.csv \
        python $OUT/one_sweep.py --chi 512 --k $k --scope site \
        || echo "ncu k=$k failed with status $?"
    done
    ;;
  *)
    echo "unknown stage $1"; exit 2 ;;
esac
