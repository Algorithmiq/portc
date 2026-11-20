#!/bin/bash
#SBATCH --account=e-dev-2026d11-370
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --job-name=portc-probe
#SBATCH --output=%x-%j.out
# Node probes, the TF32 control and env.json; with DT=c128 only the probe, its
# CholeskyQR in complex128.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/common.sh"
run probe$S uv run python "$N/probe_env.py" --cqr-dtype $DTYPE
grep '^{' "$N/logs/probe$S.log" > "$N/raw/probe$S.json"
[ -z "$S" ] || exit 0
nvidia-smi topo -m > "$N/node/topo.txt"
nvidia-smi -q > "$N/node/nvidia-smi-q.txt"
lscpu > "$N/node/lscpu.txt"
numactl -H > "$N/node/numactl.txt"
free -b > "$N/node/free.txt"
lsblk > "$N/node/lsblk.txt"
nvcc --version > "$N/node/nvcc.txt"
uname -a > "$N/node/uname.txt"
df -B1 "$W" /dev/shm > "$N/node/df.txt"
: > "$N/logs/tf32-control.log"
for t in 0 1; do
  CUPY_TF32=$t NVIDIA_TF32_OVERRIDE=$t uv run python -c "
import cupy as cp, numpy as np
rng = np.random.default_rng(0)
a, b = (cp.asarray(rng.normal(size=(4096, 4096)) + 1j * rng.normal(size=(4096, 4096)), cp.complex64) for _ in range(2))
ref = cp.asarray(a, cp.complex128) @ cp.asarray(b, cp.complex128)
print('TF32=$t', float(cp.linalg.norm(a @ b - ref) / cp.linalg.norm(ref)))" >> "$N/logs/tf32-control.log"
done
uv run python "$N/make_env.py" "$W" /dev/shm
