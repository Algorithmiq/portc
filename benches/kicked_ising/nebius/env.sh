#!/bin/bash
# Record the node: GPUs, driver, CUDA, CPU, memory, disks, packages and commit.
set -euo pipefail
N=/home/ramon/src-method/benches/kicked_ising/nebius
mkdir -p $N/node
cd /home/ramon/src-method
nvidia-smi > $N/node/nvidia-smi.txt
nvidia-smi -q > $N/node/nvidia-smi-q.txt
nvidia-smi topo -m > $N/node/topo.txt
nvcc --version > $N/node/nvcc.txt
lscpu > $N/node/lscpu.txt
free -b > $N/node/free.txt
df -B1 / /dev/shm > $N/node/df.txt
uname -a > $N/node/uname.txt
git rev-parse HEAD > $N/node/git.txt
uv pip list --python .venv/bin/python > $N/node/packages.txt
KI_PLATFORM=nebius CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/python -W ignore::UserWarning -c "
import json, sys
sys.path.insert(0, 'benches/kicked_ising')
import bench_kicked_ising as b
print(json.dumps(b.provenance(), indent=1))
" > $N/node/provenance.json
