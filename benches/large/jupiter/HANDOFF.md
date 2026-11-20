# Handoff: SRC strong scaling on Jupiter GH200 (paper Fig. 4)

Status 2026-11-18: all runs done; analysis and the figure are not.
Numbers below come from `results.json` in this folder. That file is built from
`raw/`, `logs/`, `node/` and `env.json`, copied back from Jupiter.

## Goal

Ramon asked for this in Slack, DM 2026-11-17 and channel thread C0CJ4SPTKT4. Rerun the
Fig. 4 strong-scaling experiment on one GH200 node: 4 GPUs, NVLink, G = 1, 2, 4.
Record both the measurement and the performance-model prediction. Fig. 4 then shows
four curves: RTX measured and predicted (PCIe), GH200 measured and predicted (NVLink).
His instructions:

- Use the pipeline from `bench/nebius-rtx` (`benches/large/nebius/`), but run current `main`.
- Keep the paper's conditions as far as the machine allows.

## What was done

**Branch.** `bench/jupiter-gh200` starts at `main` `6d39db3`. It is local only, never
pushed. `src/` is unchanged. Commits:

| Commit | Change |
|---|---|
| `b77ff8a` | Copied the nebius pipeline into `benches/large/jupiter/`, `src_method` renamed to `portc` |
| `0d19187` | Model and trace `main`'s distributed CholeskyQR; sbatch scripts; `bench_large.py` and its test from `bench/nebius-rtx` |
| `a37efa1` | Time runs with the shell's `time` (Jupiter has no `/usr/bin/time`); read the commit from `.git` (no `git` on compute nodes) |
| `1ef4de6` | Same `.git` fix in `make_env.py` |
| `6d1ddc1` | Start and end timestamps of each command in the Slurm output |
| `fab79fe` | Step 4 walltime raised to 4 h; `REPS` and `TRACES` select repetitions and traces |

Commits from `0d19187` on are unsigned: commit signing needs an ssh-key passphrase that
was not available. Sign them with `git commit --amend -S` or a rebase before pushing.

**Why the scripts changed.** The RTX runs ran at `e9ab83c`, which predates two `main` PRs:

- #75: one non-blocking stream per rank.
- #85: distributed CholeskyQR. When G > 1 and `chi_out >= 1024`
  (`src/portc/_sweep.py:470-475`), it replaces gather, Householder QR on rank 0 and
  scatter with two all-to-alls plus Gram allgathers.

At the reference size, `main` therefore takes a different QR path from the RTX model.
Changes made for it:

- `trace.py` also times `alltoall`, `cholesky_qr` and the Gram allgather.
- `probe_env.py` adds an all-to-all bandwidth and a per-G `cholesky_qr` timing.
- `predict.py` uses the CholeskyQR terms above the threshold and the old terms below it.

On Slack, Ramon accepted the QR difference as minor: state it in the paper, no RTX rerun.

`main`'s `bench_large.py` cannot write complex64 stacks. It also still has the complex
variance bug that overflows complex64 runs to NaN (fixed in `a49b19d`). The branch
version was copied in. It is bench-only.

**Jupiter.**

- Login: `login.jupiter.fz-juelich.de`, user `materia1`, account `e-dev-2026d11-370`,
  partition `booster`. Do not use `e-dev-2026d01-085`: it is blocked.
- Work directory: `/e/project1/e-dev-2026d11-370/materia1/portc`, copied over with rsync,
  not cloned.
- Environment: `uv` in `~/.local/bin`, venv from `uv sync --all-groups --extra gpu-nvidia`.
- Stacks are in `/e/project1/.../materia1/stacks` and are copied to `/dev/shm` before each job.
- Submit from `benches/large/jupiter/`: `probe.sbatch`, `generate.sbatch`,
  `step3.sbatch`, `step4.sbatch`. All of them source `common.sh`.
- Batch logins only work through an ssh control socket. Open
  `ssh login.jupiter.fz-juelich.de` in a terminal first; it persists for 12 h.

**Jobs run.**

| Job | What it did | Outcome |
|---|---|---|
| 2978368 | Generated the m4000 and m1000 complex64 stacks | done |
| 2978569 | Node probe, TF32 control, `env.json` | done |
| 2978570 | 1-GPU trace at m4000, then `predict.py` | Outputs written by 10:09. The script then hung until its 1 h limit. Cause unknown; `predict.py` exits in 7 s on the login node. |
| 2979643 | Step 4 repetition 1 (r1), both sizes | done; hit the 2 h limit during r2 |
| 2982618 | Step 4 r2, both sizes, plus the G = 2 and G = 4 traces | done, 1:30 h |

The first submissions (2978279/80, 2978367) failed on the missing `/usr/bin/time` and `git`.

## Results (`results.json`)

**Conditions.** complex64 with TF32 off (`CUPY_TF32=0`, `NVIDIA_TF32_OVERRIDE=0`), as in
the paper's Fig. 4. Sketch seed 0, 50 sites, depth 4.

- m4000: `D_M=4000`, `chi_out=2000`. The reference size; uses distributed CholeskyQR at G > 1.
- m1000: `D_M=1000`, `chi_out=500`. Uses the old gather/Householder path.

**Node** (`probe`, `env.json`):

- NVLink `NV6` between every GPU pair.
- Every peer collective runs at about 130 GB/s per receiving GPU (allgather, gather,
  scatter, all-to-all), at both G = 2 and 4. RTX managed 16-41 GB/s.
- Pinned host-to-device copies: about 420 GB/s into the GPU on the probe thread's NUMA
  node, 89 GB/s into the other three.
- GEMM: 55.4 TFLOP/s in complex64 and 64.1 TFLOP/s in complex128.
  - Strict FP32 never uses tensor cores. complex128 uses the H100's FP64 tensor cores.
  - So one GH200 is slower than one RTX in complex64: about 1150 s against 847 s.
- TF32 control, complex64 4096 matmul: relative error 1.6e-6 with TF32 off, 2.9e-4 with it on.

**m4000 scaling**, medians of r1 and r2 (`measured.m4000`):

| G | r1 / r2 (s) | Median (s) | Speed-up | Efficiency | Predicted `T_G` (s) | Predicted efficiency | RTX efficiency (paper) |
|---|---|---|---|---|---|---|---|
| 1 | 1138.0 / 1163.4 | 1150.7 | 1.00 | 100% | 1174.5 (traced T1) | -- | -- |
| 2 | 585.5 / 599.2 | 592.4 | 1.94 | 97.1% | 590.0 | 99.5% | 94% |
| 4 | 301.8 / 304.7 | 303.3 | 3.80 | 94.9% | 296.8 | 98.9% | 84% |

**The prediction in absolute time is within 0.4% (G = 2) and 2.1% (G = 4) of the
measured medians.** Its efficiencies are relative to the traced 1-GPU run, 1174.5 s.
That run is 2% slower than the median of the two untraced 1-GPU runs, so the predicted
efficiencies come out higher. Use one T1 for both curves when plotting.

**m1000 scaling** (`measured.m1000`), medians: 23.36, 11.51 and 6.43 s. Speed-ups 2.03
and 3.63; efficiencies 101.5% and 90.8%.

**Where the time goes**, from the traces (`traces_m4000`, rank 0, right-to-left pass,
summed over sites):

| G | Right-to-left wall (s) | Kernels (s) | all-to-all (s) | `cholesky_qr` (s) | Householder QR (s) | S allgather (s) |
|---|---|---|---|---|---|---|
| 1 | 852.6 | -- | -- | -- | 4.4 | -- |
| 2 | 431.8 | 414.1 | 5.8 | 3.4 | 0.2 | 5.9 |
| 4 | 219.7 | 207.8 | 3.2 | 2.2 | 0.2 | 4.4 |

- CholeskyQR is accepted in complex64 on these random stacks. Householder QR is left
  with 0.2 s in total.
- Communication is 4-5% of the right-to-left pass. Kernel time halves with G.

**Accuracy.** The QR distance of the G > 1 outputs to the G = 1 output is 1.9-2.7e-4 at
m4000 and 6.6-8.5e-5 at m1000. The paper reports about 2.0e-4 on RTX.

`measured.m4000_partial_r2_excluded` holds one G = 1 record (1127.5 s) from the job that
hit its time limit. It is excluded from the medians.

## Open items

1. **Figure 4.** Add the GH200 measured and predicted curves, plus one sentence: the RTX
   points used the gather/Householder QR path, the GH200 points use distributed
   CholeskyQR. Ramon offered two separate plots; not decided. The location of the
   figure's script and data is unknown.
2. **`analyze.py`, `prediction.md` and `summary.md`** are not adapted to Jupiter.
   `analyze.py` still expects the RTX steps 5 and 6 and G = 8.
3. **About 37 min of each m4000 scaling command is spent outside the timed runs**
   (71 min of wall time against 34.5 min of runs). Probably the output comparisons on
   the CPU; not checked. It may also be what hung job 2978570.
4. **Optional complex128 run on the FP64 tensor cores.** It departs from the paper's
   Fig. 4 conditions. Check the memory plan first: host peak about 2x (~600 GB), against
   921 GB of host memory per node (`node/free.txt`).
5. **Before pushing:** sign the commits, and decide whether `logs/`, `raw/` and `node/`
   are committed, as on `bench/nebius-rtx`. They are copied back but not committed yet.
