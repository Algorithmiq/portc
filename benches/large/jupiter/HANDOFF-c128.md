# Handoff: complex128 strong scaling on Jupiter GH200 (paper Fig. 4)

Status 2026-11-23: runs done. Numbers below come from `results-c128.json` in this folder,
built from `raw/*c128*` and `logs/`. The complex64 runs are in `HANDOFF.md` and
`results.json`; this file only covers what differs.

## What was run

Same experiment as the complex64 curve: 50 sites, depth 4, sketch seed 0, m4000
(`D_M=4000`, `chi_out=2000`, distributed CholeskyQR at G > 1) and m1000 (`D_M=1000`,
`chi_out=500`), on G = 1, 2, 4 of one 4x GH200 node. The stacks are complex128, so cuBLAS
ZGEMM runs on the FP64 tensor cores (DMMA, cuBLAS default; nothing to switch on). The
probe measures 64.6 TFLOP/s for complex128 GEMM against 55.9 for complex64.

**Timing rule.** Reported time is wall time minus the largest per-rank wait for stack
reads ("site stall"). The stacks are memory-mapped from the project file system
(`/e/project1`) instead of being copied to `/dev/shm`, since the complex128 run needs that
memory. Waits for host-to-device environment transfers stay in. No run spilled to disk.
Raw wall times are kept in the JSON.

| Job | Commit | What |
|---|---|---|
| 3024984-3024986 | `1cab278` | Stacks, node probe, traced 1-GPU run and prediction |
| 3073006 | `ed24168` | Both sizes on G = 1, 2, 4, twice, QR distances, traces at G = 2, 4 |
| 3077865 | `ac3e463` | m4000 G = 4 at 90 GB GPU budget, twice (`g4budget.sbatch`) |
| 3077925 | `bf59ccc` | m4000 G = 1, 2 at 60 GB GPU budget (`capped.sbatch`); cancelled after one run of each |

## Why the GPU budget matters on GH200

Each GH200's GPU memory is also a host NUMA node. Once the CPU memory (about 480 GB) is
full, host pages, file cache included, land in GPU memory. complex128 m4000 plans a
496 GB host peak, so part of it always overflows. At G = 1 and 2 the idle GPUs absorb it;
at G = 4 no GPU is idle and the sweep ran out of GPU memory (six failed jobs,
3028888-3072405, logs on Jupiter only). What made G = 4 run:

- every run in its own process, with CuPy pools freed and the file cache of the stacks
  and scratch files evicted at its start;
- `--reserve`: the run takes its GPU budget plus margin before the timer starts, so host
  pages cannot take it;
- `--host-memory 520GB`, because the reserved memory no longer counts as available and
  the default host budget then spills to disk.

With the GPU memory reserved, the file cache of the stacks has nowhere to go and is
re-read from `/e/project1`: that is the site stall subtracted below.

The GPU budget sets the sketch batch (352 at about 90 GB, 192 at 60 GB) and with it the
environment transfer waits. So the two curves below differ in budget, not in code.

## m4000 results

**Full GPU memory** (`m4000.full_gpu_memory`): planner default (about 90 GB) at G = 1, 2;
90 GB with `--reserve` at G = 4. Medians of two runs.

| G | Wall (s) | Max site stall (s) | Minus stall (s) | Speed-up | Efficiency | Predicted (s) | Predicted efficiency |
|---|---|---|---|---|---|---|---|
| 1 | 1356.1 / 1357.6 | 0.0 / 0.0 | 1356.8 | 1.00 | 100% | 1395.9 (traced T1) | -- |
| 2 | 707.9 / 709.9 | 0.0 / 0.1 | 708.9 | 1.91 | 95.7% | 702.6 | 99.3% |
| 4 | 458.1 / 401.7 | 120.5 / 62.6 | 338.4 | 4.01 | 100.2% | 354.7 | 98.4% |

**60 GB GPU budget at every G** (`m4000.gpu_budget_60GB`), `--reserve`, 520 GB host
budget. G = 1 and 2 are one run each; G = 4 is the median of two.

| G | Wall (s) | Max site stall (s) | Minus stall (s) | Speed-up | Efficiency |
|---|---|---|---|---|---|
| 1 | 1413.0 | 7.7 | 1405.3 | 1.00 | 100% |
| 2 | 734.7 | 23.1 | 711.6 | 1.97 | 98.7% |
| 4 | 384.1 / 387.1 | 8.4 / 15.5 | 373.7 | 3.76 | 94.0% |

Notes:

- The full-memory G = 4 point depends on the timing rule: its raw wall times differ by
  56 s between the two runs, almost all of it site stall, while the times minus the stall
  agree within 0.5%. In raw wall time its efficiency is 79%.
- At full memory the G = 4 environment waits drop to 58-78 s per GPU, against
  750 s / 4 = 188 s expected from G = 1; this is why the point is slightly superlinear.
  Not explained further.
- The 60 GB curve is like-for-like but slower at every G, and the prediction was built
  from a full-memory trace, so it applies to the full-memory curve.
- QR distance of the outputs to G = 1 (computed in job 3073006 only): 6.0e-13 and 4.1e-13
  at G = 2, 7.5e-13 at G = 4 (60 GB). The complex64 runs gave about 2e-4.

**Traces** (`traces_m4000`, rank 0, right-to-left pass): 1012 s, 622 s and 273 s at
G = 1, 2, 4 (G = 4 traced at 60 GB). Sketch and project kernels halve with G;
all-to-all, CholeskyQR and the S allgather together stay at 13-17 s.

## m1000

Planner default budgets (`m1000`). Medians minus site stall: 23.7, 12.6 and 7.7 s;
efficiencies 94.2% and 77.2%. Site stalls are 13-21 s of the 37-45 s G = 1 wall time,
so these runs mostly measure file-system reads. QR distances 3.9e-13.

## For the figure

`paper_csv_strong_scaling_gh200_c128` holds the rows of a
`figures/data/strong_scaling_gh200_c128.csv`, in the columns of the existing GH200 file,
for both curves. Which one goes in the paper is open:

- full GPU memory: the conditions of the complex64 runs and of the prediction, but the
  G = 4 point is 100.2% only once stack reads are excluded;
- 60 GB: one budget for every G, monotone (98.7%, 94.0%), single runs at G = 1 and 2.

Either way the caption should say that times exclude stack reads from the file system.
