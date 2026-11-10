# Out-of-core benchmark

`bench_large.py` compresses the stack `N . V . M . U` of Pauli-transfer-matrix MPOs
(physical legs of 4) on one GPU or several, where `M` has a large bond and is read
from disk.

```bash
# Write the stack to node-local NVMe (205 GB for M at the defaults, complex128).
uv run python benches/large/bench_large.py generate /local/stack --bond-m 4000

# Compress it with detected budgets, spilling to the same disk.
uv run python benches/large/bench_large.py run /local/stack --chi-out 2000 \
    --scratch-dir /local/scratch

# Check that the plan does not change the result: two GPU budgets, same seed.
uv run python benches/large/bench_large.py generate /local/medium --bond-m 1000
uv run python benches/large/bench_large.py compare /local/medium --chi-out 500 \
    --small 40GB --large 80GB --scratch-dir /local/scratch

# Split the sweep among the GPUs of the node: 1, 2, 4 and 8 of them, in turn.
uv run python benches/large/bench_large.py scaling /local/stack --chi-out 2000 \
    --devices 1 2 4 8 --scratch-dir /local/scratch --results scaling.jsonl
```

`run` logs the wall time, the size of CuPy's pool on each GPU at the end (its high-water mark,
since the pool keeps its blocks), the host peak (`ru_maxrss`), the planned peaks,
the bytes spilled and the tiers. With `--debug` before the command, `portc` also
logs its plan, the time of each pass and the stall time (`SRC stalls`).

Acceptance of the out-of-core sweep on one A100-40GB, with at least 300 GB of host
memory and node-local NVMe:

1. `run` completes at 50 sites, `D_M = 4000`, `chi_out = 2000`, complex128.
2. The pool size stays within the GPU budget plus the `max(10%, 1 GiB)` margin
   the pool cap allows for fragmentation, and the host peak within the host
   budget.
3. The stall time is below 10% of the wall time.
4. `compare` at `D_M = 1000`, `chi_out = 500` reports a distance below `1e-10`.

`scaling` logs, for each GPU count, the wall time, the speed-up and the parallel
efficiency `T_1 / (G T_G)` against the first count, and the relative distance to
its output; `--results` appends the same as JSON lines. `--devices` with
`run` splits a single run. Acceptance of the multi-GPU sweep, at the reference
size on 8xH100 or 8xH200 (and 4xA100 on Leonardo):

1. Every distance is below `1e-10`.
2. The parallel efficiency is at least 80% on all the GPUs of the node.

`scaling` reports two distances. `distance` comes from inner products and cancels
down to about `sqrt(eps)`, near `1.5e-8` in complex128, so it cannot show `1e-10`.
`QR distance` sweeps a QR over the MPO `a - b` (the direct sum of the two outputs)
and resolves differences down to rounding; the acceptance applies to it.

## Leonardo

Recorded on Booster nodes of Leonardo at CINECA; see
[HPC clusters](https://algorithmiq.github.io/portc/hpc/leonardo) for the setup
and a job script. The nodes have no local disk, so the stacks and the spilled
environments live on Lustre scratch rather than NVMe.

### One GPU: D_M = 1000, chi_out = 500

`compare --small 4GB --large 60GB`, complex128, on one A100-64GB:

| GPU budget | Wall time (s) | Pool (GB) | Planned device peak (GB) | Host peak (GB) | Tiers (device/host/disk) |
|---|---|---|---|---|---|
| 4GB  | 129.8 | 5.46  | 3.86  | 40.6 | 1 / 49 / 0 |
| 60GB | 109.9 | 60.81 | 58.84 | 40.6 | 50 / 0 / 0 |

The two plans keep opposite tiers yet give bit-identical outputs (relative distance
`0.000e+00`). Spilling 49 of 50 environments to the host costs 18 % of the wall
time. Both pools stay within their caps (budget plus 10 % of the card, 6.9 GB).

### Several GPUs: D_M = 1000, chi_out = 500

One node, 4 A100-SXM-64GB fully connected by NVLink, with peer access on every
pair, so each GPU uploads a slice of every core and gathers the rest from its
peers. `scaling --devices 1 2 4`, complex128, every environment on the device:

| GPUs | Time (s) | Speed-up | Efficiency | QR distance |
|---|---|---|---|---|
| 1 | 109.9 | 1.00 | 100% | |
| 2 | 43.6 | 2.52 | 126% | 2.8e-13 |
| 4 | 32.7 | 3.36 | 84% | 4.4e-13 |

- The single-GPU time is that of a second SRC run in its job; the first SRC run of
  a job takes about 22 s longer, for a reason not isolated.
- The 2-GPU point is superlinear. The plan and tiers are the same on every count,
  and the sketch batch drops from 480 to 224 columns per GPU, which may run the
  contractions more efficiently; this was not isolated.
- This problem fits on one GPU, so it says little about the reference size.

### Several GPUs: D_M = 4000, chi_out = 2000 (reference size)

50 sites, complex128, 184 GB of input: 49 environments of 8.19 GB, 401 GB in all.
Speed-up and efficiency are against the single-GPU run:

| GPUs | Time (s) | Speed-up | Efficiency | Left-to-right (s) | Right-to-left (s) | Environments device / host / disk | QR distance |
|---|---|---|---|---|---|---|---|
| 1 | 3925.3 | 1.00 | 100% | 794.8 | 3112.1 | 0 / 45 / 4 | |
| 2 | 2348.8 | 1.67 | 84% | 730.6 | 1606.3 | 1 / 35 / 14 | 7.5e-13 (vs 4 GPUs) |
| 4 | 1099.9 | 3.57 | 89% | 240.4 | 844.1 | 1 / 45 / 3 | |

- Both efficiencies are above the 80% target.
- Before the out-of-memory split, every run at this size failed. The live memory
  stayed within the plan, but CuPy's pool held up to 66.7 GB in split blocks and a
  12.6-14.7 GB allocation did not fit under its 67.6 GB cap. Each GPU of the
  multi-GPU runs now splits one sketch batch in two and finishes.
- Four GPUs keep only one environment on the device. The plan picks the largest
  batches that fit first, and their working set takes each GPU's budget (59.0 GB
  planned peak), although a GPU holds only 2.05 GB of each environment.
- Three or four environments go to disk on one and four GPUs: the host budget, 90%
  of the available memory (about 470 GB), falls short of the 487-496 GB that
  keeping them all in host memory takes. On two GPUs, 14 go to disk with a planned
  host peak of 381.7 GB against 462.6 GB on four; why the plan leaves that host
  memory unused was not investigated.
- The stall times the sweep logs include waiting for the GPU's own queue, so they
  cannot tell slow transfers from a GPU-bound pass.

### Einsum paths: `auto` against `dp`

The `optimize` argument of `src`, `apply` and `compress` selects the opt_einsum
path optimizer of the kernels, and the planner sizes batches for the same path.
It defaults to `dp` since these measurements; the results above ran with `auto`.
`bench_einsum_dp.py` compares the two: `kernels` times `SiteKernels.env`,
`.sketch` and `.project` on random arrays of the reference shapes, and `sweep`
with `distance` runs the `D_M = 1000` stack, `chi_out = 500`, under the 60 GB
budget, three times per optimizer and compares the outputs. Run on A100-SXM-64GB
(CuPy 14.2.0, NumPy 2.4.4, opt_einsum 3.4.0), complex128.

Full sweep, median of 3 (range), every environment on the device:

| GPUs | `auto` (s) | `dp` (s) | Speed-up | Right-to-left `auto` / `dp` (s) | Planned peak `auto` / `dp` (GB) | QR distance |
|---|---|---|---|---|---|---|
| 1 | 109.7 (109.6-130.6) | 56.2 (56.2-74.1) | 1.95x | 95.4 / 42.3 | 58.8 / 50.3 | 2.7e-13 |
| 4 | 32.4 (32.3-51.7) | 17.4 (17.4-40.1) | 1.86x | 28.6 / 13.6 | 32.3 / 12.2 per GPU | 4.0e-13 |

- The gain is in the right-to-left pass; the left-to-right pass takes the same
  time (14.3 against 13.9 s on one GPU). At these shapes `auto`'s sketch and
  projection paths cost 3.9x and 2.0x `dp`'s flops. Both optimizers give the
  same planned batches.
- The first sweep of each job is 18-23 s slower with either optimizer (the
  first-run effect noted above), and is left out of the medians. `dp`'s path
  search takes about 1 ms per shape: 0.03-0.10 s per sweep.

Kernels at the reference shapes, `D = (4, 4, 4000, 4)`, physical legs 4,
`eta = 2000`, median of 5 after 2 warm-ups (ms per row or column):

| Kernel | Batch | `auto` | `dp` | Flops `auto` / `dp` |
|---|---|---|---|---|
| project | 8 | 25.2 | 11.8 | 7.9e11 / 4.0e11 |
| project | 16 | 22.1 | 11.0 | 1.6e12 / 7.9e11 |
| project | 32, 64, 128 | over 50 GB | 10.6, 10.5, 10.5 | 9.7e12 / 1.6e12 at 32 |
| sketch | 96 | 10.27 | 10.65 | 4.78e12 / 4.76e12 |
| sketch | 192 | 10.14 | 10.53 | 9.56e12 / 9.51e12 |
| env | 224 | 7.17 | 7.18 | 7.4e12 / 7.4e12 |

- The projection runs at 16-19 TFLOP/s with either optimizer, so the time follows
  the flops. From 32 rows `auto` plans an 8.19e9-element intermediate (265-283
  GB), which is why it cannot leave 16 rows; `dp` needs 10-25 GB up to 128 rows.
- 384 sketch columns (75 GB) and 256 projection rows with `dp` (50.3 GB) exceed
  the 50 GB limit and were not run.
- `dp` is 3.7-3.8% slower on the sketch, where its path has 0.5% fewer flops.
- The outputs of the two optimizers agree to 1.6e-14 or better.
- The reference-size sweep (over an hour) was not run. Taking these per-row and
  per-column times for each site predicts its right-to-left pass at about 2060 s
  with `dp`, 1.5x faster than the 3112 s measured with `auto`.
