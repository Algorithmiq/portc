## Conclusion (Nebius: stages 1-3)

Every Leonardo run, redone on one Nebius node, then stage 3 extended to `chi`
4096 and 8192.

- **Node.** 8x RTX PRO 6000 Blackwell Server Edition (97.9 GB each), on PCIe with
  no NVLink. GPUs 0-3 sit on NUMA node 0 and GPUs 4-7 on NUMA node 1. The node has
  192 cores and 1.7 TB of RAM. Details are in `node/`.
- **Library.** src-method is PR #85 (`feat/distributed-qr`, 0.5.3.dev12) unchanged;
  `git diff origin/feat/distributed-qr -- src/` is empty. quimb is 1.15 with
  autoray 0.11, on the same GPU.
- **Precision.** Stages 1 and 2 ran in complex128. Stage 3 ran in complex64, except
  the complex128 cross-check.
- **Job layout.** Single-GPU jobs ran 8 at a time, one per GPU, each pinned to 24
  cores of its GPU's NUMA node (`driver.py`). Multi-GPU runs had the node to
  themselves.

Labels in the tables: `c64` is complex64, and a label without it is complex128.
`cut 1e-12` is the truncation cutoff, and no `cut` means cutoff 0. `+gesvd` is the
bench-side SVD fallback. `householder` is the distributed CholeskyQR disabled (see
below). `-g8` is 8 GPUs.

**Bottom line.**

- **Correct, and the same as Leonardo.** The checks reproduce dense, statevector,
  stim and the published exact references to 1e-13-1e-16, as on Leonardo.
  - 553 complex128 runs finished on both machines. Among those with errors above
    1e-11, 10 differ by more than 25%; they are listed below.
  - No run gave NaN in complex128.
- **On this node complex128 is slow, complex64 is fast.**
  - A complex128 GEMM runs at 1.5 TFLOP/s, a complex64 one at 75-82 TFLOP/s.
  - Stages 1 and 2 (complex128, one GPU per run) took 1.3-2.65x longer than on a
    Leonardo A100 (medians per method).
  - Stage 3 in complex64 took 2.8x (`chi = 1024`) and 3.2x (`chi = 2048`) less time
    than complex128 on an A100.
- **complex64 moves the stage 3 value far less than its error.** On `z62:20:8:src-k1:1024`,
  complex64 and complex128 differ by 1.9e-3 in `<Z62>`. Both errors against the
  reference are about 0.08. The Nebius and Leonardo complex128 values differ by
  8.9e-4.
- **`chi = 8192` fits on one node.** 20 steps took 9.7 h on 8 GPUs, using 49 GB of
  each GPU's memory pool, a 916 GB host peak and no disk.
- **The 20-step `<Z62>` error does not fall with `chi` at `theta_h = 8pi/32`.**
  Against the MIX reference it is 0.0796, 0.0870, 0.0881 and 0.0850 at `chi` 1024,
  2048, 4096 and 8192. At 9 steps, where an exact value exists, the same runs are
  at 5.3e-3 (`chi` 4096) and 3.6e-3 (8192). The MIX method itself is within 1.7e-4
  of exact there.
- **The distributed CholeskyQR of PR #85 never accepted a sketch of the real
  complex64 evolution.** It declined 66 of 66 attempts per step at `chi = 4096`.
  The final `chi` 4096 and 8192 runs therefore use Householder QR on rank 0. See
  `cholesky_qr.md`.
- **quimb in complex64 fails on 127 sites.** Fit returns NaN after one step at
  `chi = 1024`. Zip-up spent 1661 s in its warm-up step (about 6 s in complex128),
  and its first timed step had not finished after about 2.5 h, so it was killed.
- **Tensor cores do not speed up complex64 here.** TF32 and BF16x9 give the same
  bits and the same speed as FP32. FP64 emulation is accurate (2-4e-15) and makes
  complex128 GEMMs 11-26x faster, but whole runs only 1.00-1.46x faster.

### Decisions that differ from the brief

- **Stage 2 in complex128**, at the user's request, so that it compares
  one-to-one with Leonardo. Stage 3 is complex64 as briefed, with the complex128
  cross-check of `z62:20:8:src-k1:1024:0`.
- **Native FP64, not emulated,** for every complex128 run. The emulation runs are
  separate, in `fp64-emulation/runs/`, and appear in no table of the summary.
- **The `chi = 4096` run with CholeskyQR stopped at step 10** under the brief's
  stop rule ("CholeskyQR falling back at every site"). Its 10 steps are in
  `runs/` (the `src-k1-g8 c64` row, 0/1 done).
  - It was rerun with `run --householder`, a bench-side patch of
    `src_method._sweep.DISTRIBUTED_QR_MIN` added for this purpose, the same patch
    the QR A/B probes use.
  - The two runs agree bit for bit on steps 1-10.
  - `chi = 8192` ran with `--householder` from the start.
- **The gate on the `chi = 4096` probe** checked CholeskyQR fallbacks in
  aggregate (66 of 275 tried) and passed. Per group, 4 of the 6 groups declined
  every site they tried. That was the first sign of the problem above.
- **`chi = 8192` at `theta_h = 8pi/32` only,** for its cost.
- **The complex64 zip-up evidence run** (`z62:1:8:quimb-zipup:1024:0`) was
  killed; its empty run file was removed. The fit run (NaN) is the `z62, 1 steps`
  section.

### Comparison with Leonardo (complex128, one GPU per run)

Time on Nebius over time on Leonardo, per run, for runs finished on both:

| method | median ratio | total Nebius / Leonardo |
|---|---|---|
| quimb-fit | 1.30 | 3.24 h / 2.65 h |
| quimb-src | 1.38 | 0.45 h / 0.34 h |
| quimb-src-os | 1.56 | 0.98 h / 0.67 h |
| quimb-zipup | 1.68 | 1.77 h / 1.00 h |
| src-k1 | 1.52 | 4.44 h / 2.23 h |
| src-k1-os | 1.70 | 3.46 h / 2.03 h |
| src-k2 | 1.52 | 0.74 h / 0.54 h |
| src-k2-os | 2.02 | 1.67 h / 1.01 h |
| src-k3 | 1.97 | 2.35 h / 1.85 h |
| src-k3-os | 2.65 | 2.79 h / 1.89 h |

The 10 runs whose errors differ by more than 25% (errors above 1e-11):

- `w17:5:8:quimb-zipup:256:0`, with and without cutoff. The two machines reach
  either 6.6e-7 or 6e-6, swapped between cutoff 0 and cutoff 1e-12 (Nebius 6.6e-7
  and 6.6e-6, Leonardo 5.8e-6 and 6.7e-7).
- Randomized methods, within the seed scatter of the same cell:
  - `w17:5:8:src-k1-os:128`, seeds 1 and 2;
  - `z62:9:6:quimb-src(-os):256`, four runs;
  - `z62:9:8:src-k1-os:256:0` and `src-k2-os:256:1`.

**Runs that finished here but not on Leonardo, 16 in all.**
- Four 20-step zip-up runs at `chi = 1024` with cutoff 1e-12.
- Two 9-step src-k1-os runs at `chi = 1024`; Leonardo stopped them on its
  30-minute limit.
- Ten runs that failed or were cut off on Leonardo.

**Host SVD failures.** `LinAlgError: SVD did not converge` stopped 13 of 432
src-method runs: 2 in stage 1 and 11 in stage 2. They are listed in the summary
and kept in `runs_failed/`.
- The failing configurations differ from Leonardo's 11.
- All 13 finished when rerun with the gesvd fallback (`+gesvd`).

### Stage 1 and the precision probe

- **weight-10:** exact (1e-13-1e-16) from `chi = 384` at every angle, as on
  Leonardo.
- **weight-17:** within 1.2e-4 at `chi = 1024`, except `theta_h = 10pi/32` and
  `11pi/32` (4.0e-4 and 9.1e-4).
- **Precision probe** (`w10:5:8:src-k1:1024`, cutoff 1e-12): error 8.0e-16 in
  complex128 and 2.9e-6 in complex64.

### GEMM modes (`gemm_probe.py`, `gemm_probe.json`)

Complex GEMM on one GPU. The error is relative to a pedantic complex128 product.

| mode | 2048^3 | 8192^3 | 65536 x 2048 x 2048 | error |
|---|---|---|---|---|
| complex64, FP32 | 74.9 TFLOP/s | 80.4 TFLOP/s | 79.9 TFLOP/s | 5e-7 to 1e-6 |
| complex64, TF32 (CuPy compute type) | 75.0 | 81.6 | 81.1 | same bits as FP32 |
| complex64, BF16x9 (`CUBLAS_EMULATE_SINGLE_PRECISION`) | 74.9 | 81.7 | 82.0 | same bits as FP32 |
| complex128, FP64 | 1.5 | 1.5 | 1.5 | 0 |
| complex128, emulated (`CUBLAS_EMULATE_DOUBLE_PRECISION`) | 16.6 | 38.5 | 27.6 | 2.1e-15 to 4.3e-15 |

The `eager` emulation strategy gives the same numbers.

FP64 emulation on whole complex128 runs (`fp64-emulation/runs/` against
`runs/`, seed 0):

| run | native | emulated | error, native / emulated |
|---|---|---|---|
| `w10:5:8:src-k1:1024`, cut 1e-12 | 34.2 s | 31.1 s | 8.0e-16 / 2.0e-15 |
| `z62:9:8:quimb-zipup:256`, cut 1e-12 | 141.8 s | 141.5 s | 4.11e-3 / 4.11e-3 |
| `z62:9:8:src-k1-os:512`, cut 1e-12 | 652.0 s | 637.2 s | 1.137e-2 / 1.137e-2 |
| `z62:9:8:src-k1:1024`, cut 1e-12 | 426.3 s | 291.2 s | 1.558e-2 / 1.558e-2 |

### Stage 3: 20-step `<Z62>`, cutoff 0 unless stated

Error against Begušić, Gray and Chan's MIX values (`reference/bgc/mix.csv`).
The bench docstring states their error as below 0.01. Their 9-step MIX values
are within 8e-4 of exact at every angle (1.7e-4 at `8pi/32`). The Leonardo
column is complex128 on one A100.

| `theta_h` | MIX | src-k1, 1024 | src-k1, 2048 | src-k1-g8, 4096 | src-k1-g8, 8192 | zip-up cut 1e-12, 1024 | Leonardo src-k1, 1024 / 2048 |
|---|---|---|---|---|---|---|---|
| `5pi/32` | 0.7970 | 0.0377 | 0.0384 | - | - | 0.0486 | 0.0374 / 0.0393 |
| `7pi/32` | 0.5110 | 0.1816 | 0.1945 | - | - | 0.2000 | 0.1804 / 0.1917 |
| `8pi/32` | 0.2686 | 0.0796 | 0.0870 | 0.0881 | 0.0850 | 0.1143 | 0.0806 / 0.0847 |
| `10pi/32` | 0.0163 | 5.07e-3 | 2.83e-3 | - | - | 4.48e-4 | 5.52e-3 / 3.25e-3 |

- **Precision.** All src-method entries are complex64. Zip-up is complex128,
  since complex64 zip-up does not run (above).
- **Cross-check at `8pi/32`, `chi = 1024`.** complex128 on Nebius gives 0.18709
  (error 0.0815) in 7384 s. complex64 gives 0.18903 in 470 s, and Leonardo
  0.18797 in 1333 s.
- **`chi` 4096 and 8192 step by step.** They agree to about 1e-5 up to step 7.
  From step 8 to 16 the `chi = 8192` value is the higher one, by up to 0.019
  (step 12). From step 17 to 19 it is the lower one, and at step 20 it is 0.003
  higher.
- **quimb zip-up:**
  - At cutoff 0 it runs out of GPU memory at step 3 (`chi = 1024`) and step 2
    (`chi = 2048`), at every angle.
  - At cutoff 1e-12 and `chi = 2048` it runs out at step 9.
  - At cutoff 1e-12 and `chi = 1024` it completes 20 steps in 14977-17743 s,
    with 65-88 GB of GPU memory. On Leonardo's 64 GB A100s it ran out at steps
    13-16 at `5pi/32`, `7pi/32` and `8pi/32`. At `10pi/32` it was stopped by the
    job time limit at step 16.

### Cost of stage 3, measured on whole runs

| run | GPUs | s / step (median) | 20 steps | GPU pool / device | host peak |
|---|---|---|---|---|---|
| src-k1, 1024, complex128 | 1 | 372 | 7384 s | 9.4 GB | 32 GB |
| src-k1, 1024, complex64 | 1 | 23.5 | 470 s | 4.7 GB | 13 GB |
| src-k1, 2048, complex64 | 1 | 105.5 | 2102 s | 18.8 GB | 49 GB |
| src-k1-g8, 4096, complex64, Householder | 8 | 404 | 7897 s | 13.0 GB | 212 GB |
| src-k1-g8, 4096, complex64, CholeskyQR | 8 | 418 | stopped at step 10 | 13.0 GB | 211 GB |
| src-k1-g8, 8192, complex64, Householder | 8 | 1776 | 35062 s | 49.2 GB | 916 GB |
| Leonardo src-k1, 1024, complex128 | 1 A100 | 67 | 1333 s | | |
| Leonardo src-k1, 2048, complex128 | 1 A100 | 342 | 6772 s | | |

**Step probes** (`probe/`). Each probe times one step of a random operator at the
saturated bond.

| `chi` | CholeskyQR | Householder |
|---|---|---|
| 2048 | 138.8 s | 131.4 s |
| 4096 | 410.8 s | 357.5 s |
| 8192 | - | 1681.5 s |

- **Where CholeskyQR was accepted, it was slower.** At `chi = 4096`, groups 0
  and 1 accepted most sites, and their right-to-left sweeps took 88.4 s and
  66.6 s, against 54.3 s and 53.9 s with Householder.
- **No disk anywhere.** No probe or run used the disk tier.
- **Real runs against probes.** At `chi = 4096` the real run's steps were 13%
  slower than the probe's; at `chi = 8192` they were 6% slower (1776 s against
  1681.5 s).

### Caveats

- **Shared host.** Single-GPU runs shared the node 8 ways (host memory, and
  NUMA-local cores per GPU), so host-side work may have contended. Multi-GPU runs
  had the node to themselves.
- **Warm-up.** As on Leonardo, each method got one untimed warm-up step.
- **Environment.** quimb on CuPy 14 needs autoray 0.11. `prek`'s uv hooks
  reinstall the locked 0.8.10, so reinstall 0.11 after running prek.
- **Flops.** Counted as on Leonardo: complex64 and complex128 count the same
  nominal flops.
- **Provenance.** The provenance block in the summary is that of one run. Every
  run file carries its own (dtype, GPUs, environment variables).
- **Logs.** Driver logs are in `logs/`. The per-run logs (`s1-*`, `s2-*`, `fb-*`,
  `s3-*`) are not committed, because of their size.
