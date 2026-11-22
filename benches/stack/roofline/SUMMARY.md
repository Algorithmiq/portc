# Trotter fusion on Leonardo: retimed table and roofline

One A100-SXM-64GB of an exclusive Booster node, `portc` at `4485588` (`main`),
opt_einsum `dp`; versions and job IDs in `env.txt`. Scripts: `job.sh` (stages),
`ceilings.py`, `profile_quench.py` (patches the library at run time, never edits
it), `one_sweep.py` (ncu/nsys driver), `report.py` (merges `parts/` into the CSVs
and draws `roofline.png`).

## Short answer

Fusion wins because of **QR and per-sweep fixed costs**, not kernel efficiency.

- The QR of the sketch costs the same per sweep whatever `k` is, and it runs at
  only 0.3–2.4 TFLOP/s. It is half of every saving.
- Per-sweep copies (mostly the device-to-host copy of the output state, which
  runs at 3–14 GB/s) and the Python and launch overhead make up most of the rest.
- At `chi >= 512` the pairwise contractions are *not* bandwidth bound. They run
  at 9–18 TFLOP/s, 40–83% of the measured ZGEMM ceiling, and as fast as or faster
  than the fused ones.
- The contraction flops follow the `2**k * chi**3` model exactly: equal at
  `k = 2`, doubled at `k = 4`. That is why four steps now *lose* to two at
  `chi >= 512`.

## Hardware ceilings (`ceilings.json`)

| quantity | measured |
|---|---|
| ZGEMM 8192², complex128, 8 flop per complex multiply–add | **21.2 TFLOP/s** |
| HBM device-to-device copy, 4 GB (read + write) | **1406 GB/s** (triad 1454 GB/s) |
| ridge point (ZGEMM / copy) | **15.1 flop/B** |
| PCIe H2D / D2H, pinned | 24.8 / 26.0 GB/s |
| PCIe H2D / D2H, pageable, into an existing array | 10.7 / 10.5 GB/s |
| PCIe D2H with `cupy.ndarray.get()` into a new array (what `to_numpy` does) | **2.1 GB/s** |

The last row is first-touch page faulting on the fresh host array. It shows up in
the sweeps below.

**The part.** The Leonardo A100-SXM-64GB has **124 SMs** (the stock A100 has
108), a 4096-bit HBM2e bus (four stacks) and these clocks (`nvidia-smi -q -d
CLOCK` and CuPy device properties on lrdn1690):

| clock | idle | application (default) | max |
|---|---|---|---|
| SM / graphics | 210 MHz | 1245 MHz | 1395 MHz |
| memory | 1593 MHz | 1593 MHz | 1593 MHz |

| | peak from the clocks | measured | fraction |
|---|---|---|---|
| FP64 tensor-core ZGEMM: 124 SMs × 128 flop/clock × 1.395 GHz | 22.1 TFLOP/s | 21.2 (best of 5); 21.7 sustained over 15 s | 96–98% |
| HBM: 4096 bit × 2 × 1593 MHz | 1631 GB/s | 1406 copy, 1454 triad | 86–89% |

So the measured ceilings sit close to the hardware peaks. The ZGEMM above the
stock A100's 19.5 TFLOP/s comes from the extra SMs, not from the clocks. The
sustained ZGEMM implies the SM clock stays at or near its 1395 MHz maximum under
load. nvidia-smi was not sampled during the GEMM, so that is inferred.

## 1. Old against new table

The `n = 50` command ran twice (jobs 65154250, "new" and "repeat"). `n = 100` and
the quimb row ran once each. Speedups are against `k = 1` at the same `chi`.

| qubits | chi | method | old s | new s | repeat s | new/old | old speedup | new speedup | old infidelity | new infidelity | old z_error | new z_error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 64 | quimb cpu | 731 | 739 |  | 1.01 |  |  | 0.0368 | 0.0368 | 0.00408 | 0.00408 |
| 50 | 64 | src gpu k=1 | 6.87 | 10.9 | 11.3 | 1.59 |  |  | 0.0616 | 0.0616 | 0.0134 | 0.0134 |
| 50 | 64 | src gpu k=2 | 4.2 | 30.1* | 6.24 | 1.49* | 1.64 | 1.75* | 0.0621 | 0.0621 | 0.012 | 0.012 |
| 50 | 64 | src gpu k=4 | 2.89 | 4.88 | 4.93 | 1.69 | 2.38 | 2.23 | 0.0631 | 0.0631 | 0.0135 | 0.0135 |
| 50 | 128 | src gpu k=1 | 7.61 | 11.6 | 12.1 | 1.52 |  |  | 0.0212 | 0.0212 | 0.00554 | 0.00554 |
| 50 | 128 | src gpu k=2 | 4.49 | 6.69 | 6.78 | 1.49 | 1.69 | 1.73 | 0.0216 | 0.0216 | 0.006 | 0.006 |
| 50 | 128 | src gpu k=4 | 3.01 | 5.68 | 5.57 | 1.89 | 2.53 | 2.04 | 0.0218 | 0.0218 | 0.00596 | 0.00596 |
| 50 | 256 | src gpu k=1 | 13.9 | 17.5 | 19 | 1.26 |  |  | 0.00527 | 0.00527 | 0.00168 | 0.00168 |
| 50 | 256 | src gpu k=2 | 7.82 | 9.67 | 9.78 | 1.24 | 1.78 | 1.81 | 0.00538 | 0.00538 | 0.00191 | 0.00191 |
| 50 | 256 | src gpu k=4 | 5.86 | 8.91 | 8.84 | 1.52 | 2.37 | 1.96 | 0.0055 | 0.0055 | 0.00187 | 0.00187 |
| 50 | 512 | src gpu k=1 | 33.9 | 37 | 37.7 | 1.09 |  |  | 0.000662 | 0.000662 | 0.000235 | 0.000235 |
| 50 | 512 | src gpu k=2 | 19.2 | 21.8 | 21.9 | 1.14 | 1.77 | 1.70 | 0.000692 | 0.000692 | 0.000259 | 0.000259 |
| 50 | 512 | src gpu k=4 | 19.1 | 24.9 | 24.6 | 1.30 | 1.77 | **1.49** | 0.000711 | 0.000711 | 0.000276 | 0.000276 |
| 100 | 64 | src gpu k=1 | 13.1 | 21.2 |  | 1.62 |  |  | 0.136 | 0.136 | 0.0126 | 0.0126 |
| 100 | 64 | src gpu k=2 | 7.63 | 12.7 |  | 1.66 | 1.72 | 1.67 | 0.137 | 0.137 | 0.0151 | 0.0151 |
| 100 | 64 | src gpu k=4 | 5.22 | 7.87 |  | 1.51 | 2.51 | 2.69 | 0.138 | 0.138 | 0.0141 | 0.0141 |
| 100 | 128 | src gpu k=1 | 15.7 | 21.3 |  | 1.36 |  |  | 0.0499 | 0.0499 | 0.00592 | 0.00592 |
| 100 | 128 | src gpu k=2 | 8.77 | 11.9 |  | 1.36 | 1.79 | 1.79 | 0.0509 | 0.0509 | 0.0064 | 0.0064 |
| 100 | 128 | src gpu k=4 | 5.61 | 8.58 |  | 1.53 | 2.80 | 2.48 | 0.0512 | 0.0512 | 0.00625 | 0.00625 |
| 100 | 256 | src gpu k=1 | 28.2 | 33.3 |  | 1.18 |  |  | 0.0131 | 0.0131 | 0.00168 | 0.00168 |
| 100 | 256 | src gpu k=2 | 15.4 | 18.3 |  | 1.19 | 1.83 | 1.82 | 0.0134 | 0.0134 | 0.00166 | 0.00166 |
| 100 | 256 | src gpu k=4 | 11.6 | 15.5 |  | 1.34 | 2.43 | 2.15 | 0.0137 | 0.0137 | 0.00179 | 0.00179 |
| 100 | 512 | src gpu k=1 | 72.3 | 76.1 |  | 1.05 |  |  | 0.00175 | 0.00175 | 0.000245 | 0.000245 |
| 100 | 512 | src gpu k=2 | 40.6 | 45 |  | 1.11 | 1.78 | 1.69 | 0.00184 | 0.00184 | 0.000254 | 0.000254 |
| 100 | 512 | src gpu k=4 | 41.4 | 49.9 |  | 1.21 | 1.75 | **1.53** | 0.00188 | 0.00188 | 0.000247 | 0.000247 |

\* The first `n = 50, chi = 64, k = 2` run took 30.1 s because CuPy compiled
kernels on the first depth-3 call. The repeat, a new process reading the same
`CUPY_CACHE_DIR`, took 6.24 s, and that is the value to use (speedup 1.81 from
the repeat's own `k = 1`). The `new/old` and speedup cells marked * use the
repeat.

- **Accuracy:** every infidelity and `z_error` is identical to the old table to
  the printed three digits, quimb included. The references agree as before:
  `<Z>` = 0.667890 (`n = 50`) and 0.667891 (`n = 100`). Reference times: 150.6 /
  142.6 / 143.3 s (`n = 50`, three jobs; old 138 s) and 327.0 s (`n = 100`; old
  308 s).
- **Spread:** the two `n = 50` runs agree within 9% on every row (apart from
  the compile outlier), most within 3%. The uninstrumented `dp` baselines inside
  the profile job, a third run, are 2–15% faster than task 1 at `chi = 256, 512`.
  So treat about ±10% as the run-to-run noise of a single row.
- **Everything is slower than the old table**: by 5–30% at `chi = 512`, and by
  50–90% at the overhead-bound `chi <= 128`. Fused rows lost more than pairwise
  ones. This stack does not contain the old one, so I cannot attribute the
  regression in full. The `optimize="auto"` rerun (below) shows that the
  `dp` default accounts for the `k = 4` losses at `chi >= 256`, not for the
  `k = 1` slowdown.
- **The fusion claims hold at `k = 2`**: 1.7–1.8× at every size. At `k = 4` they
  hold only for `chi <= 256` (2.0–2.7×). At `chi = 512`, `k = 4` is now *slower*
  than `k = 2` (24.9 against 21.8 s, 49.9 against 45.0 s), where the old table
  had them equal. The README's "1.8–2.8× faster" becomes 1.5–2.7×.

## 2. Where the time goes (`breakdown.csv`, `n = 50`, `dp`, all sweeps)

Seconds over the 80 steps. "other" is the host time of the `src` calls minus the
GPU event time of the contractions and QR. Of it, H2D is GPU event time of the
staged uploads, D2H is host time of `to_numpy` after a synchronize, planning is
`make_plan` plus `contract_expression`, and "rest" is the remainder: Python,
launch latency, small allocations and host syncs inside cuSOLVER.

| chi | k | wall | uninstr. | contractions | QR | H2D | D2H | planning | rest |
|---|---|---|---|---|---|---|---|---|---|
| 256 | 1 | 17.8 | 16.1 | 3.85 | 6.51 | 0.83 | 2.19 | 0.88 | 3.53 |
| 256 | 2 | 9.3 | 9.6 | 2.50 | 3.18 | 0.39 | 0.38 | 0.78 | 2.05 |
| 256 | 4 | 8.2 | 7.6 | 3.19 | 1.58 | 0.21 | 0.53 | 1.62 | 1.03 |
| 512 | 1 | 34.9 | 36.4 | 6.09 | 15.80 | 2.34 | 6.12 | 1.02 | 3.51 |
| 512 | 2 | 19.1 | 21.0 | 5.84 | 7.91 | 1.18 | 1.00 | 0.91 | 2.24 |
| 512 | 4 | 23.1 | 22.9 | 13.93 | 3.96 | 0.60 | 1.54 | 1.93 | 1.17 |
| 1024 | 1 | 124.3 | 131.2 | 31.45 | 45.35 | 8.66 | 32.70 | 1.14 | 5.04 |
| 1024 | 2 | 79.3 | 79.5 | 32.38 | 22.67 | 4.29 | 16.49 | 1.03 | 2.41 |
| 1024 | 4 | 103.3 | 113.4 | 75.90 | 11.34 | 2.12 | 8.54 | 2.24 | 3.19 |

The `first` phase is under 0.02 s everywhere and is included in "contractions".

### The causes, in numbers

**`k = 2, chi = 512` against `k = 1`: 15.8 s saved.**

| | saved | share |
|---|---|---|
| QR (half as many sweeps) | 7.89 s | 50% |
| D2H of the output state | 5.13 s | 32% |
| H2D of the input state | 1.16 s | 7% |
| rest (Python, launches) | 1.28 s | 8% |
| planning | 0.11 s | 1% |
| contractions | 0.25 s | 2% |

Kernel efficiency (saturated, TFLOP/s): env 9.0 → 11.8, sketch 12.7 → 11.8,
projection 11.8 → 11.8. **No kernel-efficiency gain**: the contractions do the
same flops in the same time (6.90e13 against 6.84e13 real flops).

Caveat on D2H: at `k = 1` (and `k = 4`) the D2H time alternates between sweeps,
25 ms then 128 ms at `chi = 512`, while at `k = 2` it is always 25 ms. Presumably
glibc reuses the previous state's pages on every other sweep, and the slow
sweeps pay first-touch faults (the 2.1 GB/s of the ceilings). At the fast rate,
`k = 1` would spend about 2.0 s instead of 6.1 s, so about 3 s of the 5.1 s
saving is an allocator effect rather than an intrinsic per-sweep cost. Even
then, a copy of the whole state per sweep is a real per-sweep cost.

**`k = 4, chi = 256` against `k = 1` (`n = 50`): 9.6 s saved.**

| | saved | share |
|---|---|---|
| QR | 4.93 s | 51% |
| rest (Python, launches) | 2.50 s | 26% |
| D2H | 1.66 s | 17% |
| contractions | 0.66 s | 7% |
| H2D | 0.62 s | 6% |
| planning | **−0.74 s** | −8% |

Here kernel efficiency *does* change: env 1.1 → 4.9, sketch 3.4 → 5.6 and
projection 3.1 → 6.1 TFLOP/s. The `k = 4` contractions do 1.97× the flops
(1.75e13 against 8.87e12) in 0.83× the time. At `chi = 256` the pairwise
kernels are too small to fill the GPU (latency and launch bound, *not* bandwidth
bound: their modelled intensity is 36–43 flop/B, above the ridge). The absolute
saving is still only 7% of the total. Planning goes the other way: `dp`
replans the deeper equations on every `src` call, at about 80 ms per sweep at
`k = 4` against 11 ms at `k = 1`.

(The brief's example, `n = 100, chi = 256`, was not profiled. The profile runs
used `n = 50` as asked, and the decomposition above is the `n = 50` analogue.)

**QR.** `truncated_qr` of a `4 chi x chi` sketch per site runs at 0.27 TFLOP/s
(`chi = 256`), 0.88 (`chi = 512`) and 2.4 (`chi = 1024`), 1–11% of the ZGEMM
ceiling. It is 36–45% of the `k = 1` wall time. In the nsys trace it appears as
cuSOLVER `geqr2_smem_domino_fast` panel kernels, plus the host synchronisation of
`cupy.linalg.qr`'s `devInfo` check. Its modelled intensity (127–508 flop/B) is far
above the ridge, so it is latency and library bound, not bandwidth bound.

### The table's claim at `chi = 512`

The claim is that four steps are "no faster than two" because of the
`chi**2 * 2**k` cost. The measurements:

- The contraction flops double exactly from `k = 2` to `k = 4`: 6.84e13 →
  1.34e14 at `chi = 512`, 5.31e14 → 1.04e15 at `chi = 1024`. The term is
  `chi**3 * 2**k` flops, not `chi**2`.
- The `k = 4` kernels are **not** near the compute ceiling. Saturated, with
  `dp`, env, sketch and projection run at 8.6, 9.8 and 10.3 TFLOP/s, 41–48% of
  21.2. `k = 2` runs at 11.8 (56%). At `chi = 1024`, `k = 4` reaches 12.6–14.2
  TFLOP/s (59–67%) against 15.4–17.6 (73–83%) for `k = 1` and `k = 2`.
- So from `k = 2` to `k = 4` at `chi = 512` the contractions grow by 8.1 s
  (2× the flops at about 0.85× the rate). The QR saves only 4.0 s and planning
  costs 1.0 s more, hence the 4 s loss. At `chi = 1024`, `k = 4` loses 24 s to
  `k = 2` (43.5 s more in contractions).
- ncu on one bulk site (`chi = 512`, middle site, env + sketch + projection):

  | | k = 1 | k = 4 |
  |---|---|---|
  | measured FP64 work / model | 1.08 | 1.25 |
  | DMMA (Tensor Core) share of that work | 99–100% | 99–100% |
  | measured DRAM bytes / modelled lower bound | 1.8 | 2.9 |
  | measured intensity (flop/B) | 50 | 19 |
  | site kernel rate | 12.9 TFLOP/s | 11.4 TFLOP/s |
  | `cupy_copy` (tensordot transposes) share of site kernel time | 12% | 21% |

  At `k = 4` the intensity is close to the ridge (15). The extra traffic comes
  mostly from the transposes the deeper path inserts, and the 25% extra work
  probably from padding of the size-2 legs in the CUTLASS tiles (not verified).
- **Path optimizer.** Rerunning `chi = 256, 512` with `optimize="auto"` (job
  65156539, the old table's setting) gives these uninstrumented times:

  | chi | `dp` k=1 / 2 / 4 (s) | `auto` k=1 / 2 / 4 (s) |
  |---|---|---|
  | 256 | 16.1 / 9.6 / 7.6 | 23.4 / 9.2 / 6.1 |
  | 512 | 36.4 / 21.0 / 23.0 | 40.1 / 28.8 / **19.1** |

  With `auto`, the `k = 4` sketch and projection do 1.5× *more* flops than with
  `dp` (8.07e13 against 5.37e13), but run at 16.2–16.8 TFLOP/s (77–79% of the
  ceiling) against 9.8–10.3. `auto` also plans 3.6× faster at `k = 4` (0.54
  against 1.93 s). Net: `auto` reproduces the old table's `k = 4, chi = 512`
  (19.1 s), and `dp` costs `k = 4` about 4 s.

  However, the `auto` timings at `chi = 512` are erratic. `k = 1` took 40 s
  baseline and 93 s instrumented, and `k = 2` took 29 s baseline and 19 s
  instrumented. This fits the comment in `_kernels.py` that `auto` plans
  intermediates too big for the batches, so that kernels get split after running
  out of memory (logged only at DEBUG, not captured). So `dp` is the more
  reliable default, but its paths for the deep `k = 4` equations are flop-minimal
  and slow. A `dp` with a size or flop-efficiency objective, or caching the
  compiled expressions across `src` calls, would likely recover `k = 4`. That is
  a library question, outside this study.

**Verdict on the claim.** Qualitatively right: the doubled contraction work is
what stops `k = 4`. Two corrections: the cost is `chi**3 * 2**k` flops, and the
`k = 4` kernels run at about half the ceiling with `dp`, not at it. On `main`
the statement should be "four steps are slower than two at `chi = 512`", or the
path setting should be named.

## 3. Bandwidth-bound phases (`roofline.csv`, `roofline.png`)

Ridge point: 21.2 TFLOP/s / 1406 GB/s = 15.1 flop/B. The modelled intensity is an
**upper bound**, because modelled bytes (both inputs and the output of every
pairwise step, at 16 B) are a lower bound on DRAM traffic. ncu measured 1.8–2.9×
more traffic.

| phase | modelled intensity, flop/B (saturated, dp) | relative to the ridge |
|---|---|---|
| env | 36 (k=1), 39 (k=2), 19 (k=4) at chi=256; 72 / 78 / 37 at 512; 144 / 155 / 73 at 1024 | above |
| sketch, proj | 43 / 34 / 24 at 256; 85 / 68 / 46 at 512; 170 / 136 / 92 at 1024 | above |
| first | ≈ 1 | **below**, but under 0.02 s per run |
| qr | 127 (256), 254 (512), 508 (1024) | far above, yet at 1–11% of peak |

No phase that matters is below the ridge on the model. On measured bytes (ncu,
`chi = 512`), `k = 1` sits at 50 flop/B and `k = 4` at 19 flop/B. The fused
`k = 4` contractions at `chi = 256`, `env` (modelled 19, so probably measured
below 10), are the only ones likely to be truly bandwidth bound, together with
the `cupy_copy` transposes inside every contraction. Hypothesis 1 (pairwise =
bandwidth bound, fused = compute bound) is **not** supported. If anything, depth
lowers the intensity, because the path keeps `chi x chi x 2**k` intermediates and
transposes them.

## 4. Instrumentation overhead

`dp` runs, instrumented against the uninstrumented baseline in the same job:
−9% to +10% (the largest is `chi = 256, k = 1` at +10.3%). That is within
run-to-run spread. ⟨Z⟩ was bit-identical between every instrumented run and its
baseline. Against the task-1 times, the instrumented totals are within −13% to
+2% for `chi = 256, 512`. The `auto` job is the exception (+133% / −34%),
explained above.

## 5. Surprises, failures, skipped

- **Saturation is immediate.** With `cutoff = 0` and `oversample = 1`, SRC
  returns bond `chi` from the first sweep on (it is padded with near-zero
  directions), so 79 of 80 (`k = 1`) or 19 of 20 (`k = 4`) sweeps are
  "saturated". The "all" and "saturated" aggregations differ only by the first
  sweep.
- `_truncate` (the `xp.linalg.qr` around `_sweep.py:281`) and `cholesky_qr` were
  patched but never called. `oversample` defaults to 1.0, so there is no final
  truncation, and one device always takes the Householder path.
- The ceilings job's profiler smoke test failed on a CLI parsing error
  (`--ks 1 2`). It was fixed (`--ks 1 --ks 2`) before the profile job, and
  task 1 was not affected.
- **nsys** (2025.5, nvhpc/25.11) captured with the CUDA 13 compat libcuda, but its
  analysis failed ("ProcessEventsError") and the summaries keep only part of the
  kernels. The kernel names are useful, the totals are not.
  `nsight/nsys_*.csv` is kept for reference only.
- **ncu** worked: no `ERR_NVGPUCTRPERM`, and DRAM and FP64 counters are readable.
  Raw output in `nsight/ncu_site_chi512_k{1,4}.csv`. The `k = 4` file is from job 65156351. The
  `k = 1` file is from job 65156351's repeat, 65157282, which I did not submit: it
  appeared during the session with the same command line, overwrote the first
  `k = 1` file, and failed its own `k = 4` run. Its `k = 1` result matches the
  first within 1% (DRAM 4.56e8 against 4.61e8 B, identical flops).
- Two duplicate submissions (65155387, 65156394) were cancelled before
  starting, and a third (65157282, nsight) ran; see above. They were probably replayed during a dropped connection; no run was
  affected.
- The bench venv has quimb 1.15.0 while `uv.lock` pins 1.16.0 (see `env.txt`).
  The quimb row matches the old one within 1%.
- Not done: no profiling at `n = 100`; `chi = 64, 128` not profiled; the
  old-stack regression not bisected; the paper figure.

## 6. Bisect of the slowdown (`bisect/`)

All nine commits from 7631ceb (the old table) to 4485588 ran in today's
environment. None slows `k = 1`: 7631ceb itself is as slow as `main`. Two
things explain the slowdown against the old table:

- **Environment.** `TMPDIR` on Lustre makes the `shutil.disk_usage` that every
  `src` call runs cost 25–290 ms. With `TMPDIR=/tmp`, `chi = 64, k = 1` drops
  from about 11 s to 8.3 s. This per-sweep fixed cost also inflates the fusion
  speedups.
- **d2287df** (`dp` paths). It is the only commit above +15%, and only for
  `k = 4`: +18% at `chi = 512`.

See `bisect/README.md`.

## Files

| file | content |
|---|---|
| `quench_n50.md`, `quench_n50_repeat.md`, `quench_n100.md`, `quench_quimb.md` | task 1 tables |
| `ceilings.json` | task 2 |
| `roofline.csv` | (n, chi, k, optimize, phase, aggregation): calls, real flops (opt_einsum calibrated; `own_real_flops` is an independent per-step count, identical), modelled bytes, GPU s, intensity (upper bound), TFLOP/s |
| `breakdown.csv` | (n, chi, k, optimize, aggregation): wall, per-phase GPU s, other split into H2D / D2H / planning / rest, uninstrumented time, inflation |
| `sweeps.csv` | per-sweep raw rows behind both |
| `roofline.png` | sanity plot (saturated, `dp`) |
| `nsight/` | ncu per-site counters; partial nsys summaries |
| `logs/` | Slurm logs |
| `parts/` | per-job CSVs that `report.py` merges |
