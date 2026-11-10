# SRC primitive benchmarks

This directory contains synthetic micro-benchmarks of the SRC primitives,
run on multiple supercomputing systems. They isolate the cost of a single
primitive (e.g. MPO–MPO contract+compress) on randomly generated inputs of
controlled bond dimension, and compare to a `quimb` reference.

## Leonardo

The Leonardo [supercomputer](https://docs.hpc.cineca.it/general/getting_started.html) at CINECA is
a pre-exascale Tier-0 EuroHPC system; see
[HPC clusters](https://algorithmiq.github.io/portc/hpc/leonardo) for how to run there.

### MPO-MPO benchmark

The benchmark contracts two length-`n_sites` matrix product operators (MPOs) whose initial bond dimension is `chi_out` (unless `chi_id` is specified for the second MPO), then compresses the resulting MPO back to bond dimension `chi_out`. One MPO is fully random; the other is an identity MPO perturbed by summing it with another random MPO with small elements, making the contraction highly compressible. The table below compares `quimb` (CPU, `rsvd` compression) with `src` on the CPU and on one GPU: wall time of the contraction-compression, peak host memory of the job (MaxRSS), speedup over `quimb`, and relative distance to the uncompressed product.

```bash
uv run python benches/primitives/bench_mpo_mpo.py --n-sites 25 --chi-out 1000 --compare no --device gpu
```

One Booster node (32 cores, A100-64GB), complex128 unless noted, `chi_id=4` at `chi=1000`:

| Experiment        | Library             | Time (s) | MaxRSS (GB) | Speedup | Distance to reference |
|-------------------|---------------------|----------|-------------|---------|-----------------------|
| 50 sites chi=50   | quimb               | 37.76    | 0.67        |         | 2.6e-08               |
|                   | src CPU             | 0.23     | 0.57        | 164x    | 1.5e-08               |
|                   | src GPU             | 0.57     | 1.01        | 66x     | 1.5e-08               |
| 20 sites chi=100  | quimb               | 17.86    | 0.73        |         | 0.0                   |
|                   | src CPU             | 0.25     | 0.74        | 71x     | 1.5e-08               |
|                   | src GPU             | 0.46     | 1.22        | 39x     | 2.1e-08               |
| 25 sites chi=1000 | quimb               | 422.57   | 24.64       |         |                       |
|                   | src CPU             | 60.47    | 3.85        | 7x      |                       |
|                   | src GPU             | 2.32     | 3.42        | 182x    |                       |
|                   | src GPU (complex64) | 1.90     | 2.20        | 222x    |                       |
| 50 sites chi=1000 | quimb               | 1040.72  | 50.02       |         |                       |
|                   | src CPU             | 137.37   | 6.83        | 7.6x    |                       |
|                   | src GPU             | 4.47     | 6.40        | 233x    |                       |
|                   | src GPU (complex64) | 3.03     | 3.17        | 344x    |                       |

The distances sit at the floor of quimb's `distance` (about `sqrt(eps)` of the
dtype), so they show agreement rather than resolve the error. GPU times exclude a
one-off warm-up (CUDA context and kernel compilation, about 50 s on the first run
with an empty kernel cache). The CuPy pool peaks at 2.8 GB (25 sites) and 5.1 GB
(50 sites) at chi=1000 in complex128, half that in complex64.

### Earlier results (v0.3.2)

Before the refactoring, on Booster (32 CPUs) and DCGP (112 CPUs) nodes:

| Experiment        | CPUs | Library       | Time (s) | MaxRSS (GB) | Speedup | Distance to reference |
|-------------------|------|---------------|----------|-------------|---------|-----------------------|
| 50 sites chi=50   | 32   | quimb         | 164.97   |             |         | 0.0                   |
|                   |      | src           | 11.92    | 38.79       | 13.8x   | 2.1e-08               |
| 50 sites chi=50   | 112  | quimb         | 238.67   | 38.87       |         | 2.98e-08              |
|                   |      | src           | 18.88    | 38.33       | 12.6x   | 2.98e-08              |
| 20 sites chi=100  | 32   | quimb         | 2004.52  |             |         | 0.0                   |
|                   |      | src           | 94.83    | 240.79      | 21x     | 2.6e-08               |
| 25 sites chi=1000 | 112  | quimb (svd)   | 564.34   | 23.98       |         |                       |
|          chi_id=4 |      | quimb (rsvd)  | 501.82   | 24.10       |         |                       |
|                   |      | src           | 71.01    | 4.03        | 7x      |                       |
| 50 sites chi=1000 | 112  | quimb (rsvd)  | 1187.82  | 49.47       |         |                       |
|          chi_id=4 |      | src           | 150.11   | 7.01        | 8x      |                       |

### MPO-MPO strong scaling (v0.3.2, DCGP)

For this benchmark, we fix the problem size to `n_sites=25`, `chi=1000` and `chi_id=4`, and vary the number of CPUs in a single node. We tweak the OpenMP environment variables in the job script to optimize performance, for instance `OMP_PLACES` and `OMP_PROC_BIND`, so the NUMA domain is taken into account. We do this both manually and automatically.

| CPUs | Time auto (s)   | Time maual (s)|
|------|-----------------|---------------|
| 2    | 135.36          | 135.74        |
| 4    | 93.61           | 94.06         |
| 8    | 72.84           | 72.86         |
| 14   | 64.92           | 64.84         |
| 28   | 63.63           | 63.84         |
| 56   | 68.53           | 68.60         |
| 112  | 71.57           | 70.74         |

So the difference between manual and automatic binding is negligible, and the best performance is
achieved with 28 CPUs. Beyond that, the performance degrades, likely due to memory bandwidth limits or NUMA overhead.
