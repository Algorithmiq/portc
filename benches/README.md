# SRC benchmarks

This directory collects the experiments used to measure the performance of the
different SRC variants. They are grouped by the kind of workload exercised:

- [`primitives/`](primitives/) — Synthetic micro-benchmarks of the four core
  SRC primitives (MPO–MPO, MPO–MPS, …). Used to assess speedup and accuracy of
  individual primitives versus `quimb` references on a fixed problem size.
  See [`primitives/README.md`](primitives/README.md).
- [`stack/`](stack/) — One-shot SRC over stacks of trains against sequential
  pairwise application, by stack depth: accuracy against dense references and
  wall time on small chains, and quench dynamics of 50 and 100 qubits on the GPU.
  See [`stack/README.md`](stack/README.md).
- [`large/`](large/) — Out-of-core SRC of `N . V . M . U` with a large `M` read
  from disk, on one GPU or several: plan, wall time, memory peaks, stall time and
  parallel efficiency. See [`large/README.md`](large/README.md).

The scripts need the `bench` dependency group (included in `dev`):

```bash
uv sync --group bench
```

## HPC clusters

The recorded results come from the Booster partition of Leonardo at CINECA. The
[HPC clusters](https://algorithmiq.github.io/portc/hpc/leonardo) section of
the documentation covers the setup and has an example job script; pass it the
benchmark command, e.g. `python benches/large/bench_large.py run ...`.
