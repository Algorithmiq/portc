# Distributed CholeskyQR (PR #85) on the Nebius node

Findings from the kicked Ising benchmark on 8x RTX PRO 6000 Blackwell Server
Edition, connected over PCIe with no NVLink. The library is PR #85 unchanged.
The bench's `--householder` option sets `src_method._sweep.DISTRIBUTED_QR_MIN =
2**62`, so every sketch is gathered on rank 0 for a Householder QR. The probes
below compare that against the distributed CholeskyQR.

## Real evolution: complex64, chi 4096, CholeskyQR never accepted

`z62:20:8:src-k1-g8:4096:0`, complex64, cutoff 0. Run files are
`runs/z62_n20_k8_src-k1-g8_chi4096_c64_s0.jsonl` (CholeskyQR, stopped after 10
steps) and `runs/..._c64_householder_s0.jsonl` (Householder, 20 steps).

| | CholeskyQR | Householder |
|---|---|---|
| CholeskyQR tried / declined / skipped per step | 66 / 66 / 660 | 0 / 0 / 0 |
| Pattern per sub-layer group | 11 tried, 11 declined, 110 skipped | - |
| Step time, steps 1-10 | 413-423 s (median 418 s) | 361-412 s (median 378 s) |
| Step time, steps 1-20 | - | median 404 s, total 7897 s |
| <Z62> at steps 1-10 | identical bits | identical bits |

The pattern is the same in every group of every step, from step 1. The values
match bit for bit because every declined site took the Householder path. The
backoff (`CHOLESKY_QR_MAX_SKIP`) keeps the cost of the failed attempts to a few
percent of a step.

## Random operator (`step-probe`): accepted in two groups, slower there

One Trotter step on a random operator of bond chi, complex64, cutoff 0. Files
are in `probe/`.

| chi | QR | step | CholeskyQR (tried, declined) per group |
|---|---|---|---|
| 4096 | CholeskyQR | 410.8 s | (118, 7), (107, 15), (17, 11), (11, 11) x3 |
| 4096 | Householder | 357.5 s | - |
| 2048 | CholeskyQR | 138.8 s | (119, 6), (107, 14), (11, 11) x4 |
| 2048 | Householder | 131.4 s | - |

At chi 4096 the right-to-left sweep took 88.4 s against 54.3 s with Householder
in group 0, and 66.6 s against 53.9 s in group 1. Those are the groups where
CholeskyQR was mostly accepted. On this PCIe node the all-to-alls cost more than
the rank-0 Householder QR they replace.

## Why complex64 declines: the shift outgrows the matrix

The first pass of `utils/linalg.py::cholesky_qr` shifts the Gram matrix by
`11 (m n + n (n + 1)) u trace(G)`, with `m` = rows, `n` = sketch columns
(chi_out) and `u = eps / 2`. Shifted CholeskyQR3 relies on that shift being
small next to `||Y||^2`. In the sweep, `m = 4 n`, since a single-site sketch
with k = 1 has `chi x 2 x 2` rows.

| m x n | complex64 shift / trace(G) | complex128 shift / trace(G) |
|---|---|---|
| 4096 x 1024 | 3.4 | 6.4e-9 |
| 16384 x 4096 | 55 | 1.0e-7 |

In complex64, the factor reaches 1 at about n = 550 when m = 4n.
`DISTRIBUTED_QR_MIN` is 1024, so every distributed attempt in complex64 is past
that point. Beyond it the shifted pass leaves the condition number unchanged,
and the two plain passes accept only a well-conditioned sketch.
`cholesky_qr_kappa.py` measures this on the host with synthetic sketches of
condition number kappa:

| kappa | 4096 x 1024, c64 | 4096 x 1024, c128 | 16384 x 4096, c64 | 16384 x 4096, c128 |
|---|---|---|---|---|
| 1e1 | accepted, 1.6e-7 | accepted, 3.9e-16 | accepted, 1.9e-7 | accepted, 3.8e-16 |
| 1e2 | accepted, 1.6e-7 | accepted, 4.0e-16 | accepted, 2.0e-7 | accepted, 3.9e-16 |
| 1e3 | accepted, 1.6e-7 | accepted, 3.3e-16 | declined | accepted, 4.1e-16 |
| 1e4 | declined | accepted, 3.4e-16 | declined | accepted, 4.6e-16 |
| 1e6 | declined | accepted, 3.8e-16 | declined | accepted, 5.0e-16 |
| 1e8 | declined | accepted, 3.9e-16 | declined | accepted, 5.2e-16 |
| 1e12 | declined | declined | declined | declined |

Each accepted cell gives the orthogonality `||Q^H Q - I||_F / sqrt(n)`. The
residual `||Y - Q Q^H Y||_F / ||Y||_F` is of the same order. Every accepted
factorization is accurate to the working precision, and every declined one
falls back to Householder, so no result is ever wrong.

The sketches of the real evolution have a condition number above the
complex64 limit (about 1e2-1e3 at chi 4096). The random operator's sketches
are better conditioned, which is why the probes accept in two groups.

In complex128, the limit seen here is between 1e8 and 1e12, below the
`1 / u` of the method's theory. The shift scales with `trace(G)` rather than
`||Y||^2`. For these spectra `trace(G)` is 70-110 times `||Y||^2`, and in
general it can be up to n times larger.

## Not measured here

- Complex128 multi-GPU runs, which would show the condition number of real
  sketches in double precision.
- NVLink hardware. On Leonardo, with A100s over NVLink in complex128,
  CholeskyQR was reported to pay off. There are no run files for that in this
  branch.

## Possible library changes (not applied; this benchmark does not touch `src/`)

- Skip the distributed attempt when `11 (m n + n (n + 1)) u >= 1`. In complex64
  that is every sketch with n >= about 550 when m = 4n.
- Document that the distributed CholeskyQR targets complex128 (and float64).
- Optionally, scale the shift by an estimate of `||Y||^2`, such as a few power
  iterations on G, instead of `trace(G)`.
