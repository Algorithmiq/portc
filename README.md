# portc

[![Documentation](https://github.com/Algorithmiq/portc/actions/workflows/docpages.yml/badge.svg)](https://algorithmiq.github.io/portc/)
[![Test portc](https://github.com/Algorithmiq/portc/actions/workflows/test.yml/badge.svg)](https://github.com/Algorithmiq/portc/actions/workflows/test.yml)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=Algorithmiq_src-method&metric=coverage&token=ab544eca4948ef2f80dffa566c31a9c00026a825)](https://sonarcloud.io/summary/new_code?id=Algorithmiq_src-method)
[![Quality gate status](https://sonarcloud.io/api/project_badges/measure?project=Algorithmiq_src-method&metric=alert_status&token=ab544eca4948ef2f80dffa566c31a9c00026a825)](https://sonarcloud.io/summary/new_code?id=Algorithmiq_src-method)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Parallel Out-of-Core Randomized Tensor-Network Compression.**

An implementation of the SRC algorithm introduced in [Camaño, Epperly and Tropp, *Quantum* **10**, 2022 (2026)](https://doi.org/10.22331/q-2026-03-10-2022), extended to stacks of tensor trains, multi-GPU execution and memory-budgeted out-of-core compression.

### Features

The following primitives are supported:

1. MPO-MPS randomized contraction-compression.
2. MPO-MPO randomized contraction-compression.
3. MPO randomized compression.
4. MPS randomized compression.
5. Randomized contraction-compression of a whole stack of trains in one sweep:
   `MPO^k`, `MPO^k . MPS` and `MPS . MPO^k`.

`portc` has no tensor-network framework dependency: it takes and returns plain lists of per-site NumPy arrays, one array per site.

```python
from portc import apply, compress, src
```

The `apply` function covers cases 1 and 2 above, the `compress` function cases 3 and 4, and `src` all five: `apply` and `compress` are its two- and one-train special cases. All three functions are pure, meaning no in-place modification ever happens. The user should
manage the assignment of the returned objects, possibly overwriting the input variables.
See the [reference documentation](https://algorithmiq.github.io/portc/) for details, and the [tests](tests/) or [benchmarks](benches/) folders for usage examples.

Whether a train is an MPS or an MPO is inferred from the rank of its first site tensor, so no wrapper type is needed.

**NOTE**: the current implementation targets tensor networks with 3 or more sites. For smaller networks, an exact SVD-based fallback is dispatched, with a warning.

### Oversampling and Truncation

By default, `apply`, `compress` and `src` sketch at `chi_out` and return
**right-canonical** output. For better accuracy at the same `chi_out`, pass
`oversample`: a finite factor of at least `1.0` sketches at
`ceil(oversample * chi_out)`, and `None` at the paper's
`max(ceil(1.5 * chi_out), chi_out + 10)`. A final left-to-right SVD sweep then
truncates to `chi_out`, and the result is **left-canonical**, with the norm on
the last site. When positive, `cutoff` discards Schmidt values below
`cutoff * sigma_max` at each cut in this final pass. Two-site trains keep the
exact, right-canonical fallback and ignore both options.

Larger sketches improve accuracy but increase time and memory costs, often
several-fold; compare factors at the same `chi_out` when choosing an
accuracy/runtime trade-off.

### Tensor Indexing Conventions

The array layout follows the default `quimb` tensor indexing conventions, so results round-trip through [Quimb](https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/index.html) without any permutation:

```python
import quimb.tensor as qtn

result = qtn.MatrixProductOperator(apply(H1.arrays, H2.arrays, chi_out=64))
```

- **MPO Tensors:** Bulk tensors have index order `('l', 'r', 'u', 'd')`.
  Boundary tensors (at the edges) are rank-3, dropping the outer `'l'` or `'r'` index.

- **MPS Tensors:** Bulk tensors have index order `('l', 'r', 'u')`.
  Boundary tensors are rank-2, dropping the outer bond index.

Where `'l'`/`'r'` are left/right virtual bonds and `'u'`/`'d'` are the upper/lower physical legs.
Please keep this in mind when constructing or manipulating tensors directly.

### Contraction Conventions

`src` contracts a *stack* of trains and compresses the result in a single sweep:

```python
from portc import src

state = src(U3, U2, U1, psi, chi_out=64)  # U3 U2 U1 |psi>, U1 acts first
```

The stack is written in mathematical order. Each contraction joins the `'d'` leg of a train with the `'u'` leg (or the physical leg of an MPS) of the train to its right:

| Stack | Contraction | Result |
|---|---|---|
| `src(A)`, `src(psi)` | none | compressed MPO or MPS |
| `src(A, B, ...)` | `A.d` with `B.u` | MPO |
| `src(A, ..., psi)` | `A.d` with `psi` | MPS on the `'u'` leg of `A` (a ket) |
| `src(phi, A, ...)` | `phi` with `A.u` | MPS on the `'d'` leg of the last MPO (a bra) |

An MPS may appear only first or last, and not both. `apply(A, B)` and `compress(A)` are the two- and one-train cases; `apply` accepts only an MPO on the left.

A leading MPS is a row vector used **without conjugation**: `src(phi, A, B)` computes `phiᵀ A B`, which equals `src(Bᵀ, Aᵀ, phi)` with `ᵀ` swapping the `'u'` and `'d'` legs. For the physical bra `<psi| A B`, conjugate first:

```python
bra = src([t.conj() for t in psi], A, B, chi_out=64)
```

The result then pairs with a ket by plain contraction, with no further conjugation.

**Cost.** The per-site cost of the sweep grows with the sketch size squared times the product of the bond dimensions of the layers. Compressing a whole stack at once pays off for shallow stacks of thin layers, such as two or three Trotter layers; apply anything else pairwise. See [the stack depth benchmarks](benches/stack/README.md) for measurements.

### GPU Support

`portc` runs on CPU (NumPy) by default and can be accelerated on GPUs via [CuPy](https://cupy.dev/). Both **NVIDIA** (CUDA) and **AMD** (ROCm) GPUs are supported through optional install extras.

At runtime, pass `device="gpu"` to use GPU acceleration. The library handles backend dispatch automatically.

### Large Problems

Stacks that do not fit on the GPU, or in host memory, still run: the sweep reads the input cores one site at a time (from NumPy arrays, `np.memmap`, zarr or HDF5 datasets), batches every contraction to a memory budget, and keeps the sketched environments on the GPU, in host memory or on local disk. The budgets are detected, or set explicitly with `Resources`:

```python
from portc import Resources, src

out = src(
    N,
    V,
    M,
    U,
    chi_out=2000,
    device="gpu",
    resources=Resources(gpu_memory="36GB", scratch_dir="/local/scratch"),
)
```

On a node with several GPUs, `Resources(devices=4)` splits the sweep among four of them, in the same process.

The contractions follow opt_einsum's cheapest (`dp`) paths. `optimize` selects any other opt_einsum path optimizer, by name or as a `PathOptimizer` instance.

See [Large problems](https://algorithmiq.github.io/portc/features/large-problems) for the budgets, the scratch directory, several GPUs, contraction paths and how to read the logged plan.

## Installation

```bash
# CPU only (default)
uv pip install portc

# With NVIDIA GPU support (CUDA 13.x, driver >= 580)
uv pip install "portc[gpu-nvidia]"

# With AMD GPU support (ROCm)
uv pip install "portc[gpu-rocm]"
```

When including `portc` in another project's `pyproject.toml`:

```toml
# CPU only
dependencies = ["portc"]

# With NVIDIA GPU support
dependencies = ["portc[gpu-nvidia]"]

# With AMD GPU support
dependencies = ["portc[gpu-rocm]"]
```

## Nix flake

The repository ships a `flake.nix` that provides
a development shell with [uv], [Git] and the [GitHub CLI], plus the native
libraries the binary wheels need at runtime. Python itself and all project
dependencies remain managed by `uv`.

With [flakes enabled](https://nixos.wiki/wiki/Flakes#Enable_flakes_temporarily), run:

```bash
nix develop
```

Entering the shell runs `uv sync --all-groups` and activates the uv project
environment (`$UV_PROJECT_ENVIRONMENT` if set, otherwise `.venv`), so you land
in a ready-to-use environment. GPU extras are not installed by the flake: add them
explicitly with `uv sync --all-groups --extra gpu-nvidia` (or `--all-groups --extra gpu-rocm`) on a machine with the matching drivers.

If you use [direnv], the provided `.envrc` enters the shell automatically:

```bash
direnv allow
```

Unlike the DevContainer, the Nix shell does not install the git hooks for
you. Run `prek install --prepare-hooks` once after the first `nix develop`.

## Documentation

The documentation is a [Fumadocs] site under `docs/`, published [at this link].
It covers concepts, features, tutorials, the API reference generated from the
docstrings, and the developer guide.

[GitHub CLI]: https://cli.github.com/
[Git]: https://git-scm.com/
[direnv]: https://direnv.net/
[uv]: https://docs.astral.sh/uv/
[Fumadocs]: https://fumadocs.dev
[at this link]: https://algorithmiq.github.io/portc/
