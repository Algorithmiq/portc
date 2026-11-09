"""Benchmarking the MPO-MPO contraction-compression."""

from __future__ import annotations

import logging
from time import perf_counter_ns

import cyclopts
import numpy as np
import quimb.tensor as qtn

from portc import apply

logger = logging.getLogger(__name__)


# Initialize cyclopts app
app = cyclopts.App(help="Run SRC benchmark.")


def _log_distance(
    H_ref: qtn.MatrixProductOperator, H: qtn.MatrixProductOperator
) -> None:
    # quimb expands the norm of the difference, so cancellation floors this at
    # about sqrt(eps) of the dtype.
    distance = H_ref.distance(H)
    logger.info(
        " - Distance to reference: %s (relative %s)",
        distance,
        distance / abs(H_ref.norm()),
    )


@app.default
def main(
    n_sites: int = 50,
    chi_out: int = 20,
    chi_id: int = 4,
    run: str = "src",
    compare: str = "yes",
    device: str = "cpu",
    dtype: str = "complex128",
    seed: int = 0,
) -> None:
    """Main benchmarking function.

    Args:
        n_sites: Number of sites in the MPO chain.
        chi_out: Bond dimension of the random MPO and of the compressed output.
        chi_id: Bond dimension of the perturbed identity MPO.
        run: Which library to run ('quimb', 'src').
        compare: Whether to compare results to a reference ('yes', 'no').
        device: Where SRC runs ('cpu', 'gpu'); quimb always runs on the CPU.
        dtype: Data type of the MPOs ('complex128', 'complex64').
        seed: Seed of the inputs and of the SRC sketch.
    """
    phys_dim = 2
    array_type = np.dtype(dtype)

    logger.info(
        "benchmark_start: n_sites=%d, chi_out=%d, chi_id=%d, phys_dim=%d, dtype=%s, "
        "run=%s, compare=%s, device=%s",
        n_sites,
        chi_out,
        chi_id,
        phys_dim,
        dtype,
        run,
        compare,
        device,
    )

    # Generate a random MPO and a perturbed identity MPO
    logger.info("Generating MPOs...")
    H1 = qtn.MPO_rand(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type, seed=seed
    )
    # The identity adds one to the bond: a near-identity, low-rank operator.
    H2 = qtn.MPO_identity(
        n_sites, phys_dim=phys_dim, dtype=array_type
    ) + 1e-8 * qtn.MPO_rand(
        n_sites,
        bond_dim=chi_id - 1,
        phys_dim=phys_dim,
        dtype=array_type,
        seed=seed + 1,
    )

    # Quimb's contraction reference
    if compare == "yes":
        logger.info("Computing reference contraction (no compression)...")
        tms = perf_counter_ns()
        H_ref = H1.apply(H2, compress=False)
        tms = perf_counter_ns() - tms
        logger.info("Reference contraction took %s s", tms * 1e-9)

    # Quimb's contraction with compression
    if run == "quimb":
        logger.info("Computing Quimb's MPO-MPO contraction (with compression)...")
        tms = perf_counter_ns()
        H_quimb = H1.apply(H2, compress=True, max_bond=chi_out, method="rsvd")
        tms = perf_counter_ns() - tms
        logger.info(" Quimb's contraction-compression took %s s", tms * 1e-9)
        if compare == "yes":
            _log_distance(H_ref, H_quimb)

    # SRC's contraction compression
    if run == "src":
        if device == "gpu":
            # CUDA context, cuBLAS handles and kernel compilation stay out of the timing.
            logger.info("Warming up the GPU...")
            warm = qtn.MPO_rand(4, bond_dim=2, phys_dim=phys_dim, dtype=array_type)
            apply(warm.arrays, warm.arrays, chi_out=2, device=device)
        logger.info("Computing SRC's MPO-MPO contraction (with compression)...")
        tms = perf_counter_ns()
        # portc takes and returns plain lists of site arrays; quimb is only
        # used here to build the inputs and to measure the distance.
        H_src = qtn.MatrixProductOperator(
            apply(H1.arrays, H2.arrays, chi_out=chi_out, seed=seed, device=device)
        )
        tms = perf_counter_ns() - tms
        logger.info(" SRC's contraction-compression took %s s", tms * 1e-9)
        if device == "gpu":
            import cupy  # noqa: PLC0415  (optional GPU dependency)

            logger.info(
                " - CuPy pool high-water mark: %.2f GB",
                cupy.get_default_memory_pool().total_bytes() / 1e9,
            )
        if compare == "yes":
            _log_distance(H_ref, H_src)

    logger.info("benchmark_end")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    # Report the per-sweep timings of portc too.
    logging.getLogger("portc").setLevel(logging.DEBUG)
    app()
