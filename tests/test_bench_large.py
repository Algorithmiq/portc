"""Tests of the out-of-core benchmark script, ``benches/large/bench_large.py``."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("cyclopts")

from benches.large import bench_large

if TYPE_CHECKING:
    from pathlib import Path

SMALL = {"n_sites": 4, "bond": 2, "bond_m": 3}


@pytest.mark.parametrize(
    ("low", "high"), [("complex64", "complex128"), ("float32", "float64")]
)
def test_generate_rounds_the_same_draws(
    tmp_path: Path, low: bench_large.Precision, high: bench_large.Precision
) -> None:
    """A seed gives the same stack in single and double precision, up to rounding."""
    bench_large.generate(tmp_path / low, dtype=low, **SMALL)
    bench_large.generate(tmp_path / high, dtype=high, **SMALL)
    for single, double in zip(
        bench_large._load(tmp_path / low), bench_large._load(tmp_path / high)
    ):
        for a, b in zip(single, double):
            assert a.dtype == np.dtype(low)
            np.testing.assert_array_equal(a, b.astype(low))


def test_scaling_records_single_precision(tmp_path: Path) -> None:
    """``scaling`` runs a complex64 stack and records its precision and plan."""
    bench_large.generate(tmp_path / "stack", dtype="complex64", **SMALL)
    results = tmp_path / "results.jsonl"
    bench_large.scaling(
        tmp_path / "stack",
        chi_out=4,
        devices=(1, 2),
        results=results,
        scratch_dir=tmp_path,
        device="cpu",
    )
    first, second = (json.loads(line) for line in results.read_text().splitlines())
    assert first["dtype"] == second["dtype"] == "complex64"
    assert first["eps"] == pytest.approx(np.finfo(np.complex64).eps)
    assert sum(first["tiers"].values()) == SMALL["n_sites"]
    assert second["qr_distance"] < 100 * first["eps"]
