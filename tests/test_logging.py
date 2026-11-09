"""The package must leave the host application's logging configuration alone."""

from __future__ import annotations

import logging
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from portc import apply, compress


def _run(code: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter, so ``portc`` is imported anew."""
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=True,
    )


def test_import_leaves_root_logger_untouched() -> None:
    """Importing the package must not change the root logger level or handlers."""
    result = _run(
        """
        import logging

        root = logging.getLogger()
        root.setLevel(logging.CRITICAL)
        before = (root.level, list(root.handlers))

        import portc

        assert (root.level, list(root.handlers)) == before, root.handlers
        package = logging.getLogger("portc")
        assert all(isinstance(h, logging.NullHandler) for h in package.handlers)
        """
    )
    assert result.stdout == ""
    assert result.stderr == ""


def test_calls_are_silent_by_default() -> None:
    """Without logging configured, neither the SRC path nor the fallback prints."""
    result = _run(
        """
        import numpy as np
        from portc import apply, compress

        rng = np.random.default_rng(0)
        mpo = [rng.normal(size=(2, 2, 2))]
        mpo += [rng.normal(size=(2, 2, 2, 2)) for _ in range(2)]
        mpo += [rng.normal(size=(2, 2, 2))]
        mps = [rng.normal(size=(2, 2)), rng.normal(size=(2, 2, 2))]
        mps += [rng.normal(size=(2, 2, 2)), rng.normal(size=(2, 2))]

        apply(mpo, mps, chi_out=2, seed=0)
        compress(mpo, chi_out=2, seed=0)
        # The two-site fallback logs a warning, which must stay silent too.
        compress([mps[0], mps[0].T], chi_out=2)
        """
    )
    assert result.stdout == ""
    assert result.stderr == ""


@pytest.mark.parametrize("call", ["apply", "compress"])
def test_debug_records_reach_enabled_logger(call: str, caplog) -> None:
    """Progress messages are emitted at DEBUG on the ``portc`` loggers."""
    rng = np.random.default_rng(0)
    mpo = [rng.normal(size=(2, 2, 2))]
    mpo += [rng.normal(size=(2, 2, 2, 2)) for _ in range(2)]
    mpo += [rng.normal(size=(2, 2, 2))]
    caplog.set_level(logging.DEBUG, logger="portc")

    if call == "apply":
        apply(mpo, mpo, chi_out=2, seed=0)
    else:
        compress(mpo, chi_out=2, seed=0)

    records = [r for r in caplog.records if r.name.startswith("portc")]
    assert records
    assert all(r.levelno == logging.DEBUG for r in records)
    assert "SRC complete" in caplog.text
