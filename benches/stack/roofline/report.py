"""Merge the per-job CSVs of ``profile_quench.py`` and draw a sanity roofline.

Reads ``parts/<job>/{roofline,breakdown,sweeps}.csv`` and writes the merged
``roofline.csv``, ``breakdown.csv`` and ``sweeps.csv`` next to this file, plus
``roofline.png`` against the ceilings of ``ceilings.json``. Parts written before
the ``optimize`` column existed ran with ``dp``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PHASE_MARKERS = {"env": "o", "sketch": "s", "proj": "^", "first": "x", "qr": "D"}
K_COLORS = {1: "#1f77b4", 2: "#ff7f0e", 4: "#2ca02c"}


def merge(name: str) -> list[dict[str, str]]:
    """Concatenate one CSV over the jobs, with an ``optimize`` column."""
    rows: list[dict[str, str]] = []
    for part in sorted((HERE / "parts").glob(f"*/{name}")):
        for row in csv.DictReader(part.open()):
            row.setdefault("optimize", "dp")
            rows.append(row)
    columns = list(rows[0])
    columns.insert(columns.index("k") + 1, columns.pop(columns.index("optimize")))
    with (HERE / name).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, columns)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot(rows: list[dict[str, str]], ceilings: dict) -> None:
    """Saturated phases of the ``dp`` runs on the measured roofline."""
    peak, bandwidth = ceilings["zgemm_tflops"], ceilings["hbm_copy_gbs"] * 1e-3
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, chi in zip(axes, (256, 512, 1024), strict=True):
        x = [0.5, peak / bandwidth, 1e3]
        ax.plot(x, [min(peak, bandwidth * i) for i in x], "k-", lw=1)
        ax.axhline(peak, color="k", lw=0.5, ls=":")
        for row in rows:
            if (row["chi"], row["aggregation"], row["optimize"]) != (
                str(chi),
                "saturated",
                "dp",
            ):
                continue
            ax.scatter(
                float(row["intensity_flop_per_byte_upper"]),
                float(row["tflops"]),
                marker=PHASE_MARKERS[row["phase"]],
                color=K_COLORS[int(row["k"])],
                s=40,
            )
        ax.set(xscale="log", yscale="log", title=f"n = 50, chi = {chi}")
        ax.set_xlabel("modelled intensity, flop/B (upper bound)")
    axes[0].set_ylabel("achieved TFLOP/s")
    handles = [
        plt.Line2D([], [], color=c, marker="o", ls="", label=f"k = {k}")
        for k, c in K_COLORS.items()
    ] + [
        plt.Line2D([], [], color="gray", marker=m, ls="", label=p)
        for p, m in PHASE_MARKERS.items()
    ]
    axes[-1].legend(handles=handles, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(HERE / "roofline.png", dpi=120)


if __name__ == "__main__":
    roofline = merge("roofline.csv")
    merge("breakdown.csv")
    merge("sweeps.csv")
    plot(roofline, json.loads((HERE / "ceilings.json").read_text()))
