"""Predict the strong scaling of the reference sweep from a 1-GPU trace and the probes.

Per site, with ``e`` the bytes of a complex64 element:

- gather of ``M_j`` on rank 0: ``(G - 1) / G * l * eta * p * e`` at the measured
  gather bandwidth into GPU 0;
- scatter of ``Gamma_j``: ``r * eta * p * e / G`` per receiver at the measured
  per-receiver scatter bandwidth (GPU 0 sends to all others at once);
- allgather of ``S_j``: ``(G - 1) / G * r * A * e`` per receiver at the measured
  concurrent allgather bandwidth;
- cores (``t_upload``), in both passes: ``K / G`` host to device at the concurrent
  pinned rate, then an allgather of ``(G - 1) / G * K``. At ``G = 1``, ``K`` at the
  single-GPU pinned rate.

The model is ``T_G = (T_1 - S_1) / G + S_G``, with ``S_G`` the serial part at ``G``
GPUs: the QR of every site of the second pass (measured on one GPU), the
collectives and the core uploads. ``S_1`` is the serial part on one GPU (QR and
uploads), so that it is not divided by ``G``.

Usage::

    python predict.py probe.json trace_g1.json [--n-sites 50 --chi 2000 ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GB = 1e9


def main() -> None:
    """Print the prediction as JSON and as a Markdown table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--n-sites", type=int, default=50)
    parser.add_argument("--chi", type=int, default=2000)
    parser.add_argument("--bond-m", type=int, default=4000)
    parser.add_argument("--bond", type=int, default=4)
    parser.add_argument("--phys", type=int, default=4)
    parser.add_argument("--itemsize", type=int, default=8)
    args = parser.parse_args()
    probe = json.loads(args.probe.read_text())
    record = json.loads(args.trace.read_text())
    sites = record["trace"]["0"]

    n, e, chi = args.n_sites, args.itemsize, args.chi
    p = args.phys**2
    eta = r = chi  # bulk output bonds: min(chi, eta * p) = chi
    A = args.bond**3 * args.bond_m  # product of the left bonds of the four layers
    K = (args.bond_m**2 + 3 * args.bond**2) * p * e  # cores of a bulk site

    def total(pass_: str, *labels: str) -> float:
        return sum(s.get(lab, 0.0) for s in sites if s["pass"] == pass_ for lab in labels)

    spans = {
        pass_: sum(s.get("span", 0.0) for s in sites if s["pass"] == pass_)
        for pass_ in ("left_to_right", "right_to_left")
    }
    qr = total("right_to_left", "qr")
    h2d_1 = probe["h2d_gbs"]["alone"][0] * GB
    upload_1 = 2 * (n - 1) * K / h2d_1  # both passes, unsplit
    t1 = record["seconds"]
    serial_1 = qr + upload_1
    rtl_flops = total("right_to_left", "flops")
    rtl_kernel_s = total("right_to_left", "sketch", "project")
    ltr_flops = total("left_to_right", "flops")
    ltr_kernel_s = total("left_to_right", "env")

    out = {
        "T1": t1,
        "passes_1": spans,
        "qr_total": qr,
        "qr_per_site": qr / (n - 1),
        "upload_1": upload_1,
        "rtl_per_site": spans["right_to_left"] / (n - 1),
        "rtl_tflops_kernels": rtl_flops / rtl_kernel_s / 1e12,
        "rtl_tflops_span": rtl_flops / spans["right_to_left"] / 1e12,
        "ltr_tflops_kernels": ltr_flops / ltr_kernel_s / 1e12,
        "rtl_flops": rtl_flops,
        "ltr_flops": ltr_flops,
        "payload_gb": {
            "gather_M_total": chi * eta * p * e / GB,
            "scatter_Gamma_total": r * eta * p * e / GB,
            "S_total": r * A * e / GB,
            "cores_K": K / GB,
        },
        "G": {},
    }
    for g in (2, 4, 8):
        key = str(g)
        gather = (g - 1) / g * chi * eta * p * e / (probe["gather_gbs"][key] * GB)
        scatter = r * eta * p * e / g / (probe["scatter_gbs"][key] * GB)
        allgather_s = (g - 1) / g * r * A * e / (probe["allgather_gbs"][key] * GB)
        h2d = probe["h2d_gbs"]["together"][key]["per_gpu"] * GB
        cores = K / g / h2d + (g - 1) / g * K / (probe["allgather_gbs"][key] * GB)
        comm = (n - 1) * (gather + scatter + allgather_s)
        upload = 2 * (n - 1) * cores
        t_g = (t1 - serial_1) / g + qr + comm + upload
        out["G"][key] = {
            "per_site_ms": {
                "gather": 1e3 * gather,
                "scatter": 1e3 * scatter,
                "allgather_S": 1e3 * allgather_s,
                "cores_per_pass": 1e3 * cores,
                "qr": 1e3 * qr / (n - 1),
            },
            "comm_s": comm,
            "upload_s": upload,
            "qr_s": qr,
            "T_G": t_g,
            "speedup": t1 / t_g,
            "efficiency": t1 / (g * t_g),
        }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
