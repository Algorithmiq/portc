"""Print the Markdown tables of ``summary.md`` from the raw records, traces and logs.

Usage::

    python analyze.py > tables.md
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import median

HERE = Path(__file__).parent
RAW = HERE / "raw"
LOGS = HERE / "logs"
GB = 1e9
COLLECTIVES = ("gather", "scatter", "allgather_s", "allgather_cores")
KERNELS = ("env", "sketch", "project", "first", "kernel")


def _jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (RAW / name).read_text().splitlines()]


def _passes(log: str) -> list[dict[str, float]]:
    """Per run in a log: the slowest device's pass times and stalls, and the wall."""
    patterns = {
        "ltr": r"Left-to-right sweep: ([\d.]+)",
        "rtl": r"Right-to-left sweep: ([\d.]+)",
        "stall_sites": r"stalls: sites ([\d.]+)",
        "stall_envs": r"environments ([\d.]+) s",
    }
    runs, current = [], {key: 0.0 for key in patterns}
    for line in (LOGS / f"{log}.log").read_text().splitlines():
        for key, pattern in patterns.items():
            if m := re.search(pattern, line):
                current[key] = max(current[key], float(m[1]))
        if m := re.search(r"Run complete: ([\d.]+) s", line):
            runs.append({**current, "wall": float(m[1])})
            current = {key: 0.0 for key in patterns}
    return runs


def _trace(g: int) -> dict[str, list[dict]]:
    return json.loads((RAW / f"trace_g{g}_m4000.json").read_text())["trace"]


def _sum(sites: list[dict], pass_: str, *labels: str) -> float:
    return sum(s.get(label, 0.0) for s in sites if s["pass"] == pass_ for label in labels)


def scaling(size: str, predicted: dict | None) -> None:
    """Step 4: both repetitions of one size, with the predicted efficiency."""
    records = _jsonl(f"step4_{size}.jsonl")
    passes = _passes(f"step4-{size}-r1") + _passes(f"step4-{size}-r2")
    t1 = median(r["seconds"] for r in records if r["gpus"] == 1)
    print(
        "| G | Wall r1 / r2 (s) | Speed-up | Efficiency | Predicted | LTR / RTL (s) "
        "| Stalls sites / envs (s) | Tiers d/h/disk | Sketch batch "
        "| RTL TFLOP/s per GPU | QR distance r1 / r2 (eps) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    half = len(records) // 2
    for i, (a, b) in enumerate(zip(records[:half], records[half:])):
        g = a["gpus"]
        wall = median([a["seconds"], b["seconds"]])
        p = passes[i]
        tiers = "/".join(str(a["tiers"][k]) for k in ("device", "host", "disk"))
        rate = ""
        if size == "m4000":
            flops = _sum(_trace(g)["0"], "right_to_left", "flops")
            rate = f"{flops / p['rtl'] / 1e12:.1f}"
        pred = "" if predicted is None or g == 1 else f"{predicted[str(g)]:.0%}"
        qr = " / ".join(
            f"{r['qr_distance']:.2e} ({r['qr_distance'] / r['eps']:.0f})" for r in (a, b)
        )
        print(
            f"| {g} | {a['seconds']:.1f} / {b['seconds']:.1f} | {t1 / wall:.2f} "
            f"| {t1 / (g * wall):.0%} | {pred} | {p['ltr']:.1f} / {p['rtl']:.1f} "
            f"| {p['stall_sites']:.1f} / {p['stall_envs']:.1f} | {tiers} "
            f"| {a['sketch_batches'][-1 if g > 1 else 0]} | {rate} | {qr} |"
        )
    print()


def attribution() -> None:
    """Step 4: GPU time of the right-to-left pass by category, from the traces."""
    print(
        "| G | Span | Kernels | TFLOP/s in kernels | allgather S | allgather cores "
        "| gather + scatter | QR | Other | Kernels x G |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for g in (1, 2, 4, 8):
        trace = _trace(g)
        # Rank 0 runs the QR; the slowest non-root rank shows the waits for it.
        rows = []
        for rank, sites in trace.items():
            span = _sum(sites, "right_to_left", "span")
            kernels = _sum(sites, "right_to_left", *KERNELS)
            rows.append(
                {
                    "rank": rank,
                    "span": span,
                    "kernels": kernels,
                    "rate": _sum(sites, "right_to_left", "flops") / kernels / 1e12,
                    "s": _sum(sites, "right_to_left", "allgather_s"),
                    "cores": _sum(sites, "right_to_left", "allgather_cores"),
                    "gs": _sum(sites, "right_to_left", "gather", "scatter"),
                    "qr": _sum(sites, "right_to_left", "qr"),
                }
            )
        r = rows[0]
        other = r["span"] - r["kernels"] - r["s"] - r["cores"] - r["gs"] - r["qr"]
        print(
            f"| {g} | {r['span']:.1f} | {r['kernels']:.1f} | {r['rate']:.1f} "
            f"| {r['s']:.1f} | {r['cores']:.1f} | {r['gs']:.1f} | {r['qr']:.1f} "
            f"| {other:.1f} | {r['kernels'] * g:.1f} |"
        )
    print()


def collectives(probe: dict) -> None:
    """Step 4: per-site allgather time, fastest rank, against the probe."""
    check = {(r["gpus"], r["total_bytes"]): r for r in _jsonl("allgather_check.jsonl")}
    ptds = {(r["gpus"], r["total_bytes"]): r for r in _jsonl("allgather_check_ptds.jsonl")}
    print(
        "| G | allgather S, fastest rank (ms/site) | In-sweep GB/s per receiver "
        "| `DeviceGroup.allgather` alone, null / per-thread stream | Probe |"
    )
    print("|---|---|---|---|---|")
    for g in (2, 4, 8):
        per_site: dict[int, list[float]] = {}
        for sites in _trace(g).values():
            for s in sites:
                if s["pass"] == "right_to_left" and s.get("allgather_s"):
                    per_site.setdefault(s["site"], []).append(s["allgather_s"])
        fastest = median(min(v) for v in per_site.values())
        rate = (g - 1) / g * 4.096 / fastest
        key = (g, 4.096e9)
        print(
            f"| {g} | {1e3 * fastest:.0f} | {rate:.1f} "
            f"| {check[key]['per_receiver_gbs']:.1f} / "
            f"{ptds[key]['per_receiver_gbs']:.1f} | {probe['allgather_gbs'][str(g)]:.1f} |"
        )
    print()


def tiers() -> None:
    """Step 5: plans, peaks, spill, times and disk traffic."""
    names = ("a", "b", "c1", "c2")
    distances = {r["b"]: r for r in _jsonl("step5_distances.jsonl")}
    print(
        "| Plan | Tiers d/h/disk | Planned device / pool (GB) | Planned host / max RSS (GB) "
        "| Spilled (GB) | Disk written / read (GB) | Wall (s) | LTR / RTL (s) "
        "| Stalls sites / envs (s) | Bit-identical to (a) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, record in zip(names, _jsonl("step5_tiers.jsonl")):
        p = _passes(f"step5-{name}")[0]
        log = (LOGS / f"step5-{name}.log").read_text()
        blocks = {
            k: int(re.search(rf"File system {k}: (\d+)", log)[1]) * 512 / GB
            for k in ("outputs", "inputs")
        }
        rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", log)[1]) * 1024
        tiers = "/".join(str(record["tiers"][k]) for k in ("device", "host", "disk"))
        same = "--" if name == "a" else distances[f"step5-{name}.npz"]
        same = same if same == "--" else (
            "yes" if same["bit_identical"] else f"no ({same['qr_distance']:.1e})"
        )
        print(
            f"| ({name}) | {tiers} | {record['planned_device_peak'] / GB:.1f} / "
            f"{record['pools'][0] / GB:.1f} | {record['planned_host_peak'] / GB:.1f} / "
            f"{rss / GB:.1f} | {record['disk_bytes'] / GB:.1f} "
            f"| {blocks['outputs']:.1f} / {blocks['inputs']:.1f} | {record['seconds']:.1f} "
            f"| {p['ltr']:.1f} / {p['rtl']:.1f} "
            f"| {p['stall_sites']:.1f} / {p['stall_envs']:.1f} | {same} |"
        )
    print()


def per_thread_streams() -> None:
    """Supplementary: G = 4 and 8 on per-thread default streams, against step 4."""
    path = RAW / "step6_distances.jsonl"
    distances = {r["b"]: r for r in _jsonl(path.name)} if path.exists() else {}
    print(
        "| Size | G | Step 4, null stream (s) | Per-thread stream (s) | Efficiency "
        "| LTR / RTL (s) | QR distance to (a) |"
    )
    print("|---|---|---|---|---|---|---|")
    for size in ("m4000", "m1000"):
        step4 = _jsonl(f"step4_{size}.jsonl")
        t1 = median(r["seconds"] for r in step4 if r["gpus"] == 1)
        for record in _jsonl(f"step6_ptds_{size}.jsonl"):
            g = record["gpus"]
            null = median(r["seconds"] for r in step4 if r["gpus"] == g)
            p = _passes(f"step6-ptds-g{g}-{size}")[0]
            d = distances.get(f"step6-ptds-g{g}.npz")
            dist = "" if d is None else f"{d['qr_distance']:.2e}"
            print(
                f"| {size} | {g} | {null:.1f} | {record['seconds']:.1f} "
                f"| {t1 / (g * record['seconds']):.0%} | {p['ltr']:.1f} / {p['rtl']:.1f} "
                f"| {dist} |"
            )
    print()


def main() -> None:
    """Print every table."""
    probe = json.loads((RAW / "probe.json").read_text())
    predicted = {
        g: v["efficiency"]
        for g, v in json.loads((RAW / "prediction.json").read_text())["G"].items()
    }
    print("## Step 4, D_M = 4000, chi = 2000\n")
    scaling("m4000", predicted)
    print("## Step 4, D_M = 1000, chi = 500\n")
    scaling("m1000", None)
    print("## Right-to-left pass on rank 0, traced (s)\n")
    attribution()
    print("## allgather of S\n")
    collectives(probe)
    print("## Step 5\n")
    tiers()
    print("## Per-thread default streams\n")
    per_thread_streams()


if __name__ == "__main__":
    main()
