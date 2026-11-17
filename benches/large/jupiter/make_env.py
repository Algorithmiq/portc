"""Collect the node's environment probes into ``env.json``."""

from __future__ import annotations

import json
import platform
import re
import subprocess
from pathlib import Path

import cupy
import numpy
import opt_einsum

import portc

HERE = Path(__file__).parent
ENV = HERE / "node"


def _dd(name: str) -> dict[str, object]:
    line = next(
        x for x in (HERE / "logs" / f"{name}.log").read_text().splitlines() if "copied" in x
    )
    nbytes, seconds = re.match(r"(\d+) bytes .* copied, ([\d.]+) s", line).groups()
    return {"bytes": int(nbytes), "seconds": float(seconds), "gbs": int(nbytes) / float(seconds) / 1e9}


def main() -> None:
    """Write env.json."""
    runtime = cupy.cuda.runtime
    smi = (ENV / "nvidia-smi-q.txt").read_text()
    lscpu = (ENV / "lscpu.txt").read_text()
    field = lambda text, key: re.search(rf"{key}\s*:\s*(.+)", text).group(1).strip()  # noqa: E731
    probe = json.loads((HERE / "raw" / "probe.json").read_text())
    tf32 = dict(
        line.split() for line in (HERE / "logs" / "tf32-control.log").read_text().splitlines()
    )
    topo = (ENV / "topo.txt").read_text()
    df = subprocess.run(["df", "-B1", "/home/ramon/nebius-data", "/dev/shm"], capture_output=True, text=True).stdout
    env = {
        "host": platform.node(),
        "kernel": platform.release(),
        "versions": {
            "portc": portc.__version__,
            "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip(),
            "cupy": cupy.__version__,
            "numpy": numpy.__version__,
            "opt_einsum": opt_einsum.__version__,
            "python": platform.python_version(),
            "cuda_runtime": runtime.runtimeGetVersion(),
            "cuda_driver_api": runtime.driverGetVersion(),
            "driver": field(smi, "Driver Version"),
            "nvcc": (ENV / "nvcc.txt").read_text().strip().splitlines()[-1],
        },
        "gpu": {
            "model": runtime.getDeviceProperties(0)["name"].decode(),
            "count": runtime.getDeviceCount(),
            "memory_bytes": [runtime.getDeviceProperties(d)["totalGlobalMem"] for d in range(runtime.getDeviceCount())],
            "nvlink": False,
            "pcie": {
                "link_gen_max": field(smi, r"PCIe Generation\s*\n\s*Max"),
                "link_width_max": field(smi, r"Link Width\s*\n\s*Max"),
            },
            "topology": topo,
            "numa_of_gpu": {str(d): (0 if d < 4 else 1) for d in range(8)},
            "notes": "GPUs 0-3 on socket/NUMA 0, 4-7 on socket/NUMA 1. Pairs within a socket are NODE (PCIe through host bridges of one socket, no shared PCIe switch); pairs across are SYS (cross UPI).",
        },
        "cpu": {
            "model": field(lscpu, "Model name"),
            "sockets": int(field(lscpu, "Socket\\(s\\)")),
            "cores_per_socket": int(field(lscpu, "Core\\(s\\) per socket")),
            "threads_per_core": int(field(lscpu, "Thread\\(s\\) per core")),
            "numa": (ENV / "numactl.txt").read_text(),
        },
        "memory_bytes": int((ENV / "free.txt").read_text().split("\n")[1].split()[1]),
        "storage": {
            "local_nvme": None,
            "notes": "No NVMe device is exposed to this VM. The only disk is vda (300 GB virtio, ext4 on /), used as the disk tier at /home/ramon/nebius-data/scratch. Stacks live in /dev/shm (tmpfs, 859 GB).",
            "lsblk": (ENV / "lsblk.txt").read_text(),
            "df": df,
            "dd_direct_write": _dd("disk-write"),
            "dd_direct_read": _dd("disk-read"),
        },
        "peer_access": {
            "all_pairs": all(all(row) for row in probe["peer"]),
            "matrix": probe["peer"],
            "fallback": "Not needed: deviceCanAccessPeer is true for every pair, so enable_peer_access returns True and DeviceGroup.split_uploads is on. Without it, copy_into's memcpyPeerAsync would stage through the host and each GPU would upload whole cores.",
        },
        "bandwidth_gbs": {k: probe[k] for k in ("pair_gbs", "allgather_gbs", "gather_gbs", "scatter_gbs", "h2d_gbs", "d2h_gbs")},
        "bandwidth_method": f"CuPy memcpyPeerAsync / pinned memcpyAsync, {probe['block_bytes'] / 1e9:.1f} GB blocks (host: 4 GB), median of {probe['reps']} after a warm-up; pair_gbs[src][dst].",
        "fp32": {
            "CUPY_TF32": "0",
            "NVIDIA_TF32_OVERRIDE": "0",
            "complex64_matmul_4096_relative_error": probe["matmul"]["relative_error_complex64"],
            "with_tf32_enabled_control": float(tf32["TF32=1"]),
            "gemm_tflops_8192": probe["matmul"]["tflops"],
        },
    }
    (HERE / "env.json").write_text(json.dumps(env, indent=1) + "\n")


if __name__ == "__main__":
    main()
