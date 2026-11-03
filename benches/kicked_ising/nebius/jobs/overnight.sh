#!/bin/bash
# Once no bench run is left: the 8-GPU probes (step5-probes.txt), then the chi 4096
# 20-step run (step5d.txt) only if its probe passes the brief's stop rules, then the
# chi 8192 probe (step5f-probe.txt). Start detached from the repository root:
#   setsid nohup benches/kicked_ising/nebius/jobs/overnight.sh > benches/kicked_ising/nebius/logs/overnight.log 2>&1 &
set -u
N=benches/kicked_ising/nebius
drive() { .venv/bin/python $N/driver.py "$N/jobs/$1.txt" > "$N/logs/$1-driver.log" 2>&1; }
while pgrep -f "[b]ench_kicked_ising.py" > /dev/null; do sleep 60; done
echo "$(date -u) probes"
drive step5-probes
# The chi 4096 probe must be complete, in memory, without a disk tier, and CholeskyQR
# must not fall back at every site it was tried at.
if .venv/bin/python - "$N/probe/src-k1-g8_chi4096_cut0_c64.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
groups = p["groups"]
tried = sum(g["cqr_tried"] for g in groups)
fallbacks = sum(g["cqr_fallbacks"] for g in groups)
plans = [m for g in groups for m in g["src_debug"] if "plan" in m]
disk = [m for m in plans if "disk=0 B" not in m]
print(f"complete {p['complete']}, oom {'oom' in p}, CholeskyQR {fallbacks}/{tried} fallbacks, plans with disk {len(disk)}/{len(plans)}, step {p.get('step_seconds')}")
sys.exit(not (p["complete"] and "oom" not in p and not disk and (tried == 0 or fallbacks < tried)))
PY
then
    echo "$(date -u) chi 4096 run"
    drive step5d
else
    echo "$(date -u) chi 4096 probe failed the stop rules; no 20-step run"
fi
echo "$(date -u) chi 8192 probe"
drive step5f-probe
echo "$(date -u) done"
