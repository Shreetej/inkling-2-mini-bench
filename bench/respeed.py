"""Re-measure speed for every local model after leaving Low Power Mode. Archives the old numbers.

Refuses to run on battery or in Low Power Mode, since both throttle the GPU.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys

import run

pm = subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout
batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
if "AC Power" not in batt or any(ln.split()[-1] != "0" for ln in pm.splitlines() if "powermode" in ln.lower()):
    sys.exit("not on AC power / low power mode on; aborting")
ver = run.http("GET", "/api/status")["version"]
for label, (model, _extra, do_speed, _q) in run.MODELS.items():
    if not do_speed:
        continue
    sp = run.OUT / f"{label}.speed.json"
    if sp.exists() and json.loads(sp.read_text()).get("power") == "ac":
        print("skip", label)
        continue
    if sp.exists():
        shutil.copy(sp, run.OUT / f"{label}.speed-lowpower-0.8.2.json")
    s = run.speed(model)
    s |= {"power": "ac", "omlx": ver}
    sp.write_text(json.dumps(s, indent=1))
    full = run.OUT / f"{label}.json"
    if full.exists():
        d = json.loads(full.read_text())
        d["speed"] = s
        full.write_text(json.dumps(d, indent=1))
    print(label, {k: v for k, v in s.items() if not k.startswith("prefill")}, flush=True)
run.unload_all()
