"""Run a JSON list of shell jobs with N GPU workers; skips jobs whose done-file exists."""
import json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

jobs = json.loads(Path(sys.argv[1]).read_text())
n = int(sys.argv[2]) if len(sys.argv) > 2 else 1

def run(j):
    if Path(j["done"]).exists():
        return
    t = time.time()
    Path(j["log"]).parent.mkdir(parents=True, exist_ok=True)
    with open(j["log"], "w") as f:
        r = subprocess.run(j["cmd"], shell=True, stdout=f, stderr=subprocess.STDOUT)
    print(f"{j['name']} rc={r.returncode} {time.time()-t:.0f}s", flush=True)

with ThreadPoolExecutor(n) as ex:
    list(ex.map(run, jobs))
print("ALL DONE", flush=True)
