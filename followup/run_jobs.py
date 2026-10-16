"""Train every job in jobs.txt in parallel on GPUs, with the same Python that runs this script. Run from taska/:
  python followup/run_jobs.py --gpus 1,2,3,4,5,7 --workers 8
~2.5 min per 4000-step job on a shared A100 node; the whole list takes ~25 min with 8 workers."""
import argparse, itertools, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
ap = argparse.ArgumentParser(); ap.add_argument("--gpus", default="1"); ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--only", default=None, help="comma-separated job names")
a = ap.parse_args()
gpus = itertools.cycle(f"cuda:{g}" for g in a.gpus.split(","))
jobs = []
for line in open("followup/jobs.txt"):
    if not line.strip() or line.startswith("#"): continue
    name, args = line.strip().split("|")
    if a.only and name not in a.only.split(","): continue
    jobs.append((name, f"{sys.executable} followup/train_variant.py {args} --device {next(gpus)} --threads 1 --out followup/runs/{name}"))
os.makedirs("followup/logs", exist_ok=True)
def run(job):
    name, cmd = job
    with open(f"followup/logs/{name}.log", "w") as f:
        r = subprocess.run(cmd.split(), stdout=f, stderr=subprocess.STDOUT,
                           env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    print(name, "exit", r.returncode, flush=True)
with ThreadPoolExecutor(a.workers) as ex: list(ex.map(run, jobs))
