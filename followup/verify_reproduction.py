"""Check the README claims: (1) retraining with the default command gives bit-identical weights;
(2) evaluate.py reproduces runs/direct/results.json. Training logs are not shipped in the repo, so the
log comparison runs only if runs/direct/history.jsonl exists (regenerate it with the train.py command below).
  CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py --device cuda:1 --out /tmp/reproduce
  python evaluate.py --checkpoint runs/direct/model.pt --device cuda:1 --out /tmp/eval_delivered
  python followup/verify_reproduction.py /tmp/reproduce /tmp/eval_delivered [output.json]"""
import sys, json, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent; TASKA = HERE.parent
rep, ev = Path(sys.argv[1]), Path(sys.argv[2])
out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "results/reproduction.json"
a = torch.load(TASKA / "runs/direct/model.pt", map_location="cpu", weights_only=False)
b = torch.load(rep / "model.pt", map_location="cpu", weights_only=False)
r1 = json.load(open(TASKA / "runs/direct/results.json")); r2 = json.load(open(ev / "results.json"))
diffs = []
def walk(x, y, p=""):
    if isinstance(x, dict):
        for k in set(x) | set(y): walk(x.get(k), y.get(k), f"{p}.{k}")
    elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        for i, (u, v) in enumerate(zip(x, y)): walk(u, v, f"{p}[{i}]")
    elif x != y: diffs.append(p)
walk(r1, r2)
out = {"device": b["metadata"]["device_name"] + " / " + b["metadata"]["args"]["device"],
       "weights_bit_identical": a["model"].keys() == b["model"].keys() and all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"]),
       "evaluate_results_differing_fields": diffs}
orig_log, new_log = TASKA / "runs/direct/history.jsonl", rep / "history.jsonl"
if orig_log.exists() and new_log.exists():
    strip = lambda r: {k: v for k, v in r.items() if k != "elapsed_seconds"}
    h1 = [json.loads(l) for l in open(orig_log)]; h2 = [json.loads(l) for l in open(new_log)]
    out |= {"history_lines": [len(h1), len(h2)],
            "history_identical_except_elapsed": len(h1) == len(h2) and all(strip(x) == strip(y) for x, y in zip(h1, h2)),
            "original_train_seconds": h1[-1]["elapsed_seconds"], "reproduced_train_seconds": h2[-1]["elapsed_seconds"]}
json.dump(out, open(out_path, "w"), indent=1); print(json.dumps(out, indent=1))
