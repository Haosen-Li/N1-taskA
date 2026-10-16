"""Probe every trained model: in-distribution + relevant/query edits, dependency-distance sweep,
memory-offset sweep. Writes followup/results/generalization.json.  Run from anywhere:
  python followup/probe_generalization.py cuda:1"""
import sys, json, glob, os, re, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent; TASKA = HERE.parent
sys.path.insert(0, str(TASKA)); sys.path.insert(0, str(HERE))
import variants as V
from model import Config
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
N = 1024
GAPS = [0, 8, 16, 24, 36, 40, 56, 72, 96, 128, 200]   # trained gaps are 32/48/64
OFFS = [1, 2, 3, 8, 16, 32]                            # filler tokens inserted before the memory block

def runs():
    yield "default_s7", TASKA / "runs/direct", {}      # the delivered checkpoint
    for d in sorted((HERE / "runs").iterdir()):
        if (d / "model.pt").exists():
            yield d.name, d, json.load(open(d / "variant.json"))

VAL_FILE = HERE / "results/val_curves.json"
SAVED_CURVES = json.load(open(VAL_FILE)) if VAL_FILE.exists() else {}

def val_curve(name, d):
    """Validation curve from a fresh history.jsonl if present (training logs are not shipped), else the saved copy."""
    h = d / "history.jsonl"
    if h.exists():
        return [[r["step"], r["val_answer_accuracy"]] for r in map(json.loads, open(h)) if "val_answer_accuracy" in r]
    return SAVED_CURVES[name]

@torch.no_grad()
def evaluate(m, fmt, gap, offset, seed, edits=False):
    g = torch.Generator().manual_seed(seed)
    b = V.make(N, gap, g, dev, offset=offset, fmt=fmt)
    r = torch.arange(N, device=dev)
    pred = m(b.x)[:, -1].argmax(-1)
    out = {"acc": (pred == b.answer).float().mean().item()}
    if edits:
        x = b.x.clone(); new = (b.answer - 12 + 1) % 8 + 12; x[r, b.source_pos] = new
        out["follow_edit"] = (m(x)[:, -1].argmax(-1) == new).float().mean().item()
        other = (b.slot + 1) % 4; qpos = -1 if fmt == "direct" else -2
        x = b.x.clone(); x[:, qpos] = b.x[r, offset + 2 + 3 * other]
        out["query_edit"] = (m(x)[:, -1].argmax(-1) == b.x[r, offset + 3 + 3 * other]).float().mean().item()
    return out

results = {}
for name, d, var in runs():
    ck = torch.load(d / "model.pt", map_location=dev, weights_only=False)
    m = V.build(Config(**ck["config"]), var.get("noslope", False), var.get("pos", "learned")).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    fmt = var.get("fmt", "direct")
    val = val_curve(name, d)
    res = {"variant": var, "steps": ck["step"], "final_val": val[-1][1],
           "first_step_val>=0.99": next((s for s, a in val if a >= 0.99), None),
           "val_curve": val,
           "in_dist": evaluate(m, fmt, 64, 0, 1, edits=True),
           "gap": {gp: evaluate(m, fmt, gp, 0, 100 + gp)["acc"] for gp in GAPS},
           "offset": {o: evaluate(m, fmt, 64, o, 200 + o)["acc"] for o in OFFS}}
    results[name] = res
json.dump(results, open(HERE / "results/generalization.json", "w"), indent=1)
json.dump({k: v["val_curve"] for k, v in results.items()}, open(VAL_FILE, "w"), indent=0)

def key(n):
    base, s = re.match(r"(.*)_s(\d+)$", n).groups(); return base, int(s)
order = ["default", "noslope", "layers3", "sinus", "nope", "aw0", "old", "old16k", "old_nope", "randoff", "randoff16k", "nope_randoff", "nope_randoff16k"]
names = sorted(results, key=lambda n: (order.index(key(n)[0]), key(n)[1]))
print(f"{'run':17s}{'steps':>6s}{'val':>6s}{'learn@':>7s}{'g64':>6s}{'edit':>6s}{'qry':>6s} |"
      + "".join(f"{'g'+str(x):>6s}" for x in GAPS) + " |" + "".join(f"{'o'+str(x):>6s}" for x in OFFS))
for n in names:
    r = results[n]; i = r["in_dist"]
    print(f"{n:17s}{r['steps']:6d}{r['final_val']:6.2f}{str(r['first_step_val>=0.99']):>7s}{i['acc']:6.2f}{i['follow_edit']:6.2f}{i['query_edit']:6.2f} |"
          + "".join(f"{r['gap'][x]:6.2f}" for x in GAPS) + " |" + "".join(f"{r['offset'][x]:6.2f}" for x in OFFS))
