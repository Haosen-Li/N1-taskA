"""Causal test: does the learned ABSOLUTE position row at the query decide success at unseen lengths?
Only the position rows used by the last two inputs (<query>, K) are swapped; no token changes.
Writes followup/results/position_swap.json.   python followup/position_swap.py cuda:1"""
import sys, json, glob, os, re, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent; TASKA = HERE.parent
sys.path.insert(0, str(TASKA)); sys.path.insert(0, str(HERE))
import variants as V
from model import Config
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
runs = {"default_s7": TASKA / "runs/direct"} | {f"default_s{s}": HERE / f"runs/default_s{s}" for s in (8, 9, 10, 11)}

@torch.no_grad()
def acc(m, gap, rows=None, n=1024):
    b = V.make(n, gap, torch.Generator().manual_seed(500 + gap), dev)
    T = b.x.shape[1]; saved = m.positions.data.clone()
    if rows is not None:
        m.positions.data[T - 2] = saved[rows[0]]; m.positions.data[T - 1] = saved[rows[1]]
    a = (m(b.x)[:, -1].argmax(-1) == b.answer).float().mean().item()
    m.positions.data.copy_(saved); return a

out = {}
print(f"{'model':12s}{'gap':>5s}{'normal':>8s}{'rows 77,78 (trained query slot)':>33s}{'rows 200,201 (untrained)':>26s}")
for name, d in runs.items():
    ck = torch.load(d / "model.pt", map_location=dev, weights_only=False)
    m = V.build(Config(**ck["config"])).to(dev); m.load_state_dict(ck["model"]); m.eval()
    out[name] = {}
    for gap in [0, 16, 36, 40, 56]:
        r = {"normal": acc(m, gap), "rows_77_78": acc(m, gap, (77, 78)), "rows_200_201": acc(m, gap, (200, 201))}
        out[name][str(gap)] = r
        print(f"{name:12s}{gap:5d}{r['normal']:8.2f}{r['rows_77_78']:33.2f}{r['rows_200_201']:26.2f}")
    r = {"normal": acc(m, 64), "rows_50_51_filler": acc(m, 64, (50, 51))}
    out[name]["64_reverse"] = r
    print(f"{name:12s}{64:5d}{r['normal']:8.2f}   reverse: query given filler rows 50,51 -> {r['rows_50_51_filler']:.2f}")
json.dump(out, open(HERE / "results/position_swap.json", "w"), indent=1)
