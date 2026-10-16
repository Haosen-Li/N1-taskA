"""Attention of the delivered model, per head: layer-1 value->previous key (binding) and
layer-2 query->true source (retrieval), in-distribution vs shifted memory vs short gap.
Writes followup/results/attention.json.   python followup/attention_probe.py cuda:1"""
import sys, json, glob, os, re, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent; TASKA = HERE.parent
sys.path.insert(0, str(TASKA)); sys.path.insert(0, str(HERE))
import variants as V
from model import Config
import math
import model as M
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
ck = torch.load(TASKA / "runs/direct/model.pt", map_location=dev, weights_only=False)
m = M.TransformerLM(M.Config(**ck["config"])).to(dev); m.load_state_dict(ck["model"]); m.eval()
store = []
orig = M.CausalAttention.forward
def recording_forward(self, x, window=None):        # recompute the same weights as model.py and record them
    b, t, c = x.shape
    q, k, v = self.qkv(x).chunk(3, dim=-1)
    q, k, v = [z.reshape(b, t, self.n_heads, self.head_dim).transpose(1, 2) for z in (q, k, v)]
    s = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    pos = torch.arange(t, device=x.device); d = pos[:, None] - pos[None, :]
    s = s - self.distance_slopes[None, :, None, None] * d.clamp(min=0)
    store.append(M.softmax(s.masked_fill(~(d >= 0), float("-inf"))))
    return orig(self, x, window)
M.CausalAttention.forward = recording_forward

@torch.no_grad()
def run(gap, offset=0, n=512):
    b = V.make(n, gap, torch.Generator().manual_seed(5), dev, offset=offset)
    store.clear(); pred = m(b.x)[:, -1].argmax(-1)
    L1, L2 = store; r = torch.arange(n, device=dev); src = b.source_pos
    slopes = m.blocks[0].attention.distance_slopes.tolist()
    res = {"gap": gap, "offset": offset, "acc": (pred == b.answer).float().mean().item(), "heads": []}
    for h in range(4):
        res["heads"].append({"head": h, "slope": slopes[h],
                             "L1_value_to_previous_key": L1[r, h, src, src - 1].mean().item(),
                             "L2_query_to_true_source": L2[r, h, -1, src].mean().item()})
    print(f"gap={gap:3d} offset={offset}  acc={res['acc']:.3f}")
    for hh in res["heads"]:
        print(f"   head{hh['head']} slope={hh['slope']:.1f}  L1 value->prev key={hh['L1_value_to_previous_key']:.3f}"
              f"   L2 query->true source={hh['L2_query_to_true_source']:.3f}")
    return res
out = [run(64), run(64, offset=1), run(0), run(36), run(96)]
json.dump(out, open(HERE / "results/attention.json", "w"), indent=1)
