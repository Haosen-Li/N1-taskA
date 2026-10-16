"""Sanity check: the hand-written model is a working LM on REAL text (char-level Python stdlib source of the
running interpreter; no internet on the training node). Uses taska model/AdamW/clip/loss unchanged. Writes followup/results/charlm.json.
  python followup/charlm.py cuda:1   (~2.5 min)"""
import sys, glob, math, json, time, sysconfig, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from model import Config, TransformerLM, AdamW, clip_grad_norm, token_cross_entropy
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"; out = str(HERE / "results")
import os; os.makedirs(out, exist_ok=True)
torch.manual_seed(0); g = torch.Generator().manual_seed(1)
# corpus = top-level stdlib .py files of the running Python (results.json was made with CPython 3.10.18)
files = sorted(glob.glob(str(Path(sysconfig.get_paths()["stdlib"]) / "*.py")))
text = "".join(open(f, encoding="utf-8", errors="replace").read() for f in files)
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text]); n = int(0.9 * len(data))
train, val = data[:n], data[n:]          # contiguous split, no overlap
CTX, B, STEPS, LR = 128, 64, 3000, 1e-3
cfg = Config(vocab_size=len(chars), d_model=128, n_heads=4, n_layers=4, max_seq_len=256)
m = TransformerLM(cfg).to(dev); opt = AdamW(m.parameters(), lr=LR)
def batch(d, gen):
    ix = torch.randint(len(d) - CTX - 1, (B,), generator=gen)
    return (torch.stack([d[i:i+CTX] for i in ix]).to(dev), torch.stack([d[i+1:i+CTX+1] for i in ix]).to(dev))
@torch.no_grad()
def evaluate(window=None, nb=40):
    gen = torch.Generator().manual_seed(123); m.eval(); per_pos = torch.zeros(CTX, device=dev)
    for _ in range(nb):
        x, y = batch(val, gen); per_pos += token_cross_entropy(m(x, window), y).mean(0)
    m.train(); return per_pos / nb
# count baselines (nats/char)
uni = torch.bincount(train, minlength=len(chars)).float() + 1; uni /= uni.sum()
unigram = -(uni.log()[val]).mean().item()
V = len(chars); big = torch.ones(V, V); big.index_put_((train[:-1], train[1:]), torch.ones(n - 1), accumulate=True)
big = big / big.sum(1, keepdim=True); bigram = -(big.log()[val[:-1], val[1:]]).mean().item()
log = []; t0 = time.time()
for step in range(1, STEPS + 1):
    opt.lr = LR * min(1, step / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / STEPS)))
    x, y = batch(train, g); opt.zero_grad()
    loss = token_cross_entropy(m(x), y).mean(); loss.backward(); clip_grad_norm(m.parameters(), 1.0); opt.step()
    if step % 500 == 0 or step == STEPS:
        v = evaluate().mean().item(); log.append({"step": step, "train": loss.item(), "val": v})
        print(json.dumps({**log[-1], "sec": round(time.time() - t0)}), flush=True)
pp = evaluate()
res = {"corpus_chars": len(text), "vocab": V, "params": sum(p.numel() for p in m.parameters()),
       "unigram_val_nats": unigram, "bigram_val_nats": bigram, "log": log,
       "val_nats_full": pp.mean().item(), "val_bits_per_char": pp.mean().item() / math.log(2),
       "val_loss_by_position": {t: pp[t].item() for t in [0, 1, 3, 7, 15, 31, 63, 127]},
       "val_loss_by_window": {w: evaluate(w).mean().item() for w in [1, 2, 4, 16, 64]}}
@torch.no_grad()
def sample(prompt, k=500, temp=0.8):
    gen = torch.Generator(device=dev).manual_seed(0); ids = torch.tensor([[stoi[c] for c in prompt]], device=dev); m.eval()
    for _ in range(k):
        p = torch.softmax(m(ids[:, -CTX:])[:, -1] / temp, -1); ids = torch.cat([ids, torch.multinomial(p, 1, generator=gen)], 1)
    return "".join(chars[i] for i in ids[0].tolist())
res["sample"] = sample("def ")
json.dump(res, open(f"{out}/charlm.json", "w"), indent=2); print(json.dumps({k: v for k, v in res.items() if k != "sample"}, indent=1)); print(res["sample"])
