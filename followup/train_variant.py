"""Run the unchanged taska/train.py with ONE model/data variant (monkeypatched; no core file is edited).
Unknown args pass through to train.py. Run from taska/:
  python followup/train_variant.py --pos none --seed 7 --device cuda:1 --threads 1 --out followup/runs/nope_s7
"""
import sys, os, json, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import variants as V
p = argparse.ArgumentParser()
p.add_argument("--noslope", action="store_true", help="zero the fixed distance-bias slopes")
p.add_argument("--pos", default="learned", choices=["learned", "sinusoidal", "none"])
p.add_argument("--fmt", default="direct", choices=["direct", "old"], help="old = <query> K <answer> -> V")
p.add_argument("--offset-max", type=int, default=None, help="random filler before memory, U{0..k} per batch")
a, rest = p.parse_known_args()
import train as TR
TR.TransformerLM = lambda cfg: V.build(cfg, a.noslope, a.pos)
if a.fmt != "direct" or a.offset_max is not None:
    TR.make_batch = lambda bs, gap, g, device="cpu": V.make(bs, gap, g, device, fmt=a.fmt, offset_max=a.offset_max)
out = rest[rest.index("--out") + 1]
os.makedirs(out, exist_ok=True)
json.dump(vars(a), open(os.path.join(out, "variant.json"), "w"))
sys.argv = ["train.py"] + rest
TR.main()
