"""Figures from followup/results/*.json.   python followup/make_figures.py"""
import json, re
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; R = HERE / "results"; FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
gen = json.load(open(R / "generalization.json"))
def group(prefix): return {k: v for k, v in gen.items() if re.fullmatch(prefix + r"_s\d+", k)}

# 1. learning curves: one panel per single-variable change.
# One colour per seed and dashed lines for 16k-step runs, so a single legend outside the axes covers every panel.
from matplotlib.lines import Line2D
SEED_COLOR = {7: "tab:blue", 8: "tab:orange", 9: "tab:green", 10: "tab:red", 11: "tab:purple"}
panels = [("default", "delivered config (5 seeds)"), ("noslope", "no distance bias"), ("layers3", "3 layers"),
          ("sinus", "fixed sinusoidal positions"), ("nope", "no absolute positions (NoPE)"), ("aw0", "answer weight 0 (pure LM loss)"),
          ("old|old16k", "old format <query> K <answer>"), ("old_nope", "old format + NoPE"),
          ("randoff|randoff16k", "memory offset U{0..16}"), ("nope_randoff|nope_randoff16k", "NoPE + memory offset")]
fig, axes = plt.subplots(2, 5, figsize=(21, 7.8), sharey=True)
for ax, (pre, title) in zip(axes.flat, panels):
    runs = {k: v for p in pre.split("|") for k, v in group(p).items()}
    for k, v in sorted(runs.items()):
        seed = int(k.rsplit("_s", 1)[1]); s, a = zip(*v["val_curve"])
        ax.plot(s, a, color=SEED_COLOR[seed], ls="--" if "16k" in k else "-", marker="." if len(s) < 20 else None, lw=1.2)
    ok = sum(v["final_val"] >= 0.99 for v in runs.values())
    ax.axhline(0.125, color="gray", ls=":", lw=1); ax.axhline(0.412, color="tab:red", ls="--", lw=0.8)
    ax.set_title(f"{title}: {ok}/{len(runs)} runs learned", fontsize=10); ax.set_xscale("symlog", linthresh=1000); ax.set_ylim(0, 1.03)
for ax in axes[-1]: ax.set_xlabel("optimizer step")
for ax in axes[:, 0]: ax.set_ylabel("val answer accuracy")
handles = [Line2D([], [], color=c, marker=".", label=f"seed {s}") for s, c in SEED_COLOR.items()]
handles += [Line2D([], [], color="k", ls="--", label="16000-step run"), Line2D([], [], color="gray", ls=":", label="chance 1/8"),
            Line2D([], [], color="tab:red", ls="--", lw=0.8, label="query-ignoring 'memory mode' shortcut 41%")]
fig.suptitle("One change from the delivered config per panel")
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=len(handles), fontsize=9, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(FIG / "ablation_curves.png", dpi=150); plt.close(fig)

# 2. generalization: distance sweep + memory offset sweep
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.3))
style = {"default": ("tab:blue", "learned abs. pos (delivered config)"), "nope": ("tab:green", "no abs. pos (NoPE)")}
for pre, (c, lab) in style.items():
    for i, (k, v) in enumerate(sorted(group(pre).items())):
        if v["final_val"] < 0.99: continue
        g = sorted(v["gap"], key=int); a1.plot([int(x) for x in g], [v["gap"][x] for x in g], color=c, alpha=.7, marker="o", ms=3, label=lab if i == 0 else None)
        o = ["0"] + sorted(v["offset"], key=int); a2.plot([int(x) for x in o], [v["in_dist"]["acc"]] + [v["offset"][x] for x in o[1:]], color=c, alpha=.7, marker="o", ms=3, label=lab if i == 0 else None)
v = gen["randoff16k_s8"]; o = ["0"] + sorted(v["offset"], key=int)
a2.plot([int(x) for x in o], [v["in_dist"]["acc"]] + [v["offset"][x] for x in o[1:]], color="tab:orange", marker="s", ms=4, label="trained with offset U{0..16} (s8, 16k)")
for gp in (32, 48, 64): a1.axvline(gp, color="gray", ls=":", lw=1)
a1.set(xlabel="filler tokens between memory and query (gap); dotted = trained gaps", ylabel="accuracy", title="Distance sweep (successful seeds only)", ylim=(0, 1.03), xlim=(-3, 205))
a2.set(xlabel="filler tokens inserted BEFORE the memory block (gap 64)", title="Memory-position shift", ylim=(0, 1.03), xlim=(-1, 33))
for ax in (a1, a2): ax.axhline(0.125, color="gray", ls="--", lw=.8, label="chance 1/8" if ax is a2 else None)
# one shared legend above both panels (inside the axes it hid the curves)
fig.legend(*a2.get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=9, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(FIG / "generalization.png", dpi=150); plt.close(fig)

# 3. position swap
ps = json.load(open(R / "position_swap.json")); gaps = ["0", "16", "36", "40", "56"]
fig, ax = plt.subplots(figsize=(10, 4.6)); w = 0.27
for j, (key, lab, c) in enumerate([("normal", "unchanged", "tab:gray"), ("rows_77_78", "query uses position rows 77,78 (trained query slot)", "tab:blue"),
                                   ("rows_200_201", "query uses rows 200,201 (never trained)", "tab:green")]):
    for i, gp in enumerate(gaps):
        vals = [ps[m][gp][key] for m in ps]
        ax.bar(i + (j - 1) * w, sum(vals) / len(vals), w, color=c, alpha=.6, label=lab if i == 0 else None)
        ax.scatter([i + (j - 1) * w] * len(vals), vals, color="k", s=8, zorder=3)
ax.set_xticks(range(len(gaps)), [f"gap {g}" for g in gaps]); ax.set_ylim(0, 1.05); ax.set_ylabel("accuracy (bar = mean of 5 seeds, dots = seeds)")
ax.set_title("Only the absolute-position rows of the last two inputs are swapped; every token is unchanged")
# legend below the x-axis so it never covers bars or seed dots
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, frameon=False)
fig.tight_layout(); fig.savefig(FIG / "position_swap.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# 4. char-level LM on real text
c = json.load(open(R / "charlm.json"))
fig, ax = plt.subplots(figsize=(6.5, 4))
pos = sorted(c["val_loss_by_position"], key=int)
ax.plot([int(p) + 1 for p in pos], [c["val_loss_by_position"][p] for p in pos], marker="o", label="Transformer, val loss at context position t")
ax.axhline(c["unigram_val_nats"], color="tab:red", ls="--", label=f"unigram {c['unigram_val_nats']:.2f}")
ax.axhline(c["bigram_val_nats"], color="tab:orange", ls="--", label=f"count bigram {c['bigram_val_nats']:.2f}")
ax.set(xscale="log", xlabel="tokens of context available (t+1)", ylabel="nats / char", title=f"Char-level LM on Python source: val {c['val_nats_full']:.3f} nats ({c['val_bits_per_char']:.2f} bpc)")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "charlm.png", dpi=150); plt.close(fig)
print("figures:", sorted(p.name for p in FIG.iterdir()))
