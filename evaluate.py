"""Held-out test: does the model actually use the remote, query-relevant memory?

Evidence is paired input interventions (change one token, check the prediction
follows), not loss curves or attention maps.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch

from model import Config, TransformerLM, token_cross_entropy
from task import (make_batch, change_relevant_value, change_irrelevant_value, decode,
                  N_VALUES, VALUE_START, MEMORY_VALUE_POS)


def proportion(correct, total):
    p = correct / total
    z = 1.959963984540054
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"correct": correct, "n": total, "accuracy": p, "wilson95": [center - half, center + half]}


@torch.no_grad()
def evaluate_gap(model, gap, count, seed, batch_size=64, interventions=False):
    generator = torch.Generator().manual_seed(seed)
    edit_rng = torch.Generator().manual_seed(seed + 90000)
    device = next(model.parameters()).device
    counts, answer_nll, lm_nll = {}, 0.0, 0.0
    examples, distances = [], []

    def add(name, success):
        counts[name] = counts.get(name, 0) + success.sum().item()

    for start in range(0, count, batch_size):
        batch = make_batch(min(batch_size, count - start), gap, generator, device)
        n = len(batch.x)
        rows = torch.arange(n, device=device)
        logits = model(batch.x)
        pred = logits[:, -1].argmax(-1)  # unrestricted argmax over ALL vocabulary
        nll = token_cross_entropy(logits, batch.y)
        answer_nll += nll[:, -1].sum().item()
        lm_nll += nll.mean(-1).sum().item()
        add("full_context", pred == batch.answer)
        # Heuristic baselines: x ends with [..., last filler, <query>, K].
        add("last_noise_token_baseline", batch.x[:, -3] == batch.answer)
        add("memory_mode_no_query_baseline", batch.x[:, MEMORY_VALUE_POS].mode(-1).values == batch.answer)
        distances.extend((batch.x.shape[1] - batch.source_pos).tolist())
        if not interventions:
            continue

        edited, new_target = change_relevant_value(batch)
        new_pred = model(edited)[:, -1].argmax(-1)
        add("relevant_edit_new_target", new_pred == new_target)
        add("relevant_edit_both_correct", (pred == batch.answer) & (new_pred == new_target))
        add("relevant_edit_prediction_changed", pred != new_pred)
        # A controlled irrelevant edit should leave the queried answer intact.
        irrelevant_pred = model(change_irrelevant_value(batch))[:, -1].argmax(-1)
        add("irrelevant_edit_accuracy", irrelevant_pred == batch.answer)
        add("irrelevant_edit_prediction_unchanged", irrelevant_pred == pred)

        local_logits = model(batch.x, window=8)[:, -1]
        local_pred = local_logits.argmax(-1)
        add("local_window_8", local_pred == batch.answer)
        # Verify no hidden path relays source information through stacked layers.
        torch.testing.assert_close(local_logits, model(edited, window=8)[:, -1], atol=0, rtol=0)

        # Resample EVERY stored value independently, retaining a valid input.
        # Score against both the obsolete label and the correct rewritten label.
        erased = batch.x.clone()
        fresh = torch.randint(N_VALUES, (n, 4), generator=edit_rng).to(device) + VALUE_START
        erased[:, MEMORY_VALUE_POS] = fresh
        erased_pred = model(erased)[:, -1].argmax(-1)
        add("resample_values_original_target", erased_pred == batch.answer)
        add("resample_values_new_target", erased_pred == fresh[rows, batch.slot])

        # Change only the query key: disambiguates key-value binding from bag-of-values.
        other_slot = (batch.slot + 1) % 4
        query_edit = batch.x.clone()
        query_edit[:, -1] = batch.x[rows, 2 + 3 * other_slot]
        query_target = batch.x[rows, 3 + 3 * other_slot]
        query_pred = model(query_edit)[:, -1].argmax(-1)
        add("query_edit_new_target", query_pred == query_target)
        if not examples:
            for i in range(min(4, n)):
                examples.append({"input": decode(batch.x[i]), "source_position": int(batch.source_pos[i]),
                                 "answer": decode([batch.answer[i]]), "prediction": decode([pred[i]]),
                                 "edited_source_value": decode([new_target[i]]),
                                 "edited_prediction": decode([new_pred[i]]),
                                 "local_prediction": decode([local_pred[i]])})
    return {"gap": gap, "seed": seed, "answer_distance_min": min(distances),
            "answer_distance_max": max(distances), "answer_nll": answer_nll / count,
            "lm_nll": lm_nll / count, "lm_perplexity": math.exp(lm_nll / count),
            "metrics": {k: proportion(v, count) for k, v in counts.items()}, "examples": examples}


@torch.no_grad()
def exhaustive_counterfactuals(model, count=256, gap=64, seed=40001):
    """Each group contains 8 identical-suffix prompts, one per possible answer."""
    rng = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    all_correct, total_correct, rows_saved = 0, 0, []
    for start in range(0, count, 32):
        batch = make_batch(min(32, count - start), gap, rng, device)
        n = len(batch.x)
        variants = batch.x[:, None, :].repeat(1, N_VALUES, 1)
        labels = torch.arange(VALUE_START, VALUE_START + N_VALUES, device=device).expand(n, -1)
        variants[torch.arange(n, device=device)[:, None],
                 torch.arange(N_VALUES, device=device)[None, :], batch.source_pos[:, None]] = labels
        assert torch.equal(variants[:, :, 13:], batch.x[:, None, 13:].expand(-1, N_VALUES, -1))
        predicted = model(variants.flatten(0, 1))[:, -1].argmax(-1).reshape(n, N_VALUES)
        correct = predicted == labels
        all_correct += correct.all(1).sum().item()
        total_correct += correct.sum().item()
        if not rows_saved:
            rows_saved = [{"remote_value": decode([labels[0, j]]),
                           "prediction": decode([predicted[0, j]])} for j in range(N_VALUES)]
    return {"groups": count, "gap": gap, "seed": seed,
            # Variants within a group are dependent: no binomial CI over variants.
            "variant_accuracy": total_correct / (count * N_VALUES),
            "all_8_correct": proportion(all_correct, count), "example_group": rows_saved}


def make_plots(out, results, history_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    history = [json.loads(line) for line in history_path.read_text().splitlines()]
    train = [r for r in history if "lm_loss" in r]
    val = [r for r in history if "val_answer_accuracy" in r]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # Moving average improves readability; raw values remain in history.jsonl.
    for key, label in [("lm_loss", "Next-token NLL"), ("answer_loss", "Answer NLL")]:
        chunks = [train[i:i + 50] for i in range(0, len(train), 50)]
        axes[0].plot([c[-1]["step"] for c in chunks],
                     [sum(r[key] for r in c) / len(c) for c in chunks], label=label)
    axes[0].set(xlabel="Optimizer steps", ylabel="Nats", title="Training (50-step averages)")
    axes[0].legend()
    axes[1].plot([r["step"] for r in val], [r["val_answer_accuracy"] for r in val], marker=".")
    axes[1].axhline(0.125, color="gray", linestyle="--", label="Chance (1/8)")
    axes[1].set(xlabel="Optimizer steps", ylabel="Answer accuracy", ylim=(0, 1.02), title="Validation")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "training.png", dpi=180)
    plt.close(fig)
    metrics = results["primary"]["metrics"]
    keys = ["full_context", "relevant_edit_new_target", "irrelevant_edit_accuracy", "query_edit_new_target",
            "local_window_8", "resample_values_original_target"]
    labels = ["Full context", "Relevant edit\n(new answer)", "Irrelevant edit", "Query key edit",
              "Local window 8", "Resampled memory\n(old answer)"]
    values = [metrics[k]["accuracy"] for k in keys]
    errors = [[max(0, v - metrics[k]["wilson95"][0]) for k, v in zip(keys, values)],
              [max(0, metrics[k]["wilson95"][1] - v) for k, v in zip(keys, values)]]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(labels, values, yerr=errors, capsize=4, color=["#226f9e"] * 4 + ["#a86242"] * 2)
    ax.axhline(0.125, color="gray", linestyle="--", label="Chance (1/8)")
    ax.set(ylabel="Answer accuracy (95% Wilson CI)", ylim=(0, 1.08), title="Held-out causal interventions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "interventions.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/direct/model.pt")
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--count", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=30001)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model = TransformerLM(Config(**checkpoint["config"])).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    out = Path(args.out) if args.out else checkpoint_path.parent
    out.mkdir(parents=True, exist_ok=True)
    result = {"checkpoint": str(checkpoint_path), "step": checkpoint["step"],
              "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
              "device": args.device, "chance": 1 / N_VALUES,
              "primary": evaluate_gap(model, 64, args.count, args.seed, interventions=True)}
    print(json.dumps({"primary": result["primary"]}, indent=2), flush=True)
    # 40/56: unseen in-range lengths; 96/128: longer than any training sequence.
    result["lengths"] = [evaluate_gap(model, gap, 1024, args.seed + gap) for gap in (32, 40, 48, 56, 96, 128)]
    result["exhaustive_counterfactuals"] = exhaustive_counterfactuals(model)
    torch.manual_seed(checkpoint["metadata"]["args"]["seed"])
    untrained = TransformerLM(Config(**checkpoint["config"])).to(args.device).eval()
    result["untrained"] = evaluate_gap(untrained, 64, args.count, args.seed)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    history = checkpoint_path.parent / "history.jsonl"
    if history.exists():
        make_plots(out, result, history)
    print(json.dumps({"length_accuracy": {r["gap"]: r["metrics"]["full_context"]["accuracy"]
                                         for r in result["lengths"]},
                      "counterfactuals": result["exhaustive_counterfactuals"]}, indent=2), flush=True)
    print(f"saved {out / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
