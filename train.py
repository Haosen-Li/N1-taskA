"""Train with fresh data each step; validation never controls optimizer updates.

Defaults are exactly the delivered configuration (runs/direct). On GPU set
CUBLAS_WORKSPACE_CONFIG=:4096:8 so deterministic algorithms are allowed.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import random
import sys
import time

import torch
from model import Config, TransformerLM, AdamW, objective, clip_grad_norm
from task import VOCAB, TRAIN_GAPS, make_batch


@torch.no_grad()
def validate(model, seed=20001, count=384, batch_size=64):
    generator = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    correct, loss_sum = 0, 0.0
    model.eval()
    for start in range(0, count, batch_size):
        n = min(batch_size, count - start)
        gap = TRAIN_GAPS[(start // batch_size) % len(TRAIN_GAPS)]
        batch = make_batch(n, gap, generator, device)
        logits = model(batch.x)
        _, _, answer_loss = objective(logits, batch.y)
        correct += (logits[:, -1].argmax(-1) == batch.answer).sum().item()
        loss_sum += answer_loss.item() * n
    model.train()
    return {"val_answer_accuracy": correct / count, "val_answer_nll": loss_sum / count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/reproduce")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--answer-weight", type=float, default=2.0)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    rng = torch.Generator().manual_seed(args.seed + 1000)
    gap_rng = random.Random(args.seed + 2000)
    cfg = Config(vocab_size=len(VOCAB), n_layers=args.layers, d_model=args.width)
    model = TransformerLM(cfg).to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"args": vars(args), "model": asdict(cfg), "python": sys.executable,
            "python_version": platform.python_version(), "torch": torch.__version__,
            "device_name": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "CPU",
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "train_gaps": TRAIN_GAPS, "validation_seed": 20001}
    (out / "config.json").write_text(json.dumps(meta, indent=2) + "\n")
    initial = {"step": 0, **validate(model)}
    print(json.dumps({**meta, "initial": initial}), flush=True)
    start = time.monotonic()
    with (out / "history.jsonl").open("w", buffering=1) as log:
        log.write(json.dumps(initial) + "\n")
        for step in range(1, args.steps + 1):
            gap = gap_rng.choice(TRAIN_GAPS)
            batch = make_batch(args.batch_size, gap, rng, args.device)
            warmup = min(1.0, step / 50)
            progress = max(0.0, (step - 50) / max(1, args.steps - 50))
            optimizer.lr = args.lr * warmup * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
            optimizer.zero_grad()
            logits = model(batch.x)
            loss, lm_loss, answer_loss = objective(logits, batch.y, args.answer_weight)
            loss.backward()  # reverse-mode chain rule through ALL the explicit layers
            if step == 1:
                trace = {name: {"shape": list(p.shape), "grad_norm": p.grad.norm().item()}
                         for name, p in model.named_parameters()}
                (out / "first_backward.json").write_text(json.dumps(trace, indent=2) + "\n")
            grad_norm = clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()
            row = {"step": step, "loss": loss.item(), "lm_loss": lm_loss.item(),
                   "answer_loss": answer_loss.item(), "gradient_norm_before_clip": grad_norm,
                   "lr": optimizer.lr, "gap": gap, "elapsed_seconds": time.monotonic() - start}
            if step % args.eval_every == 0 or step == args.steps:
                row.update(validate(model))
                print(json.dumps(row), flush=True)
                # Fixed training budget; final checkpoint, no test-driven selection.
                torch.save({"config": asdict(cfg), "model": model.state_dict(),
                            "step": step, "metadata": meta}, out / "model.pt")
            log.write(json.dumps(row) + "\n")
    print(f"saved {out / 'model.pt'}", flush=True)


if __name__ == "__main__":
    main()
