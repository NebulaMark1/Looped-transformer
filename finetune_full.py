"""
Fine-tune a full model initialized from trained baseline checkpoint.

This ensures W_full[t] starts from W_base for each loop t, so training
residuals W_full[t] - W_base represent meaningful per-loop specializations,
not random drift from different initializations.

Usage:
    python finetune_full.py --baseline_ckpt results/baseline_best.pt \
                            --epochs 3 --output_dir ./results
"""

import argparse
import json
import math
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm

from model import create_model, LoopedTransformerConfig


# ── Data ──────────────────────────────────────────────────────────────────────

def create_dataloaders(seq_len: int, batch_size: int):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def load_split(split):
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        tokens = []
        for item in dataset:
            text = item["text"].strip()
            if text:
                tokens.extend(tokenizer.encode(text))
        return torch.tensor(tokens, dtype=torch.long)

    train_tokens = load_split("train")
    val_tokens = load_split("validation")

    def make_chunks(t, seq_len):
        chunks = []
        for i in range(0, len(t) - seq_len, seq_len):
            chunk = t[i:i + seq_len + 1]
            if len(chunk) > 1:
                chunks.append({"input_ids": chunk[:-1], "labels": chunk[1:]})
        return chunks

    train_chunks = make_chunks(train_tokens, seq_len)

    def collate(batch):
        max_len = max(item["input_ids"].size(0) for item in batch)
        inputs = [F.pad(item["input_ids"], (0, max_len - item["input_ids"].size(0)), value=0) for item in batch]
        labels = [F.pad(item["labels"], (0, max_len - item["labels"].size(0)), value=0) for item in batch]
        return {"input_ids": torch.stack(inputs), "labels": torch.stack(labels)}

    train_loader = DataLoader(train_chunks, batch_size=batch_size, shuffle=True, collate_fn=collate)

    # Val evaluation
    val_chunks = make_chunks(val_tokens, seq_len)
    val_loader = DataLoader(val_chunks, batch_size=batch_size, shuffle=False, collate_fn=collate)

    return train_loader, val_loader


# ── Init full from baseline ──────────────────────────────────────────────────

def init_full_from_baseline(baseline_ckpt: str, config_kwargs: dict):
    """
    Create a full-mode model where each loop's weights are initialized
    from the trained baseline checkpoint.
    """
    from model import LoopedTransformer

    baseline_cfg = LoopedTransformerConfig(mode="baseline", **config_kwargs)
    full_cfg = LoopedTransformerConfig(mode="full", **config_kwargs)

    baseline = LoopedTransformer(baseline_cfg)
    baseline.load_state_dict(torch.load(baseline_ckpt, map_location="cpu"))

    full_model = LoopedTransformer(full_cfg)

    full_modules = dict(full_model.named_modules())
    baseline_modules = dict(baseline.named_modules())

    for name, mod in full_modules.items():
        if mod.__class__.__name__ == "LoRALinear":
            base_mod = baseline_modules[name]
            # Copy baseline weight to each loop
            for t in range(mod.weight.shape[0]):
                mod.weight.data[t].copy_(base_mod.weight.data)
            mod.bias.data.copy_(base_mod.bias.data)

    # Copy embeddings and LNs
    full_params = dict(full_model.named_parameters())
    baseline_params = dict(baseline.named_parameters())
    for name in full_params:
        if name in baseline_params and full_params[name].shape == baseline_params[name].shape:
            full_params[name].data.copy_(baseline_params[name].data)

    return full_model


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    pbar = tqdm(loader, desc="Train")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        output = model(input_ids, labels=input_ids)
        loss = output["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        tokens = input_ids.numel()
        total_loss += loss.item() * tokens
        total_tokens += tokens
        pbar.set_postfix({"loss": f"{loss.item():.3f}", "ppl": f"{math.exp(loss.item()):.1f}"})

    return total_loss / total_tokens


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in tqdm(loader, desc="Val", leave=False):
        input_ids = batch["input_ids"].to(device)
        output = model(input_ids, labels=input_ids)
        total_loss += output["loss"].item() * input_ids.numel()
        total_tokens += input_ids.numel()
    return total_loss / total_tokens


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune full model from baseline init")
    p.add_argument("--baseline_ckpt", type=str, default="results/baseline_best.pt")
    p.add_argument("--output_dir", type=str, default="./results")
    p.add_argument("--embed_dim", type=int, default=384)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--num_loops", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config_kwargs = dict(
        embed_dim=args.embed_dim, num_heads=args.num_heads,
        num_layers=args.num_layers, num_loops=args.num_loops,
        max_seq_len=args.seq_len,
    )

    # Data
    print("Loading data...")
    train_loader, val_loader = create_dataloaders(args.seq_len, args.batch_size)

    # Model: full-mode initialized from baseline
    print("Initializing full model from baseline...")
    model = init_full_from_baseline(args.baseline_ckpt, config_kwargs).to(device)

    # Check initial PPL (should be close to baseline)
    init_val_loss = validate(model, val_loader, device)
    print(f"Initial val PPL (should ≈ baseline): {math.exp(init_val_loss):.2f}")

    # Optimizer
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if len(param.shape) < 2 or "norm" in name or "embedding" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay, "lr": args.lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": args.lr},
    ], betas=(0.9, 0.95))
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=args.warmup_steps / total_steps,
    )

    # Train
    best_val_ppl = float("inf")
    metrics = []
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = validate(model, val_loader, device)
        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(val_loss)
        metrics.append({"epoch": epoch, "train_ppl": train_ppl, "val_ppl": val_ppl})
        print(f"Epoch {epoch}: train_ppl={train_ppl:.2f}, val_ppl={val_ppl:.2f}")

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(model.state_dict(), os.path.join(args.output_dir, "full_ft_best.pt"))

    total_time = time.time() - train_start
    print(f"Best val PPL: {best_val_ppl:.2f}, Time: {total_time:.0f}s")

    out = {
        "config": vars(args),
        "best_val_ppl": best_val_ppl,
        "total_time": total_time,
        "metrics": metrics,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "full_ft_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {args.output_dir}/full_ft_best.pt")


if __name__ == "__main__":
    main()
