"""
Train Delta-Loop Transformer.

Usage:
    python train_delta.py --delta_type ffn --epochs 15
    python train_delta.py --delta_type attn_ffn --per_loop_delta --epochs 15
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

from delta_model import DeltaConfig, DeltaLoopedTransformer


# ── Data ──────────────────────────────────────────────────────────────────────

def create_dataloaders(seq_len: int, batch_size: int, num_workers: int = 0):
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

    def make_chunks(tks, sl):
        chunks = []
        for i in range(0, len(tks) - sl, sl):
            chunk = tks[i:i + sl + 1]
            if len(chunk) > 1:
                chunks.append({"input_ids": chunk[:-1], "labels": chunk[1:]})
        return chunks

    train_chunks = make_chunks(train_tokens, seq_len)
    val_chunks = make_chunks(val_tokens, seq_len)

    def collate(batch):
        max_len = max(item["input_ids"].size(0) for item in batch)
        inputs = [F.pad(item["input_ids"], (0, max_len - item["input_ids"].size(0)), value=0) for item in batch]
        labels = [F.pad(item["labels"], (0, max_len - item["labels"].size(0)), value=0) for item in batch]
        return {"input_ids": torch.stack(inputs), "labels": torch.stack(labels)}

    train_loader = DataLoader(train_chunks, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_chunks, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader, tokenizer


# ── Training ───────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss, total_tokens = 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
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
        pbar.set_postfix({"loss": f"{loss.item():.3f}", "ppl": f"{math.exp(loss.item()):.1f}",
                          "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

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
    p = argparse.ArgumentParser(description="Train Delta-Loop Transformer")
    p.add_argument("--delta_type", type=str, default="ffn", choices=["ffn", "attn_ffn"])
    p.add_argument("--delta_bottleneck", type=int, default=None)
    p.add_argument("--per_loop_delta", action="store_true")
    p.add_argument("--embed_dim", type=int, default=384)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--num_loops", type=int, default=4)
    p.add_argument("--ff_mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, default="./results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Data
    print("Loading data...")
    train_loader, val_loader, _ = create_dataloaders(args.seq_len, args.batch_size)

    # Model
    config = DeltaConfig(
        delta_type=args.delta_type,
        delta_bottleneck=args.delta_bottleneck,
        per_loop_delta=args.per_loop_delta,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_loops=args.num_loops,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        max_seq_len=args.seq_len,
    )
    model = DeltaLoopedTransformer(config).to(device)

    total, embed, trans = model.count_params()
    full_params = sum(sum(p.numel() for p in b.parameters()) for b in model.full_blocks)
    delta_params = sum(b.count_params() for b in model.delta_blocks)
    print(f"Total params: {total:,}  (embed: {embed:,}, full_block: {full_params:,}, delta: {delta_params:,})")

    # Optimizer
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or "norm" in name or "embedding" in name or "ln" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay, "lr": args.lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": args.lr},
    ], betas=(0.9, 0.95))

    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=args.warmup_steps / total_steps,
    )

    # Run name
    run_name = f"delta_{args.delta_type}"
    if args.per_loop_delta:
        run_name += "_perloop"
    if args.delta_bottleneck is not None:
        run_name += f"_b{args.delta_bottleneck}"

    # Training
    metrics_history = []
    best_val_ppl = float("inf")
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        val_loss = validate(model, val_loader, device)
        train_ppl, val_ppl = math.exp(train_loss), math.exp(val_loss)

        metrics_history.append({"epoch": epoch, "train_ppl": train_ppl, "val_ppl": val_ppl})
        print(f"Epoch {epoch}: train_ppl={train_ppl:.2f}, val_ppl={val_ppl:.2f}")

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"{run_name}_best.pt"))

    total_time = time.time() - train_start

    results = {
        "config": vars(args),
        "param_counts": {"total": total, "embedding": embed, "transformer": trans,
                         "full_block": full_params, "delta": delta_params},
        "best_val_ppl": best_val_ppl,
        "total_train_time": total_time,
        "metrics": metrics_history,
    }
    with open(os.path.join(args.output_dir, f"{run_name}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/{run_name}_results.json")
    print(f"Best val ppl: {best_val_ppl:.2f}, Time: {total_time:.0f}s")


if __name__ == "__main__":
    main()
