"""
Training script for Looped Transformer with Per-Loop LoRA experiment.

Usage:
    python train.py --mode baseline --epochs 5
    python train.py --mode lora --lora_rank 16 --epochs 5
    python train.py --mode full --epochs 5
"""

import argparse
import json
import math
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm

from model import create_model, print_param_summary


# ── Data ──────────────────────────────────────────────────────────────────────

class WikiTextDataset(IterableDataset):
    """Stream WikiText-2 as chunks of token IDs."""

    def __init__(self, split: str, seq_len: int, tokenizer):
        self.seq_len = seq_len
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        self.data = dataset

        # Pre-tokenize all texts, filter empty lines
        self.tokens = []
        for item in dataset:
            text = item["text"].strip()
            if text:
                ids = tokenizer.encode(text)
                self.tokens.extend(ids)
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        self.length = (len(self.tokens) - 1) // seq_len

    def __len__(self):
        return self.length

    def __iter__(self):
        for i in range(self.length):
            start = i * self.seq_len
            chunk = self.tokens[start:start + self.seq_len + 1]
            if len(chunk) <= 1:
                continue
            yield {
                "input_ids": chunk[:-1],
                "labels": chunk[1:],
            }


def collate_batch(batch):
    """Pad sequences in batch to the same length."""
    max_len = max(item["input_ids"].size(0) for item in batch)
    padded_inputs, padded_labels = [], []
    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad = max_len - seq_len
        padded_inputs.append(torch.cat([item["input_ids"], torch.zeros(pad, dtype=torch.long)]))
        padded_labels.append(torch.cat([item["labels"], torch.zeros(pad, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(padded_inputs),
        "labels": torch.stack(padded_labels),
    }


def create_dataloaders(seq_len: int, batch_size: int, num_workers: int = 0):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_ds = WikiTextDataset("train", seq_len, tokenizer)
    val_ds = WikiTextDataset("validation", seq_len, tokenizer)

    # Sort for val to get stable batches (converted to list since IterableDataset)
    val_data = list(val_ds)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_batch, num_workers=num_workers)

    return train_loader, val_loader, tokenizer


# ── Training ───────────────────────────────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch: int):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    start = time.time()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Create proper labels (shifted inside model)
        full_input = input_ids
        full_labels = labels

        output = model(full_input, labels=full_input)
        loss = output["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        tokens = input_ids.numel()
        total_loss += loss.item() * tokens
        total_tokens += tokens

        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "ppl": f"{math.exp(loss.item()):.1f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        })

    elapsed = time.time() - start
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)

    return {"loss": avg_loss, "ppl": ppl, "tokens_per_sec": total_tokens / elapsed, "time": elapsed}


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        input_ids = batch["input_ids"].to(device)
        output = model(input_ids, labels=input_ids)
        loss = output["loss"]

        total_loss += loss.item() * input_ids.numel()
        total_tokens += input_ids.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return {"loss": avg_loss, "ppl": ppl}


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Looped Transformer with Per-Loop LoRA")
    p.add_argument("--mode", type=str, default="baseline",
                   choices=["baseline", "lora", "full"],
                   help="Model mode")
    p.add_argument("--lora_rank", type=int, default=16,
                   help="LoRA rank (only for lora mode)")
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--num_loops", type=int, default=4)
    p.add_argument("--ff_mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, default="./results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Data
    print("Loading data...")
    train_loader, val_loader, tokenizer = create_dataloaders(
        args.seq_len, args.batch_size, args.num_workers,
    )

    # Model
    model = create_model(
        mode=args.mode,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_loops=args.num_loops,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        lora_rank=args.lora_rank,
        max_seq_len=args.seq_len,
    ).to(device)

    total, embed, trans = print_param_summary(model)

    # Optimizer
    opt_groups = model.configure_optimizer_groups(args.weight_decay, args.lr)
    optimizer = torch.optim.AdamW(opt_groups, betas=(0.9, 0.95))
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=args.warmup_steps / total_steps,
    )

    # Training
    metrics_history = []
    best_val_ppl = float("inf")
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        val_metrics = validate(model, val_loader, device)

        info = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_ppl": train_metrics["ppl"],
            "val_loss": val_metrics["loss"],
            "val_ppl": val_metrics["ppl"],
            "tokens_per_sec": train_metrics["tokens_per_sec"],
        }
        metrics_history.append(info)

        print(f"Epoch {epoch}: train_ppl={train_metrics['ppl']:.2f}, val_ppl={val_metrics['ppl']:.2f}, "
              f"tokens/s={train_metrics['tokens_per_sec']:.0f}")

        if val_metrics["ppl"] < best_val_ppl:
            best_val_ppl = val_metrics["ppl"]
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.mode}_best.pt"))

    total_time = time.time() - train_start

    # Save results
    results = {
        "config": vars(args),
        "param_counts": {"total": total, "embedding": embed, "transformer": trans},
        "best_val_ppl": best_val_ppl,
        "total_train_time": total_time,
        "metrics": metrics_history,
    }

    run_name = f"{args.mode}"
    if args.mode == "lora":
        run_name += f"_r{args.lora_rank}"
    results_path = os.path.join(args.output_dir, f"{run_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(f"Best val ppl: {best_val_ppl:.2f}")
    print(f"Total time: {total_time:.0f}s")


if __name__ == "__main__":
    main()
