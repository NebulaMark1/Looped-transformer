"""
Train Asymmetric-Depth Transformer.

Usage:
    python train_asym.py --num_full 2 --num_ffn 6 --epochs 15
    python train_asym.py --num_full 4 --num_ffn 0 --epochs 15   (standard)
    python train_asym.py --num_full 1 --num_ffn 7 --epochs 15
"""

import argparse, json, math, os, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm
from asym_model import AsymConfig, AsymTransformer


def create_dataloaders(seq_len, batch_size):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def load_split(split):
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        tokens = []
        for item in ds:
            text = item["text"].strip()
            if text: tokens.extend(tokenizer.encode(text))
        return torch.tensor(tokens, dtype=torch.long)

    def chunks(tks, sl):
        c = []
        for i in range(0, len(tks) - sl, sl):
            chunk = tks[i:i+sl+1]
            if len(chunk) > 1: c.append({"input_ids": chunk[:-1], "labels": chunk[1:]})
        return c

    train_c = chunks(load_split("train"), seq_len)
    val_c = chunks(load_split("validation"), seq_len)

    def collate(batch):
        mx = max(it["input_ids"].size(0) for it in batch)
        ins = [F.pad(it["input_ids"], (0, mx - it["input_ids"].size(0)), value=0) for it in batch]
        lbs = [F.pad(it["labels"], (0, mx - it["labels"].size(0)), value=0) for it in batch]
        return {"input_ids": torch.stack(ins), "labels": torch.stack(lbs)}

    return (DataLoader(train_c, batch_size=batch_size, shuffle=True, collate_fn=collate),
            DataLoader(val_c, batch_size=batch_size, shuffle=False, collate_fn=collate))


def train_epoch(model, loader, optim, sched, device):
    model.train()
    total_loss, total_tokens = 0.0, 0
    for batch in tqdm(loader, desc="Train", leave=False):
        ids = batch["input_ids"].to(device)
        out = model(ids, labels=ids)
        loss = out["loss"]
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        sched.step()
        total_loss += loss.item() * ids.numel()
        total_tokens += ids.numel()
    return total_loss / total_tokens


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in tqdm(loader, desc="Val", leave=False):
        ids = batch["input_ids"].to(device)
        out = model(ids, labels=ids)
        total_loss += out["loss"].item() * ids.numel()
        total_tokens += ids.numel()
    return total_loss / total_tokens


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_full", type=int, default=2)
    p.add_argument("--num_ffn", type=int, default=6)
    p.add_argument("--ffn_bottleneck", type=int, default=None)
    p.add_argument("--embed_dim", type=int, default=384)
    p.add_argument("--num_heads", type=int, default=6)
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

    train_loader, val_loader = create_dataloaders(args.seq_len, args.batch_size)

    cfg = AsymConfig(
        num_full=args.num_full, num_ffn=args.num_ffn,
        ffn_bottleneck=args.ffn_bottleneck, embed_dim=args.embed_dim,
        num_heads=args.num_heads, dropout=args.dropout,
        max_seq_len=args.seq_len,
    )
    model = AsymTransformer(cfg).to(device)

    total, embed, trans = model.count_params()
    full_p = sum(sum(p.numel() for p in b.parameters()) for b in model.full_blocks)
    ffn_p = sum(sum(p.numel() for p in b.parameters()) for b in model.ffn_blocks)
    n_total = args.num_full + args.num_ffn
    print(f"Layers: {args.num_full}F + {args.num_ffn}FFN = {n_total} total")
    print(f"Params: {total:,} total, {trans:,} trans (full: {full_p:,}, ffn: {ffn_p:,})")

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.dim() < 2 or "norm" in name or "embedding" in name or "ln" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    optim = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay, "lr": args.lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": args.lr},
    ], betas=(0.9, 0.95))

    total_steps = len(train_loader) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=args.lr, total_steps=total_steps,
        pct_start=args.warmup_steps / total_steps,
    )

    run_name = f"asym_{args.num_full}f_{args.num_ffn}ffn"
    if args.embed_dim != 384: run_name += f"_d{args.embed_dim}"

    metrics, best_val = [], float("inf")
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        tl = train_epoch(model, train_loader, optim, sched, device)
        vl = validate(model, val_loader, device)
        tp, vp = math.exp(tl), math.exp(vl)
        metrics.append({"epoch": ep, "train_ppl": tp, "val_ppl": vp})
        print(f"Epoch {ep}: train={tp:.2f}, val={vp:.2f}  [{time.time()-t0:.0f}s]")
        if vp < best_val:
            best_val = vp
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"{run_name}_best.pt"))

    results = {
        "config": vars(args),
        "param_counts": {"total": total, "embedding": embed, "transformer": trans,
                         "full": full_p, "ffn": ffn_p},
        "best_val_ppl": best_val, "total_train_time": time.time() - t0,
        "metrics": metrics,
    }
    with open(os.path.join(args.output_dir, f"{run_name}_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Best PPL: {best_val:.2f}, saved to {args.output_dir}/{run_name}_results.json")


if __name__ == "__main__":
    main()
