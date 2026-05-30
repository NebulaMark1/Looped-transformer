"""
Continue training a delta model on a new dataset.

Usage:
    python continue_train.py --pretrained results/delta_ffn_loop8_wt103_best.pt \
        --dataset fineweb --epochs 3 --output_dir ./results
"""

import argparse, json, math, os, time, itertools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm

from delta_model import DeltaConfig, DeltaLoopedTransformer


def auto_detect(state_dict, ckpt_path):
    import re
    d = state_dict["token_embedding.weight"].shape[1]
    has_full = any("full_blocks." in k for k in state_dict)
    has_delta = any("delta_blocks." in k for k in state_dict)

    prefix = "full_blocks." if (has_full and has_delta) else "blocks."
    indices = set()
    for k in state_dict:
        if k.startswith(prefix):
            m = re.match(rf"{re.escape(prefix)}(\d+)\.", k)
            if m: indices.add(int(m.group(1)))
    num_layers = max(indices) + 1 if indices else 3

    possible = [h for h in [4,5,6,7,8,9,10,12,14,16] if d % h == 0]
    num_heads = min(possible, key=lambda h: abs(d//h - 64)) if possible else 6

    num_loops = 4
    m = re.search(r"_loop(\d+)", ckpt_path)
    if m: num_loops = int(m.group(1))

    delta_bn = None
    if has_delta:
        fc1_keys = [k for k in state_dict if "delta_blocks.0.fc1" in k and "weight" in k]
        if fc1_keys:
            bn = state_dict[fc1_keys[0]].shape[0]
            if bn != d // 4: delta_bn = bn

    return d, num_heads, num_layers, num_loops, delta_bn


# ── Streaming dataset for FineWeb ──────────────────────────────────────────────

class StreamingDataset(IterableDataset):
    def __init__(self, tokenizer, seq_len, dataset="fineweb", total_tokens=200_000_000):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.total_tokens = total_tokens
        self.dataset = dataset

    def __iter__(self):
        if self.dataset == "fineweb":
            ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train",
                              streaming=True, trust_remote_code=True)
        elif self.dataset == "wikitext-103":
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        else:
            ds = load_dataset("wikitext", self.dataset, split="train")

        buffer = []
        n = 0
        for item in ds:
            text = item["text"].strip()
            if text:
                buffer.extend(self.tokenizer.encode(text))
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len:]
                yield {"input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                       "labels": torch.tensor(chunk[1:], dtype=torch.long)}
                n += self.seq_len
                if n >= self.total_tokens:
                    return


def collate(batch):
    mx = max(it["input_ids"].size(0) for it in batch)
    ins = [F.pad(it["input_ids"], (0, mx - it["input_ids"].size(0)), value=0) for it in batch]
    lbs = [F.pad(it["labels"], (0, mx - it["labels"].size(0)), value=0) for it in batch]
    return {"input_ids": torch.stack(ins), "labels": torch.stack(lbs)}


# ── Training ───────────────────────────────────────────────────────────────────

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
def validate_wt2(model, tokenizer, device, seq_len=256):
    """Quick WT-2 validation to check no catastrophic forgetting."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    tokens = []
    for item in ds:
        text = item["text"].strip()
        if text: tokens.extend(tokenizer.encode(text))

    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(tokens) - seq_len, seq_len):
        chunk = tokens[i:i + seq_len + 1]
        if len(chunk) < seq_len + 1: continue
        ids = torch.tensor([chunk[:seq_len]]).to(device)
        labels = torch.tensor(chunk[1:seq_len + 1]).to(device)
        out = model(ids)
        sl = out["logits"][0, :-1, :].float()
        ll = labels[:-1]
        total_loss += F.cross_entropy(sl, ll, reduction="sum").item()
        total_tokens += ll.numel()
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", type=str, required=True)
    p.add_argument("--dataset", type=str, default="fineweb")
    p.add_argument("--total_tokens", type=int, default=200_000_000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=500)
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

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load pretrained model
    state_dict = torch.load(args.pretrained, map_location="cpu")
    dim, heads, n_layers, n_loops, delta_bn = auto_detect(state_dict, args.pretrained)
    print(f"Loading: d={dim}, heads={heads}, layers={n_layers}, loops={n_loops}")

    cfg = DeltaConfig(embed_dim=dim, num_heads=heads, num_layers=n_layers,
                      num_loops=n_loops, delta_bottleneck=delta_bn, max_seq_len=256)
    model = DeltaLoopedTransformer(cfg).to(device)
    model.load_state_dict(state_dict, strict=True)

    # WT-2 baseline before training
    print("WT-2 PPL before:", end=" ", flush=True)
    pp_before = validate_wt2(model, tokenizer, device)
    print(f"{pp_before:.2f}")

    # Data
    print("Loading streaming data...")
    train_ds = StreamingDataset(tokenizer, 256, dataset=args.dataset, total_tokens=args.total_tokens)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collate)

    # Optimizer
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

    steps_per_epoch = args.total_tokens // (256 * args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=args.lr, total_steps=total_steps,
        pct_start=args.warmup_steps / total_steps,
    )

    run_name = f"delta_ffn_loop{n_loops}_d{dim}_{args.dataset}"
    metrics, best_val = [], float("inf")
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        tl = train_epoch(model, train_loader, optim, sched, device)
        val_ppl = validate_wt2(model, tokenizer, device)
        train_ppl = math.exp(tl)
        metrics.append({"epoch": ep, "train_ppl": train_ppl, "val_wt2_ppl": val_ppl})
        print(f"Epoch {ep}: train={train_ppl:.2f}, WT-2={val_ppl:.2f}  [{time.time()-t0:.0f}s]")
        if val_ppl < best_val:
            best_val = val_ppl
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"{run_name}_best.pt"))

    print(f"WT-2 PPL before: {pp_before:.2f} → after: {best_val:.2f}")

    out = {"config": vars(args), "wt2_before": pp_before, "best_val_wt2": best_val,
           "total_time": time.time()-t0, "metrics": metrics}
    with open(os.path.join(args.output_dir, f"{run_name}_results.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
