"""
Train per-loop bias vectors on TinyStories while preserving WT-2.
Hypothesis: bias vectors with zero-sum constraint create a task-phase rotation
— intermediate loops help TS, final loop returns to WT-2.

Usage:
    python train_loop_bias.py --checkpoint results/delta_ffn_loop8_wt103_final_best.pt \
        --epochs 3 --cancellation_weight 1.0
    python train_loop_bias.py --checkpoint results/delta_ffn_loop8_wt103_final_best.pt \
        --epochs 3 --cancellation_weight 0.0  # no cancellation (ablative control)
"""

import argparse, json, math, os, re, sys, time
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from tqdm import tqdm

from delta_model import DeltaConfig, DeltaLoopedTransformer


def auto_detect(state_dict, ckpt_path):
    d = state_dict["token_embedding.weight"].shape[1]
    prefix = "full_blocks."
    indices = set()
    for k in state_dict:
        if k.startswith(prefix):
            m = re.match(rf"{re.escape(prefix)}(\d+)\.", k)
            if m: indices.add(int(m.group(1)))
    layers = max(indices) + 1 if indices else 3
    possible = [h for h in [4, 5, 6, 7, 8, 9, 10, 12] if d % h == 0]
    heads = min(possible, key=lambda h: abs(d // h - 64)) if possible else 6
    loops = 4
    m = re.search(r"_loop(\d+)", ckpt_path)
    if m: loops = int(m.group(1))
    return d, heads, layers, loops


def load_tinystories(tokenizer, max_tokens=10_000_000):
    ds = load_dataset("roneneldan/TinyStories", split="train")
    tokens = []
    for item in ds:
        text = item["text"].strip()
        if text: tokens.extend(tokenizer.encode(text))
        if len(tokens) >= max_tokens: break
    return torch.tensor(tokens[:max_tokens], dtype=torch.long)


@torch.no_grad()
def eval_ppl_at_loop(model, tokens, device, early_exit_loop, seq_len=256):
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(tokens) - seq_len, seq_len):
        chunk = tokens[i:i + seq_len + 1].to(device)
        if len(chunk) < seq_len + 1: continue
        ids = chunk[:seq_len].unsqueeze(0)
        labels = chunk[1:seq_len + 1].unsqueeze(0)
        out = model(ids, early_exit_loop=early_exit_loop)
        logits = out["logits"][0, :-1, :].float()
        targets = labels[0, :-1]
        total_loss += F.cross_entropy(logits, targets, reduction="sum").item()
        total_tokens += targets.numel()
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


def make_chunks(tokens, seq_len=256):
    chunks = []
    for i in range(0, len(tokens) - seq_len, seq_len):
        chunk = tokens[i:i + seq_len + 1]
        if len(chunk) == seq_len + 1:
            chunks.append((chunk[:seq_len], chunk[1:seq_len + 1]))
    return chunks


def compute_cancellation_loss(model):
    """Sum of squared L2 norm of the sum of bias vectors across all delta blocks."""
    total = 0.0
    for block in model.delta_blocks:
        if hasattr(block, 'bias'):
            total += block.bias.sum(dim=0).pow(2).sum()
    return total


def compute_cumulative_bias_norms(model):
    """Returns list of cumulative bias L2 norms at each loop depth."""
    n_delta = model.config.num_loops - 1
    cum_norms = []
    for k in range(n_delta):
        cum = 0.0
        for block in model.delta_blocks:
            if hasattr(block, 'bias'):
                cum += block.bias[:k+1].sum(dim=0).pow(2).sum().item()
        cum_norms.append(math.sqrt(cum))
    return cum_norms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--cancellation_weight", type=float, default=1.0,
                   help="Weight for zero-sum cancellation loss (0=off)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, default="./results")
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    state_dict = torch.load(args.checkpoint, map_location="cpu")
    dim, heads, n_layers, n_loops = auto_detect(state_dict, args.checkpoint)
    n_delta = n_loops - 1

    print(f"Model: d={dim}, layers={n_layers}, loops={n_loops}")
    print(f"Cancellation weight: {args.cancellation_weight}")

    # Build model with bias enabled
    cfg = DeltaConfig(max_seq_len=256, embed_dim=dim, num_heads=heads,
                      num_layers=n_layers, num_loops=n_loops,
                      delta_bias=True)
    model = DeltaLoopedTransformer(cfg).to(device)
    model.load_state_dict(state_dict, strict=False)  # bias keys are new

    # Freeze everything except bias
    for name, param in model.named_parameters():
        if "bias" in name and "delta_blocks" in name and param.dim() == 2:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable bias params: {trainable:,}")

    # Profile BEFORE training
    print("\n--- Before training ---")
    model.eval()
    wt2_tokens = []
    for item in load_dataset("wikitext", "wikitext-2-raw-v1", split="validation"):
        text = item["text"].strip()
        if text: wt2_tokens.extend(tokenizer.encode(text))
    wt2_tokens = torch.tensor(wt2_tokens, dtype=torch.long)

    ts_val_tokens = load_tinystories(tokenizer, max_tokens=500_000)
    ts_val_tokens = ts_val_tokens.to(device)

    print(f"{'Loop':>5}  {'WT-2':>10}  {'TS':>10}")
    for loop_depth in range(n_loops):
        wt2_ppl = eval_ppl_at_loop(model, wt2_tokens, device, loop_depth)
        ts_ppl = eval_ppl_at_loop(model, ts_val_tokens, device, loop_depth)
        print(f"{loop_depth:>5}  {wt2_ppl:>10.2f}  {ts_ppl:>10.2f}")
    print(f"Cumulative bias norms: {[f'{x:.4f}' for x in compute_cumulative_bias_norms(model)]}")

    # Train
    ts_train_tokens = load_tinystories(tokenizer, max_tokens=10_000_000)
    chunks = make_chunks(ts_train_tokens)
    print(f"\nTraining on {len(chunks):,} TS chunks...")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    import random

    for epoch in range(1, args.epochs + 1):
        random.shuffle(chunks)
        model.train()
        total_task_loss, total_cancel_loss = 0.0, 0.0
        pbar = tqdm(range(0, len(chunks) - args.batch_size, args.batch_size),
                     desc=f"Epoch {epoch}")

        for step in pbar:
            batch_chunks = chunks[step:step + args.batch_size]
            input_ids = torch.stack([c[0] for c in batch_chunks]).to(device)
            labels = torch.stack([c[1] for c in batch_chunks]).to(device)

            out = model(input_ids, labels=labels)
            task_loss = out["loss"]
            cancel_loss = compute_cancellation_loss(model)

            loss = task_loss + args.cancellation_weight * cancel_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_task_loss += task_loss.item()
            total_cancel_loss += cancel_loss.item()
            pbar.set_postfix({
                "task": f"{math.exp(task_loss.item()):.1f}",
                "cancel": f"{cancel_loss.item():.4f}",
            })

    # Profile AFTER training
    print("\n--- After training ---")
    model.eval()
    print(f"{'Loop':>5}  {'WT-2':>10}  {'TS':>10}")
    for loop_depth in range(n_loops):
        wt2_ppl = eval_ppl_at_loop(model, wt2_tokens, device, loop_depth)
        ts_ppl = eval_ppl_at_loop(model, ts_val_tokens, device, loop_depth)
        print(f"{loop_depth:>5}  {wt2_ppl:>10.2f}  {ts_ppl:>10.2f}")
    print(f"Cumulative bias norms: {[f'{x:.4f}' for x in compute_cumulative_bias_norms(model)]}")

    # Save
    tag = f"_cw{args.cancellation_weight}"
    if args.tag: tag += f"_{args.tag}"
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"loop_bias_L{n_loops}{tag}_model.pt")
    torch.save(model.state_dict(), out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
