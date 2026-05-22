"""
Oracle LoRA Analysis: decompose trained full-model residuals via SVD to find
the ideal per-loop LoRA adapters, then evaluate.

Answers two questions:
  1. How much of the full-model gain can LoRA capture at rank r? (expressiveness)
  2. How far is trained LoRA from the oracle? (training gap)

Usage:
    python oracle_lora.py --baseline results/baseline_best.pt \
                          --full results/full_best.pt \
                          --output_dir ./results
"""

import argparse
import json
import math
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm

from model import create_model, LoopedTransformerConfig


# ── Data ──────────────────────────────────────────────────────────────────────

def load_val_data(seq_len: int, batch_size: int):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    tokens = []
    for item in dataset:
        text = item["text"].strip()
        if text:
            tokens.extend(tokenizer.encode(text))
    tokens = torch.tensor(tokens, dtype=torch.long)

    samples = []
    for i in range(0, len(tokens) - seq_len, seq_len):
        chunk = tokens[i:i + seq_len + 1]
        if len(chunk) > 1:
            samples.append({"input_ids": chunk[:-1], "labels": chunk[1:]})

    # Pad to batch_size for clean evaluation
    return samples, tokenizer


def collate(batch):
    max_len = max(item["input_ids"].size(0) for item in batch)
    padded_inputs, padded_labels = [], []
    for item in batch:
        pad = max_len - item["input_ids"].size(0)
        padded_inputs.append(F.pad(item["input_ids"], (0, pad), value=0))
        padded_labels.append(F.pad(item["labels"], (0, pad), value=0))
    return {"input_ids": torch.stack(padded_inputs), "labels": torch.stack(padded_labels)}


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in tqdm(dataloader, desc="Eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        output = model(input_ids, labels=input_ids)
        total_loss += output["loss"].item() * input_ids.numel()
        total_tokens += input_ids.numel()
    return math.exp(total_loss / total_tokens)


# ── SVD Decomposition ─────────────────────────────────────────────────────────

def svd_residual(W_base: torch.Tensor, W_full_t: torch.Tensor, rank: int):
    """
    Δ = W_full_t - W_base
    Returns (B, A) such that B@A is the best rank-r approximation of Δ
    by the Eckart-Young-Mirsky theorem.

    B: (out_features, rank), A: (rank, in_features)
    """
    delta = W_full_t - W_base
    U, S, Vh = torch.linalg.svd(delta.float())
    S_sqrt = torch.diag(torch.sqrt(S[:rank]))
    B = (U[:, :rank] @ S_sqrt).to(delta.dtype)
    A = (S_sqrt @ Vh[:rank, :]).to(delta.dtype)
    return B, A


# ── Oracle Builder ────────────────────────────────────────────────────────────

def build_oracle(baseline_ckpt: str, full_ckpt: str, rank: int, config_kwargs: dict, verbose: bool = True):
    """
    Build oracle LoRA model:
      - Base weights = baseline (per-loop linear layers)
      - Per-loop LoRA = SVD of (full[t] - baseline) truncated to `rank`
      - LN + Embedding = from the fine-tuned full model (keeps compatibility with per-loop weights)
      - Bias = from the full model (same rationale)
    """
    from model import LoopedTransformer

    baseline_cfg = LoopedTransformerConfig(mode="baseline", **config_kwargs)
    full_cfg = LoopedTransformerConfig(mode="full", **config_kwargs)
    lora_cfg = LoopedTransformerConfig(mode="lora", lora_rank=rank, **config_kwargs)

    baseline = LoopedTransformer(baseline_cfg)
    baseline.load_state_dict(torch.load(baseline_ckpt, map_location="cpu"))

    full = LoopedTransformer(full_cfg)
    full.load_state_dict(torch.load(full_ckpt, map_location="cpu"))

    oracle = LoopedTransformer(lora_cfg)

    oracle_modules = dict(oracle.named_modules())
    baseline_modules = dict(baseline.named_modules())
    full_modules = dict(full.named_modules())

    # ── Per-loop LoRA from SVD residuals ──
    residual_ratios = []
    sv_decay = []

    for name, mod in oracle_modules.items():
        if mod.__class__.__name__ != "LoRALinear":
            continue

        base_mod = baseline_modules[name]
        full_mod = full_modules[name]

        W_base = base_mod.weight.data
        W_full = full_mod.weight.data  # (num_loops, out, in)

        # Use full model's bias (trained with these per-loop weights)
        mod.weight.data.copy_(W_base)
        mod.bias.data.copy_(full_mod.bias.data)

        base_norm = W_base.norm().item()
        for t in range(W_full.shape[0]):
            delta = W_full[t] - W_base
            ratio = delta.norm().item() / (base_norm + 1e-8)
            residual_ratios.append(ratio)

            # SVD and truncate
            U, S, Vh = torch.linalg.svd(delta.float())
            if t == 0 and S.numel() > 0:
                sv_decay.append((S[0].item(), S[:rank].norm().item() / S.norm().item()))

            S_sqrt = torch.diag(torch.sqrt(S[:rank]))
            B = (U[:, :rank] @ S_sqrt).to(delta.dtype)
            A = (S_sqrt @ Vh[:rank, :]).to(delta.dtype)
            mod.lora_A.data[t].copy_(A)
            mod.lora_B.data[t].copy_(B)

    # ── Copy LN + Embedding from FULL model (not baseline) ──
    # Per-loop weights evolved with full's LN/embedding; using baseline's creates mismatch.
    oracle_params = dict(oracle.named_parameters())
    full_params = dict(full.named_parameters())
    for name in oracle_params:
        if name in full_params and oracle_params[name].shape == full_params[name].shape:
            oracle_params[name].data.copy_(full_params[name].data)

    if verbose:
        mean_ratio = sum(residual_ratios) / len(residual_ratios) if residual_ratios else 0
        print(f"  Mean ||Δ|| / ||W_base||: {mean_ratio:.6f}")
        if sv_decay:
            top_sv, energy = sv_decay[0]
            print(f"  First layer top SV: {top_sv:.6f}, top-{rank} energy ratio: {energy:.4f}")

    return oracle


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Oracle LoRA analysis via SVD residuals")
    p.add_argument("--baseline", type=str, default="results/baseline_best.pt")
    p.add_argument("--full", type=str, default="results/full_best.pt")
    p.add_argument("--trained_results", type=str, default="results",
                   help="Directory with *_results.json for trained LoRA comparison")
    p.add_argument("--output_dir", type=str, default="./results")
    p.add_argument("--embed_dim", type=int, default=384)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--num_loops", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config_kwargs = dict(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_loops=args.num_loops,
        max_seq_len=args.seq_len,
    )

    # Load val data once
    samples, _ = load_val_data(args.seq_len, args.batch_size)
    val_loader = DataLoader(samples, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # Baseline evaluation (reference)
    from model import LoopedTransformer as LT
    baseline_cfg = LoopedTransformerConfig(mode="baseline", **config_kwargs)
    baseline_ref = LT(baseline_cfg).to(device)
    baseline_ref.load_state_dict(torch.load(args.baseline, map_location=device))
    baseline_ppl = evaluate(baseline_ref, val_loader, device)
    print(f"Baseline PPL: {baseline_ppl:.2f}")

    # Full model evaluation
    full_cfg = LoopedTransformerConfig(mode="full", **config_kwargs)
    full_ref = LT(full_cfg).to(device)
    full_ref.load_state_dict(torch.load(args.full, map_location=device))
    full_ppl = evaluate(full_ref, val_loader, device)
    print(f"Full PPL:    {full_ppl:.2f}")

    # Try different ranks for oracle
    results = {}
    for rank in args.ranks:
        print(f"\nBuilding oracle rank={rank}...")
        oracle = build_oracle(args.baseline, args.full, rank, config_kwargs)
        oracle = oracle.to(device)
        ppl = evaluate(oracle, val_loader, device)
        results[rank] = ppl
        print(f"  Oracle r={rank}: PPL={ppl:.2f}")

        # Save best oracle
        if rank == max(args.ranks):
            torch.save(oracle.state_dict(), os.path.join(args.output_dir, f"oracle_r{rank}_best.pt"))

    # Load trained LoRA result if available
    trained_ppl = None
    lora_results_path = os.path.join(args.trained_results, "lora_r8_results.json")
    if os.path.exists(lora_results_path):
        with open(lora_results_path) as f:
            data = json.load(f)
        trained_ppl = data["best_val_ppl"]

    # ── Print summary ──
    print("\n" + "=" * 72)
    print("  Oracle LoRA Analysis — SVD Residual Decomposition")
    print("=" * 72)
    print(f"  Baseline PPL:              {baseline_ppl:.2f}")
    print(f"  Full (upper bound) PPL:    {full_ppl:.2f}")
    if trained_ppl:
        print(f"  Trained LoRA r=8 PPL:      {trained_ppl:.2f}")
    print()

    print(f"  {'Rank':<8} {'Oracle PPL':<14} {'Δ vs Baseline':<16} {'Recovery %':<14}")
    print(f"  {'-'*52}")
    full_gain = baseline_ppl - full_ppl
    for rank in args.ranks:
        oracle_ppl = results[rank]
        delta = baseline_ppl - oracle_ppl
        recovery = (delta / full_gain * 100) if full_gain > 0 else 0.0
        print(f"  {rank:<8} {oracle_ppl:<14.2f} {delta:+.2f} ({delta/baseline_ppl*100:+.2f}%)   {recovery:.1f}%")

    print()
    print(f"  Full gain over baseline: {full_gain:.2f} PPL points ({full_gain/baseline_ppl*100:.1f}%)")
    print()

    # Save results
    out = {
        "baseline_ppl": baseline_ppl,
        "full_ppl": full_ppl,
        "trained_lora_ppl": trained_ppl,
        "oracle_results": {str(r): ppl for r, ppl in results.items()},
        "full_gain": full_gain,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "oracle_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {args.output_dir}/oracle_results.json")


if __name__ == "__main__":
    main()
