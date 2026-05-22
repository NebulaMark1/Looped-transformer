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

def build_oracle(baseline_ckpt: str, full_ckpt: str, rank: int, config_kwargs: dict,
                 shared_A: bool = False, verbose: bool = True):
    """
    Build oracle LoRA model.

    Independent-A (shared_A=False):  W_t = W_base + B_t @ A_t
        SVD each Δ_t = W_full[t] - W_base independently.
        Params: num_loops * rank * (in + out) per layer.

    Shared-A (shared_A=True):       W_t = W_base + B_t @ A_shared
        Stack all Δ_t vertically, SVD the stacked matrix → shared A.
        Per-loop B_t from the corresponding U-block.
        Params: rank * in + num_loops * rank * out per layer.
        Saves (num_loops - 1) * rank * in params vs independent-A.
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

    residual_ratios = []
    sv_decay = []

    for name, mod in oracle_modules.items():
        if mod.__class__.__name__ != "LoRALinear":
            continue

        base_mod = baseline_modules[name]
        full_mod = full_modules[name]

        W_base = base_mod.weight.data                    # (out, in)
        W_full = full_mod.weight.data                    # (num_loops, out, in)
        num_loops = W_full.shape[0]
        out_dim, in_dim = W_base.shape

        mod.weight.data.copy_(W_base)
        mod.bias.data.copy_(full_mod.bias.data)

        base_norm = W_base.norm().item()

        if shared_A:
            # ── Shared-A SVD ──
            # Stack all loop residuals: Δ_stacked (num_loops * out, in)
            deltas = (W_full - W_base.unsqueeze(0)).reshape(num_loops * out_dim, in_dim)
            ratio = deltas.norm().item() / (base_norm * (num_loops ** 0.5) + 1e-8)
            residual_ratios.append(ratio)

            U_stacked, S_stacked, Vh = torch.linalg.svd(deltas.float())
            if sv_decay == [] or name == list(oracle_modules.keys())[0]:
                sv_decay.append((S_stacked[0].item(),
                                 S_stacked[:rank].norm().item() / (S_stacked.norm().item() + 1e-8)))

            S_sqrt = torch.diag(torch.sqrt(S_stacked[:rank]))
            A_shared = (S_sqrt @ Vh[:rank, :]).to(deltas.dtype)   # (rank, in)

            # Set same A for all loops
            for t in range(num_loops):
                mod.lora_A.data[t].copy_(A_shared)

            # Extract per-loop B from U blocks
            for t in range(num_loops):
                U_t = U_stacked[t * out_dim:(t + 1) * out_dim, :rank]  # (out, rank)
                B_t = (U_t @ S_sqrt).to(deltas.dtype)                  # (out, rank)
                mod.lora_B.data[t].copy_(B_t)

        else:
            # ── Independent-A SVD ──
            for t in range(num_loops):
                delta = W_full[t] - W_base
                ratio = delta.norm().item() / (base_norm + 1e-8)
                residual_ratios.append(ratio)

                U, S, Vh = torch.linalg.svd(delta.float())
                if t == 0 and name == list(oracle_modules.keys())[0]:
                    sv_decay.append((S[0].item(),
                                     S[:rank].norm().item() / (S.norm().item() + 1e-8)))

                S_sqrt = torch.diag(torch.sqrt(S[:rank]))
                B = (U[:, :rank] @ S_sqrt).to(delta.dtype)
                A = (S_sqrt @ Vh[:rank, :]).to(delta.dtype)
                mod.lora_A.data[t].copy_(A)
                mod.lora_B.data[t].copy_(B)

    # ── Copy LN + Embedding from FULL model ──
    oracle_params = dict(oracle.named_parameters())
    full_params = dict(full.named_parameters())
    for name in oracle_params:
        if name in full_params and oracle_params[name].shape == full_params[name].shape:
            oracle_params[name].data.copy_(full_params[name].data)

    if verbose:
        mean_ratio = sum(residual_ratios) / len(residual_ratios) if residual_ratios else 0
        tag = "shared-A" if shared_A else "independent-A"
        print(f"  [{tag}] Mean ||Δ|| / ||W_base||: {mean_ratio:.6f}")
        if sv_decay:
            top_sv, energy = sv_decay[0]
            print(f"  [{tag}] First layer top SV: {top_sv:.6f}, top-{rank} energy: {energy:.4f}")

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

    # Try different ranks for both independent-A and shared-A oracle
    results_ind = {}
    results_shared = {}

    for rank in args.ranks:
        print(f"\nBuilding oracle rank={rank} (independent-A)...")
        oracle_ind = build_oracle(args.baseline, args.full, rank, config_kwargs, shared_A=False)
        oracle_ind = oracle_ind.to(device)
        ppl_ind = evaluate(oracle_ind, val_loader, device)
        results_ind[rank] = ppl_ind
        print(f"  Oracle r={rank} independent-A: PPL={ppl_ind:.2f}")

        print(f"Building oracle rank={rank} (shared-A)...")
        oracle_shared = build_oracle(args.baseline, args.full, rank, config_kwargs, shared_A=True)
        oracle_shared = oracle_shared.to(device)
        ppl_shared = evaluate(oracle_shared, val_loader, device)
        results_shared[rank] = ppl_shared
        print(f"  Oracle r={rank} shared-A: PPL={ppl_shared:.2f}")

        # Save best
        if rank == max(args.ranks):
            torch.save(oracle_ind.state_dict(), os.path.join(args.output_dir, f"oracle_r{rank}_best.pt"))
            torch.save(oracle_shared.state_dict(), os.path.join(args.output_dir, f"oracle_shared_r{rank}_best.pt"))

    # Load trained LoRA results
    trained_ppl = {}
    for fname in ["lora_r8_results.json", "lora_r16_frozen_results.json", "lora_r8_frozen_results.json"]:
        path = os.path.join(args.trained_results, fname)
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            key = fname.replace("_results.json", "")
            trained_ppl[key] = d["best_val_ppl"]

    # ── Print summary ──
    print("\n" + "=" * 80)
    print("  Oracle LoRA Analysis — Independent-A vs Shared-A")
    print("=" * 80)
    print(f"  Baseline PPL:              {baseline_ppl:.2f}")
    print(f"  Full (upper bound) PPL:    {full_ppl:.2f}")
    for key, ppl in sorted(trained_ppl.items()):
        print(f"  Trained {key:<30} {ppl:.2f}")
    print()

    full_gain = baseline_ppl - full_ppl

    # Independent-A table
    print(f"  ── Independent A (B_t @ A_t, {args.num_loops}*r*(in+out) params/layer) ──")
    print(f"  {'Rank':<8} {'Oracle PPL':<14} {'Δ vs Baseline':<18} {'Recovery %':<14}")
    print(f"  {'-'*56}")
    for rank in args.ranks:
        ppl = results_ind[rank]
        delta = baseline_ppl - ppl
        recovery = (delta / full_gain * 100) if full_gain > 0 else 0.0
        print(f"  {rank:<8} {ppl:<14.2f} {delta:+.2f} ({delta/baseline_ppl*100:+.2f}%)     {recovery:.1f}%")

    # Shared-A table
    print(f"\n  ── Shared A (B_t @ A_shared, r*in + {args.num_loops}*r*out params/layer) ──")
    print(f"  {'Rank':<8} {'Oracle PPL':<14} {'Δ vs Baseline':<18} {'Recovery %':<14}")
    print(f"  {'-'*56}")
    for rank in args.ranks:
        ppl = results_shared[rank]
        delta = baseline_ppl - ppl
        recovery = (delta / full_gain * 100) if full_gain > 0 else 0.0
        print(f"  {rank:<8} {ppl:<14.2f} {delta:+.2f} ({delta/baseline_ppl*100:+.2f}%)     {recovery:.1f}%")

    print()
    print(f"  Full gain over baseline: {full_gain:.2f} PPL points ({full_gain/baseline_ppl*100:.1f}%)")
    print()

    # Save results
    out = {
        "baseline_ppl": baseline_ppl,
        "full_ppl": full_ppl,
        "trained_ppl": trained_ppl,
        "oracle_independent": {str(r): ppl for r, ppl in results_ind.items()},
        "oracle_shared": {str(r): ppl for r, ppl in results_shared.items()},
        "full_gain": full_gain,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "oracle_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {args.output_dir}/oracle_results.json")


if __name__ == "__main__":
    main()
