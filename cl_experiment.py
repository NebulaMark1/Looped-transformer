"""
Continual learning experiment: fine-tune baseline on TinyStories,
measure forgetting on WT-2.

Strategies:
  1. full:       all params trainable (standard CL baseline)
  2. protect:    freeze embedding + attention, only train FFN + LN

Usage:
    python cl_experiment.py --checkpoint results/baseline_loop8_best.pt \
        --strategy full --epochs 1 --output_dir ./results

    python cl_experiment.py --checkpoint results/baseline_loop8_best.pt \
        --strategy protect --epochs 1 --output_dir ./results
"""

import argparse, json, math, os, re, time
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
from tqdm import tqdm


def auto_detect(state_dict, ckpt_path):
    d = state_dict["token_embedding.weight"].shape[1]
    has_full = any("full_blocks." in k for k in state_dict)
    has_delta = any("delta_blocks." in k for k in state_dict)

    model_type = "delta" if (has_full and has_delta) else "baseline"
    prefix = "full_blocks." if model_type == "delta" else "blocks."
    indices = set()
    for k in state_dict:
        if k.startswith(prefix):
            m = re.match(rf"{re.escape(prefix)}(\d+)\.", k)
            if m: indices.add(int(m.group(1)))
    layers = max(indices) + 1 if indices else 3
    possible = [h for h in [4,5,6,7,8,9,10,12] if d % h == 0]
    heads = min(possible, key=lambda h: abs(d//h - 64)) if possible else 6
    loops = 4
    m = re.search(r"_loop(\d+)", ckpt_path)
    if m: loops = int(m.group(1))
    return d, heads, layers, loops, model_type


def make_model(dim, heads, n_layers, n_loops, device, model_type="baseline"):
    if model_type == "delta":
        from delta_model import DeltaConfig, DeltaLoopedTransformer
        cfg = DeltaConfig(max_seq_len=256, embed_dim=dim, num_heads=heads,
                          num_layers=n_layers, num_loops=n_loops)
        return DeltaLoopedTransformer(cfg).to(device)
    else:
        from model import LoopedTransformerConfig, LoopedTransformer
        cfg = LoopedTransformerConfig(mode="baseline", max_seq_len=256,
                                      embed_dim=dim, num_heads=heads,
                                      num_layers=n_layers, num_loops=n_loops)
        return LoopedTransformer(cfg).to(device)


def load_tinystories(tokenizer, seq_len=256, max_tokens=5_000_000):
    """Load TinyStories tokens (subset for quick experiment)."""
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train")
    tokens = []
    for item in ds:
        text = item["text"].strip()
        if text: tokens.extend(tokenizer.encode(text))
        if len(tokens) >= max_tokens: break
    return torch.tensor(tokens[:max_tokens], dtype=torch.long)


@torch.no_grad()
def eval_ppl(model, tokens, device, seq_len=256):
    """Compute PPL on tokenized data."""
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(tokens) - seq_len, seq_len):
        chunk = tokens[i:i + seq_len + 1].to(device)
        if len(chunk) < seq_len + 1: continue
        ids = chunk[:seq_len].unsqueeze(0)
        labels = chunk[1:seq_len + 1].unsqueeze(0)
        out = model(ids)
        sl = out["logits"][0, :-1, :].float()
        ll = labels[0, :-1]
        total_loss += F.cross_entropy(sl, ll, reduction="sum").item()
        total_tokens += ll.numel()
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


def apply_strategy(model, strategy):
    """Set which params are trainable based on strategy."""
    if strategy == "full":
        return  # all trainable by default

    if strategy == "protect":
        for name, param in model.named_parameters():
            is_attn = any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj"])
            is_emb = "embedding" in name
            is_delta = "delta_blocks." in name
            # FFN + LN + delta stay trainable; attn + emb frozen
            if is_attn or is_emb:
                param.requires_grad = False
            # delta blocks always trainable (our architecture's strength)
            if is_delta:
                param.requires_grad = True

    if strategy == "delta_only":
        for name, param in model.named_parameters():
            param.requires_grad = ("delta_blocks." in name)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--strategy", type=str, default="full", choices=["full", "protect", "delta_only"])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, default="./results")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    state_dict = torch.load(args.checkpoint, map_location="cpu")
    dim, heads, n_layers, n_loops, model_type = auto_detect(state_dict, args.checkpoint)
    print(f"Model: {model_type}, d={dim}, layers={n_layers}, loops={n_loops}")
    print(f"Strategy: {args.strategy}")

    model = make_model(dim, heads, n_layers, n_loops, device, model_type)
    model.load_state_dict(state_dict, strict=True)

    # ── Pre-training performance ──
    # WT-2 val
    from datasets import load_dataset
    wt2_tokens = []
    for item in load_dataset("wikitext", "wikitext-2-raw-v1", split="validation"):
        text = item["text"].strip()
        if text: wt2_tokens.extend(tokenizer.encode(text))
    wt2_tokens = torch.tensor(wt2_tokens, dtype=torch.long)

    ppl_wt2_before = eval_ppl(model, wt2_tokens, device)
    print(f"WT-2 PPL before: {ppl_wt2_before:.2f}")

    # TinyStories val (quick check)
    ts_val_tokens = load_tinystories(tokenizer, max_tokens=500_000)
    ts_val_tokens = ts_val_tokens.to(device)

    # ── Apply strategy ──
    apply_strategy(model, args.strategy)

    # ── Load TinyStories training data ──
    print("Loading TinyStories...")
    ts_train_tokens = load_tinystories(tokenizer, max_tokens=10_000_000)
    print(f"  {len(ts_train_tokens):,} tokens")

    # Make training chunks
    seq_len = 256
    chunks = []
    for i in range(0, len(ts_train_tokens) - seq_len, seq_len):
        chunk = ts_train_tokens[i:i + seq_len + 1]
        if len(chunk) == seq_len + 1:
            chunks.append((chunk[:seq_len], chunk[1:seq_len + 1]))

    print(f"  {len(chunks):,} training batches")

    # Optimizer
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Optimizing {sum(p.numel() for p in trainable):,} params")
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.1)

    # ── Train ──
    model.train()
    total_steps = len(chunks) // args.batch_size

    for epoch in range(1, args.epochs + 1):
        # Shuffle
        import random
        random.shuffle(chunks)

        pbar = tqdm(range(0, len(chunks) - args.batch_size, args.batch_size),
                     desc=f"Epoch {epoch}")
        total_loss = 0.0
        for step in pbar:
            batch_chunks = chunks[step:step + args.batch_size]
            # Pad to same length (should be 256 for all but we're safe)
            input_ids = torch.stack([c[0] for c in batch_chunks]).to(device)
            labels = torch.stack([c[1] for c in batch_chunks]).to(device)

            out = model(input_ids)
            sl = out["logits"][:, :-1, :].float()
            ll = labels[:, :-1]
            loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), ll.reshape(-1))

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            total_loss += loss.item()
            if step % 100 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        avg_loss = total_loss / (total_steps + 1)
        print(f"  Train loss: {avg_loss:.4f}, PPL: {math.exp(avg_loss):.1f}")

    # ── Post-training evaluation ──
    model.eval()
    ppl_wt2_after = eval_ppl(model, wt2_tokens, device)
    print(f"\nWT-2 PPL after: {ppl_wt2_after:.2f}")

    ppl_ts = eval_ppl(model, ts_val_tokens, device)
    print(f"TinyStories PPL: {ppl_ts:.2f}")

    forgetting = ppl_wt2_after - ppl_wt2_before
    print(f"\n{'='*50}")
    print(f"  WT-2 PPL: {ppl_wt2_before:.1f} → {ppl_wt2_after:.1f}  (Δ = +{forgetting:.1f})")
    print(f"  Forgetting: {forgetting/ppl_wt2_before*100:+.1f}%")
    print(f"  TinyStories PPL: {ppl_ts:.1f}")
    print(f"{'='*50}")

    run_name = f"cl_{args.strategy}_L{n_loops}"
    out = {"strategy": args.strategy, "ppl_wt2_before": ppl_wt2_before,
           "ppl_wt2_after": ppl_wt2_after, "ppl_tinystories": ppl_ts,
           "forgetting_pct": forgetting/ppl_wt2_before*100}
    with open(os.path.join(args.output_dir, f"{run_name}_results.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
