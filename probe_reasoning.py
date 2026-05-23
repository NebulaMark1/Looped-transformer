"""
Probe reasoning capabilities of trained models on synthetic tasks.
Tests: copying, induction head, token position recall.

Usage:
    python probe_reasoning.py --checkpoint results/delta_ffn_loop8_best.pt
    python probe_reasoning.py --checkpoint results/baseline_loop8_best.pt
"""

import argparse
import re
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast


def auto_detect(state_dict, ckpt_path):
    embed_dim = state_dict["token_embedding.weight"].shape[1]
    has_full = any("full_blocks." in k for k in state_dict)
    has_delta = any("delta_blocks." in k for k in state_dict)
    has_lora = any(".lora_A" in k for k in state_dict)

    if has_full and has_delta: model_type = "delta"
    elif has_lora: model_type = "lora"
    else: model_type = "baseline"

    prefix = "full_blocks." if model_type == "delta" else "blocks."
    indices = set()
    for k in state_dict:
        if k.startswith(prefix):
            m = re.match(rf"{re.escape(prefix)}(\d+)\.", k)
            if m: indices.add(int(m.group(1)))
    num_layers = max(indices) + 1 if indices else 3

    possible = [h for h in [4, 5, 6, 7, 8, 9, 10, 12, 14, 16] if embed_dim % h == 0]
    num_heads = min(possible, key=lambda h: abs(embed_dim // h - 64)) if possible else 6

    num_loops = 4
    m = re.search(r"_loop(\d+)", ckpt_path)
    if m: num_loops = int(m.group(1))

    delta_bn = None
    if model_type == "delta":
        fc1_keys = [k for k in state_dict if "delta_blocks.0.fc1" in k and "weight" in k]
        if fc1_keys:
            bn = state_dict[fc1_keys[0]].shape[0]
            if bn != embed_dim // 4: delta_bn = bn

    return {"model_type": model_type, "embed_dim": embed_dim, "num_heads": num_heads,
            "num_layers": num_layers, "num_loops": num_loops, "delta_bottleneck": delta_bn}


def load_model(ckpt_path, device):
    state_dict = torch.load(ckpt_path, map_location="cpu")
    cfg = auto_detect(state_dict, ckpt_path)
    print(f"Model: {cfg['model_type']}, d={cfg['embed_dim']}, "
          f"L={cfg['num_layers']}, loops={cfg['num_loops']}")

    if cfg["model_type"] == "delta":
        from delta_model import DeltaConfig, DeltaLoopedTransformer
        dc = DeltaConfig(max_seq_len=256, embed_dim=cfg["embed_dim"],
                         num_heads=cfg["num_heads"], num_layers=cfg["num_layers"],
                         num_loops=cfg["num_loops"], delta_bottleneck=cfg["delta_bottleneck"])
        model = DeltaLoopedTransformer(dc)
    else:
        from model import LoopedTransformerConfig, LoopedTransformer
        lc = LoopedTransformerConfig(mode=cfg["model_type"], max_seq_len=256,
                                     embed_dim=cfg["embed_dim"], num_heads=cfg["num_heads"],
                                     num_layers=cfg["num_layers"], num_loops=cfg["num_loops"])
        model = LoopedTransformer(lc)

    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval(), cfg


# ── Tasks ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def test_copying(model, tokenizer, device, max_len=8):
    """Can model copy a prefix? Input: 'A B C → A B C' (should output A B C again)."""
    n_correct = 0
    n_total = 0
    test_tokens = list(range(100, 200))  # random-ish tokens unlikely to be special
    import random
    random.seed(42)

    for _ in range(50):
        length = random.randint(2, max_len)
        seq = random.sample(test_tokens, length)
        # Repeat: [A, B, C, A, B, C], model should predict [B, C, A, B, C, <end>]
        prompt = seq + seq[:-1]
        target = seq[1:] + [tokenizer.eos_token_id]

        input_ids = torch.tensor([prompt], device=device)
        out = model(input_ids)
        logits = out["logits"][0, -len(target):]
        preds = logits.argmax(dim=-1)

        for p, t in zip(preds.tolist(), target):
            if p == t:
                n_correct += 1
            n_total += 1

    return n_correct / n_total if n_total > 0 else 0


@torch.no_grad()
def test_induction_head(model, tokenizer, device):
    """
    Induction head: [A, B, ..., A] → model should predict B.
    Classic test of in-context pattern matching.
    """
    n_correct = 0
    n_total = 0
    import random
    random.seed(42)

    for _ in range(100):
        a = random.randint(100, 1000)
        b = random.randint(100, 1000)
        # Pattern: [a, b, ...distractors..., a] → expect b
        distractors = [random.randint(100, 1000) for _ in range(6)]
        prompt = [a, b] + distractors + [a]
        target = b

        input_ids = torch.tensor([prompt], device=device)
        out = model(input_ids)
        logits = out["logits"][0, -1]
        pred = logits.argmax().item()

        if pred == target:
            n_correct += 1
        n_total += 1

    return n_correct / n_total


@torch.no_grad()
def test_position_recall(model, tokenizer, device, max_dist=10):
    """Given 'token at position 0 is X' later ask 'what was at position 0?'"""
    n_correct = 0
    n_total = 0
    import random
    random.seed(42)

    for _ in range(100):
        dist = random.randint(1, max_dist)
        target_token = random.randint(100, 1000)
        distractors = [random.randint(100, 1000) for _ in range(dist - 1)]
        prompt = [target_token] + distractors + [target_token]
        target = target_token

        input_ids = torch.tensor([prompt], device=device)
        out = model(input_ids)
        logits = out["logits"][0, -1]
        pred = logits.argmax().item()

        if pred == target:
            n_correct += 1
        n_total += 1

    return n_correct / n_total


@torch.no_grad()
def test_token_accuracy(model, tokenizer, device):
    """Standard token-level accuracy on WikiText-2 validation (without padding)."""
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    tokens = []
    for item in dataset:
        text = item["text"].strip()
        if text:
            tokens.extend(tokenizer.encode(text))

    n_correct = 0
    n_total = 0
    seq_len = 128
    for i in range(0, len(tokens) - seq_len, seq_len):
        chunk = tokens[i:i + seq_len + 1]
        if len(chunk) < 2: continue
        input_ids = torch.tensor([chunk[:-1]], device=device)
        labels = torch.tensor([chunk[1:]], device=device)
        out = model(input_ids)
        logits = out["logits"][0]
        preds = logits.argmax(dim=-1)
        for p, l in zip(preds.tolist(), labels[0].tolist()):
            if p == l:
                n_correct += 1
            n_total += 1

    return n_correct / n_total if n_total > 0 else 0


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--compare", type=str, nargs="*", default=[],
                   help="Additional checkpoints to compare")
    return p.parse_args()


def main():
    args = parse_args()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoints = [args.checkpoint] + args.compare
    results = {}

    for ckpt in checkpoints:
        print(f"\n{'='*60}")
        print(f"Probing: {ckpt}")
        print(f"{'='*60}")

        model, cfg = load_model(ckpt, device)
        name = ckpt.replace("results/", "").replace("_best.pt", "")

        r = {}
        r["copying"] = test_copying(model, tokenizer, device)
        print(f"  Copying accuracy:     {r['copying']:.3f}")
        r["induction"] = test_induction_head(model, tokenizer, device)
        print(f"  Induction head acc:   {r['induction']:.3f}")
        r["position"] = test_position_recall(model, tokenizer, device)
        print(f"  Position recall acc:  {r['position']:.3f}")

        # Token accuracy on val set (sampled)
        r["token_acc"] = test_token_accuracy(model, tokenizer, device)
        print(f"  Token accuracy (val): {r['token_acc']:.3f}")

        results[name] = r

    # ── Summary table ──
    if len(results) > 1:
        print(f"\n{'='*80}")
        print(f"{'Model':<35} {'Copy':>8} {'Induction':>10} {'Position':>10} {'TokenAcc':>10}")
        print(f"{'-'*75}")
        for name, r in results.items():
            print(f"{name:<35} {r['copying']:>8.3f} {r['induction']:>10.3f} "
                  f"{r['position']:>10.3f} {r['token_acc']:>10.3f}")


if __name__ == "__main__":
    main()
