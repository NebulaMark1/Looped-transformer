"""
Evaluate trained model on multiple validation sets.

Usage:
    python eval_ppl.py --checkpoint results/delta_ffn_loop8_wt103_best.pt

Tests: WT-2, WT-103, Penn Treebank
"""

import argparse
import json
import math
import re
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast


def auto_detect(state_dict, ckpt_path):
    embed_dim = state_dict["token_embedding.weight"].shape[1]
    has_full = any("full_blocks." in k for k in state_dict)
    has_delta = any("delta_blocks." in k for k in state_dict)
    has_asym = any("ffn_blocks." in k for k in state_dict)
    has_lora = any(".lora_A" in k for k in state_dict)

    if has_asym and has_full: model_type = "asym"
    elif has_full and has_delta: model_type = "delta"
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

    num_full, num_ffn, delta_bn = None, None, None
    if model_type == "asym":
        fi = set()
        for k in state_dict:
            if k.startswith("full_blocks."):
                m = re.match(r"full_blocks\.(\d+)\.", k)
                if m: fi.add(int(m.group(1)))
        num_full = max(fi) + 1 if fi else 0
        ffi = set()
        for k in state_dict:
            if k.startswith("ffn_blocks."):
                m = re.match(r"ffn_blocks\.(\d+)\.", k)
                if m: ffi.add(int(m.group(1)))
        num_ffn = max(ffi) + 1 if ffi else 0
    if model_type == "delta":
        fc1_keys = [k for k in state_dict if "delta_blocks.0.fc1" in k and "weight" in k]
        if fc1_keys:
            bn = state_dict[fc1_keys[0]].shape[0]
            if bn != embed_dim // 4: delta_bn = bn

    return {"model_type": model_type, "embed_dim": embed_dim, "num_heads": num_heads,
            "num_layers": num_layers, "num_loops": num_loops,
            "num_full": num_full, "num_ffn": num_ffn, "delta_bottleneck": delta_bn}


def load_model(ckpt_path, device):
    state_dict = torch.load(ckpt_path, map_location="cpu")
    cfg = auto_detect(state_dict, ckpt_path)
    print(f"Model: {cfg['model_type']}, d={cfg['embed_dim']}, layers={cfg['num_layers']}, loops={cfg['num_loops']}")

    if cfg["model_type"] == "delta":
        from delta_model import DeltaConfig, DeltaLoopedTransformer
        dc = DeltaConfig(max_seq_len=256, embed_dim=cfg["embed_dim"],
                         num_heads=cfg["num_heads"], num_layers=cfg["num_layers"],
                         num_loops=cfg["num_loops"], delta_bottleneck=cfg["delta_bottleneck"])
        model = DeltaLoopedTransformer(dc)
    elif cfg["model_type"] == "asym":
        from asym_model import AsymConfig, AsymTransformer
        ac = AsymConfig(max_seq_len=256, embed_dim=cfg["embed_dim"],
                        num_heads=cfg["num_heads"], num_full=cfg["num_full"], num_ffn=cfg["num_ffn"])
        model = AsymTransformer(ac)
    else:
        from model import LoopedTransformerConfig, LoopedTransformer
        lc = LoopedTransformerConfig(mode=cfg["model_type"], max_seq_len=256,
                                     embed_dim=cfg["embed_dim"], num_heads=cfg["num_heads"],
                                     num_layers=cfg["num_layers"], num_loops=cfg["num_loops"])
        model = LoopedTransformer(lc)

    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval(), cfg


@torch.no_grad()
def eval_ppl(model, tokenizer, dataset_config, device, seq_len=256, max_samples=None):
    """Compute perplexity on a HuggingFace dataset split."""
    from datasets import load_dataset

    tokens = []
    name, subset, split = dataset_config
    ds = load_dataset(name, subset, split=split)
    for item in ds:
        text = item["text"].strip()
        if text: tokens.extend(tokenizer.encode(text))

    total_loss, total_tokens = 0.0, 0
    batch_size = 4
    batch_loss, batch_tokens = 0.0, 0
    n_seen = 0

    for i in range(0, len(tokens) - seq_len, seq_len):
        chunk = tokens[i:i + seq_len + 1]
        if len(chunk) < seq_len + 1: continue
        input_ids = torch.tensor([chunk[:seq_len]], device=device)
        labels = torch.tensor(chunk[1:seq_len + 1], device=device)

        out = model(input_ids)
        shift_logits = out["logits"][0, :-1, :].float()
        shift_labels = labels[:-1]
        loss = F.cross_entropy(shift_logits, shift_labels, reduction="sum")
        n_tokens = shift_labels.numel()

        batch_loss += loss.item()
        batch_tokens += n_tokens
        n_seen += n_tokens

        if batch_tokens >= seq_len * batch_size:
            total_loss += batch_loss
            total_tokens += batch_tokens
            batch_loss, batch_tokens = 0.0, 0

        if max_samples and n_seen >= max_samples:
            break

    if batch_tokens > 0:
        total_loss += batch_loss
        total_tokens += batch_tokens

    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model, cfg = load_model(args.checkpoint, device)
    total, embed, trans = model.count_params()
    print(f"Params: {total:,} total, {trans:,} transformer\n")

    benchmarks = [
        ("WikiText-2",          "wikitext", "wikitext-2-raw-v1", "test"),
        ("WikiText-103",        "wikitext", "wikitext-103-raw-v1", "test"),
        ("TinyStories",         "roneneldan/TinyStories", None, "validation"),
    ]

    results = {}
    for name, dname, dsubset, split in benchmarks:
        try:
            ppl = eval_ppl(model, tokenizer, (dname, dsubset, split), device)
            print(f"  {name:<18} PPL = {ppl:.2f}")
            results[name] = ppl
        except Exception as e:
            print(f"  {name:<18} ERROR: {e}")
            results[name] = None

    print(f"\n{'='*45}")
    print(f"{'Benchmark':<18} {'PPL':>10}")
    print(f"{'-'*30}")
    for name, ppl in results.items():
        if ppl: print(f"{name:<18} {ppl:>10.2f}")

    # Save
    out = {"checkpoint": args.checkpoint, "config": cfg, "results": results}
    with open(args.checkpoint.replace(".pt", "_eval.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
