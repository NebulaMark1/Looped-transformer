"""
Probe PPL at each loop depth for different datasets.
Tests the hypothesis: different loop depths naturally prefer different tasks.

Usage:
    python probe_loop_depth.py --checkpoint results/delta_ffn_loop8_wt103_final_best.pt
    python probe_loop_depth.py --checkpoint results/delta_ffn_fineweb_epoch4.pt \
        --datasets wikitext-2 fineweb tinystories
"""

import argparse, math, re
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


@torch.no_grad()
def eval_ppl_at_loop(model, tokens, device, early_exit_loop, seq_len=256):
    total_loss, total_tokens = 0.0, 0
    chunk_indices = list(range(0, len(tokens) - seq_len, seq_len))
    for i in tqdm(chunk_indices, desc=f"  exit={early_exit_loop}", leave=False):
        chunk = tokens[i:i + seq_len + 1].to(device)
        if len(chunk) < seq_len + 1:
            continue
        ids = chunk[:seq_len].unsqueeze(0)
        labels = chunk[1:seq_len + 1].unsqueeze(0)
        out = model(ids, early_exit_loop=early_exit_loop)
        logits = out["logits"][0, :-1, :].float()
        targets = labels[0, :-1]
        total_loss += F.cross_entropy(logits, targets, reduction="sum").item()
        total_tokens += targets.numel()
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


def load_dataset_tokens(dataset_name, tokenizer, max_tokens):
    if dataset_name == "tinystories":
        ds = load_dataset("roneneldan/TinyStories", split="train")
    elif dataset_name == "wikitext-2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    elif dataset_name == "fineweb":
        ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    else:
        raise ValueError(f"Unknown: {dataset_name}")

    tokens = []
    for item in ds:
        text = item["text"].strip()
        if text:
            tokens.extend(tokenizer.encode(text))
        if len(tokens) >= max_tokens:
            break
    return torch.tensor(tokens[:max_tokens], dtype=torch.long)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--max_tokens", type=int, default=2_000_000)
    p.add_argument("--datasets", type=str, nargs="+",
                   default=["wikitext-2", "tinystories"])
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    state_dict = torch.load(args.checkpoint, map_location="cpu")
    dim, heads, n_layers, n_loops = auto_detect(state_dict, args.checkpoint)

    print(f"Model: d={dim}, layers={n_layers}, loops={n_loops}")
    print(f"Checkpoint: {args.checkpoint}\n")

    datasets_tokens = {}
    for ds_name in args.datasets:
        print(f"Loading {ds_name}...")
        datasets_tokens[ds_name] = load_dataset_tokens(
            ds_name, tokenizer, args.max_tokens)

    model = DeltaLoopedTransformer(DeltaConfig(
        max_seq_len=256, embed_dim=dim, num_heads=heads,
        num_layers=n_layers, num_loops=n_loops)).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(f"\n{'Loop':>5}", end="")
    for ds_name in args.datasets:
        print(f"  {ds_name:>14}", end="")
    print()

    for loop_depth in range(n_loops):
        print(f"{loop_depth:>5}", end="", flush=True)
        for ds_name in args.datasets:
            ppl = eval_ppl_at_loop(model, datasets_tokens[ds_name], device, loop_depth)
            print(f"  {ppl:>14.2f}", end="", flush=True)
        print()


if __name__ == "__main__":
    main()
