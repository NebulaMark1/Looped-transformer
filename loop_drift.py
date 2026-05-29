"""
Track noise amplification through loop iterations.

Answers: Does parameter noise grow linearly or exponentially across loops?
Which loop position is most sensitive? (early perturbation compound, or late?)

Usage:
    python loop_drift.py --checkpoint results/delta_ffn_loop8_wt103_best.pt
    python loop_drift.py --checkpoint results/baseline_loop8_best.pt
"""

import argparse, json, math, re
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast


def auto_detect(state_dict, ckpt_path):
    d = state_dict["token_embedding.weight"].shape[1]
    has_full = any("full_blocks." in k for k in state_dict)
    has_delta = any("delta_blocks." in k for k in state_dict)

    prefix = "full_blocks." if (has_full and has_delta) else "blocks."
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

    delta_bn = None
    if has_delta:
        fc1_keys = [k for k in state_dict if "delta_blocks.0.fc1" in k and "weight" in k]
        if fc1_keys:
            bn = state_dict[fc1_keys[0]].shape[0]
            if bn != d // 4: delta_bn = bn

    return d, heads, layers, loops, delta_bn, "delta" if has_delta else "baseline"


def add_noise_to_params(model, target_names, noise_scale=0.01):
    for name, param in model.named_parameters():
        for tn in target_names:
            if tn in name:
                noise = torch.randn_like(param) * param.norm() * noise_scale
                param.data.add_(noise)
                break


@torch.no_grad()
def kl_div(logits_a, logits_b):
    log_p = F.log_softmax(logits_a.float(), dim=-1)
    log_q = F.log_softmax(logits_b.float(), dim=-1)
    p = F.softmax(logits_a.float(), dim=-1)
    return F.kl_div(log_q, p, reduction="batchmean", log_target=False).item()


# ── Per-loop tracking for delta model ──

class DeltaTracker(torch.nn.Module):
    """Delta model that intercepts outputs after each block's full+delta loop."""
    def __init__(self, orig_model):
        super().__init__()
        self.token_embedding = orig_model.token_embedding
        self.position_embedding = orig_model.position_embedding
        self.full_blocks = orig_model.full_blocks
        self.delta_blocks = orig_model.delta_blocks
        self.ln_final = orig_model.ln_final
        self.lm_head = orig_model.lm_head
        self.dropout = orig_model.dropout
        self.config = orig_model.config
        self.intermediate = {}

    def forward(self, input_ids):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)
        x = self.dropout(x)

        self.intermediate = {}
        for layer_idx in range(self.config.num_layers):
            # Full pass (loop 0)
            x = self.full_blocks[layer_idx](x)
            self.intermediate[f"L{layer_idx}_loop0"] = x.clone()

            # Delta passes (loop 1..n_loops-1)
            for d in range(self.config.num_loops - 1):
                x = x + self.delta_blocks[layer_idx](x, d)
                self.intermediate[f"L{layer_idx}_loop{d+1}"] = x.clone()

        x = self.ln_final(x)
        logits = F.linear(x, self.token_embedding.weight) if self.config.tie_embedding else self.lm_head(x)
        return {"logits": logits}


# ── Per-loop tracking for baseline model ──

class BaselineTracker(torch.nn.Module):
    """Baseline model that intercepts outputs after each loop iteration."""
    def __init__(self, orig_model):
        super().__init__()
        self.token_embedding = orig_model.token_embedding
        self.position_embedding = orig_model.position_embedding
        self.blocks = orig_model.blocks
        self.ln_final = orig_model.ln_final
        self.lm_head = orig_model.lm_head
        self.dropout = orig_model.dropout
        self.config = orig_model.config
        self.intermediate = {}

    def forward(self, input_ids):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)
        x = self.dropout(x)

        self.intermediate = {}
        for blk_idx, blk in enumerate(self.blocks):
            for loop_idx in range(self.config.num_loops):
                x = blk(x, loop_idx)
                self.intermediate[f"B{blk_idx}_loop{loop_idx}"] = x.clone()

        x = self.ln_final(x)
        logits = F.linear(x, self.token_embedding.weight) if self.config.tie_embedding else self.lm_head(x)
        return {"logits": logits}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--noise_scale", type=float, default=0.01)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    state_dict = torch.load(args.checkpoint, map_location="cpu")
    dim, heads, n_layers, n_loops, delta_bn, model_type = auto_detect(state_dict, args.checkpoint)
    print(f"Model: {model_type}, d={dim}, layers={n_layers}, loops={n_loops}")

    prompts = ["The history of", "According to the study"]
    input_ids_list = [tokenizer.encode(p, return_tensors="pt").to(device) for p in prompts]

    # ── Create tracker models ──
    if model_type == "delta":
        from delta_model import DeltaConfig, DeltaLoopedTransformer
        cfg = DeltaConfig(max_seq_len=256, embed_dim=dim, num_heads=heads,
                          num_layers=n_layers, num_loops=n_loops, delta_bottleneck=delta_bn)
        orig = DeltaLoopedTransformer(cfg).to(device)
        noisy = DeltaLoopedTransformer(cfg).to(device)
    else:
        from model import LoopedTransformerConfig, LoopedTransformer
        cfg = LoopedTransformerConfig(mode="baseline", max_seq_len=256,
                                      embed_dim=dim, num_heads=heads,
                                      num_layers=n_layers, num_loops=n_loops)
        orig = LoopedTransformer(cfg).to(device)
        noisy = LoopedTransformer(cfg).to(device)

    orig.load_state_dict(state_dict, strict=True)
    noisy.load_state_dict(state_dict, strict=True)

    tracker_orig = DeltaTracker(orig) if model_type == "delta" else BaselineTracker(orig)
    tracker_noisy = DeltaTracker(noisy) if model_type == "delta" else BaselineTracker(noisy)

    # ── Perturb all parameters of the block ──
    prefix = "full_blocks." if model_type == "delta" else "blocks."
    all_block_params = [f"{prefix}0."]
    add_noise_to_params(noisy, all_block_params, args.noise_scale)

    # ── Track KL divergence at each checkpoint ──
    all_kl = {}
    for ids in input_ids_list:
        out_orig = tracker_orig(ids)
        out_noisy = tracker_noisy(ids)

        for key in tracker_orig.intermediate:
            if key not in all_kl:
                all_kl[key] = []

            orig_h = tracker_orig.intermediate[key]
            noisy_h = tracker_noisy.intermediate[key]

            # Cosine similarity of hidden states
            cos = F.cosine_similarity(orig_h.view(-1), noisy_h.view(-1), dim=0).item()
            l2 = (orig_h - noisy_h).norm().item() / orig_h.norm().item()

            all_kl[key].append((cos, l2))

    # ── Print growth curve ──
    print(f"\nNoise scale = {args.noise_scale*100:.1f}% of param norm")
    print(f"\n{'Checkpoint':<18} {'Cos Sim':>10} {'L2 Norm':>10} {'L2/Base':>10}")
    print("-" * 52)

    for key in sorted(all_kl.keys(), key=lambda k: (int(k[1]) if k[1].isdigit() else 0, k)):
        cos_list = [v[0] for v in all_kl[key]]
        l2_list = [v[1] for v in all_kl[key]]
        avg_cos = sum(cos_list) / len(cos_list)
        avg_l2 = sum(l2_list) / len(l2_list)
        # Show L2 deviation
        print(f"{key:<18} {avg_cos:>10.6f} {avg_l2:>10.4f}")

    # ── Show growth ratios ──
    keys_sorted = sorted(all_kl.keys(),
                         key=lambda k: (int(k.split('_')[0][1]) if '_' in k else 0,
                                        int(k.split('loop')[-1]) if 'loop' in k.split('_')[-1] else 0))
    print(f"\n── Growth analysis ──")
    for blk in range(n_layers):
        l2_vals = []
        for loop in range(n_loops):
            key = f"B{blk}_loop{loop}" if model_type == "baseline" else f"L{blk}_loop{loop}"
            if key in all_kl:
                vals = all_kl[key]
                avg_l2 = sum(v[1] for v in vals) / len(vals)
                l2_vals.append(avg_l2)
        if l2_vals:
            # Check linear vs exponential
            diffs = [l2_vals[i+1] - l2_vals[i] for i in range(len(l2_vals)-1)]
            ratios = [l2_vals[i+1] / max(l2_vals[i], 1e-8) for i in range(len(l2_vals)-1)]
            print(f"  Block {blk}: L2 trajectory = {l2_vals}")
            if diffs:
                print(f"    Δ per loop   = {diffs}")
                print(f"    Ratio per loop = {ratios}")
                print(f"    Growth type: {'~exponential' if max(ratios) > 1.3 else '~linear'} "
                      f"(max_ratio={max(ratios):.2f})")


if __name__ == "__main__":
    main()
