"""
Loop-wise sensitivity analysis: which parts of a looped model amplify drift?

Adds controlled noise to different components and measures output divergence
to find the most "catastrophic" parameter positions.

Usage:
    python sensitivity.py --checkpoint results/delta_ffn_loop8_wt103_best.pt
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


@torch.no_grad()
def measure_divergence(model_orig, model_perturbed, input_ids):
    """KL divergence of original vs perturbed model outputs."""
    out_orig = model_orig(input_ids)["logits"][0].float()
    out_pert = model_perturbed(input_ids)["logits"][0].float()

    log_p = F.log_softmax(out_orig, dim=-1)
    log_q = F.log_softmax(out_pert, dim=-1)
    p = F.softmax(out_orig, dim=-1)

    kl = F.kl_div(log_q, p, reduction="batchmean", log_target=False)
    return kl.item()


def add_noise_to_params(model, target_names, noise_scale=0.01):
    """Add Gaussian noise (relative to param norm) to target parameter groups."""
    for name, param in model.named_parameters():
        for tn in target_names:
            if tn in name:
                noise = torch.randn_like(param) * param.norm() * noise_scale
                param.data.add_(noise)
                break


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

    # Load original model
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    dim, heads, n_layers, n_loops, delta_bn, model_type = auto_detect(state_dict, args.checkpoint)
    print(f"Model: {model_type}, d={dim}, layers={n_layers}, loops={n_loops}")

    if model_type == "delta":
        from delta_model import DeltaConfig, DeltaLoopedTransformer
        cfg = DeltaConfig(max_seq_len=256, embed_dim=dim, num_heads=heads,
                          num_layers=n_layers, num_loops=n_loops,
                          delta_bottleneck=delta_bn)
    else:
        from model import LoopedTransformerConfig, LoopedTransformer as LoopedTM
        cfg = LoopedTransformerConfig(mode="baseline", max_seq_len=256,
                                      embed_dim=dim, num_heads=heads,
                                      num_layers=n_layers, num_loops=n_loops)

    # Test inputs: diverse prompts
    prompts = [
        "The history of",
        "According to the study",
        "In the early years",
        "The theory of",
    ]
    input_ids_list = [tokenizer.encode(p, return_tensors="pt").to(device) for p in prompts]

    # ── Define perturbation targets ──
    prefix = "full_blocks." if model_type == "delta" else "blocks."

    targets = {}
    targets["attn"] = [f"{prefix}0.q_proj", f"{prefix}0.k_proj",
                       f"{prefix}0.v_proj", f"{prefix}0.o_proj"]
    targets["ffn"]  = [f"{prefix}0.ff_up", f"{prefix}0.ff_down"]
    targets["ln"]   = [f"{prefix}0.ln1", f"{prefix}0.ln2"]
    targets["embedding"] = ["token_embedding", "position_embedding"]
    targets["ln_final"]  = ["ln_final"]

    if model_type == "delta":
        targets["delta_ffn"] = ["delta_blocks.0.fc1", "delta_blocks.0.fc2"]
        targets["delta_ln"]  = ["delta_blocks.0.ln_ffn"]

    # Per-layer breakdown
    for l in range(n_layers):
        targets[f"layer{l}"] = [f"{prefix}{l}."]

    # Per-component sensitivity for all layers
    for lay in range(n_layers):
        targets[f"attn_L{lay}"]    = [f"{prefix}{lay}.q_proj", f"{prefix}{lay}.k_proj",
                                       f"{prefix}{lay}.v_proj", f"{prefix}{lay}.o_proj"]
        targets[f"ffn_L{lay}"]     = [f"{prefix}{lay}.ff_up", f"{prefix}{lay}.ff_down"]
        targets[f"ln_L{lay}"]      = [f"{prefix}{lay}.ln1", f"{prefix}{lay}.ln2"]

    print(f"\n{'Target':<25} {'KL Div':>10} {'Rel%':>8}")
    print(f"{'-'*45}")

    results = {}
    for name, target_patterns in targets.items():
        # Check if these params exist
        if model_type == "delta":
            model = DeltaLoopedTransformer(cfg).to(device)
            model_orig = DeltaLoopedTransformer(cfg).to(device)
        else:
            model = LoopedTM(cfg).to(device)
            model_orig = LoopedTM(cfg).to(device)

        model.load_state_dict(state_dict, strict=True)
        model_orig.load_state_dict(state_dict, strict=True)
        model.eval()
        model_orig.eval()

        # Inject noise into target params
        add_noise_to_params(model, target_patterns, args.noise_scale)

        total_kl = 0.0
        for ids in input_ids_list:
            total_kl += measure_divergence(model_orig, model, ids)

        avg_kl = total_kl / len(input_ids_list)
        print(f"{name:<25} {avg_kl:>10.4f}")
        results[name] = avg_kl

    # ── Loop sensitivity: perturb full_blocks at different loop positions ──
    # (This is conceptual for delta — the full block only runs once.
    #  For baseline, the same block runs n_loops times.)
    if model_type == "baseline":
        print(f"\n── Loop sensitivity (baseline: same block runs {n_loops}×) ──")
        # Perturb all loop-invariant things and compare to delta's per-loop structure
        model = DeltaLoopedTransformer(cfg).to(device)
        model_orig = DeltaLoopedTransformer(cfg).to(device)
        model_orig.load_state_dict(state_dict, strict=True)
        model_orig.eval()

    # Save
    out = {"noise_scale": args.noise_scale, "results": results}
    with open(args.checkpoint.replace(".pt", "_sensitivity.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.checkpoint.replace('.pt', '_sensitivity.json')}")


if __name__ == "__main__":
    main()
