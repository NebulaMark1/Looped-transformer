"""
Compute FLOPs for each model variant and print a comparison table.

Usage:
    python compute_analysis.py
"""

import math
import torch


def count_linear_flops(module, input_shape, output_shape):
    """Estimate FLOPs for a linear layer: 2 * in * out per token."""
    in_features = input_shape[-1]
    out_features = output_shape[-1]
    return 2 * in_features * out_features


def count_attn_flops(seq_len, embed_dim):
    """Estimate FLOPs for scaled_dot_product_attention.
       QK^T: seq_len * head_dim * seq_len  → T * d * T = T^2 * d
       softmax * V: same → T^2 * d
       Total: 2 * T^2 * d (per head, but d already accounts for all heads)
    """
    return 2 * seq_len * seq_len * embed_dim


def model_flops(model, seq_len=256):
    """Count FLOPs for one forward pass."""
    d = model.config.embed_dim
    flops = 0

    if hasattr(model, 'delta_blocks'):
        # DeltaLoopedTransformer
        L = model.config.num_layers
        n_loops = model.config.num_loops
        b = model.config.delta_bottleneck or (d // 4)

        for _ in range(L):
            # First loop: full block
            # Q, K, V, O: 4 * (d * d)
            flops += 4 * 2 * d * d * seq_len
            # Attention: 2 * T^2 * d
            flops += 2 * seq_len * seq_len * d
            # FFN up: d * 4d, down: 4d * d
            flops += 2 * 2 * d * 4 * d * seq_len

            # Subsequent loops: delta only
            for _ in range(n_loops - 1):
                flops += 2 * d * b * seq_len * 2  # fc1 + fc2
                if model.config.attn_inject_every > 0:
                    # injected attention
                    flops += 4 * 2 * d * d * seq_len / 2  # half heads
                    flops += 2 * seq_len * seq_len * d / 2

    else:
        # LoopedTransformer
        L = model.config.num_layers
        n_loops = model.config.num_loops

        for _ in range(L):
            for _ in range(n_loops):
                flops += 4 * 2 * d * d * seq_len  # QKVO
                flops += 2 * seq_len * seq_len * d  # Attention
                flops += 2 * 2 * d * 4 * d * seq_len  # FFN

    return flops / 1e9  # billions


def main():
    from model import create_model
    from delta_model import DeltaConfig, DeltaLoopedTransformer

    d = 384
    seq_len = 256

    configs = []

    # Baseline and variants
    for L in [4, 8]:
        for mode in ["baseline"]:
            configs.append((f"{mode} L={L}", mode, L, d))
    for mode in ["lora"]:
        configs.append((f"{mode}_r8 L=4", mode, 4, d))
        configs.append((f"{mode}_r16 L=4", mode, 4, d))

    # Full
    configs.append(("full L=4", "full", 4, d))

    # Delta variants
    for L in [4, 8]:
        for dt, b, inj in [("ffn", d//4, 0), ("ffn_b48", d//8, 0), ("ffn_inj2", d//4, 2)]:
            if L == 4 and ("b48" not in dt):
                continue
            configs.append((f"delta_{dt} L={L}", "delta", L, d, dt, b, inj))

    # Matched-compute baselines: wider models
    for wide_d, heads in [(256, 4), (300, 5), (500, 10), (600, 10)]:
        for L in [8]:
            configs.append((f"baseline d={wide_d} L={L}", "baseline", L, wide_d))

    print(f"\n{'Config':<28} {'FLOPs (G)':>10} {'Params (M)':>12} {'FLOPs/Param':>14}")
    print("-" * 68)

    results = []
    for name, mode, L, d_val, *extra in configs:
        try:
            if mode == "delta":
                dt, b, inj = extra
                cfg = DeltaConfig(delta_type=dt, delta_bottleneck=b, attn_inject_every=inj,
                                  num_loops=L, embed_dim=d_val, num_heads=6 if d_val==384 else (d_val//64),
                                  max_seq_len=seq_len, num_layers=3)
                model = DeltaLoopedTransformer(cfg)
            else:
                if mode == "lora":
                    rank = extra[0] if extra else 8
                    rank = 8 if "r8" in name else 16
                    model = create_model(mode, embed_dim=d_val, num_loops=L,
                                         num_heads=d_val//64, lora_rank=rank,
                                         max_seq_len=seq_len, num_layers=3)
                else:
                    nh = d_val // 64
                    if d_val == 300: nh = 5
                    elif d_val == 500: nh = 10
                    elif d_val == 600: nh = 10
                    model = create_model(mode, embed_dim=d_val, num_loops=L,
                                         num_heads=nh, max_seq_len=seq_len, num_layers=3)

            flops = model_flops(model, seq_len)
            _, _, trans = model.count_params()
            params_m = trans / 1e6
            results.append((name, flops, params_m))
            print(f"{name:<28} {flops:>10.2f} {params_m:>12.1f} {flops/params_m:>14.2f}")
        except Exception as e:
            print(f"{name:<28} {'ERROR':>10} {'—':>12} {'—':>14}  ({e})")

    print()

    if results:
        print("Best FLOPs-to-PPL bar (lower is better PPL at same compute):")
        print(f"{'Config':<28} {'FLOPs':>10} {'Val PPL':>12} {'PPL×FLOPs':>14}")
        print("-" * 68)
        # Known PPL values (hardcoded from our experiments)
        ppl_map = {
            "baseline L=4": 1041, "baseline L=8": 706,
            "delta_ffn L=4": 848, "delta_ffn L=8": 552,
            "delta_ffn_b48 L=8": 577, "delta_ffn_inj2 L=8": 555,
            "full L=4": 940, "lora_r16 L=4": 966, "lora_r8 L=4": 1042,
            "delta_ffn_frombl_frz L=8": 628,
        }
        for name, flops, params_m in results:
            ppl = ppl_map.get(name)
            if ppl:
                score = ppl * flops
                print(f"{name:<28} {flops:>10.2f} {ppl:>12} {score:>14.1f}")


if __name__ == "__main__":
    main()
