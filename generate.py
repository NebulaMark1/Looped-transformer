"""Generate text from a trained model checkpoint. Auto-detects config from weights."""

import argparse
import re
import torch
from transformers import GPT2TokenizerFast


def auto_detect(state_dict, ckpt_path):
    """Infer model config from checkpoint weights and filename."""
    # -- embed_dim from token embedding --
    embed_dim = state_dict["token_embedding.weight"].shape[1]

    # -- model type from key structure --
    has_full_blocks = any("full_blocks." in k for k in state_dict)
    has_delta_blocks = any("delta_blocks." in k for k in state_dict)
    has_lora = any(".lora_A" in k for k in state_dict)
    has_blocks = any(k.startswith("blocks.") for k in state_dict)

    if has_full_blocks and has_delta_blocks:
        model_type = "delta"
    elif has_lora:
        model_type = "lora"
    elif has_blocks:
        # Check if full mode: blocks.X.q_proj.weight is 3D
        first_block = [k for k in state_dict if k.startswith("blocks.0") and "weight" in k and "q_proj" in k]
        if first_block and state_dict[first_block[0]].ndim == 3:
            model_type = "full"
        else:
            model_type = "baseline"
    else:
        model_type = "baseline"

    # -- num_layers from block indices --
    prefix = "full_blocks." if model_type == "delta" else "blocks."
    layer_indices = set()
    for k in state_dict:
        if k.startswith(prefix):
            m = re.match(rf"{re.escape(prefix)}(\d+)\.", k)
            if m:
                layer_indices.add(int(m.group(1)))
    num_layers = max(layer_indices) + 1 if layer_indices else 3

    # -- num_heads: prefer head_dim ≈ 64 (standard practice) --
    possible_heads = [h for h in [4, 5, 6, 7, 8, 9, 10, 12, 14, 16]
                      if embed_dim % h == 0]
    if possible_heads:
        # Pick the head_dim closest to 64
        num_heads = min(possible_heads, key=lambda h: abs(embed_dim // h - 64))
    else:
        num_heads = 6

    # -- num_loops: from filename first, then infer --
    num_loops = 4
    m = re.search(r"_loop(\d+)", ckpt_path)
    if m:
        num_loops = int(m.group(1))
    elif model_type == "full":
        # full mode: blocks.X.weight has shape (num_loops, ...)
        num_loops = state_dict[first_block[0]].shape[0]
    elif model_type == "delta":
        # Check if per-loop delta: fc1 has ModuleList
        fc1_keys = [k for k in state_dict if "delta_blocks.0.fc1" in k and "weight" in k]
        if fc1_keys and ".0.weight" in fc1_keys[0]:
            # per_loop: delta_blocks.0.fc1.0.weight
            indices = set()
            for k in fc1_keys:
                m2 = re.search(r"fc1\.(\d+)\.weight", k)
                if m2:
                    indices.add(int(m2.group(1)))
            num_loops = (max(indices) + 2) if indices else 4  # +1 for 0-index, +1 for full first loop
        else:
            # shared delta: can't infer from weights, trust filename
            pass

    # -- delta_bottleneck for delta model --
    delta_bottleneck = None
    if model_type == "delta":
        fc1_keys = [k for k in state_dict if "delta_blocks.0.fc1" in k and "weight" in k]
        if fc1_keys:
            # Shared: delta_blocks.0.fc1.weight shape (bottleneck, embed_dim)
            # Perloop: delta_blocks.0.fc1.0.weight shape (bottleneck, embed_dim)
            bottleneck = state_dict[fc1_keys[0]].shape[0]
            if bottleneck != embed_dim // 4:
                delta_bottleneck = bottleneck

    return {
        "model_type": model_type, "embed_dim": embed_dim, "num_heads": num_heads,
        "num_layers": num_layers, "num_loops": num_loops,
        "delta_bottleneck": delta_bottleneck,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--prompt", type=str, default="The capital of France is")
    p.add_argument("--max_tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    return p.parse_args()


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_tokens: int, temperature: float, top_k: int):
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()
    model.eval()

    for _ in range(max_tokens):
        ctx = generated[:, -256:]
        out = model(ctx)
        logits = out["logits"][:, -1, :] / max(temperature, 0.01)
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        generated = torch.cat([generated, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main():
    args = parse_args()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    state_dict = torch.load(args.checkpoint, map_location="cpu")
    cfg = auto_detect(state_dict, args.checkpoint)
    print(f"Auto-detected: {cfg}")

    if cfg["model_type"] == "delta":
        from delta_model import DeltaConfig, DeltaLoopedTransformer
        dc = DeltaConfig(
            max_seq_len=256, num_loops=cfg["num_loops"],
            embed_dim=cfg["embed_dim"], num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            delta_bottleneck=cfg["delta_bottleneck"],
        )
        model = DeltaLoopedTransformer(dc).to(device)
    else:
        from model import LoopedTransformerConfig, LoopedTransformer
        from model import LoopedTransformerConfig as LTC
        lc = LTC(
            mode=cfg["model_type"], max_seq_len=256,
            embed_dim=cfg["embed_dim"], num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"], num_loops=cfg["num_loops"],
            lora_rank=8 if cfg["model_type"] == "lora" else 16,
        )
        model = LoopedTransformer(lc).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(f"\nPrompt: {args.prompt}")
    print(f"Output:  ", end="", flush=True)
    text = generate(model, tokenizer, args.prompt, args.max_tokens, args.temperature, args.top_k)
    print(text)


if __name__ == "__main__":
    main()
