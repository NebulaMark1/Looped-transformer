"""Generate text from a trained model checkpoint."""

import argparse
import torch
from transformers import GPT2TokenizerFast


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model_type", type=str, default="delta",
                   choices=["baseline", "delta", "lora", "full"])
    p.add_argument("--prompt", type=str, default="The capital of France is")
    p.add_argument("--max_tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--num_loops", type=int, default=4)
    return p.parse_args()


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_tokens: int, temperature: float, top_k: int):
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    generated = input_ids.clone()
    model.eval()

    for _ in range(max_tokens):
        # Truncate to max_seq_len if needed
        ctx = generated[:, -256:]

        if args.model_type == "delta":
            out = model(ctx)
        else:
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
    global args
    args = parse_args()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.model_type == "delta":
        from delta_model import DeltaConfig, DeltaLoopedTransformer
        cfg = DeltaConfig(max_seq_len=args.seq_len, num_loops=args.num_loops,
                          embed_dim=384, num_heads=6, num_layers=3)
        model = DeltaLoopedTransformer(cfg).to(device)
    else:
        from model import create_model
        model = create_model(args.model_type, max_seq_len=args.seq_len,
                             num_loops=args.num_loops).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    print(f"\nPrompt: {args.prompt}")
    print(f"Output:  ", end="", flush=True)

    text = generate(model, tokenizer, args.prompt, args.max_tokens, args.temperature, args.top_k)
    print(text)


if __name__ == "__main__":
    main()
