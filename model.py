"""
Looped Transformer with Per-Loop LoRA.

Three configurations:
  - baseline: traditional looped transformer (fully shared params)
  - lora:     shared W_base + per-loop independent B_t @ A_t (trainable)
  - full:     per-loop fully independent parameters (upper-bound reference)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class LoopedTransformerConfig:
    vocab_size: int = 50257
    embed_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    num_loops: int = 4
    ff_mult: int = 4
    max_seq_len: int = 512
    dropout: float = 0.1
    mode: str = "baseline"       # "baseline" | "lora" | "full"
    lora_rank: int = 16
    tie_embedding: bool = True


class LoRALinear(nn.Module):
    """
    Linear layer supporting three modes:
      - baseline: standard weight, shared across all loops
      - lora:     shared weight + per-loop B_t @ A_t
      - full:     independent weight per loop
    """

    def __init__(self, in_features: int, out_features: int, config: LoopedTransformerConfig):
        super().__init__()
        self.mode = config.mode
        self.num_loops = config.num_loops
        self.rank = config.lora_rank
        self.in_features = in_features
        self.out_features = out_features

        if self.mode == "full":
            self.weight = nn.Parameter(torch.empty(config.num_loops, out_features, in_features))
        else:
            self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if self.mode == "lora":
            # Per-loop LoRA: B_t (out, rank), A_t (rank, in)
            self.lora_A = nn.Parameter(torch.zeros(config.num_loops, self.rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(config.num_loops, out_features, self.rank))

        self.bias = nn.Parameter(torch.zeros(out_features))
        self._reset_parameters()

    def _reset_parameters(self):
        if self.mode == "full":
            for i in range(self.num_loops):
                nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.mode == "lora":
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # lora_B stays zero-initialized

        fan_in = self.weight.shape[-1]
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, loop_idx: int) -> torch.Tensor:
        if self.mode == "full":
            w = self.weight[loop_idx]
        else:
            w = self.weight

        out = F.linear(x, w, self.bias)

        if self.mode == "lora":
            A = self.lora_A[loop_idx]          # (rank, in)
            B = self.lora_B[loop_idx]          # (out, rank)
            lora_out = F.linear(F.linear(x, A), B)
            out = out + lora_out

        return out

    def count_params(self):
        """Return (trainable, total) parameter counts for this layer."""
        base = self.weight.numel() + self.bias.numel()
        if self.mode == "full":
            return base, base
        if self.mode == "lora":
            lora = self.lora_A.numel() + self.lora_B.numel()
            return base + lora, base + lora
        return base, base


class LoopedTransformerBlock(nn.Module):
    """One transformer block that can be iterated multiple times."""

    def __init__(self, config: LoopedTransformerConfig):
        super().__init__()
        self.config = config
        dim = config.embed_dim
        head_dim = dim // config.num_heads
        self.num_heads = config.num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # Multi-head attention
        self.q_proj = LoRALinear(dim, dim, config)
        self.k_proj = LoRALinear(dim, dim, config)
        self.v_proj = LoRALinear(dim, dim, config)
        self.o_proj = LoRALinear(dim, dim, config)

        # Feed-forward
        ff_dim = dim * config.ff_mult
        self.ff_up = LoRALinear(dim, ff_dim, config)
        self.ff_down = LoRALinear(ff_dim, dim, config)

        # Layer norms (not LoRA-affected)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(config.dropout)

    def _attention(self, x: torch.Tensor, loop_idx: int, causal: bool = True):
        B, T, D = x.shape

        q = self.q_proj(x, loop_idx).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x, loop_idx).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x, loop_idx).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.config.dropout if self.training else 0.0,
            is_causal=causal,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(attn_out, loop_idx)

    def forward(self, x: torch.Tensor, loop_idx: int) -> torch.Tensor:
        # Pre-LN attention
        x = x + self.dropout(self._attention(self.ln1(x), loop_idx))
        # Pre-LN FFN
        x = x + self.dropout(self.ff_down(self.ff_up(self.ln2(x), loop_idx), loop_idx))
        return x

    def count_params(self):
        total = sum(
            mod.count_params()[0]
            for mod in [self.q_proj, self.k_proj, self.v_proj, self.o_proj, self.ff_up, self.ff_down]
        )
        return total


class LoopedTransformer(nn.Module):
    """Looped transformer with per-loop LoRA support."""

    def __init__(self, config: LoopedTransformerConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)

        self.blocks = nn.ModuleList([
            LoopedTransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.ln_final = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

        if config.tie_embedding:
            self.lm_head = self.token_embedding
        else:
            self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        B, T = input_ids.shape

        # Embeddings
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        # Apply each block for num_loops iterations
        for block in self.blocks:
            for loop_idx in range(self.config.num_loops):
                x = block(x, loop_idx)

        x = self.ln_final(x)
        logits = self.lm_head(x) if not self.config.tie_embedding else F.linear(x, self.token_embedding.weight)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return {"logits": logits, "loss": loss}

    def count_params(self):
        """Return (total_trainable, embedding_params, transformer_params)."""
        embed_params = self.token_embedding.weight.numel() + self.position_embedding.weight.numel()
        if not self.config.tie_embedding:
            embed_params += self.lm_head.weight.numel()

        block_params = sum(b.count_params() for b in self.blocks)
        ln_params = sum(p.numel() for p in self.ln_final.parameters())
        for block in self.blocks:
            ln_params += sum(p.numel() for p in block.ln1.parameters())
            ln_params += sum(p.numel() for p in block.ln2.parameters())

        transformer_params = block_params + ln_params
        total = embed_params + transformer_params
        return total, embed_params, transformer_params

    def configure_optimizer_groups(self, weight_decay: float, lr: float):
        """Separate weight-decay and no-weight-decay parameter groups."""
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) < 2:
                no_decay_params.append(param)
            elif "embedding" in name or "ln" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return [
            {"params": decay_params, "weight_decay": weight_decay, "lr": lr},
            {"params": no_decay_params, "weight_decay": 0.0, "lr": lr},
        ]


def create_model(mode: str, **kwargs) -> LoopedTransformer:
    """Factory to create a LoopedTransformer with the given mode."""
    config = LoopedTransformerConfig(mode=mode, **kwargs)
    return LoopedTransformer(config)


def print_param_summary(model: LoopedTransformer):
    total, embed, trans = model.count_params()
    print(f"Mode: {model.config.mode}")
    print(f"  Total params:      {total:,}")
    print(f"  Embedding params:  {embed:,}")
    print(f"  Transformer params: {trans:,}")
    return total, embed, trans


# Quick test
if __name__ == "__main__":
    for mode in ["baseline", "lora", "full"]:
        model = create_model(mode, embed_dim=256, num_layers=2, num_loops=4, num_heads=4, lora_rank=16)
        total, embed, trans = print_param_summary(model)
        print()

        # Test forward pass
        x = torch.randint(0, 50257, (2, 64))
        out = model(x, labels=x)
        print(f"  Loss: {out['loss'].item():.4f}")
        print()
