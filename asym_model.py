"""
Asymmetric-Depth Transformer: standard layers have [Attn + FFN],
but later layers can be FFN-only. Tests whether attention is
needed at every depth position.

Usage:
    python train_asym.py --num_full 2 --num_ffn 6 --epochs 15
    python train_asym.py --num_full 4 --num_ffn 0 --epochs 15  (= standard)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class AsymConfig:
    vocab_size: int = 50257
    embed_dim: int = 384
    num_heads: int = 6
    num_full: int = 2          # layers with full [Attn + FFN]
    num_ffn: int = 6            # layers with FFN only
    ff_mult: int = 4
    ffn_bottleneck: int | None = None  # None → embed_dim // 4
    max_seq_len: int = 512
    dropout: float = 0.1
    tie_embedding: bool = True


class FullTransformerBlock(nn.Module):
    """Standard pre-LN block: attention + FFN."""

    def __init__(self, config: AsymConfig):
        super().__init__()
        dim = config.embed_dim
        head_dim = dim // config.num_heads
        self.num_heads = config.num_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.ff_up = nn.Linear(dim, dim * config.ff_mult, bias=False)
        self.ff_down = nn.Linear(dim * config.ff_mult, dim, bias=False)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, D = x.shape
        # Attention
        h = self.ln1(x)
        q = self.q_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.dropout(self.o_proj(attn))
        # FFN
        h = self.ln2(x)
        x = x + self.dropout(self.ff_down(F.gelu(self.ff_up(h))))
        return x


class FFNOnlyBlock(nn.Module):
    """FFN-only layer. Much cheaper than a full block."""

    def __init__(self, config: AsymConfig):
        super().__init__()
        dim = config.embed_dim
        bottleneck = config.ffn_bottleneck or (dim // 4)
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, bottleneck, bias=False)
        self.fc2 = nn.Linear(bottleneck, dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        h = self.ln(x)
        return x + self.dropout(self.fc2(F.gelu(self.fc1(h))))


class AsymTransformer(nn.Module):
    """Transformer where only the first num_full layers have attention."""

    def __init__(self, config: AsymConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.full_blocks = nn.ModuleList([FullTransformerBlock(config) for _ in range(config.num_full)])
        self.ffn_blocks = nn.ModuleList([FFNOnlyBlock(config) for _ in range(config.num_ffn)])
        self.ln_final = nn.LayerNorm(config.embed_dim)

        self.lm_head = self.token_embedding if config.tie_embedding else nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)
        x = self.dropout(x)

        for blk in self.full_blocks:
            x = blk(x)
        for blk in self.ffn_blocks:
            x = blk(x)

        x = self.ln_final(x)
        logits = F.linear(x, self.token_embedding.weight) if self.config.tie_embedding else self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {"logits": logits, "loss": loss}

    def count_params(self):
        embed = self.token_embedding.weight.numel() + self.position_embedding.weight.numel()
        full = sum(sum(p.numel() for p in b.parameters()) for b in self.full_blocks)
        ffn = sum(sum(p.numel() for p in b.parameters()) for b in self.ffn_blocks)
        ln = sum(p.numel() for p in self.ln_final.parameters())
        trans = full + ffn + ln
        return embed + trans, embed, trans


# ── Quick test ──
if __name__ == "__main__":
    for nf, nfn in [(4, 0), (2, 6), (1, 7), (2, 10)]:
        cfg = AsymConfig(num_full=nf, num_ffn=nfn, embed_dim=384)
        model = AsymTransformer(cfg)
        total, embed, trans = model.count_params()
        # FLOPs estimate
        d = 384
        flops_full = 4 * 2 * d * d + 2 * 2 * d * 4 * d  # QKVO + FFN
        flops_ffn = 2 * 2 * d * (cfg.ffn_bottleneck or d // 4)
        total_flops = nf * flops_full + nfn * flops_ffn
        print(f"full={nf}, ffn={nfn}: total_params={total:,}, trans={trans:,}, "
              f"linear_FLOPs_per_token={total_flops:,} ({total_flops/(nf*flops_full)*100:.0f}% of {nf}-layer standard)")
        x = torch.randint(0, 50257, (2, 64))
        out = model(x, labels=x)
        print(f"  loss={out['loss'].item():.4f}")
