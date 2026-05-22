"""
Delta-Loop Transformer: first loop full computation, subsequent loops
use a lightweight delta block that predicts the residual correction.

Variants:
  - ffn:      small FFN delta  (embed_dim → bottleneck → embed_dim)
  - attn_ffn: scaled-down attention + small FFN

Comparison point: per-loop LoRA adds params, Delta-Loop saves compute.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class DeltaConfig:
    vocab_size: int = 50257
    embed_dim: int = 384
    num_heads: int = 6
    num_layers: int = 3
    num_loops: int = 4
    ff_mult: int = 4
    max_seq_len: int = 512
    dropout: float = 0.1
    delta_type: str = "ffn"        # "ffn" | "attn_ffn"
    delta_bottleneck: int | None = None  # None → embed_dim // 4
    per_loop_delta: bool = False   # separate delta block per loop?
    attn_inject_every: int = 0     # 0=never, 2=attention injection every 2nd delta step
    tie_embedding: bool = True


# ── Building blocks ───────────────────────────────────────────────────────────

class FullBlock(nn.Module):
    """Standard pre-LN transformer block, used for the first loop iteration."""

    def __init__(self, config: DeltaConfig):
        super().__init__()
        dim = config.embed_dim
        head_dim = dim // config.num_heads
        self.num_heads = config.num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        ff_dim = dim * config.ff_mult
        self.ff_up = nn.Linear(dim, ff_dim, bias=False)
        self.ff_down = nn.Linear(ff_dim, dim, bias=False)

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(config.dropout)

    def _attention(self, x: torch.Tensor):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                             dropout_p=self.dropout.p if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self._attention(self.ln1(x)))
        x = x + self.dropout(self.ff_down(F.gelu(self.ff_up(self.ln2(x)))))
        return x


class DeltaBlock(nn.Module):
    """
    Lightweight block that predicts a residual correction Δx.
    Can be shared across subsequent loops or per-loop.
    """

    def __init__(self, config: DeltaConfig):
        super().__init__()
        dim = config.embed_dim
        bottleneck = config.delta_bottleneck or (dim // 4)
        self.delta_type = config.delta_type
        self.per_loop = config.per_loop_delta
        self.attn_inject_every = config.attn_inject_every
        n = config.num_loops - 1  # number of delta loops
        self.n_delta = n

        # Build attention if delta_type always uses it, OR if injection is enabled
        self.has_attn = (self.delta_type == "attn_ffn") or (self.attn_inject_every > 0)
        if self.has_attn:
            delta_heads = max(1, config.num_heads // 2)
            head_dim = dim // delta_heads
            self.attn_q = self._make_param(n, dim, dim)
            self.attn_k = self._make_param(n, dim, dim)
            self.attn_v = self._make_param(n, dim, dim)
            self.attn_o = self._make_param(n, dim, dim)
            self.num_delta_heads = delta_heads
            self.delta_head_dim = head_dim
            self.attn_scale = head_dim ** -0.5
            self.ln_attn = self._make_ln(n, dim)

        self.ln_ffn = self._make_ln(n, dim)
        self.fc1 = self._make_linear(n, dim, bottleneck)
        self.fc2 = self._make_linear(n, bottleneck, dim)
        self.dropout = nn.Dropout(config.dropout)

    def _make_param(self, n, *shape):
        if self.per_loop:
            return nn.ParameterList([nn.Parameter(torch.empty(*shape)) for _ in range(n)])
        return nn.Parameter(torch.empty(*shape))

    def _make_linear(self, n, in_f, out_f):
        if self.per_loop:
            return nn.ModuleList([nn.Linear(in_f, out_f, bias=False) for _ in range(n)])
        return nn.Linear(in_f, out_f, bias=False)

    def _make_ln(self, n, dim):
        if self.per_loop:
            return nn.ModuleList([nn.LayerNorm(dim) for _ in range(n)])
        return nn.LayerNorm(dim)

    def _get_linear(self, module, idx: int):
        return module[idx] if self.per_loop else module

    def _get_ln(self, module, idx: int):
        return module[idx] if self.per_loop else module

    def _get_param(self, param, idx: int):
        return param[idx] if self.per_loop else param

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            elif p.dim() == 1:
                nn.init.zeros_(p)

    def _apply_attn(self, x: torch.Tensor, delta_idx: int) -> torch.Tensor:
        B, T, D = x.shape
        h = self._get_ln(self.ln_attn, delta_idx)(x)
        nh = self.num_delta_heads
        hd = self.delta_head_dim
        q = F.linear(h, self._get_param(self.attn_q, delta_idx)).view(B, T, nh, hd).transpose(1, 2)
        k = F.linear(h, self._get_param(self.attn_k, delta_idx)).view(B, T, nh, hd).transpose(1, 2)
        v = F.linear(h, self._get_param(self.attn_v, delta_idx)).view(B, T, nh, hd).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                  dropout_p=self.dropout.p if self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        return self.dropout(F.linear(attn_out, self._get_param(self.attn_o, delta_idx)))

    def _should_apply_attn(self, delta_idx: int) -> bool:
        if self.delta_type == "attn_ffn":
            return True  # always
        if self.attn_inject_every > 0:
            return (delta_idx + 1) % self.attn_inject_every == 0
        return False

    def forward(self, x: torch.Tensor, delta_idx: int) -> torch.Tensor:
        delta = torch.zeros_like(x)

        if self.has_attn and self._should_apply_attn(delta_idx):
            delta = delta + self._apply_attn(x, delta_idx)

        # FFN delta (always present)
        h = self._get_ln(self.ln_ffn, delta_idx)(x)
        h = self._get_linear(self.fc1, delta_idx)(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self._get_linear(self.fc2, delta_idx)(h)
        delta = delta + self.dropout(h)

        return delta

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Baseline init ─────────────────────────────────────────────────────────────

def load_full_blocks_from_baseline(model: "DeltaLoopedTransformer", baseline_ckpt: str):
    """
    Initialize FullBlocks from a trained baseline checkpoint.
    Baseline block params (e.g. blocks.0.q_proj.weight) map to
    full_blocks.0.q_proj.weight.
    """
    baseline_state = torch.load(baseline_ckpt, map_location="cpu")

    # Remap keys: blocks.N.xxx → full_blocks.N.xxx
    remapped = {}
    for key, val in baseline_state.items():
        if key.startswith("blocks."):
            remapped["full_" + key] = val
        elif key.startswith("token_embedding.") or key.startswith("position_embedding.") or key.startswith("ln_final."):
            remapped[key] = val

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print(f"  Loaded {len(remapped)} params from baseline, "
          f"skipped {len(missing)} delta-only params")


# ── Full model ────────────────────────────────────────────────────────────────

class DeltaLoopedTransformer(nn.Module):
    """
    Looped transformer where:
      - First loop: full transformer block (standard computation)
      - Subsequent loops: delta block predicts residual correction
    """

    def __init__(self, config: DeltaConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)

        self.full_blocks = nn.ModuleList([FullBlock(config) for _ in range(config.num_layers)])
        self.delta_blocks = nn.ModuleList([DeltaBlock(config) for _ in range(config.num_layers)])

        self.ln_final = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.lm_head = self.token_embedding if config.tie_embedding else nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        B, T = input_ids.shape

        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        for layer_idx in range(self.config.num_layers):
            # First loop: full block
            x = self.full_blocks[layer_idx](x)

            # Subsequent loops: delta correction
            for delta_idx in range(self.config.num_loops - 1):
                x = x + self.delta_blocks[layer_idx](x, delta_idx)

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
        delta = sum(b.count_params() for b in self.delta_blocks)
        ln = sum(p.numel() for p in self.ln_final.parameters())
        trans = full + delta + ln
        return embed + trans, embed, trans


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("ffn", False, 0),
        ("ffn", False, 2),
        ("attn_ffn", False, 0),
        ("ffn", True, 3),
    ]
    for dt, pl, inj in tests:
        cfg = DeltaConfig(delta_type=dt, per_loop_delta=pl, attn_inject_every=inj,
                          embed_dim=384, num_layers=3, num_loops=4, num_heads=6)
        model = DeltaLoopedTransformer(cfg)
        total, embed, trans = model.count_params()
        full_block = sum(sum(p.numel() for p in b.parameters()) for b in model.full_blocks)
        delta_block = sum(b.count_params() for b in model.delta_blocks)
        print(f"{dt}, per_loop={pl}, inject_every={inj}: "
              f"total={total:,}, trans={trans:,}, full={full_block:,}, delta={delta_block:,}")
        x = torch.randint(0, 50257, (2, 64))
        out = model(x, labels=x)
        print(f"  loss={out['loss'].item():.4f}")
