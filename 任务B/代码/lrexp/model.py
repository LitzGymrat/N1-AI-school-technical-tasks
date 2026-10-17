"""Small decoder-only Transformer for long-range dependency experiments.

Efficiency choices:
  * fused QKV projection, bias-free linears
  * F.rms_norm (fused by torch.compile), SwiGLU MLP
  * F.scaled_dot_product_attention(is_causal=True) -> FlashAttention-2 backend for rope / nope / logn
  * FlexAttention (compiled Triton kernel) for ALiBi so we never materialise an LxL bias tensor
  * RoPE tables precomputed once as buffers
A slow "diagnostic" path computes attention probabilities only for selected query rows.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    _HAS_FLEX = True
except Exception:  # pragma: no cover
    _HAS_FLEX = False


@dataclass
class ModelCfg:
    vocab: int = 512
    d_model: int = 256
    n_layer: int = 4
    n_head: int = 4
    head_dim: int = 0          # 0 -> d_model // n_head; otherwise attention inner width = n_head * head_dim
    mlp_mult: float = 4.0
    max_len: int = 4096
    pos: str = "rope"          # rope | nope | alibi
    rope_base: float = 10000.0
    rope_frac: float = 1.0     # fraction of each head's dims that are rotated (partial RoPE); rest is NoPE
    alibi_scale: float = 1.0   # multiplies standard ALiBi slopes (bigger = stronger recency bias)
    attn_scale: str = "std"    # std | logn (scalable-softmax style, learnable per-head s * log(n))
    qk_norm: bool = False
    smear: bool = False        # add learnable embedding of previous token (gives prev-token info for free)
    init_std: float = 0.02
    tie_emb: bool = False      # tie input embedding and output head weights


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.w, eps=1e-6)


def rope_tables(max_len, hd, base, device=None):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=device).float() / hd))
    t = torch.arange(max_len, device=device).float()
    f = torch.outer(t, inv)  # (L, hd/2)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, L, hd); rotate-half convention over the first 2*cos.shape[-1] dims (partial RoPE if < hd)
    L = x.shape[-2]
    cos = cos[:L].to(x.dtype)
    sin = sin[:L].to(x.dtype)
    rd = 2 * cos.shape[-1]
    if rd < x.shape[-1]:
        return torch.cat([apply_rope(x[..., :rd], cos, sin), x[..., rd:]], -1)
    x1, x2 = x.chunk(2, dim=-1)
    c = torch.cat([cos, cos], -1)
    s = torch.cat([sin, sin], -1)
    rot = torch.cat([-x2, x1], -1)
    return x * c + rot * s


_ALIBI_CACHE = {}


def _alibi_slopes(n_head):
    # standard ALiBi geometric slopes
    start = 2 ** (-8.0 / n_head)
    return torch.tensor([start ** (i + 1) for i in range(n_head)])


class Attention(nn.Module):
    def __init__(self, cfg: ModelCfg, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.nh = cfg.n_head
        self.hd = cfg.head_dim or cfg.d_model // cfg.n_head
        self.inner = self.nh * self.hd
        self.qkv = nn.Linear(cfg.d_model, 3 * self.inner, bias=False)
        self.proj = nn.Linear(self.inner, cfg.d_model, bias=False)
        if cfg.qk_norm:
            self.qn = RMSNorm(self.hd)
            self.kn = RMSNorm(self.hd)
        if cfg.attn_scale == "logn":
            # q_i <- q_i * s_h * log(i+1) / log(L_ref); init s_h = 1 so scale==1 at i+1 == L_ref
            self.s = nn.Parameter(torch.ones(self.nh))
        if cfg.pos == "alibi":
            self.register_buffer("slopes", _alibi_slopes(self.nh) * cfg.alibi_scale, persistent=False)
        self.diag = False
        self.saved = None

    def _qkv(self, x, cos, sin):
        B, L, _ = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
        if self.cfg.qk_norm:
            q, k = self.qn(q), self.kn(k)
        if self.cfg.pos == "rope":
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.cfg.attn_scale == "logn":
            pos = torch.arange(1, L + 1, device=x.device, dtype=torch.float32)
            sc = (torch.log(pos) / math.log(512.0)).clamp_min(1.0 / math.log(512.0))
            q = q * (self.s.view(1, -1, 1, 1) * sc.view(1, 1, -1, 1)).to(q.dtype)
        return q, k, v

    def forward(self, x, cos, sin, block_mask=None):
        B, L, D = x.shape
        q, k, v = self._qkv(x, cos, sin)
        if self.diag:
            self.saved = (q.detach(), k.detach())
        if self.cfg.pos == "alibi":
            slopes = self.slopes

            def score_mod(score, b, h, qi, ki):
                return score + slopes[h] * (ki - qi).to(score.dtype)

            y = flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, L, self.inner)
        return self.proj(y)

    @torch.no_grad()
    def attn_probs_rows(self, rows_b, rows_q):
        """Attention probabilities for selected (batch, query) rows. Returns (N, H, L) float32."""
        q, k = self.saved
        qs = q[rows_b, :, rows_q].float()                 # (N, H, hd)
        ks = k[rows_b].float()                            # (N, H, L, hd)
        s = torch.einsum("nhd,nhld->nhl", qs, ks) / math.sqrt(self.hd)
        L = ks.shape[2]
        idx = torch.arange(L, device=s.device)
        if self.cfg.pos == "alibi":
            s = s + self.slopes.view(1, -1, 1).float() * (idx.view(1, 1, -1) - rows_q.view(-1, 1, 1)).float()
        s = s.masked_fill(idx.view(1, 1, -1) > rows_q.view(-1, 1, 1), float("-inf"))
        return s.softmax(-1)

    @torch.no_grad()
    def attn_probs_q(self, rows_q):
        """Memory-efficient version: rows_q (B, n) query positions per sequence -> probs (B, n, H, L) float32."""
        q, k = self.saved                                  # (B, H, L, hd)
        B, H, L, hd = q.shape
        n = rows_q.shape[1]
        qs = q.gather(2, rows_q[:, None, :, None].expand(B, H, n, hd)).float()
        s = qs @ k.float().transpose(-1, -2) / math.sqrt(hd)  # (B, H, n, L)
        idx = torch.arange(L, device=s.device).view(1, 1, 1, L)
        rq = rows_q[:, None, :, None]
        if self.cfg.pos == "alibi":
            s = s + self.slopes.view(1, -1, 1, 1).float() * (idx - rq).float()
        s = s.masked_fill(idx > rq, float("-inf"))
        return s.softmax(-1).permute(0, 2, 1, 3)


class MLP(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        h = int(cfg.mlp_mult * cfg.d_model * 2 / 3)
        h = (h + 63) // 64 * 64
        self.w12 = nn.Linear(cfg.d_model, 2 * h, bias=False)
        self.w3 = nn.Linear(h, cfg.d_model, bias=False)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, -1)
        return self.w3(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, cfg, i):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg, i)
        self.n2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, block_mask=None):
        x = x + self.attn(self.n1(x), cos, sin, block_mask)
        x = x + self.mlp(self.n2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d_model)
        if cfg.smear:
            self.emb_prev = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.nf = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        if cfg.tie_emb:
            self.head.weight = self.emb.weight
        hd = cfg.head_dim or cfg.d_model // cfg.n_head
        rd = max(2, int(hd * cfg.rope_frac) // 2 * 2)   # rotated dims (== hd for rope_frac=1.0)
        cos, sin = rope_tables(cfg.max_len, rd, cfg.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for b in self.blocks:  # scaled init of residual projections
            nn.init.normal_(b.attn.proj.weight, std=cfg.init_std / math.sqrt(2 * cfg.n_layer))
            nn.init.normal_(b.mlp.w3.weight, std=cfg.init_std / math.sqrt(2 * cfg.n_layer))
        self._bm = {}
        self.collect_resid = False
        self.resid = []

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=self.cfg.init_std)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=self.cfg.init_std)

    def block_mask(self, L, device):
        if self.cfg.pos != "alibi":
            return None
        key = (L, str(device))
        if key not in self._bm:
            def causal(b, h, qi, ki):
                return qi >= ki
            self._bm[key] = create_block_mask(causal, None, None, L, L, device=device)
        return self._bm[key]

    def forward(self, idx, block_mask=None):
        x = self.emb(idx)
        if self.cfg.smear:
            prev = F.pad(idx[:, :-1], (1, 0), value=0)
            x = x + self.emb_prev(prev)
        if self.collect_resid:
            self.resid = [x.detach()]
        for b in self.blocks:
            x = b(x, self.cos, self.sin, block_mask)
            if self.collect_resid:
                self.resid.append(x.detach())
        return self.head(self.nf(x))

    def set_diag(self, on: bool):
        for b in self.blocks:
            b.attn.diag = on
            if not on:
                b.attn.saved = None
        self.collect_resid = on
        if not on:
            self.resid = []
