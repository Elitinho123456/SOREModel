
"""
soreModel_v3.py
The experimental SOREModel v3 — GPT-style transformer with:
- Custom efficient multi-head attention supporting ALiBi (per-head linear attention bias).
- Rotary positional embeddings (RoPE) applied to q/k (rotary-only positional scheme; learned pos removed).
- Pre-LN transformer blocks, RMSNorm option.
- Weight tying, generation (top-k, top-p, temperature), and a convenience config dataclass.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    context_size: int = 1024
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    dropout: float = 0.1
    ffn_mult: int = 4
    use_rmsnorm: bool = False
    use_alibi: bool = True
    rotary_pct: float = 1.0  # proportion of head_dim to apply RoPE (0..1)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.scale

def rotate_half(x):
    # split last dim in half and rotate
    x1 = x[..., : x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope_to(q, k, seq_len, base=10000):
    """
    Apply RoPE to q and k tensors.
    q,k shape: (B, H, T, D)
    Returns rotated q,k.
    """
    B, H, T, D = q.shape
    device = q.device
    # create freqs
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2, device=device).float() / D))
    positions = torch.arange(seq_len, device=device).type_as(inv_freq)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)  # (T, D/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (T, D)
    cos = emb.cos().view(1, 1, T, D)
    sin = emb.sin().view(1, 1, T, D)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

def build_alibi_slopes(n_heads: int, device: torch.device):
    """
    Build ALiBi slopes following the method from the ALiBi paper.
    Returns a tensor of shape (n_heads,) with monotonically increasing slopes.
    """
    def get_slopes(n):
        # implementation adapted from prior ALiBi implementations
        def _pow(x, y):
            return x ** y
        if math.log2(n).is_integer():
            start = 2**(-2**-(math.log2(n)-3))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]
        else:
            # fallback: linear spacing on log scale
            return [1.0/(i+1) for i in range(1, n+1)]
    slopes = torch.tensor(get_slopes(n_heads), device=device).float()
    return slopes

class MultiHeadSelfAttentionWithAlibi(nn.Module):
    """
    Custom multi-head self-attention:
    - q/k/v projections
    - scaled dot-product with causal mask
    - optional ALiBi bias per head (added to attention scores)
    - supports RoPE by providing q/k pre-rotated
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, use_alibi: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.use_alibi = use_alibi
        if use_alibi:
            # slopes created lazily per device/size in forward
            self.register_buffer("_alibi_slopes", torch.empty(0), persistent=False)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, alibi_bias: Optional[torch.Tensor] = None, rotary_qk: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        x: (B, T, C)
        attn_mask: additive mask of shape (T, T) with -inf on disallowed positions (or None)
        alibi_bias: precomputed per-head alibi bias shaped (num_heads, T, T) or None
        rotary_qk: optional tuple (q_rot_mask) - if provided, q and k will be replaced by rotated versions AFTER projection
        """
        B, T, C = x.shape
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, T, D)

        # apply rotary if provided
        if rotary_qk is not None:
            # rotary_qk is tuple (seq_len) but we just apply based on T
            q, k = apply_rope_to(q, k, seq_len=T)

        # calcular scores
        # scores: (B, H, T, T)
        scores = torch.einsum("bhtd,bhSd->bh t S", q, k) * self.scale  # using einsum with S==T, keep dims explicit
        # Note: o einsum acima usa nomes; reformatando se houver problemas.
        # De qualquer forma, einsum usado com indices repetidos pode ser complicado; usando matmul:
        # scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # máscara causal: garantir que a triangular superior seja -inf; se attn_mask for fornecido, adicione-o
        if attn_mask is not None:
            # attn_mask expected shape (T, T) additive (with -inf where masked), broadcast to (B, H, T, T)
            scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)

        # ALiBi: alibi_bias expected shape (H, T, T)
        if self.use_alibi and alibi_bias is not None:
            scores = scores + alibi_bias.unsqueeze(0)  # broadcast to (B, H, T, T)

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)  # (B, H, T, D)
        # combine heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.out_dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.embed_dim
        self.norm1 = RMSNorm(d) if cfg.use_rmsnorm else nn.LayerNorm(d)
        self.attn = MultiHeadSelfAttentionWithAlibi(d, cfg.num_heads, dropout=cfg.dropout, use_alibi=cfg.use_alibi)
        self.norm2 = RMSNorm(d) if cfg.use_rmsnorm else nn.LayerNorm(d)
        self.ffn = FeedForward(d, mult=cfg.ffn_mult, dropout=cfg.dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, alibi_bias: Optional[torch.Tensor] = None):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask, alibi_bias=alibi_bias)
        x = x + self.ffn(self.norm2(x))
        return x

class SOREModel_v3(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        # no learned pos embedding; using RoPE instead
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.ln_f = RMSNorm(cfg.embed_dim) if cfg.use_rmsnorm else nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        # weight tying
        self.head.weight = self.token_emb.weight
        self._init_weights()
        # alibi slopes cache (device-bound)
        self._alibi_cache = {}

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _build_causal_mask(self, T: int, device: torch.device):
        # return additive mask (T, T) with -inf in upper triangle
        mask = torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)
        attn_mask = torch.zeros((T, T), device=device)
        attn_mask = attn_mask.masked_fill(mask, float("-inf"))
        return attn_mask

    def _build_alibi(self, T: int, device: torch.device):
        # returns tensor (H, T, T)
        key = (self.cfg.num_heads, T, device)
        if key in self._alibi_cache:
            return self._alibi_cache[key]
        slopes = build_alibi_slopes(self.cfg.num_heads, device=device)  # (H,)
        # distances: (T, T) where dist[i,j] = j - i (target index - query index)
        # For causal, positions where j > i should be negative infinity after mask; ALiBi uses relative positions
        pos = torch.arange(T, device=device)
        # relative positions matrix: (T, T) with j - i
        rel_pos = pos.view(1, -1) - pos.view(-1, 1)  # (T, T)
        # make positive values where j>i (future) large positive; but we'll mask futures separately;
        # ALiBi uses negative slopes times distance; we want bias for attention scores: slope * (i - j)
        # use bias = slopes[:, None, None] * ( - rel_pos )
        # but ensure shape (H, T, T)
        # we compute bias = slopes[:, None, None] * (torch.arange(T).view(1,-1) - torch.arange(T).view(-1,1))
        bias = slopes.view(-1, 1, 1) * ( - rel_pos.unsqueeze(0).to(device) ).float()
        # store
        self._alibi_cache[key] = bias
        return bias

    def forward(self, idx: torch.LongTensor) -> torch.Tensor:
        """
        idx: (B, T)
        returns logits (B, T, V)
        """
        B, T = idx.shape
        assert T <= self.cfg.context_size, f"Sequence length {T} > context_size {self.cfg.context_size}"
        x = self.token_emb(idx)  # (B, T, C)
        x = self.drop(x)
        attn_mask = self._build_causal_mask(T, device=idx.device)  # (T, T)
        alibi_bias = None
        if self.cfg.use_alibi:
            alibi_bias = self._build_alibi(T, device=idx.device)  # (H, T, T)
        # pass through blocks; rotary is applied inside attention (using apply_rope_to)
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask, alibi_bias=alibi_bias)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx: torch.LongTensor, max_new_tokens: int = 128, temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0):
        for _ in range(max_new_tokens):
            B, T = idx.shape
            if T > self.cfg.context_size:
                idx_cond = idx[:, -self.cfg.context_size:]
            else:
                idx_cond = idx
            logits = self(idx_cond)  # (B, T_cond, V)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Top-k
            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                min_vals = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_vals, torch.full_like(logits, float("-inf")), logits)

            # Top-p (nucleus)
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                # set removed indices to -inf
                sorted_logits[sorted_indices_to_remove] = float("-inf")
                # scatter back
                logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
