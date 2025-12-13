
"""
soreModel_v4.py
SOREModel v4 - The "Home-made LLM" Architecture.
Evolution of v3 with strict optimizations for production/distillation:
- RMSNorm by default.
- RoPE (Rotary Positional Embeddings) on q/k.
- ALiBi (Attention with Linear Biases) for context extrapolation.
- SwiGLU (planned/optional) or standard GEGLU/GELU.
- Weight Tying explicitly enforced.
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
    context_size: int = 2048  # Increased default for v4
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    dropout: float = 0.1
    ffn_mult: int = 4
    use_rmsnorm: bool = True  # Default to True for v4
    use_alibi: bool = True
    rotary_pct: float = 1.0

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.scale

def rotate_half(x):
    x1 = x[..., : x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope_to(q, k, seq_len, base=10000):
    B, H, T, D = q.shape
    device = q.device
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2, device=device).float() / D))
    positions = torch.arange(seq_len, device=device).type_as(inv_freq)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().view(1, 1, T, D)
    sin = emb.sin().view(1, 1, T, D)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

def build_alibi_slopes(n_heads: int, device: torch.device):
    def get_slopes(n):
        def _pow(x, y): return x ** y
        if math.log2(n).is_integer():
            start = 2**(-2**-(math.log2(n)-3))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]
        else:
            return [1.0/(i+1) for i in range(1, n+1)]
    slopes = torch.tensor(get_slopes(n_heads), device=device).float()
    return slopes

class MultiHeadSelfAttentionWithAlibi(nn.Module):
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

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, alibi_bias: Optional[torch.Tensor] = None, rotary_qk: bool = True):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if rotary_qk:
            q, k = apply_rope_to(q, k, seq_len=T)

        # Flash Attention is cleaner, but keeping explicit math for custom bias/alibi compatibility
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)

        if self.use_alibi and alibi_bias is not None:
            scores = scores + alibi_bias.unsqueeze(0)

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
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

class SOREModel_v4(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.ln_f = RMSNorm(cfg.embed_dim) if cfg.use_rmsnorm else nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        
        # Weight Tying: Critical for v4
        self.head.weight = self.token_emb.weight
        
        self._init_weights()
        self._alibi_cache = {}

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _build_causal_mask(self, T: int, device: torch.device):
        mask = torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)
        attn_mask = torch.zeros((T, T), device=device)
        attn_mask = attn_mask.masked_fill(mask, float("-inf"))
        return attn_mask

    def _build_alibi(self, T: int, device: torch.device):
        key = (self.cfg.num_heads, T, device)
        if key in self._alibi_cache: return self._alibi_cache[key]
        slopes = build_alibi_slopes(self.cfg.num_heads, device=device)
        pos = torch.arange(T, device=device)
        rel_pos = pos.view(1, -1) - pos.view(-1, 1)
        bias = slopes.view(-1, 1, 1) * ( - rel_pos.unsqueeze(0).to(device) ).float()
        self._alibi_cache[key] = bias
        return bias

    def forward(self, idx: torch.LongTensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.cfg.context_size, f"Sequence length {T} > context_size {self.cfg.context_size}"
        x = self.token_emb(idx)
        x = self.drop(x)
        attn_mask = self._build_causal_mask(T, device=idx.device)
        alibi_bias = self._build_alibi(T, device=idx.device) if self.cfg.use_alibi else None
        
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask, alibi_bias=alibi_bias)
            
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx: torch.LongTensor, max_new_tokens: int = 128, temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0):
        for _ in range(max_new_tokens):
            B, T = idx.shape
            idx_cond = idx[:, -self.cfg.context_size:] if T > self.cfg.context_size else idx
            logits = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            
            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                logits = torch.where(logits < values[:, -1].unsqueeze(-1), torch.full_like(logits, float("-inf")), logits)
            
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                sorted_logits[sorted_indices_to_remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
