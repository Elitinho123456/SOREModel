"""
soreModel_v4.1.py
SOREModel v4.1 - The "Home-made LLM" Architecture.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    context_size: int = 2048
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    dropout: float = 0.1
    ffn_mult: int = 4
    use_alibi: bool = False  # v4.1 uses RoPE only, but we keep this for API compatibility
    use_rmsnorm: bool = True

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

class PrecomputedRoPE(nn.Module):
    """Pré-calcula as frequências do RoPE para evitar alocação de tensores no forward"""
    def __init__(self, dim: int, max_seq_len: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Registra como buffer (vai para GPU, salva no state_dict, mas não tem gradiente)
        self.register_buffer("cos_cached", emb.cos().view(1, 1, max_seq_len, dim))
        self.register_buffer("sin_cached", emb.sin().view(1, 1, max_seq_len, dim))

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int, offset: int = 0):
        cos = self.cos_cached[:, :, offset : offset + seq_len, :]
        sin = self.sin_cached[:, :, offset : offset + seq_len, :]
        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.embed_dim % cfg.num_heads == 0
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.embed_dim // cfg.num_heads
        
        self.qkv_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)
        
        self.attn_dropout = cfg.dropout
        self.out_dropout = nn.Dropout(cfg.dropout)
        self.rope = PrecomputedRoPE(self.head_dim, cfg.context_size)

    def forward(
        self, 
        x: torch.Tensor, 
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
        use_cache: bool = False
    ):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Lógica do KV Cache: Descobre a posição atual baseada no cache passado
        offset = past_kv[0].shape[2] if past_kv is not None else 0
        
        # Aplica o RoPE pré-calculado com o offset correto
        q, k = self.rope(q, k, seq_len=T, offset=offset)

        # Atualiza o KV Cache
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        
        new_kv = (k, v) if use_cache else None

        # Flash Attention (SDPA): Muito mais rápido e gasta O(N) memória em vez de O(N^2)
        # Se offset == 0 e T > 1 (fase de prompt), usamos máscara causal. 
        # Na geração token a token, a máscara não é necessária pois Q(tam 1) só olha pro K,V passado.
        is_causal = (past_kv is None and T > 1)
        
        out = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=is_causal
        )
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_dropout(self.out_proj(out))
        
        return out, new_kv

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
        self.attn = MultiHeadSelfAttention(cfg)
        self.norm2 = RMSNorm(d) if cfg.use_rmsnorm else nn.LayerNorm(d)
        self.ffn = FeedForward(d, mult=cfg.ffn_mult, dropout=cfg.dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
        use_cache: bool = False
    ):
        attn_out, new_kv = self.attn(self.norm1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_kv

class SOREModel_v4_1(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.ln_f = RMSNorm(cfg.embed_dim) if cfg.use_rmsnorm else nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        
        # Weight Tying
        self.head.weight = self.token_emb.weight
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, 
        idx: torch.LongTensor, 
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ):
        B, T = idx.shape
        x = self.token_emb(idx)
        x = self.drop(x)
        
        next_kv = [] if use_cache else None
        
        for i, blk in enumerate(self.blocks):
            past_kv_blk = past_key_values[i] if past_key_values is not None else None
            x, new_kv = blk(x, past_kv=past_kv_blk, use_cache=use_cache)
            if use_cache:
                next_kv.append(new_kv)
                
        x = self.ln_f(x)
        logits = self.head(x)
        
        if use_cache:
            return logits, next_kv
        return logits

    @torch.no_grad()
    def generate(self, idx: torch.LongTensor, max_new_tokens: int = 128, temperature: float = 1.0, top_p: float = 0.9):
        past_kv = None
        
        for _ in range(max_new_tokens):
            # Se já temos o cache, passamos apenas o último token gerado!
            idx_input = idx if past_kv is None else idx[:, -1:]
            
            logits, past_kv = self(idx_input, past_key_values=past_kv, use_cache=True)
            
            # Pega o logit apenas do último token do batch
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            
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