import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_rotary_emb_torch(xq, xk, cos, sin, start_pos):
    B, H, S, D = xq.shape
    
    cos = cos[start_pos:start_pos+S].view(1, 1, S, -1)
    sin = sin[start_pos:start_pos+S].view(1, 1, S, -1)
    
    def rotate_half(x):
        x1 = x[..., : D // 2]
        x2 = x[..., D // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    
    return xq_out, xk_out

class TorchLlamaBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.eps = eps

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.attn_norm = nn.RMSNorm(hidden_dim, eps=eps)
        self.mlp_norm = nn.RMSNorm(hidden_dim, eps=eps)

        hidden_mlp = int(8 * hidden_dim / 3) 
        self.gate_proj = nn.Linear(hidden_dim, hidden_mlp, bias=False)
        self.up_proj = nn.Linear(hidden_dim, hidden_mlp, bias=False)
        self.down_proj = nn.Linear(hidden_mlp, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, kv_cache=None):
        B, S, D = x.shape
        residual = x

        x_norm = self.attn_norm(x)

        q = self.q_proj(x_norm).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        start_pos = 0 if kv_cache is None else kv_cache[0].shape[2]
        q, k = apply_rotary_emb_torch(q, k, cos, sin, start_pos)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        current_kv = (k, v)

        # PyTorch SDPA uses FlashAttention internally when possible
        is_causal = (kv_cache is None) and (S > 1)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        x = residual + self.o_proj(attn_out)

        residual = x
        x_norm = self.mlp_norm(x)
        
        gate = F.silu(self.gate_proj(x_norm))
        up = self.up_proj(x_norm)
        mlp_out = self.down_proj(gate * up)

        x = residual + mlp_out

        return x, current_kv

class TorchTinyLlama(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_heads, num_layers, max_seq_length):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)

        self.layers = nn.ModuleList([
            TorchLlamaBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])

        self.norm = nn.RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        scale = 1.0 / math.sqrt(2 * num_layers)
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.q_proj.weight)
            nn.init.xavier_uniform_(layer.k_proj.weight)
            nn.init.xavier_uniform_(layer.v_proj.weight)
            nn.init.xavier_uniform_(layer.o_proj.weight)
            layer.o_proj.weight.data.mul_(scale)
            nn.init.xavier_uniform_(layer.gate_proj.weight)
            nn.init.xavier_uniform_(layer.up_proj.weight)
            nn.init.xavier_uniform_(layer.down_proj.weight)
            layer.down_proj.weight.data.mul_(scale)

        cos, sin = self._precompute_rope(max_seq_length, hidden_dim // num_heads)
        self.register_buffer("cos_cached", cos)
        self.register_buffer("sin_cached", sin)

    def _precompute_rope(self, seq_length, head_dim):
        theta = 10000.0 ** (-2 * torch.arange(0, head_dim // 2).float() / head_dim)
        seq = torch.arange(seq_length).float()
        freqs = torch.outer(seq, theta)
        return torch.cos(freqs).to(torch.float16), torch.sin(freqs).to(torch.float16)

    @torch.inference_mode()
    def forward(self, x, kv_cache=None):
        B, S = x.shape
        x = self.embed(x)
        
        cos, sin = self.cos_cached, self.sin_cached
        
        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, new_kv = layer(x, cos, sin, kv_cache=layer_cache)
            new_kv_cache.append(new_kv)
            
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_kv_cache
