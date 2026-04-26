import torch
import torch.nn as nn
import torch.nn.functional as F

from fused_kernels.attention import CustomFlashAttention
from fused_kernels.rmsnorm import triton_rmsnorm
from fused_kernels.rope import triton_rope

class CustomLlamaBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.eps = eps

        # Attention Projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.attn_norm_weight = nn.Parameter(torch.ones(hidden_dim))
        self.mlp_norm_weight = nn.Parameter(torch.ones(hidden_dim))

        hidden_mlp = int(8 * hidden_dim / 3) 
        self.gate_proj = nn.Linear(hidden_dim, hidden_mlp, bias=False)
        self.up_proj = nn.Linear(hidden_dim, hidden_mlp, bias=False)
        self.down_proj = nn.Linear(hidden_mlp, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, kv_cache=None):
        B, S, D = x.shape
        sm_scale = 1.0 / (self.head_dim ** 0.5)

        residual = x

        x_norm = triton_rmsnorm(x, self.attn_norm_weight, self.eps)

        q = self.q_proj(x_norm).view(B, S, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, S, D]
        k = self.k_proj(x_norm).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        start_pos = 0 if kv_cache is None else kv_cache[0].shape[2]
        
        q = triton_rope(q, cos, sin, start_pos=start_pos)
        k = triton_rope(k, cos, sin, start_pos=start_pos)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        current_kv = (k, v)

        attn_out = CustomFlashAttention.forward(q, k, v, sm_scale)

        # Reshape and Output Projection
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        x = residual + self.o_proj(attn_out)

        residual = x
        
        x_norm = triton_rmsnorm(x, self.mlp_norm_weight, self.eps)
        
        # SwiGLU
        gate = F.silu(self.gate_proj(x_norm))
        up = self.up_proj(x_norm)
        mlp_out = self.down_proj(gate * up)

        x = residual + mlp_out

        return x, current_kv