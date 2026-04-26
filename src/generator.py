import math
import torch
import torch.nn as nn
from llama_block import CustomLlamaBlock
from fused_kernels.rmsnorm import triton_rmsnorm

class TinyLlama(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_heads, num_layers, max_seq_length):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)

        self.layers = nn.ModuleList([
            CustomLlamaBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])

        self.norm_weight = nn.Parameter(torch.ones(hidden_dim))
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Scale weights by 1/sqrt(2 * layers) for activation stability
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
        
        # We pass the full precomputed cos and sin; rope kernel will index them using start_pos
        cos, sin = self.cos_cached, self.sin_cached
        
        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, new_kv = layer(x, cos, sin, kv_cache=layer_cache)
            new_kv_cache.append(new_kv)
            
        x = triton_rmsnorm(x, self.norm_weight)
        logits = self.lm_head(x)
        return logits, new_kv_cache

class Generator:
    def __init__(self, model):
        self.model = model

    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens):
        # Prefill Phase
        logits, kv_cache = self.model(prompt)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
        
        generated_tokens = [next_token]
        
        # Decode Phase Loop
        for _ in range(max_new_tokens - 1):
            logits, kv_cache = self.model(next_token, kv_cache=kv_cache)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            generated_tokens.append(next_token)
            
        return torch.cat(generated_tokens, dim=-1)
