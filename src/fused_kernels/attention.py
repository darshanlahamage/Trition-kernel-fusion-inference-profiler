# triton_llm/kernels/attention.py
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=1),
    ],
    key=['seq_len_q', 'seq_len_kv', 'head_dim']
)
@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, Out,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    seq_len_q, seq_len_kv,
    head_dim: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    q_tile_idx = tl.program_id(2)

    offs_m = q_tile_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, head_dim)

    q_ptrs = Q + (batch_idx * stride_qb + head_idx * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim), other=0.0).to(tl.float16)

    for start_n in range(0, seq_len_kv, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_ptrs = K + (batch_idx * stride_kb + head_idx * stride_kh + (start_n + offs_n[None, :]) * stride_kn + offs_d[:, None] * stride_kd)
        v_ptrs = V + (batch_idx * stride_vb + head_idx * stride_vh + (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vd)

        k = tl.load(k_ptrs, mask=((start_n + offs_n[None, :]) < seq_len_kv) & (offs_d[:, None] < head_dim), other=0.0).to(tl.float16)
        v = tl.load(v_ptrs, mask=((start_n + offs_n[:, None]) < seq_len_kv) & (offs_d[None, :] < head_dim), other=0.0).to(tl.float16)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k) * sm_scale
        
        # Mask out-of-bounds computation
        qk = tl.where((offs_m[:, None] < seq_len_q) & ((start_n + offs_n[None, :]) < seq_len_kv), qk, float("-inf"))
        
        # Causal mask (only apply if seq_len_q == seq_len_kv, otherwise it's decoding and all KVs are valid)
        if seq_len_q == seq_len_kv:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(tl.float16), v)
        m_i = m_new

    l_i = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i[:, None]
    
    out_ptrs = Out + (batch_idx * stride_ob + head_idx * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    tl.store(out_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim))

class CustomFlashAttention:
    @staticmethod
    def forward(q, k, v, sm_scale):
        B, H, S_q, D = q.shape
        _, _, S_kv, _ = k.shape
        
        out = torch.empty_like(q)
        
        grid = lambda META: (B, H, triton.cdiv(S_q, META['BLOCK_M']))
        
        _fwd_kernel[grid](
            q, k, v, sm_scale, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            S_q, S_kv, D
        )
        
        return out