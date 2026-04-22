# triton_llm/kernels/attention.py
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=1),
    ],
    key=['seq_len', 'head_dim']
)
@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, Out, LSE,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    seq_len, head_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
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

    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim), other=0.0)

    for start_n in range(0, seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_ptrs = K + (batch_idx * stride_kb + head_idx * stride_kh + (start_n + offs_n[None, :]) * stride_kn + offs_d[:, None] * stride_kd)
        v_ptrs = V + (batch_idx * stride_vb + head_idx * stride_vh + (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vd)

        k = tl.load(k_ptrs, mask=((start_n + offs_n[None, :]) < seq_len) & (offs_d[:, None] < head_dim), other=0.0)
        v = tl.load(v_ptrs, mask=((start_n + offs_n[:, None]) < seq_len) & (offs_d[None, :] < head_dim), other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k) * sm_scale
        qk = tl.where((offs_m[:, None] < seq_len) & ((start_n + offs_n[None, :]) < seq_len), qk, float("-inf"))

        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(tl.float16), v)
        m_i = m_new

    # Store LogSumExp for the backward pass
    l_i = tl.where(l_i == 0, 1.0, l_i)
    lse = m_i + tl.math.log(l_i)
    lse_ptrs = LSE + (batch_idx * head_idx * seq_len) + (head_idx * seq_len) + offs_m
    tl.store(lse_ptrs, lse, mask=offs_m < seq_len)

    acc = acc / l_i[:, None]
    out_ptrs = Out + (batch_idx * stride_ob + head_idx * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    tl.store(out_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim))

@triton.jit
def _bwd_kernel(
    Q, K, V, sm_scale, Out, dO, dQ, dK, dV, LSE,
    stride_qb, stride_qh, stride_qm, stride_qd,
    seq_len, head_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    q_tile_idx = tl.program_id(2)

    offs_m = q_tile_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, head_dim)

    q_ptrs = Q + (batch_idx * stride_qb + head_idx * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    o_ptrs = Out + (batch_idx * stride_qb + head_idx * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    do_ptrs = dO + (batch_idx * stride_qb + head_idx * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    lse_ptrs = LSE + (batch_idx * head_idx * seq_len) + (head_idx * seq_len) + offs_m

    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim), other=0.0)
    o = tl.load(o_ptrs, mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim), other=0.0)
    do = tl.load(do_ptrs, mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim), other=0.0)
    lse = tl.load(lse_ptrs, mask=offs_m < seq_len, other=0.0)

    Di = tl.sum(do * o, axis=1)
    dq = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    for start_n in range(0, seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        k_ptrs = K + (batch_idx * stride_qb + head_idx * stride_qh + (start_n + offs_n[None, :]) * stride_qm + offs_d[:, None] * stride_qd)
        v_ptrs = V + (batch_idx * stride_qb + head_idx * stride_qh + (start_n + offs_n[:, None]) * stride_qm + offs_d[None, :] * stride_qd)
        
        k = tl.load(k_ptrs, mask=((start_n + offs_n[None, :]) < seq_len) & (offs_d[:, None] < head_dim), other=0.0)
        v = tl.load(v_ptrs, mask=((start_n + offs_n[:, None]) < seq_len) & (offs_d[None, :] < head_dim), other=0.0)

        qk = tl.dot(q, k) * sm_scale
        qk = tl.where((offs_m[:, None] < seq_len) & ((start_n + offs_n[None, :]) < seq_len), qk, float("-inf"))
        
        p = tl.exp(qk - lse[:, None])
        dv = tl.dot(tl.trans(p.to(tl.float16)), do)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - Di[:, None])
        dq += tl.dot(ds.to(tl.float16), k)

    dq_ptrs = dQ + (batch_idx * stride_qb + head_idx * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    tl.store(dq_ptrs, dq.to(tl.float16), mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim))

class CustomFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        B, H, S, D = q.shape
        out = torch.empty_like(q)
        lse = torch.empty((B, H, S), device=q.device, dtype=torch.float32)
        
        grid = lambda META: (B, H, triton.cdiv(S, META['BLOCK_M']))
        
        _fwd_kernel[grid](
            q, k, v, sm_scale, out, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            S, D
        )
        
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.sm_scale = sm_scale
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, out, lse = ctx.saved_tensors
        B, H, S, D = q.shape
        
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        
        grid = lambda META: (B, H, triton.cdiv(S, META['BLOCK_M']))
        
        _bwd_kernel[grid](
            q, k, v, ctx.sm_scale, out, do, dq, dk, dv, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            S, D, BLOCK_M=64, BLOCK_N=64
        )
        
        return dq, dk, dv, None