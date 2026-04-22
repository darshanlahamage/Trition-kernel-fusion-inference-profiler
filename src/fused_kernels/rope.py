# triton_llm/kernels/rope.py
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 64}, num_warps=4),
        triton.Config({'BLOCK_S': 128}, num_warps=4),
    ],
    key=['seq_len']
)
@triton.jit
def _rope_fwd_2d(
    Q_ptr, Out_ptr, Cos_ptr, Sin_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_cs, stride_cd, seq_len,
    BLOCK_S: tl.constexpr, HALF_D: tl.constexpr
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_block_idx = tl.program_id(2)

    offs_s = seq_block_idx * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_d1 = tl.arange(0, HALF_D)
    offs_d2 = HALF_D + tl.arange(0, HALF_D)
    mask_s = offs_s < seq_len

    q_base = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    out_base = Out_ptr + batch_idx * stride_ob + head_idx * stride_oh

    q1 = tl.load(q_base + offs_s[:, None] * stride_qs + offs_d1[None, :] * stride_qd, mask=mask_s[:, None], other=0.0).to(tl.float32)
    q2 = tl.load(q_base + offs_s[:, None] * stride_qs + offs_d2[None, :] * stride_qd, mask=mask_s[:, None], other=0.0).to(tl.float32)
    cos = tl.load(Cos_ptr + offs_s[:, None] * stride_cs + offs_d1[None, :] * stride_cd, mask=mask_s[:, None], other=0.0).to(tl.float32)
    sin = tl.load(Sin_ptr + offs_s[:, None] * stride_cs + offs_d1[None, :] * stride_cd, mask=mask_s[:, None], other=0.0).to(tl.float32)

    out1 = q1 * cos - q2 * sin
    out2 = q2 * cos + q1 * sin

    tl.store(out_base + offs_s[:, None] * stride_os + offs_d1[None, :] * stride_od, out1.to(tl.float16), mask=mask_s[:, None])
    tl.store(out_base + offs_s[:, None] * stride_os + offs_d2[None, :] * stride_od, out2.to(tl.float16), mask=mask_s[:, None])

def triton_rope(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(q)
    B, H, S, D = out.shape
    grid = lambda META: (B, H, triton.cdiv(S, META['BLOCK_S']))
    
    _rope_fwd_2d[grid](
        q, out, cos, sin,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        cos.stride(0), cos.stride(1), S, HALF_D=D // 2
    )
    return out