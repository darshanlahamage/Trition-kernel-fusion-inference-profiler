# triton_llm/kernels/rmsnorm.py
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=['N_COLS']
)
@triton.jit
def _rmsnorm_fwd(
    X_ptr, Y_ptr, W_ptr, 
    stride_x_row, stride_y_row, 
    N_COLS, eps, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    X_row_ptr = X_ptr + row_idx * stride_x_row
    Y_row_ptr = Y_ptr + row_idx * stride_y_row
    
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_COLS
    
    x = tl.load(X_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    variance = tl.sum(x * x, axis=0) / N_COLS
    rsqrt = tl.math.rsqrt(variance + eps)
    y = x * rsqrt * w
    
    tl.store(Y_row_ptr + cols, y.to(tl.float16), mask=mask)

def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    original_shape = x.shape
    x_2d = x.view(-1, original_shape[-1])
    M_ROWS, N_COLS = x_2d.shape
    y_2d = torch.empty_like(x_2d)
    BLOCK_SIZE = triton.next_power_of_2(N_COLS)
    
    grid = lambda META: (M_ROWS,)
    _rmsnorm_fwd[grid](
        x_2d, y_2d, weight, x_2d.stride(0), y_2d.stride(0),
        N_COLS, eps, BLOCK_SIZE=BLOCK_SIZE
    )
    return y_2d.view(original_shape)