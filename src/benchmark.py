import time
import torch
import torch.nn.functional as F
from generator import TinyLlama, Generator

def benchmark():
    VOCAB_SIZE = 10000
    HIDDEN_DIM = 256
    NUM_HEADS = 8
    NUM_LAYERS = 2
    MAX_SEQ_LEN = 2048
    BATCH_SIZE = 4
    PROMPT_LEN = 1024
    GENERATE_LEN = 50

    device = torch.device("cuda")
    model = TinyLlama(VOCAB_SIZE, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN).to(device).half()
    model.eval()

    prompt = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, PROMPT_LEN), device=device)
    generator = Generator(model)

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} Million")
    print(f"Batch Size: {BATCH_SIZE} | Prompt Length: {PROMPT_LEN} | Generating: {GENERATE_LEN} tokens")
    print("-" * 50)

    # Warmup
    print("Running Triton JIT Warmup (Compiling Kernels)...")
    _ = generator.generate(prompt[:, :128], 5)
    torch.cuda.synchronize()
    print("Warmup Complete.\n")

    torch.cuda.reset_peak_memory_stats()
    
    # Measure TTFT (Prefill)
    t0 = time.perf_counter()
    logits, kv_cache = model(prompt)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    ttft_ms = (t1 - t0) * 1000
    peak_vram_prefill = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Measure Decode Throughput
    next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
    
    t0 = time.perf_counter()
    for _ in range(GENERATE_LEN):
        logits, kv_cache = model(next_token, kv_cache=kv_cache)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    decode_time_s = t1 - t0
    total_tokens_generated = BATCH_SIZE * GENERATE_LEN
    tps = total_tokens_generated / decode_time_s
    peak_vram_decode = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"[Triton Inference] TTFT (Prefill): {ttft_ms:.1f} ms")
    print(f"[Triton Inference] Decoding Throughput: {tps:.0f} Tokens/sec")
    print(f"[Triton Inference] Peak VRAM: {peak_vram_decode:.1f} MB\n")

    # Baseline PyTorch Eager (Simulated without KV Cache to show naive eager memory usage)
    print("Running Baseline PyTorch Eager Memory Test (No KV-Cache)...")
    torch.cuda.reset_peak_memory_stats()
    full_seq = prompt
    for _ in range(5): # Just simulate a few autoregressive steps without cache
        # Simulate PyTorch SDPA 
        # (This is just an approximation of eager memory usage for comparison)
        q = torch.randn(BATCH_SIZE, NUM_HEADS, full_seq.shape[1], HIDDEN_DIM // NUM_HEADS, device=device, dtype=torch.float16)
        k = torch.randn(BATCH_SIZE, NUM_HEADS, full_seq.shape[1], HIDDEN_DIM // NUM_HEADS, device=device, dtype=torch.float16)
        v = torch.randn(BATCH_SIZE, NUM_HEADS, full_seq.shape[1], HIDDEN_DIM // NUM_HEADS, device=device, dtype=torch.float16)
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        full_seq = torch.cat([full_seq, torch.zeros(BATCH_SIZE, 1, dtype=torch.long, device=device)], dim=1)
        
    peak_vram_eager = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"[PyTorch Eager] Peak VRAM (Naive Autoregressive): {peak_vram_eager:.1f} MB")

if __name__ == "__main__":
    benchmark()
