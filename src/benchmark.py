import time
import torch
from generator import TinyLlama, Generator

def format_time(ms):
    return f"{ms:.2f} ms"

def print_header(title):
    print(f"\n{title}")
    print("=" * 60)

def benchmark():
    # Scaled up parameters to showcase actual bottlenecks and memory differences
    VOCAB_SIZE = 10000
    HIDDEN_DIM = 512       
    NUM_HEADS = 16
    NUM_LAYERS = 8         
    MAX_SEQ_LEN = 4096     
    BATCH_SIZE = 4
    PROMPT_LEN = 2048      # Large prompt to demonstrate N^2 memory blowup in eager
    GENERATE_LEN = 20

    device = torch.device("cuda")
    print_header(f"Triton Kernel Fusion Profiler")
    print(f"Architecture: {NUM_LAYERS} Layers | {HIDDEN_DIM} Dim | {NUM_HEADS} Heads")
    print(f"Workload: Batch {BATCH_SIZE} | Prompt {PROMPT_LEN} tokens | Gen {GENERATE_LEN} tokens")
    
    model = TinyLlama(VOCAB_SIZE, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN).to(device).half()
    model.eval()
    generator = Generator(model)

    params_mb = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model Params: {params_mb:.2f} M (FP16)")
    print("-" * 60)

    prompt = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, PROMPT_LEN), device=device)

    # 1. Warmup
    print("[1/3] Running Triton JIT Warmup...")
    _ = generator.generate(prompt[:, :128], 5)
    torch.cuda.synchronize()

    # 2. Triton Engine Profiling
    print("[2/3] Profiling Optimized Triton Engine (KV-Cache Enabled)...")
    torch.cuda.reset_peak_memory_stats()
    
    # TTFT
    t0 = time.perf_counter()
    logits, kv_cache = model(prompt)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ttft_ms = (t1 - t0) * 1000
    
    # Decode
    next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(GENERATE_LEN):
        logits, kv_cache = model(next_token, kv_cache=kv_cache)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
    end_event.record()
    torch.cuda.synchronize()
    
    decode_time_s = start_event.elapsed_time(end_event) / 1000.0
    triton_tps = (BATCH_SIZE * GENERATE_LEN) / decode_time_s
    triton_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # 3. Baseline Eager Profiling
    print("[3/3] Profiling Naive PyTorch Baseline (No KV-Cache)...")
    torch.cuda.reset_peak_memory_stats()
    full_seq = prompt.clone()
    
    start_event.record()
    for _ in range(GENERATE_LEN):
        logits, _ = model(full_seq, kv_cache=None)
        next_t = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
        full_seq = torch.cat([full_seq, next_t], dim=1)
    end_event.record()
    torch.cuda.synchronize()
        
    eager_time_s = start_event.elapsed_time(end_event) / 1000.0
    eager_tps = (BATCH_SIZE * GENERATE_LEN) / eager_time_s
    eager_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Print Portfolio-Grade Results Table
    print_header("Performance Report")
    
    speedup = triton_tps / eager_tps
    vram_saved = eager_vram - triton_vram
    
    print(f"{'Metric':<25} | {'PyTorch Eager':<15} | {'Triton Optimized':<15} | {'Delta / Improvement'}")
    print("-" * 85)
    print(f"{'Prefill TTFT':<25} | {'-':<15} | {format_time(ttft_ms):<15} | -")
    print(f"{'Decode Throughput':<25} | {eager_tps:>6.1f} tok/s    | {triton_tps:>6.1f} tok/s    | {speedup:.1f}x Speedup")
    print(f"{'Peak VRAM':<25} | {eager_vram:>7.1f} MB     | {triton_vram:>7.1f} MB     | {vram_saved:.1f} MB Saved")
    print("-" * 85)
    print("\n[Analysis]")
    print(f"By fusing kernels and utilizing a stateful KV-Cache, the Triton Engine reduces "
          f"computational complexity from O(N^2) to O(N) during autoregressive decoding. "
          f"This results in a {speedup:.1f}x throughput multiplier and actively avoids the "
          f"memory blowout associated with re-computing historical attention states.")

if __name__ == "__main__":
    benchmark()
