import time
import torch
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity
from generator import TinyLlama, Generator
from torch_model import TorchTinyLlama

def format_time(ms):
    return f"{ms:.2f} ms"

def print_header(title):
    print(f"\n{title}")
    print("=" * 80)

def plot_results(tps_dict, vram_dict):
    labels = list(tps_dict.keys())
    tps = list(tps_dict.values())
    vram = list(vram_dict.values())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Throughput Chart
    bars1 = ax1.bar(labels, tps, color=['#ff6b6b', '#4ecdc4', '#ffe66d'])
    ax1.set_title('Decoding Throughput (Tokens/sec)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Tokens per Second')
    ax1.bar_label(bars1, fmt='%.1f')

    # VRAM Chart
    bars2 = ax2.bar(labels, vram, color=['#ff6b6b', '#4ecdc4', '#ffe66d'])
    ax2.set_title('Peak VRAM Footprint (MB)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Megabytes (MB)')
    ax2.bar_label(bars2, fmt='%.1f')

    plt.tight_layout()
    plt.savefig('profiling_results.png', dpi=300)
    print("\n[Visualizations] Saved side-by-side charts to 'profiling_results.png'")

def benchmark():
    VOCAB_SIZE = 10000
    HIDDEN_DIM = 512       
    NUM_HEADS = 16
    NUM_LAYERS = 8         
    MAX_SEQ_LEN = 4096     
    BATCH_SIZE = 4
    PROMPT_LEN = 2048
    GENERATE_LEN = 20

    device = torch.device("cuda")
    print_header(f"Advanced Profiling Suite: PyTorch vs Triton")
    print(f"Architecture: {NUM_LAYERS} Layers | {HIDDEN_DIM} Dim | {NUM_HEADS} Heads")
    print(f"Workload: Batch {BATCH_SIZE} | Prompt {PROMPT_LEN} tokens | Gen {GENERATE_LEN} tokens")
    
    # Initialize both models
    triton_model = TinyLlama(VOCAB_SIZE, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN).to(device).half()
    triton_model.eval()
    
    torch_model = TorchTinyLlama(VOCAB_SIZE, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN).to(device).half()
    # Copy weights for absolute fairness (same parameters)
    torch_model.load_state_dict(triton_model.state_dict(), strict=False)
    torch_model.eval()
    
    triton_generator = Generator(triton_model)

    prompt = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, PROMPT_LEN), device=device)

    results_tps = {}
    results_vram = {}

    # ---------------------------------------------------------
    # 1. Triton Engine (Custom Kernels + KV Cache)
    # ---------------------------------------------------------
    print("\n[1/3] Profiling Triton Engine (KV-Cache) + Nsight Profiler...")
    _ = triton_generator.generate(prompt[:, :128], 5) # Warmup
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    t0 = time.perf_counter()
    logits, kv_cache = triton_model(prompt)
    torch.cuda.synchronize()
    ttft_triton = (time.perf_counter() - t0) * 1000
    
    next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Run torch.profiler
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        start_event.record()
        for _ in range(GENERATE_LEN):
            logits, kv_cache = triton_model(next_token, kv_cache=kv_cache)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
        end_event.record()
        torch.cuda.synchronize()

    prof.export_chrome_trace("trace.json")
    print("      Exported execution trace to 'trace.json'")
    
    triton_time_s = start_event.elapsed_time(end_event) / 1000.0
    results_tps["Triton\n(KV-Cache)"] = (BATCH_SIZE * GENERATE_LEN) / triton_time_s
    results_vram["Triton\n(KV-Cache)"] = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # ---------------------------------------------------------
    # 2. PyTorch Optimized (Native SDPA FlashAttention + KV Cache)
    # ---------------------------------------------------------
    print("[2/3] Profiling PyTorch Optimized (Native SDPA FlashAttention + KV Cache)...")
    torch.cuda.reset_peak_memory_stats()
    
    t0 = time.perf_counter()
    logits, kv_cache = torch_model(prompt)
    torch.cuda.synchronize()
    ttft_torch = (time.perf_counter() - t0) * 1000
    
    next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
    
    start_event.record()
    for _ in range(GENERATE_LEN):
        logits, kv_cache = torch_model(next_token, kv_cache=kv_cache)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
    end_event.record()
    torch.cuda.synchronize()
    
    torch_time_s = start_event.elapsed_time(end_event) / 1000.0
    results_tps["PyTorch\n(KV-Cache)"] = (BATCH_SIZE * GENERATE_LEN) / torch_time_s
    results_vram["PyTorch\n(KV-Cache)"] = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # ---------------------------------------------------------
    # 3. PyTorch Naive (No KV Cache)
    # ---------------------------------------------------------
    print("[3/3] Profiling PyTorch Naive (No KV Cache)...")
    torch.cuda.reset_peak_memory_stats()
    full_seq = prompt.clone()
    
    t0 = time.perf_counter()
    logits, _ = torch_model(full_seq, kv_cache=None)
    torch.cuda.synchronize()
    ttft_naive = (time.perf_counter() - t0) * 1000
    
    start_event.record()
    for _ in range(GENERATE_LEN):
        logits, _ = torch_model(full_seq, kv_cache=None)
        next_t = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
        full_seq = torch.cat([full_seq, next_t], dim=1)
    end_event.record()
    torch.cuda.synchronize()
    
    naive_time_s = start_event.elapsed_time(end_event) / 1000.0
    results_tps["PyTorch Naive\n(O(N^2))"] = (BATCH_SIZE * GENERATE_LEN) / naive_time_s
    results_vram["PyTorch Naive\n(O(N^2))"] = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Output Formatting
    print_header("Performance Report")
    print(f"{'Metric':<25} | {'PyTorch Naive':<15} | {'PyTorch KV-Cache':<17} | {'Triton KV-Cache'}")
    print("-" * 80)
    print(f"{'Prefill TTFT':<25} | {format_time(ttft_naive):<15} | {format_time(ttft_torch):<17} | {format_time(ttft_triton)}")
    print(f"{'Decode Throughput':<25} | {results_tps['PyTorch Naive\n(O(N^2))']:>6.1f} tok/s    | {results_tps['PyTorch\n(KV-Cache)']:>7.1f} tok/s      | {results_tps['Triton\n(KV-Cache)']:>7.1f} tok/s")
    print(f"{'Peak VRAM':<25} | {results_vram['PyTorch Naive\n(O(N^2))']:>7.1f} MB     | {results_vram['PyTorch\n(KV-Cache)']:>8.1f} MB       | {results_vram['Triton\n(KV-Cache)']:>8.1f} MB")
    print("-" * 80)
    
    print_header("Profiler Micro-Kernel Averages (Triton KV-Cache Decode Loop)")
    print("These are the CUDA level execution times for the Triton Decode iteration:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Generate Visualization
    plot_results(results_tps, results_vram)

if __name__ == "__main__":
    benchmark()
