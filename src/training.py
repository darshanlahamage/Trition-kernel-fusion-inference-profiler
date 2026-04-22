import torch
import torch.nn as nn
import torch.optim as optim
import time
from llama_block import CustomLlamaBlock

class TinyLlama(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_heads, num_layers, max_seq_length):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)

        self.layers = nn.ModuleList([
            CustomLlamaBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])

        self.norm = nn.RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        cos, sin = self._precompute_rope(max_seq_length, hidden_dim // num_heads)
        self.register_buffer("cos_cached", cos)
        self.register_buffer("sin_cached", sin)

    def _precompute_rope(self, seq_length, head_dim):
        theta = 10000.0 ** (-2 * torch.arange(0, head_dim // 2).float() / head_dim)
        seq = torch.arange(seq_length).float()
        freqs = torch.outer(seq, theta)
        return torch.cos(freqs).to(torch.float16), torch.sin(freqs).to(torch.float16)
        
    def forward(self, x):
        B, S = x.shape
        x = self.embed(x)
        cos, sin = self.cos_cached[:S], self.sin_cached[:S]
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
    
def train():

    VOCAB_SIZE = 10000
    HIDDEN_DIM = 256
    NUM_HEADS = 8
    NUM_LAYERS = 2
    MAX_SEQ_LEN = 2048
    BATCH_SIZE = 4
    SEQ_LEN = 1024
    STEPS = 50

    device = torch.device("cuda")
    model = TinyLlama(VOCAB_SIZE, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} Million")
    print(f"Batch Size: {BATCH_SIZE} | Sequence Length: {SEQ_LEN}")
    print("-" * 50)

    # Dummy Data Generator (To test raw hardware throughput without dataloader bottlenecks)
    def get_batch():
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        return x, y
    print("Running JIT Warmup (Compiling Triton Kernels)...")
    x, y = get_batch()
    logits = model(x)
    loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    print("Warmup Complete. Starting Profiling Loop.\n")

    total_tokens_per_step = BATCH_SIZE * SEQ_LEN

    for step in range(STEPS):
        x, y = get_batch()

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()

        logits = model(x)
        loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        step_time_ms = (t1 - t0) * 1000
        tps = total_tokens_per_step / (t1 - t0) # Tokens Per Second
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        if step % 10 == 0 or step == STEPS - 1:
            print(f"Step {step:02d} | Loss: {loss.item():.4f} | "
                  f"Latency: {step_time_ms:.1f}ms | "
                  f"Throughput: {tps:.0f} Tokens/sec | "
                  f"Peak VRAM: {peak_vram_mb:.1f} MB")

if __name__ == "__main__":
    train()






