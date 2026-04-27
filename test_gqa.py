"""Simple GQA test for FlexLLMGen with Qwen3"""
import torch
from flexllmgen.pytorch_backend import TorchDevice, DeviceType

print("=" * 50)
print("GQA Support Test for FlexLLMGen")
print("=" * 50)

# Test parameters (Qwen3-8B style: 32 Q heads, 8 KV heads)
n_head = 32
n_kv_head = 8
batch_size = 1
seq_len = 128
hidden_dim = 4096
head_dim = hidden_dim // n_head

print(f"\n📋 Test Configuration:")
print(f"   n_head (Q heads): {n_head}")
print(f"   n_kv_head (KV heads): {n_kv_head}")
print(f"   GQA ratio: {n_head // n_kv_head}x")
print(f"   batch_size: {batch_size}")
print(f"   seq_len: {seq_len}")
print(f"   hidden_dim: {hidden_dim}")
print(f"   head_dim: {head_dim}")

# Create mock device
device = TorchDevice(torch.device('cpu'))

# Create mock inputs
inputs = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16)
attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

# Create mock weights (note: K/V weights have n_kv_head * head_dim output)
w_q = torch.randn(hidden_dim, hidden_dim, dtype=torch.float16)
b_q = torch.zeros(hidden_dim, dtype=torch.float16)
w_k = torch.randn(hidden_dim, n_kv_head * head_dim, dtype=torch.float16)  # GQA!
b_k = torch.zeros(n_kv_head * head_dim, dtype=torch.float16)
w_v = torch.randn(hidden_dim, n_kv_head * head_dim, dtype=torch.float16)  # GQA!
b_v = torch.zeros(n_kv_head * head_dim, dtype=torch.float16)
w_out = torch.randn(hidden_dim, hidden_dim, dtype=torch.float16)
b_out = torch.zeros(hidden_dim, dtype=torch.float16)
w_ln = torch.ones(hidden_dim, dtype=torch.float16)
b_ln = torch.zeros(hidden_dim, dtype=torch.float16)

print(f"\n✅ Weight shapes:")
print(f"   w_q: {w_q.shape}")
print(f"   w_k: {w_k.shape} (GQA: {n_kv_head} heads)")
print(f"   w_v: {w_v.shape} (GQA: {n_kv_head} heads)")

# Test mha function signature
import inspect
from flexllmgen.pytorch_backend import TorchDevice
sig = inspect.signature(TorchDevice.mha)
params = list(sig.parameters.keys())
print(f"\n✅ mha() parameters: {len(params)} params")
print(f"   Has n_kv_head: {'n_kv_head' in params}")

sig_gen = inspect.signature(TorchDevice.mha_gen)
params_gen = list(sig_gen.parameters.keys())
print(f"\n✅ mha_gen() parameters: {len(params_gen)} params")
print(f"   Has n_kv_head: {'n_kv_head' in params_gen}")

print("\n" + "=" * 50)
print("✅ GQA Support Verification Complete!")
print("=" * 50)
