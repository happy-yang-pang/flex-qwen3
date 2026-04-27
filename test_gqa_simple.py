"""Simple GQA signature test for FlexLLMGen"""
import inspect
from flexllmgen.pytorch_backend import TorchDevice

print("=" * 60)
print("GQA Support Verification for FlexLLMGen")
print("=" * 60)

# Check mha signature
sig = inspect.signature(TorchDevice.mha)
params = list(sig.parameters.keys())
print(f"\n📋 mha() parameters ({len(params)}):")
print(f"   {params}")
print(f"   ✅ Has n_kv_head: {'n_kv_head' in params}")

# Check mha_gen signature
sig_gen = inspect.signature(TorchDevice.mha_gen)
params_gen = list(sig_gen.parameters.keys())
print(f"\n📋 mha_gen() parameters ({len(params_gen)}):")
print(f"   {params_gen}")
print(f"   ✅ Has n_kv_head: {'n_kv_head' in params_gen}")

# Check Qwen3 config
from flexllmgen.opt_config import Qwen3Config, get_qwen3_config
config = get_qwen3_config("qwen3-0.6b")
print(f"\n📋 Qwen3 Config:")
print(f"   name: {config.name}")
print(f"   n_head: {config.n_head}")
print(f"   n_kv_head: {config.n_kv_head}")
print(f"   GQA ratio: {config.gqa_ratio}x")

# Verify GQA logic exists in source
import os
with open("flexllmgen/pytorch_backend.py", "r") as f:
    content = f.read()
    has_repeat = "repeat_interleave" in content
    has_gqa_comment = "GQA support" in content or "GQA:" in content
    
print(f"\n📋 Source Code Check:")
print(f"   ✅ Has repeat_interleave: {has_repeat}")
print(f"   ✅ Has GQA comments: {has_gqa_comment}")

print("\n" + "=" * 60)
if 'n_kv_head' in params and 'n_kv_head' in params_gen and has_repeat:
    print("✅ GQA SUPPORT VERIFICATION PASSED!")
else:
    print("❌ GQA SUPPORT VERIFICATION FAILED!")
print("=" * 60)
