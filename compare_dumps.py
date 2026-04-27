import os
import numpy as np

d = r"D:\peku_task\ai\FlexLLMGen_stage1\debug"
flex_path = os.path.join(d, "flex_mha_decode_layer27.npz")
hf_path = os.path.join(d, "hf_mha_decode_layer27.npz")

f = np.load(flex_path)
h = np.load(hf_path)

print("flex keys:", f.files)
print("hf keys:", h.files)

common = sorted(set(f.files) & set(h.files))
print("common:", common)

for k in common:
    a = f[k]
    b = h[k]
    if a.shape != b.shape:
        print(f"{k}: shape mismatch {a.shape} vs {b.shape}")
        continue
    diff = np.abs(a - b)
    print(f"{k}: shape={a.shape}, max={diff.max():.6f}, mean={diff.mean():.6f}")