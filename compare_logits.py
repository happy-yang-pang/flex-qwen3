import os
import numpy as np


def main():
    debug_dir = r"D:\peku_task\ai\FlexLLMGen_stage1\debug"
    flex_path = os.path.join(debug_dir, "flex_output_logits.npz")
    hf_path = os.path.join(debug_dir, "hf_output_logits.npz")

    flex = np.load(flex_path)
    hf = np.load(hf_path)

    print("flex keys:", flex.files)
    print("hf keys:", hf.files)

    f_logits = flex["last_token_logits"]
    h_logits = hf["last_token_logits"]

    if f_logits.shape != h_logits.shape:
        print(f"shape mismatch: flex={f_logits.shape}, hf={h_logits.shape}")
        return

    diff = np.abs(f_logits - h_logits)
    print(f"logits shape: {f_logits.shape}")
    print(f"logits max abs diff: {diff.max():.6f}")
    print(f"logits mean abs diff: {diff.mean():.6f}")

    k = min(10, f_logits.shape[-1])
    f_topk_idx = np.argsort(-f_logits, axis=-1)[:, :k]
    h_topk_idx = np.argsort(-h_logits, axis=-1)[:, :k]

    print("\nTop-10 token ids (first sample):")
    print("flex:", f_topk_idx[0].tolist())
    print("hf  :", h_topk_idx[0].tolist())

    overlap = len(set(f_topk_idx[0].tolist()) & set(h_topk_idx[0].tolist()))
    print(f"top-10 overlap: {overlap}/10")

    f_top1 = int(f_topk_idx[0, 0])
    h_top1 = int(h_topk_idx[0, 0])
    print(f"top-1 same: {f_top1 == h_top1} (flex={f_top1}, hf={h_top1})")


if __name__ == "__main__":
    main()
