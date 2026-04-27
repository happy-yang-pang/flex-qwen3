import os
import numpy as np


def main():
    debug_dir = r"D:\peku_task\ai\FlexLLMGen_stage1\debug"
    flex_path = os.path.join(debug_dir, "flex_output_logits.npz")
    hf_path = os.path.join(debug_dir, "hf_output_logits.npz")

    flex = np.load(flex_path)
    hf = np.load(hf_path)

    f = flex["last_token_logits"]
    h = hf["last_token_logits"]

    if f.shape != h.shape:
        print("shape mismatch:", f.shape, h.shape)
        return

    diff = np.abs(f - h)
    print("shape:", f.shape)
    print("max_abs_diff:", float(diff.max()))
    print("mean_abs_diff:", float(diff.mean()))

    k = 10
    f_top = np.argsort(-f[0])[:k]
    h_top = np.argsort(-h[0])[:k]

    print("flex top10:", f_top.tolist())
    print("hf   top10:", h_top.tolist())
    print("top1 same:", int(f_top[0]) == int(h_top[0]), "(flex=", int(f_top[0]), ", hf=", int(h_top[0]), ")")
    print("top10 overlap:", len(set(f_top.tolist()) & set(h_top.tolist())), "/10")


if __name__ == "__main__":
    main()
