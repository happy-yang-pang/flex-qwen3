import os
import numpy as np
from transformers import AutoTokenizer


def main():
    debug_dir = r"D:\peku_task\ai\FlexLLMGen_stage1\debug"
    flex_ids = np.load(os.path.join(debug_dir, "flex_output_ids.npy"))
    hf_ids = np.load(os.path.join(debug_dir, "hf_output_ids.npy"))
    flex_prompt = np.load(os.path.join(debug_dir, "flex_prompt_ids.npy"))
    hf_prompt = np.load(os.path.join(debug_dir, "hf_prompt_ids.npy"))

    if flex_ids.shape != hf_ids.shape:
        print("shape mismatch:", flex_ids.shape, hf_ids.shape)
        return

    model_id = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", use_fast=False, local_files_only=True)

    # detect prompt length from saved prompt ids
    prompt_len = flex_prompt.shape[1]
    print("Total shape:", flex_ids.shape)
    print("Prompt shape flex/hf:", flex_prompt.shape, hf_prompt.shape)
    print("Assumed prompt_len:", prompt_len)

    if flex_prompt.shape != hf_prompt.shape:
        print("prompt shape mismatch:", flex_prompt.shape, hf_prompt.shape)
        return

    prompt_diff = np.where(flex_prompt[0] != hf_prompt[0])[0]
    print("prompt token mismatches:", len(prompt_diff))
    if len(prompt_diff) > 0:
        print("first 20 prompt mismatch positions:", prompt_diff[:20].tolist())

    first_mismatch = None
    total_steps = flex_ids.shape[1] - prompt_len
    for t in range(total_steps):
        pos = prompt_len + t
        f = int(flex_ids[0, pos])
        h = int(hf_ids[0, pos])
        if f != h and first_mismatch is None:
            first_mismatch = t
        if t < 16:
            print(f"step {t:02d}: flex={f} ({repr(tokenizer.decode([f]))}) | hf={h} ({repr(tokenizer.decode([h]))})")

    print("first mismatch step:", first_mismatch)


if __name__ == "__main__":
    main()
