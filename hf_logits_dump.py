import os
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def to_np(x):
    return x.detach().float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", type=str, default="请用三句话介绍一下北京的历史与文化。")
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--out-dir", type=str, default=os.path.join(os.path.dirname(__file__), "debug"))
    parser.add_argument("--local-files-only", action="store_true", default=True,
                        help="Load tokenizer/model from local cache only.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model_id = args.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side="left",
        use_fast=False,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    inputs = tokenizer([args.prompt], padding="max_length", max_length=args.prompt_len,
                       truncation=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_token_logits = out.logits[:, -1, :]
        topk = torch.topk(last_token_logits, k=min(10, last_token_logits.shape[-1]), dim=-1)

    out_path = os.path.join(args.out_dir, "hf_output_logits.npz")
    np.savez(
        out_path,
        last_token_logits=to_np(last_token_logits),
        topk_values=to_np(topk.values),
        topk_indices=to_np(topk.indices),
    )
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
