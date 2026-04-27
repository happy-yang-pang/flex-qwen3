import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B-Instruct")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--gen-len", type=int, default=64)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        padding_side="left",
        use_fast=False,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    enc = tokenizer([args.prompt], padding="max_length", truncation=True,
                    max_length=args.prompt_len, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.gen_len,
            do_sample=args.do_sample,
            temperature=args.temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
    print("Outputs:")
    print("-" * 70)
    print(f"0: {text}")
    print("-" * 70)


if __name__ == "__main__":
    main()
