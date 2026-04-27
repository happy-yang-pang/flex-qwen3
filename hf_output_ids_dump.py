import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--gen-len", type=int, default=64)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out-dir", type=str, default=os.path.join(os.path.dirname(__file__), "debug"))
    parser.add_argument("--local-files-only", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

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
        # Manual decoding loop to avoid model.generation_config side effects
        out_ids = input_ids.clone()
        cur_input = input_ids
        cur_mask = attention_mask
        past = None
        for _ in range(args.gen_len):
            out = model(
                input_ids=cur_input,
                attention_mask=cur_mask,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            logits = out.logits[:, -1, :]
            if args.do_sample:
                probs = torch.softmax(logits / max(args.temperature, 1e-5), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            out_ids = torch.cat([out_ids, next_token], dim=1)
            past = out.past_key_values
            cur_input = next_token
            cur_mask = torch.cat([
                cur_mask,
                torch.ones((cur_mask.shape[0], 1), dtype=cur_mask.dtype, device=cur_mask.device)
            ], dim=1)

            if tokenizer.eos_token_id is not None and bool((next_token == tokenizer.eos_token_id).all()):
                break

    out_np = out_ids.detach().cpu().numpy()
    np.save(os.path.join(args.out_dir, "hf_output_ids.npy"), out_np)
    np.save(os.path.join(args.out_dir, "hf_prompt_ids.npy"), input_ids.detach().cpu().numpy())
    print("Saved:", os.path.join(args.out_dir, "hf_output_ids.npy"))


if __name__ == "__main__":
    main()
