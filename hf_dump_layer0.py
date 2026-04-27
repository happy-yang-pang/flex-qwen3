import os
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def to_np(x):
    return x.detach().float().cpu().numpy()


def rms_norm(x, weight, eps=1e-6):
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


def apply_rope_interleaved(x, positions, theta=1_000_000.0):
    # x: (b, s, n_head, head_dim)
    head_dim = x.shape[-1]
    half_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half_dim, device=x.device, dtype=torch.float32) / half_dim))
    freqs = torch.einsum("bs,d->bsd", positions.float(), inv_freq)
    cos = freqs.cos().unsqueeze(2).to(dtype=x.dtype)
    sin = freqs.sin().unsqueeze(2).to(dtype=x.dtype)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos

    out = torch.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--prompt", type=str, default="请用三句话介绍一下北京的历史与文化。")
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--out-dir", type=str, default=os.path.join(os.path.dirname(__file__), "debug"))
    args = parser.parse_args()

    model_id = "Qwen/Qwen3-0.6B"
    prompt = args.prompt
    prompt_len = args.prompt_len
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", use_fast=False)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    inputs = tokenizer([prompt], padding="max_length", max_length=prompt_len,
                       truncation=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        # Get the true input hidden states of the target layer from HF forward pass.
        out = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        # hidden_states[0] is embedding output; hidden_states[i] is input to layer i
        hidden = out.hidden_states[args.layer]

        layer = model.model.layers[args.layer]
        hidden_norm = layer.input_layernorm(hidden)

        q = layer.self_attn.q_proj(hidden_norm)
        k = layer.self_attn.k_proj(hidden_norm)
        v = layer.self_attn.v_proj(hidden_norm)

        b, s, q_dim = q.shape
        _, _, k_dim = k.shape
        n_head = model.config.num_attention_heads
        n_kv_head = model.config.num_key_value_heads
        # Qwen3 may have q_proj dim != hidden_size (e.g., 2048 for 0.6B)
        head_dim = q_dim // n_head

        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, s, n_kv_head, k_dim // n_kv_head)
        v = v.view(b, s, n_kv_head, k_dim // n_kv_head)

        if hasattr(layer.self_attn, "q_norm") and layer.self_attn.q_norm is not None:
            q = rms_norm(q, layer.self_attn.q_norm.weight.view(1, 1, 1, -1))
        if hasattr(layer.self_attn, "k_norm") and layer.self_attn.k_norm is not None:
            k = rms_norm(k, layer.self_attn.k_norm.weight.view(1, 1, 1, -1))

        positions = attention_mask.long().cumsum(dim=1) - 1
        positions = torch.clamp(positions, min=0)
        q = apply_rope_interleaved(q, positions)
        k = apply_rope_interleaved(k, positions)

        if n_kv_head < n_head:
            rep = n_head // n_kv_head
            k_attn = k.repeat_interleave(rep, dim=2)
            v_attn = v.repeat_interleave(rep, dim=2)
        else:
            k_attn = k
            v_attn = v

        q2 = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        k2 = k_attn.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, s)
        v2 = v_attn.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        attn_scores = torch.bmm(q2, k2) * (head_dim ** -0.5)
        attn_scores = attn_scores.view(b, n_head, s, s)

        causal = torch.tril(torch.ones((s, s), device=device, dtype=torch.bool)).view(1, 1, s, s)
        mask = attention_mask.view(b, 1, 1, s).bool() & causal
        attn_scores = torch.where(mask, attn_scores, torch.tensor(-1e4, device=device, dtype=attn_scores.dtype))
        attn_probs = torch.softmax(attn_scores.view(b * n_head, s, s), dim=2)
        value = torch.bmm(attn_probs, v2).view(b, n_head, s, head_dim)
        qkv_dim = n_head * head_dim
        value = value.transpose(1, 2).reshape(b, s, qkv_dim)
        value = layer.self_attn.o_proj(value)

    np.savez(
        os.path.join(out_dir, f"hf_mha_prefill_layer{args.layer}.npz"),
        hidden_input=to_np(hidden),
        hidden=to_np(hidden_norm),
        q=to_np(q2),
        k_attn=to_np(k2),
        v_attn=to_np(v2),
        attn_scores=to_np(attn_scores),
        attn_probs=to_np(attn_probs.view(b, n_head, s, s)),
        value_before_residual=to_np(value),
        attention_mask=to_np(attention_mask),
        input_ids=to_np(input_ids),
    )
    print("Saved:", os.path.join(out_dir, f"hf_mha_prefill_layer{args.layer}.npz"))
    print("Layer:", args.layer)
    print("Prompt:", prompt)


if __name__ == "__main__":
    main()
