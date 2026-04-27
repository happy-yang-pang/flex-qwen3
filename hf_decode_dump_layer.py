import os
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def to_np(x):
    return x.detach().float().cpu().numpy()


def rms_norm(x, weight, eps=1e-6):
    x_fp32 = x.float()
    var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x_fp32 * torch.rsqrt(var + eps)
    return x_norm.to(dtype=x.dtype) * weight


def apply_rope_interleaved(x, positions, theta=1_000_000.0):
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
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--prompt", type=str, default="请用三句话介绍一下北京的历史与文化。")
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--out-dir", type=str, default=os.path.join(os.path.dirname(__file__), "debug"))
    args = parser.parse_args()

    model_id = "Qwen/Qwen3-0.6B"
    os.makedirs(args.out_dir, exist_ok=True)

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

    enc = tokenizer([args.prompt], padding="max_length", max_length=args.prompt_len,
                    truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # First decode token: run model once to get next token id
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past = out.past_key_values

        # Build decode-step inputs (tgt_s = 1)
        dec_input_ids = next_token
        dec_attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)], dim=1)

        # Get hidden input for target layer at decode step via hidden_states
        dec_out = model.model(
            input_ids=dec_input_ids,
            attention_mask=dec_attention_mask,
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_in = dec_out.hidden_states[args.layer]  # (b, 1, h)
        layer = model.model.layers[args.layer]
        hidden = layer.input_layernorm(hidden_in)

        q = layer.self_attn.q_proj(hidden)
        k = layer.self_attn.k_proj(hidden)
        v = layer.self_attn.v_proj(hidden)

        b, tgt_s, q_dim = q.shape
        _, _, k_dim = k.shape
        n_head = model.config.num_attention_heads
        n_kv_head = model.config.num_key_value_heads
        head_dim = q_dim // n_head

        q = q.view(b, tgt_s, n_head, head_dim)
        k = k.view(b, tgt_s, n_kv_head, k_dim // n_kv_head)
        v = v.view(b, tgt_s, n_kv_head, k_dim // n_kv_head)

        if hasattr(layer.self_attn, "q_norm") and layer.self_attn.q_norm is not None:
            q = rms_norm(q, layer.self_attn.q_norm.weight.view(1, 1, 1, -1))
        if hasattr(layer.self_attn, "k_norm") and layer.self_attn.k_norm is not None:
            k = rms_norm(k, layer.self_attn.k_norm.weight.view(1, 1, 1, -1))

        # decode position is current valid length - 1
        positions = dec_attention_mask.long().sum(dim=1, keepdim=True) - 1
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

        q2 = q.permute(0, 2, 1, 3).reshape(b * n_head, tgt_s, head_dim)
        k_new_attn = k_attn.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
        v_new_attn = v_attn.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)

        # compute one-step value using full cache from past
        pk, pv = past[args.layer]  # shape typically (b, n_kv_head, src_s, head_dim)
        # convert to (src_s, b*n_kv_head, head_dim)
        k_cache = pk.permute(2, 0, 1, 3).reshape(pk.shape[2], b * n_kv_head, head_dim)
        v_cache = pv.permute(2, 0, 1, 3).reshape(pv.shape[2], b * n_kv_head, head_dim)

        # append new kv only if cache does not already include current decode token
        k_new_cache = k.permute(1, 0, 2, 3).reshape(tgt_s, b * n_kv_head, head_dim)
        v_new_cache = v.permute(1, 0, 2, 3).reshape(tgt_s, b * n_kv_head, head_dim)
        if k_cache.shape[0] < dec_attention_mask.shape[1]:
            k_full = torch.cat([k_cache, k_new_cache], dim=0)
            v_full = torch.cat([v_cache, v_new_cache], dim=0)
        else:
            # Some model variants already include the current token in returned cache.
            k_full = k_cache[:dec_attention_mask.shape[1]]
            v_full = v_cache[:dec_attention_mask.shape[1]]
        src_s = k_full.shape[0]

        if n_kv_head < n_head:
            rep = n_head // n_kv_head
            k_full_attn = k_full.view(src_s, b, n_kv_head, head_dim).repeat_interleave(rep, dim=2).reshape(src_s, b * n_head, head_dim)
            v_full_attn = v_full.view(src_s, b, n_kv_head, head_dim).repeat_interleave(rep, dim=2).reshape(src_s, b * n_head, head_dim)
        else:
            k_full_attn = k_full
            v_full_attn = v_full

        k_mat = k_full_attn.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)
        v_mat = v_full_attn.permute(1, 0, 2).reshape(b * n_head, src_s, head_dim)

        attn_scores = torch.bmm(q2, k_mat) * (head_dim ** -0.5)
        mask = dec_attention_mask.view(b, 1, 1, src_s).bool()
        attn_scores = attn_scores.view(b, n_head, 1, src_s)
        attn_scores = torch.where(mask, attn_scores, torch.tensor(-1e4, device=device, dtype=attn_scores.dtype))
        attn_probs = torch.softmax(attn_scores.view(b * n_head, 1, src_s), dim=2)
        value = torch.bmm(attn_probs, v_mat).view(b, n_head, 1, head_dim)
        value = value.transpose(1, 2).reshape(b, 1, n_head * head_dim)
        value = layer.self_attn.o_proj(value)

    out_path = os.path.join(args.out_dir, f"hf_mha_decode_layer{args.layer}.npz")
    np.savez(
        out_path,
        hidden_input=to_np(hidden_in),
        q=to_np(q2),
        k_new_attn=to_np(k_new_attn),
        v_new_attn=to_np(v_new_attn),
        value_before_residual=to_np(value),
        attention_mask=to_np(dec_attention_mask),
        k_cache_full=to_np(k_full),
        v_cache_full=to_np(v_full),
    )
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
