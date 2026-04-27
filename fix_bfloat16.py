"""Fix bfloat16 conversion issue"""

with open("flexllmgen/opt_config.py", "r") as f:
    content = f.read()

# 找到旧的 convert_qwen3_weights_to_np 函数并替换
old_func_start = "def convert_qwen3_weights_to_np(hf_folder, output_path, model_name):"
old_func_end = '    print(f"✅ Qwen3 weights converted to {np_output_dir}")'

# 找到函数起始和结束位置
start_idx = content.find(old_func_start)
if start_idx != -1:
    # 找到函数结束（下一个 def 或文件末尾）
    end_idx = content.find('\n\ndef ', start_idx)
    if end_idx == -1:
        end_idx = len(content)
    else:
        # 包含最后一个 print 语句
        end_idx = content.find(old_func_end, start_idx) + len(old_func_end)
    
    # 新的转换函数（使用 torch 作为中间格式）
    new_func = '''def convert_qwen3_weights_to_np(hf_folder, output_path, model_name):
    """Convert Qwen3 safetensors weights to numpy format."""
    import os
    import numpy as np
    import torch
    from safetensors.torch import load_file
    
    print(f"Converting Qwen3 weights from {hf_folder} to numpy format...")
    
    # Create output directory
    np_output_dir = os.path.join(output_path, model_name + "-np")
    os.makedirs(np_output_dir, exist_ok=True)
    
    # Find safetensors file
    safetensors_file = None
    for f in os.listdir(hf_folder):
        if f.endswith(".safetensors"):
            safetensors_file = os.path.join(hf_folder, f)
            break
    
    if safetensors_file is None:
        raise FileNotFoundError(f"No safetensors file found in {hf_folder}")
    
    print(f"Found safetensors file: {safetensors_file}")
    
    # Load all tensors using torch (supports bfloat16)
    print("Loading tensors with torch...")
    tensors = load_file(safetensors_file, device="cpu")
    print(f"Found {len(tensors)} tensors in safetensors file")
    
    # Map Qwen3 weight names to OPT-style names
    # Qwen3-0.6B has 28 layers
    num_layers = 28
    
    weight_mapping = {
        "model.embed_tokens.weight": "decoder.embed_tokens.weight",
        "model.norm.weight": "decoder.final_layer_norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    
    # Add layer mappings
    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        weight_mapping.update({
            f"{prefix}.input_layernorm.weight": f"decoder.layers.{i}.self_attn_layer_norm.weight",
            f"{prefix}.self_attn.q_proj.weight": f"decoder.layers.{i}.self_attn.q_proj.weight",
            f"{prefix}.self_attn.k_proj.weight": f"decoder.layers.{i}.self_attn.k_proj.weight",
            f"{prefix}.self_attn.v_proj.weight": f"decoder.layers.{i}.self_attn.v_proj.weight",
            f"{prefix}.self_attn.o_proj.weight": f"decoder.layers.{i}.self_attn.out_proj.weight",
            f"{prefix}.post_attention_layernorm.weight": f"decoder.layers.{i}.final_layer_norm.weight",
            f"{prefix}.mlp.gate_proj.weight": f"decoder.layers.{i}.fc1.weight",
            f"{prefix}.mlp.up_proj.weight": f"decoder.layers.{i}.fc2.weight",
            f"{prefix}.mlp.down_proj.weight": f"decoder.layers.{i}.fc3.weight",
        })
    
    # Convert and save each tensor
    saved_count = 0
    for key, tensor in tensors.items():
        # Convert bfloat16 to float32 for numpy compatibility
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
        elif tensor.dtype == torch.float16:
            tensor = tensor.to(torch.float32)
        
        # Map to OPT-style name
        opt_name = weight_mapping.get(key, key)
        
        # Convert to numpy
        np_tensor = tensor.numpy()
        
        # Save as numpy file
        np_path = os.path.join(np_output_dir, opt_name + ".npy")
        os.makedirs(os.path.dirname(np_path), exist_ok=True)
        np.save(np_path, np_tensor)
        saved_count += 1
        
        if saved_count % 50 == 0:
            print(f"  Progress: {saved_count}/{len(tensors)} tensors saved...")
    
    print(f"✅ Qwen3 weights converted: {saved_count} tensors saved to {np_output_dir}")'''
    
    # 替换函数
    content = content[:start_idx] + new_func + content[end_idx:]
    
    with open("flexllmgen/opt_config.py", "w") as f:
        f.write(content)
    
    print("✅ 代码已更新！")
else:
    print("❌ 未找到目标函数")
