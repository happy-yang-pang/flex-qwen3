"""Fix Qwen3 weight download and conversion"""

with open("flexllmgen/opt_config.py", "r") as f:
    content = f.read()

# 1. 修改 download_opt_weights 函数，支持 safetensors 下载
old_qwen3_code = '''    # Handle Qwen3 models
    if "qwen3" in model_name.lower():
        hf_model_name = "Qwen/Qwen3-0.6B"
        print(f"Loading Qwen3 model from {hf_model_name}")
    elif "opt" in model_name:
        hf_model_name = "facebook/" + model_name
    elif "galactica" in model_name:
        hf_model_name = "facebook/" + model_name

    folder = snapshot_download(hf_model_name, allow_patterns="*.bin")'''

new_qwen3_code = '''    # Handle Qwen3 models
    if "qwen3" in model_name.lower():
        hf_model_name = "Qwen/Qwen3-0.6B"
        print(f"Loading Qwen3 model from {hf_model_name}")
        # Qwen3 uses safetensors format
        folder = snapshot_download(hf_model_name, allow_patterns="*.safetensors")
        # Convert safetensors to numpy for Qwen3
        convert_qwen3_weights_to_np(folder, path, model_name)
        return
    elif "opt" in model_name:
        hf_model_name = "facebook/" + model_name
    elif "galactica" in model_name:
        hf_model_name = "facebook/" + model_name

    folder = snapshot_download(hf_model_name, allow_patterns="*.bin")'''

content = content.replace(old_qwen3_code, new_qwen3_code)
# 2. 在文件末尾添加 Qwen3 权重转换函数
qwen3_convert_func = '''

def convert_qwen3_weights_to_np(hf_folder, output_path, model_name):
    """Convert Qwen3 safetensors weights to numpy format."""
    import os
    import numpy as np
    from safetensors import safe_open
    
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
    
    # Map Qwen3 weight names to OPT-style names
    # This is a simplified mapping - may need adjustment based on actual Qwen3 architecture
    weight_mapping = {
        "model.embed_tokens.weight": "decoder.embed_tokens.weight",
        "model.norm.weight": "decoder.final_layer_norm.weight",
        "model.norm.bias": "decoder.final_layer_norm.bias",
        "lm_head.weight": "lm_head.weight",
    }
    
    # Add layer mappings
    for i in range(28):  # Qwen3-0.6B has 28 layers
        prefix = f"model.layers.{i}"
        weight_mapping.update({
            f"{prefix}.input_layernorm.weight": f"decoder.layers.{i}.self_attn_layer_norm.weight",
            f"{prefix}.input_layernorm.bias": f"decoder.layers.{i}.self_attn_layer_norm.bias",
            f"{prefix}.self_attn.q_proj.weight": f"decoder.layers.{i}.self_attn.q_proj.weight",
            f"{prefix}.self_attn.k_proj.weight": f"decoder.layers.{i}.self_attn.k_proj.weight",
            f"{prefix}.self_attn.v_proj.weight": f"decoder.layers.{i}.self_attn.v_proj.weight",
            f"{prefix}.self_attn.o_proj.weight": f"decoder.layers.{i}.self_attn.out_proj.weight",
            f"{prefix}.post_attention_layernorm.weight": f"decoder.layers.{i}.final_layer_norm.weight",
            f"{prefix}.post_attention_layernorm.bias": f"decoder.layers.{i}.final_layer_norm.bias",
            f"{prefix}.mlp.gate_proj.weight": f"decoder.layers.{i}.fc1.weight",
            f"{prefix}.mlp.up_proj.weight": f"decoder.layers.{i}.fc2.weight",
            f"{prefix}.mlp.down_proj.weight": f"decoder.layers.{i}.fc3.weight",
        })
    
    with safe_open(safetensors_file, framework="np") as f:
        keys = f.keys()
        print(f"Found {len(keys)} tensors in safetensors file")
        
        for key in keys:
            tensor = f.get_tensor(key)
            
            # Map to OPT-style name
            opt_name = weight_mapping.get(key, key)
            
            # Save as numpy file
            np_path = os.path.join(np_output_dir, opt_name + ".npy")
            os.makedirs(os.path.dirname(np_path), exist_ok=True)
            np.save(np_path, tensor)
            print(f"  Saved: {opt_name}.npy (shape: {tensor.shape})")
    
    print(f"✅ Qwen3 weights converted to {np_output_dir}")
'''
# 检查是否已存在该函数
if "def convert_qwen3_weights_to_np" not in content:
    content = content.rstrip() + qwen3_convert_func

with open("flexllmgen/opt_config.py", "w") as f:
    f.write(content)

print("✅ 代码已更新！")
