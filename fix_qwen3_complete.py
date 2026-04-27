"""Complete fix for Qwen3 weight conversion"""

with open("flexllmgen/opt_config.py", "r") as f:
    content = f.read()

# 1. 确保导入 os
if "import os" not in content.split("def download_opt_weights")[0]:
    content = content.replace(
        "import dataclasses",
        "import dataclasses\nimport os"
    )

# 2. 修复 download_opt_weights 函数
old_func = """def download_opt_weights(model_name, path):
    from huggingface_hub import snapshot_download
    import torch"""

new_func = """def download_opt_weights(model_name, path):
    from huggingface_hub import snapshot_download
    import torch
    # Expand ~ in path
    path = os.path.expanduser(path)"""

content = content.replace(old_func, new_func)
# 3. 修复 convert_qwen3_weights_to_np 函数
old_convert = """def convert_qwen3_weights_to_np(hf_folder, output_path, model_name):
    \"\"\"Convert Qwen3 safetensors weights to numpy format.\"\"\"
    import os
    import numpy as np
    import torch
    from safetensors.torch import load_file
    
    print(f"Converting Qwen3 weights from {hf_folder} to numpy format...")
    
    # Create output directory
    np_output_dir = os.path.join(output_path, model_name + "-np")"""

new_convert = """def convert_qwen3_weights_to_np(hf_folder, output_path, model_name):
    \"\"\"Convert Qwen3 safetensors weights to numpy format.\"\"\"
    import os
    import numpy as np
    import torch
    from safetensors.torch import load_file
    
    print(f"Converting Qwen3 weights from {hf_folder} to numpy format...")
    
    # Expand ~ in output_path
    output_path = os.path.expanduser(output_path)
    
    # Create output directory
    np_output_dir = os.path.join(output_path, model_name + "-np")"""

content = content.replace(old_convert, new_convert)
# 4. 更新权重映射（添加缺失的映射）
old_mapping = """    weight_mapping = {
        "model.embed_tokens.weight": "decoder.embed_tokens.weight",
        "model.norm.weight": "decoder.final_layer_norm.weight",
        "lm_head.weight": "lm_head.weight",
    }"""

new_mapping = """    weight_mapping = {
        "model.embed_tokens.weight": "decoder.embed_tokens.weight",
        "model.norm.weight": "decoder.final_layer_norm.weight",
        "model.norm.bias": "decoder.final_layer_norm.bias",
        "lm_head.weight": "lm_head.weight",
    }
    
    # Add q_norm and k_norm mappings (for GQA)
    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        weight_mapping.update({
            f"{prefix}.self_attn.q_norm.weight": f"decoder.layers.{i}.self_attn.q_norm.weight",
            f"{prefix}.self_attn.q_norm.bias": f"decoder.layers.{i}.self_attn.q_norm.bias",
            f"{prefix}.self_attn.k_norm.weight": f"decoder.layers.{i}.self_attn.k_norm.weight",
            f"{prefix}.self_attn.k_norm.bias": f"decoder.layers.{i}.self_attn.k_norm.bias",
        })"""

content = content.replace(old_mapping, new_mapping)

with open("flexllmgen/opt_config.py", "w") as f:
    f.write(content)

print("✅ 完整修复完成！")
