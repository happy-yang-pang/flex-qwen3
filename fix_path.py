"""Fix path expansion issue"""

with open("flexllmgen/opt_config.py", "r") as f:
    content = f.read()

# 1. 确保导入 os.path.expanduser
if "import os" not in content.split("def download_opt_weights")[0]:
    # 在文件开头添加 import os
    content = content.replace(
        "import dataclasses",
        "import dataclasses\nimport os"
    )

# 2. 修复 convert_qwen3_weights_to_np 中的路径
old_code = '''    np_output_dir = os.path.join(output_path, model_name + "-np")'''
new_code = '''    # Expand ~ in output_path
    output_path = os.path.expanduser(output_path)
    np_output_dir = os.path.join(output_path, model_name + "-np")'''

content = content.replace(old_code, new_code)
# 3. 确保 download_opt_weights 函数也展开路径
old_download = '''def download_opt_weights(model_name, path):'''
new_download = '''def download_opt_weights(model_name, path):
    # Expand ~ in path
    path = os.path.expanduser(path)'''

content = content.replace(old_download, new_download)

with open("flexllmgen/opt_config.py", "w") as f:
    f.write(content)

print("✅ 路径修复完成！")
