"""Fix opt_config.py to add Qwen3 support"""

with open("flexllmgen/opt_config.py", "r") as f:
    lines = f.readlines()

# 找到 'else:' 行和 'raise ValueError' 行
else_line_idx = None
raise_line_idx = None

for i, line in enumerate(lines):
    if line.strip() == "else:":
        else_line_idx = i
    if "Invalid model name" in line:
        raise_line_idx = i
        break

if else_line_idx is None or raise_line_idx is None:
    print("❌ 未找到目标行")
    print(f"else_line_idx: {else_line_idx}, raise_line_idx: {raise_line_idx}")
    exit(1)
print(f"✅ 找到目标位置：else 在第 {else_line_idx + 1} 行，raise 在第 {raise_line_idx + 1} 行")

# 在 else: 之后插入 Qwen3 判断逻辑（保持正确缩进）
qwen3_code = [
    "        # Check for Qwen3 models (GQA support)\n",
    "        if \"qwen3\" in name.lower():\n",
    "            return get_qwen3_config(name, **kwargs)\n",
    "        \n",
]

# 插入代码
new_lines = lines[:else_line_idx + 1] + qwen3_code + lines[else_line_idx + 1:]

# 写回文件
with open("flexllmgen/opt_config.py", "w") as f:
    f.writelines(new_lines)

print("✅ 代码已更新！")
# 验证
with open("flexllmgen/opt_config.py", "r") as f:
    content = f.read()
    if "qwen3" in content and "get_qwen3_config" in content:
        print("✅ 验证成功：Qwen3 支持已添加")
    else:
        print("❌ 验证失败")

# 显示修改后的代码片段
print("\n=== 修改后的代码片段 ===")
with open("flexllmgen/opt_config.py", "r") as f:
    all_lines = f.readlines()
    start = max(0, else_line_idx - 2)
    end = min(len(all_lines), else_line_idx + 8)
    for i in range(start, end):
        print(f"{i + 1:4d}: {all_lines[i]}", end="")
