"""Fix download_opt_weights to support Qwen3"""

with open("flexllmgen/opt_config.py", "r") as f:
    lines = f.readlines()

# 找到 download_opt_weights 函数
func_start = None
func_end = None

for i, line in enumerate(lines):
    if "def download_opt_weights(" in line:
        func_start = i
    if func_start is not None and i > func_start and line.strip().startswith("def "):
        func_end = i
        break

if func_start is None:
    print("❌ 未找到 download_opt_weights 函数")
    exit(1)

print(f"✅ 找到 download_opt_weights 函数：第 {func_start + 1} 行 到 第 {func_end} 行")

# 显示原函数内容
print("\n=== 原函数内容 ===")
for i in range(func_start, min(func_end or func_start + 60, len(lines))):
    print(f"{i + 1:4d}: {lines[i]}", end="")

# 我们需要在函数开头添加 Qwen3 的判断
# 查找 "if "opt" in name" 或类似的模式
insert_line = None
for i in range(func_start, min(func_end or func_start + 60, len(lines))):
    if 'if "opt"' in lines[i] or "if name" in lines[i]:
        insert_line = i
        break

if insert_line is None:
    # 备用：在函数定义后第一行插入
    insert_line = func_start + 1

print(f"\n✅ 将在第 {insert_line + 1} 行插入 Qwen3 支持代码")

# 构建新代码
new_lines = lines[:insert_line]

# 插入 Qwen3 处理逻辑
qwen3_code = [
    "    # Handle Qwen3 models\n",
    "    if \"qwen3\" in name.lower():\n",
    "        hf_model_name = \"Qwen/Qwen3-0.6B\"\n",
    "        print(f\"Loading Qwen3 model from {hf_model_name}\")\n",
    "    el",  # 故意截断，后面会接原代码
]

# 处理原代码中的第一个 if 语句
original_line = lines[insert_line]
if original_line.strip().startswith("if "):
    # 将原 if 改为 elif
    modified_line = original_line.replace("if ", "elif ", 1)
    qwen3_code[-1] = "    elif "  # 接上原代码的剩余部分
    new_lines.extend(qwen3_code[:-1])  # 不包含最后一个截断的元素
    new_lines.append(modified_line)
    new_lines.extend(lines[insert_line + 1:])
else:
    new_lines.extend(qwen3_code[:-1])
    new_lines.extend(lines[insert_line:])

with open("flexllmgen/opt_config.py", "w") as f:
    f.writelines(new_lines)

print("✅ 代码已更新！")
