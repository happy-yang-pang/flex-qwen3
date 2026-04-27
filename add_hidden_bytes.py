"""Add hidden_bytes method to Qwen3Config"""

with open("flexllmgen/opt_config.py", "r") as f:
    content = f.read()

# 查找 Qwen3Config 类中的 cache_bytes 方法
# 在 cache_bytes 方法之后添加 hidden_bytes 方法

# 定义 hidden_bytes 方法（参考 OPT 的实现）
hidden_bytes_method = """
    def hidden_bytes(self, batch_size, seq_len):
        \"\"\"Calculate hidden state memory size.\"\"\"
        # Hidden states: batch_size * seq_len * hidden_size * dtype_bytes
        dtype_bytes = 2 if self.dtype == np.float16 else 4
        return batch_size * seq_len * self.hidden_size * dtype_bytes
"""
import re

# 匹配 cache_bytes 方法的完整定义
pattern = r'(    def cache_bytes\(self, batch_size, seq_len\):.*?return .*?\n)'

def add_method(match):
    cache_method = match.group(1)
    return cache_method + hidden_bytes_method

new_content = re.sub(pattern, add_method, content, flags=re.DOTALL)

if new_content == content:
    print("⚠️ 未找到 cache_bytes 方法，尝试其他方式...")
    # 备用方案：在 model_bytes 之后插入
    pattern2 = r'(    def model_bytes\(self\):.*?return .*?\n)'
    new_content = re.sub(pattern2, add_method, content, flags=re.DOTALL)

with open("flexllmgen/opt_config.py", "w") as f:
    f.write(new_content)

print("✅ 代码已更新！")
with open("flexllmgen/opt_config.py", "r") as f:
    content = f.read()
    if "def hidden_bytes" in content and "Qwen3Config" in content:
        print("✅ 验证成功：hidden_bytes 方法已添加")
    else:
        print("❌ 验证失败")

# 显示 Qwen3Config 类的完整定义
print("\n=== Qwen3Config 完整定义 ===")
lines = content.split('\n')
in_qwen3 = False
indent_level = 0
for i, line in enumerate(lines):
    if "class Qwen3Config:" in line:
        in_qwen3 = True
        start = i
    if in_qwen3 and line.strip().startswith("def get_qwen3_config"):
        end = i
        break
else:
    end = len(lines)

if in_qwen3:
    for i in range(start, min(end, start + 50)):
        print(f"{i + 1:4d}: {lines[i]}")
