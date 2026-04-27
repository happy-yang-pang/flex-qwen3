"""Fix np.save filename extension"""

with open("flexllmgen/opt_config.py", "r") as f:
    content = f.read()

# 查找并修复 np.save 相关代码
import re

# 模式1：np_path = os.path.join(np_output_dir, opt_name + ".npy")
pattern1 = r'np_path = os\.path\.join\(np_output_dir, opt_name \+ "\.npy"\)'
replacement1 = 'np_path = os.path.join(np_output_dir, opt_name)  # No .npy extension'

if re.search(pattern1, content):
    content = re.sub(pattern1, replacement1, content)
    print("✅ 修复模式 1 完成")
else:
    print("⚠️ 未找到模式 1")

# 模式2：直接 np.save 带 .npy
pattern2 = r'np\.save\(os\.path\.join\(np_output_dir, [^)]+ \+ "\.npy"\)'
replacement2 = 'np.save(os.path.join(np_output_dir, opt_name))'

if re.search(pattern2, content):
    content = re.sub(pattern2, replacement2, content)
    print("✅ 修复模式 2 完成")
else:
    print("⚠️ 未找到模式 2")

with open("flexllmgen/opt_config.py", "w") as f:
    f.write(content)

print("✅ 转换脚本修复完成！")
