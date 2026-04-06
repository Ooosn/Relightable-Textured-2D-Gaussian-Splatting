import re

# 你要乘的比例因子
scaling_factor = 1.2

# 源 bash 文件路径
input_file = "test.sh"
output_file = "testtest.sh"  # 可与原始不同，防止覆盖

# 需要更新的变量名列表
param_keys = [
    "iterations",
    "asg_freeze_step",
    "spcular_freeze_step",
    "fit_linear_step",
    "asg_lr_freeze_step",
    "asg_lr_max_steps",
    "train_cam_freeze_step",
    "opt_cam_lr_max_steps",
    "train_pl_freeze_step",
    "opt_pl_lr_max_steps",
    "local_q_lr_freeze_step",
    "local_q_lr_max_steps",
    "freeze_phasefunc_steps",
    "neural_phasefunc_lr_max_steps",
    "position_lr_max_steps",
    "densify_until_iter",
    "unfreeze_iterations",
    "end_iterations"
]

# 正则匹配 e.g., iterations=50000
pattern = re.compile(rf"^({'|'.join(param_keys)})=(\d+)", re.MULTILINE)

# 读取文件内容
with open(input_file, 'r') as f:
    content = f.read()

# 替换内容
def scale_match(match):
    key = match.group(1)
    original = int(match.group(2))
    scaled = int(original * scaling_factor)
    return f"{key}={scaled}"

new_content = pattern.sub(scale_match, content)

# 保存到新文件
with open(output_file, 'w') as f:
    f.write(new_content)

print(f"✅ 参数已按 {scaling_factor} 倍缩放，写入 {output_file}")
