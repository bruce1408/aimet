# 修正的代码 - 使用配置文件获取AIMET支持的量化算子

import os
import json
from pprint import pprint

# 获取AIMET安装路径下的配置文件目录
aimet_config_dir = "/mnt/share_disk/bruce_trie/miniforge3/envs/torch113_cuda116/lib/python3.10/site-packages/aimet_common/quantsim_config"

# 列出可用的配置文件
print("AIMET配置文件:")
config_files = [f for f in os.listdir(aimet_config_dir) if f.endswith('.json')]
for i, config_file in enumerate(config_files):
    print(f"{i+1}. {config_file}")

# 选择后端感知的CPU量化配置文件查看 (这里包含了最全面的支持算子信息)
backend_config_file = os.path.join(aimet_config_dir, "backend_aware_cpu_quantsim_config.json")

# 读取配置文件
with open(backend_config_file, 'r') as f:
    config = json.load(f)

# 提取支持的算子类型
supported_ops = list(config["op_type"].keys())
supported_ops.sort()

print("\n支持量化的算子类型:")
for op in supported_ops:
    print(op)

# 打印带有支持内核信息的详细算子配置
print("\n支持的量化配置详情:")
for op in supported_ops[:10]:  # 只显示前10个，避免输出过多
    if op in config["op_type"] and "supported_kernels" in config["op_type"][op]:
        print(f"\n{op}:")
        pprint(config["op_type"][op]["supported_kernels"])