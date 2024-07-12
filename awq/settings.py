import torch

# 模型路径
model_path = '/mnt/models/opt-125m'
# 存储AWQ结果的路径
dump_awq_path = 'awq_cache/opt-125m-w4-g128.pt'
# 加载AWQ结果的路径
load_awq_path = 'awq_cache/opt-125m-w4-g128.pt'
# 存储伪量化模型的路径
dump_fake_path = 'quant_cache/opt-125m-w4-g128-fake.pt'
# 存储量化后权重的路径
dump_real_path = 'quant_cache/opt-125m-w4-g128-real.pt'
# 加载量化后的权重的路径
load_quant_path = 'quant_cache/opt-125m-w4-g128-real.pt'

cache_dir = 'models'
kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
w_bit = 4
q_config = {
    "zero_point": True,  # by default True
    "q_group_size": 128,  # whether to use group quantization
}
