from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from awq.quantize.pre_quant import run_awq, apply_awq
import os
from awq.settings import *

#model_path = '/mnt/models/opt-125m'
#dump_awq_path = 'awq_cache/opt-125m-w4-g128.pt'
#cache_dir = 'models'
#kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
#w_bit = 4
#q_config = {
#    "zero_point": True,  # by default True
#    "q_group_size": 128,  # whether to use group quantization
#}

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # Note (Haotian): To avoid OOM after huggingface transformers 4.36.2
config.use_cache = False

if "mpt" in config.__class__.__name__.lower():
    enc = AutoTokenizer.from_pretrained(
        config.tokenizer_name, trust_remote_code=True
    )
else:
    enc = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )

model = AutoModelForCausalLM.from_pretrained(
    model_path, config=config, trust_remote_code=True, **kwargs
)

model.eval()

awq_results = run_awq(
    model,
    enc,
    w_bit=w_bit,
    q_config=q_config,
    n_samples=128,
    seqlen=512,
)

dirpath = os.path.dirname(dump_awq_path)
os.makedirs(dirpath, exist_ok=True)

torch.save(awq_results, dump_awq_path)
print("AWQ results saved at", dump_awq_path)

