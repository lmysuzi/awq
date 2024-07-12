from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from awq.quantize.pre_quant import run_awq, apply_awq
import os
from awq.quantize.quantizer import (
    pseudo_quantize_model_weight,
    real_quantize_model_weight,
)

load_awq_path = 'awq_cache/opt-125m-w4-g128.pt'
model_path = '/mnt/models/opt-125m'
dump_fake_path = 'quant_cache/opt-125m-w4-g128-fake.pt'
kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
w_bit = 4
q_config = {
    "zero_point": True,  # by default True
    "q_group_size": 128,  # whether to use group quantization
}

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

print("Loading pre-computed AWQ results from", load_awq_path)
awq_results = torch.load(load_awq_path, map_location="cpu")
print("Applying awq")
apply_awq(model, awq_results)
print('Pseudo_quantizing model weight')
pseudo_quantize_model_weight(model, w_bit=w_bit, q_config=q_config)

dirpath = os.path.dirname(dump_fake_path)
os.makedirs(dirpath, exist_ok=True)

model.save_pretrained(dump_fake_path)
print("Pseudo-quantized models saved at", dump_fake_path)
