from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from awq.quantize.pre_quant import run_awq, apply_awq
import os
from awq.quantize.quantizer import (
    pseudo_quantize_model_weight,
    real_quantize_model_weight,
)
from awq.settings import *

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
print('Real_quantizing model weight')
real_quantize_model_weight(model, w_bit=w_bit, q_config=q_config)

dirpath = os.path.dirname(dump_real_path)
os.makedirs(dirpath, exist_ok=True)

print(f"Saving the quantized model at {dump_real_path}...")
torch.save(model.cpu().state_dict(), dump_real_path)
