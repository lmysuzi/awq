from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from awq.quantize.pre_quant import run_awq, apply_awq
import os
from awq.quantize.quantizer import (
    pseudo_quantize_model_weight,
    real_quantize_model_weight,
)
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    dispatch_model,
    load_checkpoint_in_model,
)
from awq.utils.utils import simple_dispatch_model
from awq.test_model import test
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

print("Loading pre-computed quantized weights...")
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(
        config=config, torch_dtype=torch.float16, trust_remote_code=True
    )
real_quantize_model_weight(
    model, w_bit=w_bit, q_config=q_config, init_only=True
)

model.tie_weights()

torch.cuda.reset_peak_memory_stats('cuda')

# Infer device map
kwargs = {}
device_map = infer_auto_device_map(
    model,
    no_split_module_classes=[
        "OPTDecoderLayer",
        "LlamaDecoderLayer",
        "BloomBlock",
        "MPTBlock",
        "DecoderLayer",
    ],
    **kwargs,
)
# Load checkpoint in the model
load_checkpoint_in_model(
    model,
    checkpoint=load_quant_path,
    device_map=device_map,
    offload_state_dict=True,
)

# Dispatch model
model = simple_dispatch_model(model, device_map=device_map)
# model = dispatch_model(model, device_map=device_map)

model.eval()

#peak_memory_allocated = torch.cuda.max_memory_allocated('cuda')
# print(f"最大显存开销（GB）: {peak_memory_allocated / 1e9}")
test(model, enc, 'result_cache/quant_results.json')
