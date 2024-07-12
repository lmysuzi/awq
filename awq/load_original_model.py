from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from awq.quantize.pre_quant import run_awq, apply_awq
import os
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    dispatch_model,
    load_checkpoint_in_model,
)
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

model = AutoModelForCausalLM.from_pretrained(
    model_path, config=config, trust_remote_code=True, **kwargs
)

torch.cuda.reset_peak_memory_stats('cuda')

kwargs = {
    "max_memory": None
}
device_map = infer_auto_device_map(
    model,
    # TODO: can we remove this?
    no_split_module_classes=[
        "OPTDecoderLayer",
        "LlamaDecoderLayer",
        "BloomBlock",
        "MPTBlock",
        "DecoderLayer",
    ],
    **kwargs,
)
model = dispatch_model(model, device_map=device_map)

test(model, enc, 'result_cache/original_model.json')
