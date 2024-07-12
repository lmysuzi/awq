from datasets import load_dataset
import tqdm
import torch
from torch import nn
import json
import os
import time
import numpy as np
from accelerate.utils.modeling import get_balanced_memory
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    dispatch_model,
    load_checkpoint_in_model,
)




def test(model, enc, output_path, DEV='cuda'):

    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()

    print('tests begin')
    
    testenc = load_dataset("/mnt/data/wikitext", split="validation")
    #testenc = load_dataset('/mnt/data/mit-han-lab___pile-val-backup', split='validation')
    testenc = enc("\n\n".join(testenc["text"]), return_tensors="pt")
    model.seqlen = 2048
    testenc = testenc.input_ids.to(model.device)
    nsamples = testenc.numel() // model.seqlen
    model = model.eval()
    nlls = []
    times = []
    
    sync()

    # 重置当前设备的最大显存统计
    torch.cuda.reset_peak_memory_stats(DEV)

    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
            model.device
        )
        # 测试时间
        tick = time.time()
        with torch.no_grad():
            lm_logits = model(batch).logits
        sync()
        times.append(time.time() - tick)

        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[
            :, (i * model.seqlen) : ((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    # 计算ppl
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f'ppl: {ppl.item()}')
    # 计算平均时间
    median_time = np.median(times)
    print(f'median_time: {median_time}s')
    # 获取推理后的最大显存使用量
    peak_memory_allocated = torch.cuda.max_memory_allocated(DEV)
    print(f"推理时的最大显存开销（GB）: {peak_memory_allocated / 1e9}")

    results = {"ppl": ppl.item(), 'median_time': median_time, '最大显存开销': peak_memory_allocated/1e9}

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
