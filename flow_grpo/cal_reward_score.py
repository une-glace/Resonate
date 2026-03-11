## 使用reward score计算模型结果

import os
import sys
import json
import pandas as pd
import torch
import torchaudio
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from rewards import (
    qwen25_omni_semantic_align_score,
    qwen3_omni_thinking_semantic_align_score,
    qwen3_omni_semantic_align_score,
    clap_score
)
from tqdm import tqdm

"""
python flow_grpo/eval_reward.py qwen25_omni_semantic_align_score
python flow_grpo/eval_reward.py qwen3_omni_thinking_semantic_align_score
python flow_grpo/eval_reward.py qwen25_omni_thinking_semantic_align_score
python flow_grpo/eval_reward.py qwen3_omni_semantic_align_score
python flow_grpo/eval_reward.py clap_score
"""

DEVICE = "cuda"

# ====== load reward fn ======
## 默认使用Qwen3 Omni评测
reward_fn_str = "qwen3_omni_semantic_align_score"

if reward_fn_str == "qwen25_omni_semantic_align_score":
    reward_fn = qwen25_omni_semantic_align_score(DEVICE)
elif reward_fn_str == "qwen25_omni_thinking_semantic_align_score": 
    reward_fn = qwen25_omni_thinking_semantic_align_score(DEVICE)
elif reward_fn_str == "qwen3_omni_thinking_semantic_align_score":
    reward_fn = qwen3_omni_thinking_semantic_align_score(DEVICE)
elif reward_fn_str == "qwen3_omni_semantic_align_score": 
    reward_fn = qwen3_omni_semantic_align_score(DEVICE)
elif reward_fn_str == "clap_score": 
    reward_fn = clap_score(DEVICE)
else:
    raise NotImplementedError


def get_prompt_text(prompt_id: str) -> str:
    """Return the textual prompt for a given prompt id.

    Tries the 'text' field first; if unavailable, tries common variants.
    """
    prompt_path = "./sets/acc_prompt.json"

    prompts = json.load(open(prompt_path, 'r'))
    
    # Build an id->object map if needed
    index = {p["id"]: p for p in prompts}
    obj = index.get(prompt_id)
    return obj['prompt_text']


if __name__ == '__main__': 
    import sys
    input_jsonl = sys.argv[1]
    result_jsonl = sys.argv[2]
    
    with open(input_jsonl, "r") as f:
        lines = f.readlines()
    
    output = []
    for line in tqdm(lines):
        data = json.loads(line)
        file_path = data['path']
        prompt_id = file_path.split('/')[-1].replace('.wav','')
        prompt_text = get_prompt_text(prompt_id)

        wav, sr = torchaudio.load(file_path)  # [C, T]
        wav = wav.mean(dim=0, keepdim=True)  # [1, T]
        with torch.no_grad():
            score, _ = reward_fn(
                audios=wav,
                prompts=[prompt_text],
                vae_sr=sr,
            )
            score = float(score.item())
        
        output.append({"prompt_id": prompt_id, "prompt_text": prompt_text, reward_fn_str: score})
    
    with open(result_jsonl, 'w', encoding='utf-8') as out_f:
        for d in output: 
            json.dump(d, out_f)
            out_f.write('\n')
