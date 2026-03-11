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
    qwen25_omni_thinking_semantic_align_score, 
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

# ====== 配置 ======
CSV_PATH = ""
AUDIO_DIR = ""
DEVICE = "cuda"

# ====== load reward fn ======
reward_fn_str = sys.argv[1]
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

OUT_JSONL = f"./flow_grpo/eval_reward/{reward_fn_str}_test.jsonl"

# ====== 读取人工评测 ======
df = pd.read_csv(CSV_PATH)

rewards = []
records = []

print(f"Evaluating {reward_fn_str}")
for _, row in tqdm(df.iterrows(), total=len(df)):
    text = row["Text"]
    file_name = row["File Name"]
    model_name = row["Model"]

    wav_path = os.path.join(AUDIO_DIR, model_name, f"{file_name}.wav")
    assert os.path.exists(wav_path), wav_path

    wav, sr = torchaudio.load(wav_path)  # [C, T]
    wav = wav.mean(dim=0, keepdim=True)  # [1, T]

    with torch.no_grad():
        score, _ = reward_fn(
            audios=wav,
            prompts=[text],
            vae_sr=sr,
        )

    reward_val = score.item()
    rewards.append(reward_val)
    print(f"Reward: {reward_val} | REL: {float(row['REL'])} | OVL: {float(row['OVL'])}")

    records.append({
        "audio_path": wav_path,
        "model": model_name,
        "text": text,
        "reward": reward_val,
        "REL": float(row["REL"]),
        "OVL": float(row["OVL"]),
    })

rewards = np.array(rewards)

# ====== 相关性指标 ======
def compute_metrics(pred, gt):
    return {
        "LCC": pearsonr(pred, gt)[0],
        "SRCC": spearmanr(pred, gt)[0],
        "KTAU": kendalltau(pred, gt)[0],
    }

metrics_rel = compute_metrics(rewards, df["REL"].values)
metrics_ovl = compute_metrics(rewards, df["OVL"].values)

print("\n=== Alignment Metrics (REL) ===")
print(f"LCC  (Pearson): {metrics_rel['LCC']:.4f}")
print(f"SRCC (Spearman): {metrics_rel['SRCC']:.4f}")
print(f"KTAU (Kendall):  {metrics_rel['KTAU']:.4f}")

print("\n=== Alignment Metrics (OVL) ===")
print(f"LCC  (Pearson): {metrics_ovl['LCC']:.4f}")
print(f"SRCC (Spearman): {metrics_ovl['SRCC']:.4f}")
print(f"KTAU (Kendall):  {metrics_ovl['KTAU']:.4f}")

# ====== JSONL 输出 ======
with open(OUT_JSONL, "w") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

    f.write(json.dumps({
        "summary": {
            "REL": {
                "LCC": metrics_rel["LCC"],
                "SRCC": metrics_rel["SRCC"],
                "KTAU": metrics_rel["KTAU"],
            },
            "OVL": {
                "LCC": metrics_ovl["LCC"],
                "SRCC": metrics_ovl["SRCC"],
                "KTAU": metrics_ovl["KTAU"],
            },
        }
    }) + "\n")
