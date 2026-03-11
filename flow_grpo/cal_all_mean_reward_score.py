import os
import json

if __name__ == "__main__":
    """计算每个系统每个维度的平均 qwen3_omni_semantic_align_score 分数"""
    import sys
    result_jsonl = sys.argv[1]
    outfile = sys.argv[2]

    with open(outfile, 'w') as f:
        total, count = 0, 0
        with open(result_jsonl, 'r') as r_f:
            for line in r_f:
                total += json.loads(line)["qwen3_omni_semantic_align_score"]
                count += 1
        avg = total / count if count > 0 else 0
        print(f"count: {count} Average qwen3_omni_semantic_align_score: {avg}")
        f.write(f"{avg}\n")
