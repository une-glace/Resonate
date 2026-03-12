"""
根据prompt包含event的个数做平均
"""

import sys
import json
from collections import defaultdict

result_jsonl = sys.argv[1]
outfile = sys.argv[2]

prompt_path = "../sets/acc_prompt.json"

# ------------------------------------------------------------------
# Load prompt metadata: id -> (event_count, event_relation)
# ------------------------------------------------------------------
with open(prompt_path, "r") as f:
    prompts = json.load(f)

id2meta = {
    p["id"]: {
        "event_count": p["event_count"],
        "event_relation": p["event_relation"]
    }
    for p in prompts
}

# ------------------------------------------------------------------
# Statistics containers
# ------------------------------------------------------------------
# event_count -> [sum, count]
stats_by_cnt = defaultdict(lambda: [0.0, 0])

# event_relation -> [sum, count]
stats_by_rel = defaultdict(lambda: [0.0, 0])

# event_count + event_relation -> [sum, count]
stats_by_cnt_rel = defaultdict(lambda: [0.0, 0])

# overall
overall_sum = 0.0
overall_cnt = 0

# ------------------------------------------------------------------
# Aggregate
# ------------------------------------------------------------------
with open(result_jsonl, "r") as f:
    for line in f:
        d = json.loads(line)
        prompt_id = d["prompt_id"]
        score = d["qwen3_omni_semantic_align_score"]

        if prompt_id not in id2meta:
            continue

        event_cnt = id2meta[prompt_id]["event_count"]
        event_rel = id2meta[prompt_id]["event_relation"]

        stats_by_cnt[event_cnt][0] += score
        stats_by_cnt[event_cnt][1] += 1

        stats_by_rel[event_rel][0] += score
        stats_by_rel[event_rel][1] += 1

        stats_by_cnt_rel[(event_cnt, event_rel)][0] += score
        stats_by_cnt_rel[(event_cnt, event_rel)][1] += 1

        overall_sum += score
        overall_cnt += 1

# ------------------------------------------------------------------
# Write results
# ------------------------------------------------------------------
with open(outfile, "w") as out_f:
    out_f.write("=== Overall ===\n")
    overall_avg = overall_sum / overall_cnt if overall_cnt > 0 else 0.0
    line = f"overall_count={overall_cnt}\toverall_avg_score={overall_avg:.6f}"
    print(line)
    out_f.write(line + "\n\n")

    out_f.write("=== By event_count ===\n")
    for event_cnt in sorted(stats_by_cnt.keys()):
        total, cnt = stats_by_cnt[event_cnt]
        avg = total / cnt if cnt > 0 else 0.0
        line = f"event_count={event_cnt}\tcount={cnt}\tavg_score={avg:.6f}"
        print(line)
        out_f.write(line + "\n")

    out_f.write("\n=== By event_relation ===\n")
    for event_rel in sorted(stats_by_rel.keys()):
        total, cnt = stats_by_rel[event_rel]
        avg = total / cnt if cnt > 0 else 0.0
        line = f"event_relation={event_rel}\tcount={cnt}\tavg_score={avg:.6f}"
        print(line)
        out_f.write(line + "\n")

    # out_f.write("\n=== By event_count + event_relation ===\n")
    # for (event_cnt, event_rel) in sorted(stats_by_cnt_rel.keys()):
    #     total, cnt = stats_by_cnt_rel[(event_cnt, event_rel)]
    #     avg = total / cnt if cnt > 0 else 0.0
    #     line = (
    #         f"event_count={event_cnt}\t"
    #         f"event_relation={event_rel}\t"
    #         f"count={cnt}\t"
    #         f"avg_score={avg:.6f}"
    #     )
    #     print(line)
    #     out_f.write(line + "\n")
    
    
print(f"Results saved in {outfile}")
