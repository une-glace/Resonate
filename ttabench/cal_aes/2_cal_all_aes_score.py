"""
Step 2: Calculate AES scores for all wav samples listed in prepared JSONL files.
"""
import os
# from utils.config import SYS_NAMES, EVAL_DIMS, PATHS, AES_RESULTS_JSON_DIR
# from utils.common import ensure_dir

# Select GPU device (optional)
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
BATCH_SIZE = 4



def calculate_all_aes_scores():
    """Iterate all systems/dimensions and run AES CLI to produce JSONL scores."""
    import sys
    
    input_jsonl = sys.argv[1]
    result_jsonl = sys.argv[2]

    command = f"audio-aes {input_jsonl} --batch-size {BATCH_SIZE} > {result_jsonl}"
    print(f"Compute AES for {input_jsonl}...")
    os.system(command)
    print(f"AES results saved to {result_jsonl}")

if __name__ == "__main__":
    calculate_all_aes_scores()
