"""
Calculate clap scores for all wav samples
"""
import os
import json
import torch
from msclap import CLAP
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm


def get_prompt_text(prompt_id: str) -> str:
    """Return the textual prompt for a given prompt id.

    Tries the 'text' field first; if unavailable, tries common variants.
    """
    prompt_path = "./prompts/acc_prompt.json"

    prompts = json.load(open(prompt_path, 'r'))
    
    # Build an id->object map if needed
    index = {p["id"]: p for p in prompts}
    obj = index.get(prompt_id)
    return obj['prompt_text']


def cal_clap_score_for_jsonl(input_jsonl: str, result_jsonl: str):
    """Compute CLAP scores for all entries in a JSONL file."""
    clap_model = CLAP(version='2023', use_cuda=True)
    with open(input_jsonl, "r") as f:
        lines = f.readlines()
    
    output = []
    for line in tqdm(lines):
        data = json.loads(line)
        file_path = data['path']
        prompt_id = file_path.split('/')[-1].replace('.wav','')
        prompt_text = get_prompt_text(prompt_id)

        audio_emb = clap_model.get_audio_embeddings([file_path])
        text_emb = clap_model.get_text_embeddings([prompt_text])
        score = torch.nn.functional.cosine_similarity(audio_emb, text_emb).item()
        output.append({"prompt_id": prompt_id, "prompt_text": prompt_text, "clap_score": score})

    with open(result_jsonl, 'w', encoding='utf-8') as out_f:
        for d in output: 
            json.dump(d, out_f)
            out_f.write('\n')

if __name__ == "__main__":
    import sys
    input_jsonl = sys.argv[1]
    result_jsonl = sys.argv[2]
    cal_clap_score_for_jsonl(input_jsonl, result_jsonl)