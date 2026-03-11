import json


a = '/inspire/hdd/project/embodied-multimodality/public/xqli/TTA/Resonate/data/AudioCaps/GRPO_Meta/train_metadata.jsonl'
b = '/inspire/hdd/project/embodied-multimodality/public/xqli/TTA/Resonate/data/AudioCaps/GRPO_Meta/train_metadata_m.jsonl'

data = [json.loads(d) for d in open(a, 'r')]
new_data = [{"prompt": d['prompt'], "audio_id": d['audio_id']} for d in data]
with open(b, 'w') as f:
    for d in new_data: 
        f.write(json.dumps(d) + '\n')