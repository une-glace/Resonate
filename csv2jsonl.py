import csv
import json
import os

csv_path = "/inspire/hdd/project/embodied-multimodality/public/datasets/AudioCaps/train.csv"  # 替换为实际路径
wav_dir = "/inspire/hdd/project/embodied-multimodality/public/datasets/AudioCaps/train"
out_path = "/inspire/hdd/project/embodied-multimodality/chenxie-25019/qingya/Resonate/data/AudioCaps/train_audiocaps_wduration.jsonl"  # 替换为实际路径

with open(csv_path, encoding="utf-8", newline="") as fin, \
     open(out_path, "w", encoding="utf-8") as fout:
    reader = csv.DictReader(fin)
    for row in reader:
        youtube_id = row.get("youtube_id", "").strip()
        if not youtube_id:
            continue

        audio_id = f"Y{youtube_id}.wav"
        audio_path = os.path.join(wav_dir, audio_id)

        # 检查文件是否存在
        if not os.path.exists(audio_path):
            print(f"音频文件缺失: {audio_path}")
            continue

        # 构造 JSON 对象
        json_obj = {
            "audio_id": audio_id,
            "audio_path": audio_path,
            "caption": row.get("caption", "").strip(),
            "duration": 10.0
        }
        fout.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

print("转换完成：", out_path)