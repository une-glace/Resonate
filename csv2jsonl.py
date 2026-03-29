import csv
import json
import os

# 输入和输出文件路径
csv_path = r"/inspire/hdd/project/embodied-multimodality/public/datasets/AudioCaps/train.csv"
output_path = r"/inspire/hdd/project/embodied-multimodality/chenxie-25019/qingya/Resonate/data/AudioCaps/train_audiocaps_wduration.jsonl"

# 转换脚本
with open(csv_path, 'r', encoding='utf-8') as csv_file, open(output_path, 'w', encoding='utf-8') as jsonl_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        youtube_id = row['youtube_id']
        audio_id = f"Y{youtube_id}.wav"  # 构造音频文件名
        audio_path = f"/inspire/hdd/project/embodied-multimodality/public/datasets/AudioCaps/train/{audio_id}"  # 构造音频路径
        caption = row['caption']
        duration = 10.0  # 假设所有音频时长为10秒

        # 构造JSON对象
        json_obj = {
            "audio_id": audio_id,
            "audio_path": audio_path,
            "caption": caption,
            "duration": duration
        }

        # 写入jsonl文件
        jsonl_file.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

print(f"转换完成，JSONL文件已保存到: {output_path}")