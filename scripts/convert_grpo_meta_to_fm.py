import argparse
import json
from pathlib import Path


def normalize_audio_id(audio_id: str, suffix: str) -> str:
    if not audio_id.endswith(suffix):
        return f"{audio_id}{suffix}"
    return audio_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert GRPO metadata JSONL into Flow Matching metadata JSONL."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/AudioCaps/GRPO_Meta/train_metadata.jsonl"),
        help="Path to the GRPO metadata JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/AudioCaps/train_audiocaps_from_grpo_wduration.jsonl"),
        help="Path to write the converted Flow Matching metadata JSONL.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split name used to build audio_path, e.g. train/val/test.",
    )
    parser.add_argument(
        "--audio-root",
        default="../datasets/AudioCaps",
        help="Base path prefix used in audio_path entries.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Fallback duration to write when the GRPO metadata does not include it.",
    )
    parser.add_argument(
        "--audio-suffix",
        default=".wav",
        help="Suffix to append when audio_id has no file extension.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    num_records = 0
    with args.input.open("r", encoding="utf-8") as src, args.output.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for line_no, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            if "audio_id" not in item or "prompt" not in item:
                raise ValueError(
                    f"Line {line_no} in {args.input} is missing required keys. "
                    "Expected at least 'audio_id' and 'prompt'."
                )

            audio_id = normalize_audio_id(str(item["audio_id"]), args.audio_suffix)
            converted = {
                "audio_id": audio_id,
                "audio_path": f"{args.audio_root}/{args.split}/{audio_id}",
                "caption": item["prompt"],
                "duration": item.get("duration", args.duration),
            }
            dst.write(json.dumps(converted, ensure_ascii=False) + "\n")
            num_records += 1

    print(f"Converted {num_records} records")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
