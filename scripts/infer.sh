export CUDA_VISIBLE_DEVICES=0

ckpt_path=./weights/Resonate_GRPO.pth
prompt="Keys typing repeatedly"
config_name=GRPO_flant5_44kMMVAE_fluxaudio_audiocaps_qwen25omni_semantic
output_path=./output/resonate_grpo/output

python infer.py \
    --config ${config_name}.yaml \
    --prompt "$prompt" \
    --model_path "$ckpt_path" \
    --output $output_path \
    --num_steps 25 \
    --cfg_strength 4.5 \
    --duration 10 \
    --seed 123 \
    --full_precision