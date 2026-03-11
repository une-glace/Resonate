export CUDA_VISIBLE_DEVICES=1

export HF_HOME="/inspire/hdd/global_user/chenxie-25019/cache/meanaudio"
export HF_HUB_OFFLINE=1

ckpt_path=./weights/Resonate_GRPO.pth
config_name=GRPO_flant5_44kMMVAE_fluxaudio_audiocaps_qwen25omni_semantic
output_path=/inspire/hdd/project/embodied-multimodality/public/xqli/TTA/Resonate/output/resonate_grpo_h200/tta-bench

python eval.py \
    --config ${config_name}.yaml \
    --model_path "$ckpt_path" \
    --output $output_path \
    --num_steps 25 \
    --cfg_strength 4.5 \
    --eval_dataset tta-bench-acc

bash /inspire/hdd/project/embodied-multimodality/public/xqli/TTA/TTA-Bench-tools/run_eval.sh $output_path