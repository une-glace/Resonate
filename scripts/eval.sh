ckpt_path=./exps/TTA/GRPO_flant5_44kMMVAE_fluxaudio_audiocaps_qwen25omni_semantic_numgpus2/model_1000.pth
config_name=GRPO_flant5_44kMMVAE_fluxaudio_audiocaps_qwen25omni_semantic
output_path=./output/GRPO_flant5_44kMMVAE_fluxaudio_audiocaps_qwen25omni_semantic_numgpus2

output_path="$(realpath -m "$output_path")"

python eval.py \
    --config ${config_name}.yaml \
    --model_path "$ckpt_path" \
    --output $output_path \
    --num_steps 25 \
    --cfg_strength 4.5 \
    --eval_dataset tta-bench-acc

cd ttabench
bash run_eval.sh $output_path