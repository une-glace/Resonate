DEBUG=False
export WANDB_MODE=offline

export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=$(echo ${CUDA_VISIBLE_DEVICES:-""} | tr ',' '\n' | wc -l)
config_name=GRPO_flant5_44kMMVAE_fluxaudio_audiocaps_qwen25omni_semantic

if [ "$DEBUG" = True ]; then
    exp_id=debug
else
    exp_id=${config_name}_numgpus$NUM_GPUS
fi

OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
torchrun --standalone --nproc_per_node=$NUM_GPUS \
    train_grpo.py \
    --config-name $config_name \
    exp_id=$exp_id \
    use_wandb=True \
    debug=$DEBUG