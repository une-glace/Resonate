DEBUG=False

export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=$(echo ${CUDA_VISIBLE_DEVICES:-""} | tr ',' '\n' | wc -l)
config_name=DPO_flant5_44kMMVAE_fluxaudio_audiocaps_qwen25omni_semantic_offline
beta=100

if [ "$DEBUG" = True ]; then
    exp_id=debug
else
    exp_id=${config_name}_numgpus${NUM_GPUS}_beta${beta}
fi

OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
torchrun --standalone --nproc_per_node=$NUM_GPUS \
    train_dpo.py \
    --config-name $config_name \
    exp_id=$exp_id \
    use_wandb=True \
    train.beta=${beta} \
    debug=$DEBUG