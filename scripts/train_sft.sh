DEBUG=False

export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=$(echo ${CUDA_VISIBLE_DEVICES:-""} | tr ',' '\n' | wc -l)


num_iterations=500_000
config_name=SFT_Temporal_online_feature_qwen3_06b_fluxaudio_44kMMVAE
bsz=128 # only for fixed bsz


if [ "$DEBUG" = True ]; then
    bsz=16
    exp_id=debug_${config_name}
else
    exp_id=${config_name}_numgpus${NUM_GPUS}_niter${num_iterations}
fi

OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
torchrun --standalone --nproc_per_node=$NUM_GPUS \
    train_fm.py \
    --config-name $config_name \
    exp_id=$exp_id \
    compile=False \
    batch_size=${bsz} \
    eval_batch_size=32 \
    num_iterations=${num_iterations} \
    use_meanflow=False \
    cfg_strength=4.5 \
    ++use_rope=True \
    ++use_wandb=False \
    ++debug=$DEBUG