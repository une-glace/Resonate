import os
import datetime
import time
import json
import hashlib
import contextlib
import random
from collections import defaultdict
from concurrent import futures
from functools import partial
from resonate.model.flow_matching import FlowMatching

import torch
import torch.nn.functional as F
import torch.distributed as distributed
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import wandb
import soundfile as sf
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from peft import LoraConfig, get_peft_model, PeftModel

# resonate imports
from torch.distributed.elastic.multiprocessing.errors import record
from resonate.model.sequence_config import CONFIG_16K, CONFIG_44K
from resonate.model.utils.features_utils import FeaturesUtils
from resonate.model.networks import get_model

# FlowGRPO specific imports
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.fluxaudio_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.ema import EMAModuleWrapper
from flow_grpo.rewards import multi_score

def pbar(iterable, desc=None, position=0, leave=True, disable=True, **kwargs):
    if disable:
        return iterable
    else: 
        from tqdm.auto import tqdm
        return tqdm(iterable, desc=desc, position=position, leave=leave, dynamic_ncols=True, **kwargs)

logger = get_logger(__name__)

# --- Datasets and Samplers ---

class AudioPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas
    
class AudioTemporalDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
            self.metadatas = [item['phrases'] for item in self.data]
            from resonate.data.online_audio import format_variant1
            self.format_fn = [format_variant1]
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        phrases = self.metadatas[idx]
        format_fn = random.choice(self.format_fn)
        prompt = format_fn(phrases)
        return {"prompt": prompt, "metadata": phrases}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            # shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            # shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                # per_card_samples.append(shuffled_samples[start:end])
                per_card_samples.append(repeated_indices[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch

# --- Utils ---

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big') 
        seed = (base_seed + prompt_hash_int) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    return model

def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_path = os.path.join(save_dir)
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, f'model_{global_step}.pth')
    
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrapped = unwrap_model(transformer, accelerator)
        if isinstance(unwrapped, PeftModel):
             unwrapped.save_pretrained(os.path.join(save_path, f"lora_{global_step}"))
        else:
             torch.save(unwrapped.state_dict(), model_path)
        logger.info(f'Network weights saved to {model_path}.')
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)

# --- Eval Logic ---
def eval(pipeline, test_dataloader, features, config, accelerator, global_step, reward_fn, executor, autocast, ema, transformer_trainable_parameters):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    
    with torch.no_grad():
        neg_prompt_embed, neg_pooled_prompt_embed = features.encode_text([''])
        neg_prompt_embed = neg_prompt_embed[0].unsqueeze(0).to(accelerator.device)
        if neg_pooled_prompt_embed is not None:
             neg_pooled_prompt_embed = neg_pooled_prompt_embed[0].unsqueeze(0).to(accelerator.device)
        else:
             neg_pooled_prompt_embed = neg_prompt_embed.mean(dim=1)

    all_rewards = defaultdict(list)
    
    for test_batch in pbar(test_dataloader, desc="Eval: ", disable=not accelerator.is_main_process):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = features.encode_text(prompts)
        prompt_embeds = prompt_embeds.to(accelerator.device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
        
        with autocast():
            with torch.no_grad():
                latents, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    noise_level=0,
                )
        
        last_latent = latents[-1]
        mel = features.decode(last_latent)
        audios = features.vocode(mel).squeeze(1)

        rewards_future = executor.submit(reward_fn, audios, prompts, prompt_metadata, vae_sr=16000 if config.audio_sample_rate==16000 else 44100, only_strict=False)
        rewards, _ = rewards_future.result()

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)
    
    if accelerator.is_main_process and config.use_wandb:
         wandb.log({f"eval_reward_{k}": np.concatenate(v).mean() for k, v in all_rewards.items()}, step=global_step)

    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)


# --- Main Training Loop ---
@record
@hydra.main(version_base='1.3.2', config_path='config', config_name='train_config.yaml')
def train(cfg: DictConfig):
    
    if cfg.get("debug", False): 
        import debugpy
        if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
            debugpy.listen(6665) 
            print(f'Waiting for debugger attach (rank {os.environ["RANK"]})...')
            debugpy.wait_for_client()  
            
    config = cfg
    if config.sample.num_audio_per_prompt % 2 != 0:
        raise ValueError("For DPO training, num_audio_per_prompt (k) must be even.")

    save_dir = os.path.join(HydraConfig.get().run.dir)
    accelerator_config = ProjectConfiguration(project_dir=save_dir, automatic_checkpoint_naming=True, total_limit=config.num_checkpoint_limit)
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps, 
    )

    if accelerator.is_main_process and config.use_wandb:
        wandb.init(project="flow_dpo", name=cfg.exp_id, config=dict(config))
    
    logger.info(f"\n{config}")
    set_seed(config.seed, device_specific=True)

    if cfg.audio_sample_rate == 16000:
        mode, seq_cfg, vae_sr = '16k', CONFIG_16K, 16000
    elif config.audio_sample_rate == 44100:
        mode, seq_cfg, vae_sr = '44k', CONFIG_44K, 44100
    else:
        raise ValueError(f'Invalid sample rate: {cfg.audio_sample_rate}')

    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16": inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16": inference_dtype = torch.bfloat16

    if mode == '16k':
        features = FeaturesUtils(tod_vae_ckpt=cfg['vae_16k_ckpt'], bigvgan_vocoder_ckpt=cfg['bigvgan_vocoder_ckpt'], encoder_name=cfg['text_encoder_name'], enable_conditions=True, mode=mode, need_vae_encoder=cfg.get('online_feature_extraction', False))
    else:
        features = FeaturesUtils(tod_vae_ckpt=cfg['vae_44k_ckpt'], encoder_name=cfg['text_encoder_name'], enable_conditions=True, mode=mode, need_vae_encoder=cfg.get('online_feature_extraction', False))
    
    features = features.eval().to(accelerator.device, dtype=torch.float32)
    if cfg.compile: features.compile()

    # Precompute negative embeddings for CFG
    with torch.no_grad():
        neg_prompt_embed, neg_pooled_prompt_embed = features.encode_text([''])
        neg_prompt_embed = neg_prompt_embed[0].to(accelerator.device) # [Seq, D]
        if neg_pooled_prompt_embed is not None: 
            neg_pooled_prompt_embed = neg_pooled_prompt_embed[0].to(accelerator.device)
        else: 
            neg_pooled_prompt_embed = neg_prompt_embed.mean(dim=0)
    
    latent_mean, latent_std = torch.load(cfg.latent_mean), torch.load(cfg.latent_std)
    transformer = get_model(cfg.model, text_dim=cfg.text_dim, text_c_dim=cfg.text_c_dim, latent_mean=latent_mean, latent_std=latent_std, empty_string_feat=neg_prompt_embed, empty_string_feat_c=neg_pooled_prompt_embed, use_rope=cfg.use_rope)
    transformer.load_weights(torch.load(cfg.weight, map_location=accelerator.device, weights_only=True))
    transformer.to(accelerator.device, dtype=inference_dtype)

    if config.use_lora:
        transformer_lora_config = LoraConfig(
            r=32, lora_alpha=64, init_lora_weights="gaussian",
            target_modules=["attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out", "attn.to_k", "attn.to_out.0", "attn.to_q", "attn.to_v"]
        )
        if config.train.lora_path:
            transformer = PeftModel.from_pretrained(transformer, config.train.lora_path)
            transformer.set_adapter("default")
        else:
            transformer = get_peft_model(transformer, transformer_lora_config)
        ref_transformer = None 
    else:
        import copy
        ref_transformer = copy.deepcopy(transformer)
        for p in ref_transformer.parameters(): p.requires_grad = False
        ref_transformer.eval()

    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)

    if config.allow_tf32: torch.backends.cuda.matmul.allow_tf32 = True
    if config.train.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(transformer_trainable_parameters, lr=config.train.learning_rate, betas=(config.train.adam_beta1, config.train.adam_beta2), weight_decay=config.train.adam_weight_decay, eps=config.train.adam_epsilon)
    else:
        optimizer = torch.optim.AdamW(transformer_trainable_parameters, lr=config.train.learning_rate, betas=(config.train.adam_beta1, config.train.adam_beta2), weight_decay=config.train.adam_weight_decay, eps=config.train.adam_epsilon)

    if config.prompt_fn == "audioprompt":
        train_dataset = AudioPromptDataset(config.dataset, 'train')
        test_dataset = AudioPromptDataset(config.dataset, 'test')
        collate_fn = AudioPromptDataset.collate_fn
    elif config.prompt_fn == "audio_temporal_prompt":
        train_dataset = AudioTemporalDataset(config.dataset, 'train')
        test_dataset = AudioTemporalDataset(config.dataset, 'test')
        collate_fn = AudioTemporalDataset.collate_fn
    else:
        raise NotImplementedError
    
    train_sampler = DistributedKRepeatSampler(train_dataset, batch_size=config.sample.train_batch_size, k=config.sample.num_audio_per_prompt, num_replicas=accelerator.num_processes, rank=accelerator.process_index, seed=42)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=1, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=config.sample.test_batch_size, collate_fn=collate_fn, shuffle=False, num_workers=4)

    reward_fn = multi_score(accelerator.device, config.reward_fn)
    eval_reward_fn = multi_score(accelerator.device, config.reward_fn)
    
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    transformer, optimizer, train_dataloader, test_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader, test_dataloader)
    executor = futures.ThreadPoolExecutor(max_workers=8)

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)
    online_dpo = config.get('online_dpo', True)
    logger.info(f"Starting DPO Training: Batch Size {config.train.batch_size}, K={config.sample.num_audio_per_prompt}, Online: {online_dpo}")

    while epoch < config.total_epoch:
        if epoch % config.eval_freq == 0 and config.do_eval and epoch >= 0:
            transformer.eval()
            eval(transformer, test_dataloader, features, config, accelerator, global_step, eval_reward_fn, executor, autocast, ema, transformer_trainable_parameters)
        if epoch % config.save_freq == 0 and epoch > 0:
            save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)

        transformer.eval()
        samples = []
        
        for i in pbar(range(config.sample.num_batches_per_epoch), desc=f"Epoch {epoch}: sampling", disable=not accelerator.is_main_process):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            try:
                prompts, prompt_metadata = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = features.encode_text(prompts)
            generator = create_generator(prompts, base_seed=epoch*10000+i) if config.sample.same_latent else None
            
            with autocast(), torch.no_grad():
                # latents, _, _ = pipeline_with_logprob(
                #     transformer,  
                #     prompt_embeds=prompt_embeds,
                #     pooled_prompt_embeds=pooled_prompt_embeds,
                #     num_inference_steps=config.sample.num_steps,
                #     guidance_scale=config.sample.guidance_scale,
                #     noise_level=0,
                #     generator=generator,
                # )
                def generate(transformer, prompts):
                    if hasattr(transformer, 'module'): 
                        transformer = transformer.module
                        
                    bs = len(prompts)
                    x0 = torch.randn(bs, transformer.latent_seq_len, transformer.latent_dim, device=transformer.device, generator=generator)
                    preprocessed_conditions = transformer.preprocess_conditions(prompt_embeds, pooled_prompt_embeds)

                    empty_conditions = transformer.get_empty_conditions(bs)
                    cfg_ode_wrapper = lambda t, x: transformer.ode_wrapper(t, 
                                                                        x, 
                                                                        preprocessed_conditions, 
                                                                        empty_conditions, 
                                                                        config.sample.guidance_scale)
                    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=config.sample.num_steps)
                    x1 = fm.to_data(cfg_ode_wrapper, x0)
                    latents = [x1]
                    return latents
                
                if online_dpo: 
                    latents = generate(transformer, prompts)
                else: 
                    latents = generate(ref_transformer, prompts)

            last_latent = latents[-1] # [Batch, Seq, D]
            mel = features.decode(last_latent)
            audios = features.vocode(mel).squeeze(1)

            rewards_future = executor.submit(reward_fn, audios, prompts, prompt_metadata, vae_sr=vae_sr, only_strict=True)
            time.sleep(0) 
            
            samples.append({
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "latents": last_latent.detach().cpu(), 
                "rewards_future": rewards_future,
                "audios": audios.detach().cpu(), 
                "prompts": prompts               
            })

        paired_samples = []
        for batch_idx, sample in enumerate(pbar(samples, desc="Pairing", disable=not accelerator.is_main_process)):
            rewards_dict, _ = sample["rewards_future"].result()
            
            # Save Debug Rollout (JSONL + Audio)
            if epoch % 100 == 0 and accelerator.is_main_process: 
                rollout_root = os.path.join(save_dir, "dpo_rollout")
                os.makedirs(rollout_root, exist_ok=True)
                
                epoch_dir = os.path.join(rollout_root, f"epoch_{epoch:04d}")
                audio_dir = os.path.join(epoch_dir, "audio")
                os.makedirs(audio_dir, exist_ok=True)

                jsonl_path = os.path.join(epoch_dir, "metadata.jsonl")
                
                current_audios = sample["audios"]
                current_prompts = sample["prompts"]

                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for b in range(len(current_prompts)):
                        audio_name = f"epoch{epoch:04d}_batch{batch_idx:03d}_sample{b:02d}.wav"
                        audio_path = os.path.join(audio_dir, audio_name)

                        sf.write(
                            audio_path,
                            current_audios[b].numpy(), 
                            samplerate=vae_sr,
                        )
                        
                        record = {
                            "epoch": epoch,
                            "batch_idx": batch_idx,
                            "sample_idx": b,
                            "audio_path": audio_path,
                            "prompt": current_prompts[b]
                        }
                        
                        for r_key, r_val in rewards_dict.items():
                            if hasattr(r_val, '__getitem__'):
                                record[r_key] = float(r_val[b])
                            else:
                                record[r_key] = float(r_val)
                                
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                if batch_idx == 0:
                    logger.info(f"[DEBUG] Saving rollout samples to {epoch_dir}")

            scores = torch.tensor(rewards_dict['avg'], device=accelerator.device)
            
            if accelerator.is_main_process and batch_idx == 0:
                logger.info(f"Step {global_step} | reward mean: {scores.mean().item()}")
                if config.use_wandb: 
                    wandb.log({"train_reward_mean": scores.mean().item()}, step=global_step)
            
            k = config.sample.num_audio_per_prompt
            bsz_total = scores.shape[0]
            num_prompts = bsz_total // k
            
            scores = scores.reshape(num_prompts, k)
            latents = sample["latents"].to(accelerator.device).reshape(num_prompts, k, *sample["latents"].shape[1:])
            p_embeds = sample["prompt_embeds"].reshape(num_prompts, k, *sample["prompt_embeds"].shape[1:])
            pp_embeds = sample["pooled_prompt_embeds"].reshape(num_prompts, k, *sample["pooled_prompt_embeds"].shape[1:])

            swap_mask = scores[:, 1] > scores[:, 0]
            
            latents_w = torch.where(swap_mask.view(-1, 1, 1), latents[:, 1], latents[:, 0])
            latents_l = torch.where(swap_mask.view(-1, 1, 1), latents[:, 0], latents[:, 1])
            
            pe_w = p_embeds[:, 0]
            pe_l = p_embeds[:, 0]
            ppe_w = pp_embeds[:, 0]
            ppe_l = pp_embeds[:, 0]

            batch_latents = torch.cat([latents_w, latents_l], dim=0)
            batch_pe = torch.cat([pe_w, pe_l], dim=0)
            batch_ppe = torch.cat([ppe_w, ppe_l], dim=0)

            paired_samples.append({
                "latents": batch_latents,
                "prompt_embeds": batch_pe,
                "pooled_prompt_embeds": batch_ppe
            })

        transformer.train()
        for inner_epoch in range(config.train.num_inner_epochs):
            random.shuffle(paired_samples)
            for step, batch in enumerate(pbar(paired_samples, desc=f"Epoch {epoch}.{inner_epoch}: Train", disable=not accelerator.is_main_process)):
                
                model_input = batch["latents"].to(accelerator.device)
                embeds = batch["prompt_embeds"].to(accelerator.device)
                pooled_embeds = batch["pooled_prompt_embeds"].to(accelerator.device)
                
                bsz = model_input.shape[0]
                N = bsz // 2
                
                noise = torch.randn_like(model_input)
                # Ensure noise is identical for Winner and Loser to reduce variance in DPO
                noise_w = noise[:N]
                noise = torch.cat([noise_w, noise_w], dim=0)
                
                t = torch.rand((N,), device=accelerator.device)
                t = torch.cat([t, t], dim=0)
                t_expand = t.view(-1, 1, 1)
                
                # Path: Audio -> Noise
                # x_t = t * x0(noise) + (1-t) * x1(data/audio)
                noisy_latents = t_expand * noise + (1 - t_expand) * model_input
                target_v = noise - model_input

                # [Modified CFG Training Logic]
                # Instead of applying CFG formula on outputs, we randomly drop condition inputs
                if config.train.cfg:
                    cfg_drop_rate = config.train.get("cfg_drop_rate", 0.1)
                    # Create drop mask [B]
                    drop_mask = torch.rand(bsz, device=accelerator.device) < cfg_drop_rate
                    
                    if drop_mask.any():
                        # neg_prompt_embed: [Seq, D] -> [1, Seq, D] -> expand to batch
                        neg_pe_batch = neg_prompt_embed.unsqueeze(0).expand(bsz, -1, -1)
                        neg_ppe_batch = neg_pooled_prompt_embed.unsqueeze(0).expand(bsz, -1)
                        
                        embeds[drop_mask] = neg_pe_batch[drop_mask]
                        pooled_embeds[drop_mask] = neg_ppe_batch[drop_mask]

                with accelerator.accumulate(transformer):
                    with autocast():
                        # Standard conditional flow matching forward (with some prompts potentially dropped)
                        model_pred = transformer(
                            latent=noisy_latents,
                            text_f=embeds,
                            text_f_c=pooled_embeds,
                            t=t
                        )
                    
                    # Reference Model Forward
                    with torch.no_grad():
                        with autocast():
                            if config.use_lora:
                                with transformer.module.disable_adapter():
                                    ref_pred = transformer(
                                        latent=noisy_latents,
                                        text_f=embeds, # Same inputs (including drops) to Ref
                                        text_f_c=pooled_embeds,
                                        t=t
                                    )
                            else:
                                ref_pred = ref_transformer(
                                    latent=noisy_latents,
                                    text_f=embeds,
                                    text_f_c=pooled_embeds,
                                    t=t
                                )
                    
                    target_v = target_v.float()
                    model_pred = model_pred.float()
                    ref_pred = ref_pred.float()
                    
                    raw_model_loss = (model_pred - target_v).pow(2).mean(dim=[1, 2])
                    raw_ref_loss = (ref_pred - target_v).pow(2).mean(dim=[1, 2])
                    
                    model_losses_w, model_losses_l = raw_model_loss[:N], raw_model_loss[N:]
                    ref_losses_w, ref_losses_l = raw_ref_loss[:N], raw_ref_loss[N:]
                    
                    w_diff = model_losses_w - ref_losses_w
                    l_diff = model_losses_l - ref_losses_l
                    
                    inside_term = -0.5 * config.train.beta * (w_diff - l_diff)
                    loss = -F.logsigmoid(inside_term).mean()
                    
                    reward_accuracies = (inside_term > 0).float().mean()
                    
                    accelerator.backward(loss)
                    
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(transformer.parameters(), config.train.max_grad_norm)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                
                if accelerator.sync_gradients:
                    if accelerator.is_main_process and config.use_wandb:
                        wandb.log({
                            "loss": loss.item(),
                            "accuracy": reward_accuracies.item(),
                            "margin": inside_term.mean().item(),
                            "lr": optimizer.param_groups[0]["lr"],
                            "epoch": epoch
                        }, step=global_step)
                    
                    if config.train.ema:
                        ema.step(transformer_trainable_parameters, global_step)
                    
                    global_step += 1

        epoch += 1

if __name__ == "__main__":
    train()