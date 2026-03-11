# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py

from typing import Any, Dict, List, Optional, Union, Callable
import torch
import numpy as np
from flow_grpo.fluxaudio_sde_with_logprob import sde_step_with_logprob
from tqdm import tqdm

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

@torch.no_grad()
def pipeline_with_logprob(
    transformer,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    noise_level: float = 0.7,
    timesteps: Optional[List[float]] = None,
):
    batch_size = prompt_embeds.shape[0]
    device = transformer.device

    if hasattr(transformer, 'module'): 
        transformer = transformer.module
    conditions = transformer.preprocess_conditions(prompt_embeds, pooled_prompt_embeds)
    empty_conditions = transformer.get_empty_conditions(batch_size)
    cfg_ode_wrapper = lambda t, x: transformer.ode_wrapper(
        t, x, conditions, empty_conditions, guidance_scale)
        
    latents = torch.empty(
        batch_size,
        transformer.latent_seq_len,
        transformer.latent_dim,
        device=device
    ).normal_(generator=generator)
    # timesteps = torch.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if timesteps is None else timesteps
    timesteps = torch.linspace(1.0, 0, num_inference_steps+1) if timesteps is None else timesteps
    timesteps = timesteps.to(device)


    all_latents = [latents]
    all_log_probs = []

    # Denoising loop
    for i, t in enumerate(timesteps[:-1]):
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        noise_pred = cfg_ode_wrapper(t, latents)
        latents_dtype = latents.dtype

        latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
            timesteps,
            noise_pred.float(), 
            t.unsqueeze(0).repeat(latents.shape[0]), 
            latents.float(),
            noise_level=noise_level,
        )
        if latents.dtype != latents_dtype:
            latents = latents.to(latents_dtype)
        all_latents.append(latents)
        all_log_probs.append(log_prob)

    return all_latents, all_log_probs, timesteps
