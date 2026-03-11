import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from resonate.ext.rotary_embeddings import compute_rope_rotations
from resonate.model.embeddings import TimestepEmbedder
from resonate.model.low_level import MLP, ChannelLastConv1d, ConvMLP
from resonate.model.transformer_layers import (FinalBlock, JointBlock, MMDitSingleBlock, TransformerBlock, VanillaFinalBlock)

log = logging.getLogger()


@dataclass
class PreprocessedConditions:
    text_f: torch.Tensor
    text_f_c: Optional[torch.Tensor]


class FluxAudio(nn.Module):
    # Flux style latent transformer for TTA, single time step embedding

    def __init__(self,
                 *,
                 latent_dim: int,
                 text_dim: int,
                 text_c_dim: int, 
                 hidden_dim: int,
                 depth: int,
                 fused_depth: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 latent_seq_len: int,
                 text_seq_len: int = 77,
                 latent_mean: Optional[torch.Tensor] = None,
                 latent_std: Optional[torch.Tensor] = None,
                 empty_string_feat: Optional[torch.Tensor] = None,
                 empty_string_feat_c: Optional[torch.Tensor] = None,
                 use_rope: bool = False) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self._latent_seq_len = latent_seq_len  # used as the default latent len during infer
        self._text_seq_len = text_seq_len
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.mm_depth = depth - fused_depth

        self.audio_input_proj = nn.Sequential(
            ChannelLastConv1d(latent_dim, hidden_dim, kernel_size=7, padding=3),
            nn.SELU(),
            ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=7, padding=3),
        )

        self.text_input_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            MLP(hidden_dim, hidden_dim * 4),
        )

        self.text_cond_proj = nn.Sequential(
            nn.Linear(text_c_dim, hidden_dim),
            MLP(hidden_dim, hidden_dim*4)
        )

        self.final_layer = FinalBlock(hidden_dim, latent_dim)  

        self.t_embed = TimestepEmbedder(hidden_dim,
                                        frequency_embedding_size=256,
                                        max_period=10000)
        
        self.joint_blocks = nn.ModuleList([
            JointBlock(hidden_dim,
                         num_heads,
                         mlp_ratio=mlp_ratio,
                         pre_only=(i == depth - fused_depth - 1)) for i in range(depth - fused_depth)  # last layer is pre-only (only appllied to text and vision)
        ])

        self.fused_blocks = nn.ModuleList([
            MMDitSingleBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio, kernel_size=3, padding=1)
            for i in range(fused_depth)
        ])

        if latent_mean is None:
            # these values are not meant to be used
            # if you don't provide mean/std here, we should load them later from a checkpoint
            assert latent_std is None
            latent_mean = torch.ones(latent_dim).view(1, 1, -1).fill_(float('nan'))
            latent_std = torch.ones(latent_dim).view(1, 1, -1).fill_(float('nan'))
        else:
            assert latent_std is not None
            assert latent_mean.numel() == latent_dim, f'{latent_mean.numel()=} != {latent_dim=}'

        if empty_string_feat is None:
            empty_string_feat = torch.zeros((text_seq_len, text_dim))
        if empty_string_feat_c is None: 
            empty_string_feat_c = torch.zeros((text_c_dim))

        assert empty_string_feat.shape[-1] == text_dim, f'{empty_string_feat.shape[-1]} == {text_dim}'
        assert empty_string_feat_c.shape[-1] == text_c_dim, f'{empty_string_feat_c.shape[-1]} == {text_c_dim}'

        self.latent_mean = nn.Parameter(latent_mean.view(1, 1, -1), requires_grad=False)  # (1, 1, d)
        self.latent_std = nn.Parameter(latent_std.view(1, 1, -1), requires_grad=False)   # (1, 1, d)

        self.empty_string_feat = nn.Parameter(empty_string_feat, requires_grad=False) 
        self.empty_string_feat_c = nn.Parameter(empty_string_feat_c, requires_grad=False)


        self.initialize_weights()
        if self.use_rope: 
            log.info("Network: Enabling RoPE embeddings")
            max_latent_len = 2048  # 2048 // 43 = 47.628 seconds of audio
            max_text_len = 77
            head_dim = self.hidden_dim // self.num_heads
            base_freq = 1.0
            
            latent_rot = compute_rope_rotations(max_latent_len,
                                               head_dim,
                                               10000,
                                               freq_scaling=base_freq,
                                               device=self.device)
            text_rot = compute_rope_rotations(max_text_len,
                                             head_dim,
                                             10000,
                                             freq_scaling=base_freq,
                                             device=self.device)
            
            self.register_buffer('_latent_rot_buffer', latent_rot, persistent=False)
            self.register_buffer('_text_rot_buffer', text_rot, persistent=False)
        else: 
            log.info("Network: RoPE embedding disabled")

    def update_seq_lengths(self, latent_seq_len: int) -> None: 
        self._latent_seq_len = latent_seq_len

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.joint_blocks:
            nn.init.constant_(block.latent_block.adaLN_modulation[-1].weight, 0)  # the linear layer -> 6 coefficients
            nn.init.constant_(block.latent_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].bias, 0)
        for block in self.fused_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv.weight, 0)
        nn.init.constant_(self.final_layer.conv.bias, 0)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # return (x - self.latent_mean) / self.latent_std
        return x.sub_(self.latent_mean).div_(self.latent_std)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        # return x * self.latent_std + self.latent_mean
        return x.mul_(self.latent_std).add_(self.latent_mean)

    def preprocess_conditions(self, text_f: torch.Tensor, text_f_c: torch.Tensor) -> PreprocessedConditions:  
        """
        cache computations that do not depend on the latent/time step
        i.e., the features are reused over steps during inference
        """

        bs = text_f.shape[0]

        # get global and local text features
        # NOTE here the order of projection has been changed so global and local features are projected seperately 
        text_f_c = self.text_cond_proj(text_f_c)  # (B, D)
        text_f = self.text_input_proj(text_f)  # (B, VN, D)

        return PreprocessedConditions(text_f=text_f,
                                      text_f_c=text_f_c)

    def predict_flow(self, latent: torch.Tensor, t: torch.Tensor,
                     conditions: PreprocessedConditions) -> torch.Tensor:
        """
        for non-cacheable computations
        """

        text_f = conditions.text_f
        text_f_c = conditions.text_f_c

        latent = self.audio_input_proj(latent)  # (B, N, D)

        if self.use_rope:
            latent_len = latent.shape[1]
            text_len = text_f.shape[1]
            
            latent_rot = self._latent_rot_buffer[:, :latent_len, :, :, :]
            text_rot = self._text_rot_buffer[:, :text_len, :, :, :]
        else:
            latent_rot = None
            text_rot = None

        global_c = self.t_embed(t).unsqueeze(1) + text_f_c.unsqueeze(1)  # (B, 1, D)

        extended_c = global_c  # extended_c: Latent_c, global_c: Text_c

        for block in self.joint_blocks:
            latent, text_f = block(latent, text_f, global_c, extended_c, latent_rot, text_rot)  # (B, N, D)

        for block in self.fused_blocks:
            latent = block(latent, extended_c, latent_rot)

        flow = self.final_layer(latent, extended_c)  # (B, N, out_dim), remove t
        return flow

    def forward(self, latent: torch.Tensor, text_f: torch.Tensor, text_f_c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        latent: (B, N, C) 
        text_f: (B, T, D)
        t: (B,)
        """
        conditions = self.preprocess_conditions(text_f, text_f_c)  # cachable operations 
        flow = self.predict_flow(latent, t, conditions)  # non-cachable operations
        return flow

    def get_empty_string_sequence(self, bs: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.empty_string_feat.unsqueeze(0).expand(bs, -1, -1), \
            self.empty_string_feat_c.unsqueeze(0).expand(bs, -1)  # (b, d)

    def get_empty_conditions(
            self,
            bs: int,
            *,
            negative_text_features: Optional[torch.Tensor] = None) -> PreprocessedConditions:
        if negative_text_features is not None:  
            empty_string_feat, empty_string_feat_c = negative_text_features  
        else:
            empty_string_feat, empty_string_feat_c = self.get_empty_string_sequence(1)

        conditions = self.preprocess_conditions(empty_string_feat,
                                                empty_string_feat_c)  # use encoder's empty features
        
        if negative_text_features is None:
            conditions.text_f = conditions.text_f.expand(bs, -1, -1)
            
            conditions.text_f_c = conditions.text_f_c.expand(bs, -1)

        return conditions

    def ode_wrapper(self, t: torch.Tensor, latent: torch.Tensor, conditions: PreprocessedConditions,
                    empty_conditions: PreprocessedConditions, cfg_strength: float) -> torch.Tensor:
        t = t * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)

        if cfg_strength < 1.0:
            return self.predict_flow(latent, t, conditions)
        else:
            return (cfg_strength * self.predict_flow(latent, t, conditions) +
                    (1 - cfg_strength) * self.predict_flow(latent, t, empty_conditions))

    def load_weights(self, src_dict) -> None:
        if 't_embed.freqs' in src_dict:
            del src_dict['t_embed.freqs']
        if 'latent_rot' in src_dict:
            del src_dict['latent_rot']
        if 'text_rot' in src_dict:
            del src_dict['text_rot']

        # if 'empty_string_feat_c' not in src_dict.keys():  # FIXME: issue of version mismatch here
        #     src_dict['empty_string_feat_c'] = src_dict['empty_string_feat'].mean(dim=0)
        self.load_state_dict(src_dict, strict=True)

    @property
    def device(self) -> torch.device:
        return self.latent_mean.device

    @property
    def latent_seq_len(self) -> int:
        return self._latent_seq_len
    
    
def fluxaudio_m_44k(**kwargs) -> FluxAudio: 
    num_heads = 7
    return FluxAudio(latent_dim=40,
                     hidden_dim=64 * num_heads,  # 7*64=448
                     depth=52,
                     fused_depth=36,
                     num_heads=num_heads,
                     latent_seq_len=430,  # for 10s audio
                     **kwargs)


def get_model(name: str, **kwargs) -> nn.Module:
    log.info(f'Getting model: {name}')

    if name == 'fluxaudio_m_44k': 
        return fluxaudio_m_44k(**kwargs)

    raise ValueError(f'Unknown model name: {name}')