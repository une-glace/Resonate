from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.transforms import Normalize
from transformers import T5EncoderModel, AutoTokenizer

from resonate.ext.autoencoder import AutoEncoderModule
from resonate.ext.mel_converter import get_mel_converter
from resonate.model.utils.distributions import DiagonalGaussianDistribution
import logging
from typing import Union, List


def patch_clip(clip_model):
    # a hack to make it output last hidden states
    # https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/model.py#L269
    def new_encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        return F.normalize(x, dim=-1) if normalize else x

    clip_model.encode_text = new_encode_text.__get__(clip_model)
    return clip_model


class FeaturesUtils(nn.Module):

    def __init__(
        self,
        *,
        tod_vae_ckpt: Optional[str] = None,
        bigvgan_vocoder_ckpt: Optional[str] = None,
        enable_conditions: bool = True,
        encoder_name=Literal['clip', 'flan-t5', 'flan-t5-clap', 'flan-t5-clap-cat', 'umT5', 'qwen3-4b', 'qwen25-omni-7b'], 
        mode=Literal['16k', '44k'],
        need_vae_encoder: bool = True,
    ):
        super().__init__()
        
        if enable_conditions:
            self.encoder_name = encoder_name
            if encoder_name == 'flan-t5': 
                logging.info('FeatureUtils: Loading google/flan-t5-large ... ')   
                self.tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large')
                self.text_encoder = T5EncoderModel.from_pretrained('google/flan-t5-large').eval()

            elif encoder_name == 'flan-t5-clap' or encoder_name == 'flan-t5-clap-cat':
                import laion_clap
                self.tokenizer = AutoTokenizer.from_pretrained('../../models/flan-t5-large')
                self.text_encoder = T5EncoderModel.from_pretrained('../../models/flan-t5-large').eval()
                self.laion_clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base').eval()
                self._clap_ckpt_path = "./weights/music_speech_audioset_epoch_15_esc_89.98.pt"  
                self.laion_clap_model.load_ckpt(self._clap_ckpt_path, verbose=False)
            elif encoder_name == 'qwen3-06b':
                logging.info('FeatureUtils: Loading Qwen/Qwen2.5-0.5B ...')
                self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
                # Qwen is a decoder-only model, but used here for feature extraction
                from transformers import AutoModel
                self.text_encoder = AutoModel.from_pretrained('Qwen/Qwen2.5-0.5B').eval()
            else: 
                raise ValueError(f"Encoder {encoder_name} is not allowed, select from ['clip', 'flan-t5', 'flan-t5-clap', 'flan-t5-clap-cat', 'umT5', 'qwen3-4b', 'qwen25-omni-7b', 'qwen3-06b']")

        else:
            self.text_encoder = None
            self.tokenizer = None

        if tod_vae_ckpt is not None:
            self.mel_converter = get_mel_converter(mode)
            self.tod = AutoEncoderModule(vae_ckpt_path=tod_vae_ckpt,
                                         vocoder_ckpt_path=bigvgan_vocoder_ckpt,
                                         mode=mode,
                                         need_vae_encoder=need_vae_encoder)
        else:
            self.tod = None

    def compile(self):
        if self.text_encoder is not None and hasattr(self.text_encoder, 'encode_text'):
            self.text_encoder.encode_text = torch.compile(self.text_encoder.encode_text)
        self.encode_text = torch.compile(self.encode_text)
        self.decode = torch.compile(self.decode)
        self.vocode = torch.compile(self.vocode)

    def train(self, mode: bool) -> None:
        return super().train(False)

    # @torch.inference_mode()
    @torch.no_grad()
    def encode_text(self, text: list[str]) -> torch.Tensor:
        assert self.text_encoder is not None, 'Text encoder is not loaded'
        assert self.tokenizer is not None, 'Tokenizer is not loaded'
        # x: (B, L)
        if self.encoder_name == 'clip': 
            tokens = self.tokenizer(text).to(self.device)
            text_features = self.text_encoder.encode_text(tokens, normalize=True)
            text_features_c = None
        elif self.encoder_name == 'flan-t5': 
            tokens = self.tokenizer(
                text, 
                max_length=77, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            )
            input_ids, attention_mask = tokens.input_ids.cuda(), tokens.attention_mask.cuda()
            text_features = self.text_encoder(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )[0]
            text_features_c = text_features.mean(dim=1)
        elif self.encoder_name == 'flan-t5-clap' or self.encoder_name == 'flan-t5-clap-cat': 
            tokens = self.tokenizer(
                text, 
                max_length=77, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            )
            input_ids, attention_mask = tokens.input_ids.cuda(), tokens.attention_mask.cuda()
            text_features = self.text_encoder(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )[0]
            text_features_c = self.laion_clap_model.get_text_embedding(text, use_tensor=True)
            
            if self.encoder_name == 'flan-t5-clap-cat': 
                text_features_c = torch.cat([text_features.mean(dim=-2), text_features_c], dim=-1)
        elif self.encoder_name == 'qwen3-06b':
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            tokens = self.tokenizer(
                text,
                max_length=77,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids, attention_mask = tokens.input_ids.to(self.device), tokens.attention_mask.to(self.device)
            # Get hidden states from the last layer
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            text_features = outputs.hidden_states[-1]
            # Use mean pooling or last token for global context
            text_features_c = text_features.mean(dim=1)
            
        return text_features, text_features_c

    # @torch.inference_mode()
    @torch.no_grad()
    def encode_audio(self, x) -> DiagonalGaussianDistribution:
        assert self.tod is not None, 'VAE is not loaded'
        # x: (B * L)
        mel = self.mel_converter(x)
        dist = self.tod.encode(mel)
        return dist

    @torch.inference_mode()
    def vocode(self, mel: torch.Tensor) -> torch.Tensor:
        assert self.tod is not None, 'VAE is not loaded'
        return self.tod.vocode(mel)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        assert self.tod is not None, 'VAE is not loaded'
        return self.tod.decode(z.transpose(1, 2))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


if __name__ == '__main__': 
    # features = FeaturesUtilsAT(
    #     tod_vae_ckpt='./ext_weights/v1-16.pth',
    #     bigvgan_vocoder_ckpt='./ext_weights/best_netG.pt',
    #     mode='16k',
    #     encoder_name='t5'
    # )
    # print(features)

    # clap_ckpt = "./weights/music_speech_audioset_epoch_15_esc_89.98.pt"
    # weights = torch.load(clap_ckpt, weights_only=False)
    # print(weights.keys())

    features = FeaturesUtils(
        tod_vae_ckpt='./weights/v1-16.pth',
        bigvgan_vocoder_ckpt='./weights/best_netG.pt',
        mode='16k',
        encoder_name='byt5-large'
    )
    print(f'Text encoder parameters: {sum(p.numel() for p in features.text_encoder.parameters()):,d}')
    text_features, text_features_c = features.encode_text(['A dog is barking'])
    print(text_features[0, 5:15])
    print(text_features.shape)