import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
from argparse import ArgumentParser
from pathlib import Path
import torch
import torchaudio
from hydra import compose, initialize
from huggingface_hub import snapshot_download
from resonate.eval_utils import generate_fm, setup_eval_logging
from resonate.model.flow_matching import FlowMatching
from resonate.model.networks import FluxAudio, get_model
from resonate.model.utils.features_utils import FeaturesUtils
from resonate.model.sequence_config import CONFIG_16K, CONFIG_44K
from torchaudio.transforms import Resample

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from tqdm import tqdm
log = logging.getLogger()


# python demo.py --prompt "A cat is meowing, followed by guitar sound"
@torch.inference_mode()
def main():
    setup_eval_logging()

    parser = ArgumentParser()
    parser.add_argument('--prompt', type=str, help='Input prompt', default='')
    parser.add_argument('--negative_prompt', type=str, help='Negative prompt', default='')
    parser.add_argument('--duration', type=float, default=10)
    parser.add_argument('--cfg_strength', type=float, default=4.5)
    parser.add_argument('--num_steps', type=int, default=25)

    parser.add_argument('--output', type=Path, help='Output directory', default='./output')
    parser.add_argument('--seed', type=int, help='Random seed', default=123)
    parser.add_argument('--full_precision', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug: 
        import debugpy
        debugpy.listen(6666) 
        print("Waiting for debugger attach (rank 0)...")
        debugpy.wait_for_client()  
    
    with initialize(version_base="1.3.2", config_path="config"):
        cfg = compose(config_name='GRPO_flant5_44kMMVAE_fluxaudio_audiocaps_qwen25omni_semantic')
    
    if cfg.audio_sample_rate == 16000:
        seq_cfg = CONFIG_16K
    elif cfg.audio_sample_rate == 44100:
        seq_cfg = CONFIG_44K
    else:
        raise ValueError(f'Invalid audio sample rate: {cfg.audio_sample_rate}')

    negative_prompt: str = args.negative_prompt
    output_dir: str = args.output.expanduser()
    seed: int = args.seed
    num_steps: int = args.num_steps
    duration: float = args.duration
    cfg_strength: float = args.cfg_strength

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        log.warning('CUDA/MPS are not available, running on CPU')
    dtype = torch.float32 if args.full_precision else torch.bfloat16

    output_dir.mkdir(parents=True, exist_ok=True)
    # load a pretrained model with train-style params (only overwrite model-loading related fields)
    use_rope = cfg.get('use_rope', True)
    text_dim = cfg.get('text_dim', None)
    text_c_dim = cfg.get('text_c_dim', None)
    
    model_path = Path('./weights/Resonate_GRPO.pth')
    if not model_path.exists():
        log.info(f'Model not found at {model_path}')
        log.info('Downloading models to "./weights/"...')
        try:
            weights_dir = Path('./weights')
            weights_dir.mkdir(exist_ok=True)
            snapshot_download(repo_id="AndreasXi/resonate", local_dir="./weights" )
        except Exception as e:
            log.error(f"Failed to download model: {e}")
            raise FileNotFoundError(f"Model file not found and download failed: {model_path}, you may need to download the model manually.")
    
    net: FluxAudio = get_model(cfg.model,
                                use_rope=use_rope,
                                text_dim=text_dim,
                                text_c_dim=text_c_dim).to(device, dtype).eval()
    net.load_weights(torch.load(model_path, map_location=device, weights_only=True))
    log.info(f'Loaded weights from {model_path}')

    # misc setup
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

    encoder_name = cfg.get('text_encoder_name', 'flan-t5')
    if cfg.audio_sample_rate == 16000:
        feature_utils = FeaturesUtils(tod_vae_ckpt=cfg.get('vae_16k_ckpt'),
                                    enable_conditions=True,
                                    encoder_name=encoder_name,
                                    mode='16k',
                                    bigvgan_vocoder_ckpt=cfg.get('bigvgan_vocoder_ckpt'),
                                    need_vae_encoder=True)
    elif cfg.audio_sample_rate == 44100:
        feature_utils = FeaturesUtils(tod_vae_ckpt=cfg.get('vae_44k_ckpt'),
                                    enable_conditions=True,
                                    encoder_name=encoder_name,
                                    mode='44k',
                                    need_vae_encoder=True)
    else:
        raise ValueError(f'Invalid audio sample rate: {cfg.audio_sample_rate}')
        
    feature_utils = feature_utils.to(device, dtype).eval()

    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len)
    log.info(f'Updated seq_cfg latent_seq_len: {seq_cfg.latent_seq_len}')
    # prompts: str = [args.prompt]
    if args.prompt != "": 
        prompts = [args.prompt]
    else: 
        prompts = ['A dog is barking']  # default vanilla prompt
   
    for prompt in tqdm(prompts): 
        log.info(f'Prompt: {prompt}')
        log.info(f'Negative prompt: {negative_prompt}')
        audios = generate_fm([prompt],
                            negative_text=[negative_prompt],
                            feature_utils=feature_utils,
                            net=net,
                            fm=fm,
                            rng=rng,
                            cfg_strength=cfg_strength)
        audio = audios.float().cpu()[0]
        safe_filename = prompt.replace(' ', '_').replace('/', '_').replace('.', '')
        safe_filename = safe_filename[:200]
        save_path = output_dir / f'{safe_filename}--numsteps{num_steps}--seed{args.seed}--duration{args.duration}.wav'
        torchaudio.save(save_path, audio, seq_cfg.sampling_rate)

        log.info(f'Audio saved to {save_path}')
        
        log.info('Memory usage: %.2f GB', torch.cuda.max_memory_allocated() / (2**30))


if __name__ == '__main__':
    main()
