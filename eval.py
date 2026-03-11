import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
import math
from argparse import ArgumentParser
from pathlib import Path
import torch
import torchaudio
from hydra import compose, initialize
import pandas as pd
from tqdm import tqdm

from resonate.eval_utils import generate_fm, setup_eval_logging
from resonate.model.flow_matching import FlowMatching
from resonate.model.networks import FluxAudio, get_model
from resonate.model.utils.features_utils import FeaturesUtils
from resonate.model.sequence_config import CONFIG_16K, CONFIG_44K

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from tqdm import tqdm
log = logging.getLogger()


@torch.inference_mode()
def main():
    setup_eval_logging()

    parser = ArgumentParser()
    parser.add_argument('--config_name', type=str, required=True, help='config file name under config/ (e.g., train_config_online_feature_umt5.yaml)')
    parser.add_argument('--eval_dataset', type=str, required=True, help='eval dataset name, e.g. librispeech-pc')
    parser.add_argument('--cfg_strength', type=float, default=4.5)
    parser.add_argument('--num_steps', type=int, default=25)

    parser.add_argument('--output', type=Path, help='Output directory', default='./output')
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--full_precision', action='store_true')
    parser.add_argument('--model_path', type=str, help='Path of trained model')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--enable_speech_prompt', action='store_true', help='Whether or not explicitly instruct the speech')
    args = parser.parse_args()

    if args.debug: 
        import debugpy
        debugpy.listen(6666) 
        print("Waiting for debugger attach (rank 0)...")
        debugpy.wait_for_client()  
    
    with initialize(version_base="1.3.2", config_path="config"):
        cfg = compose(config_name=args.config_name)

    if cfg.audio_sample_rate == 16000:
        seq_cfg = CONFIG_16K
    elif cfg.audio_sample_rate == 44100:
        seq_cfg = CONFIG_44K
    else:
        raise ValueError(f'Invalid audio sample rate: {cfg.audio_sample_rate}')

    output_dir: str = args.output.expanduser()
    seed: int = args.seed
    num_steps: int = args.num_steps
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
    
    net: FluxAudio = get_model(cfg.model,
                                    use_rope=use_rope,
                                    text_dim=text_dim,
                                    text_c_dim=text_c_dim).to(device, dtype).eval()
    net.load_weights(torch.load(args.model_path, map_location=device, weights_only=True))
    log.info(f'Loaded weights from {args.model_path}')
    net.update_seq_lengths(seq_cfg.latent_seq_len)

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
                                    need_vae_encoder=False)
    elif cfg.audio_sample_rate == 44100:
        feature_utils = FeaturesUtils(tod_vae_ckpt=cfg.get('vae_44k_ckpt'),
                                    enable_conditions=True,
                                    encoder_name=encoder_name,
                                    mode='44k',
                                    need_vae_encoder=False)
    feature_utils = feature_utils.to(device, dtype).eval()

    if args.eval_dataset.lower() == 'librispeech-pc':
        metadata_file = '/inspire/hdd/project/embodied-multimodality/public/xqli/TTA/F5-TTS/data/librispeech_pc_test_clean_cross_sentence.lst'
        f = open(metadata_file)
        lines = f.readlines()
        f.close()

        test_set = []
        for line in tqdm(lines):
            ref_utt, ref_dur, ref_txt, gen_utt, gen_dur, gen_txt = line.strip().split("\t")
            test_set.append((gen_utt, gen_txt, float(gen_dur)))

        for file_name, prompt, duration in tqdm(test_set):
            latent_seq_len = int(math.ceil(duration * seq_cfg.sampling_rate / seq_cfg.spectrogram_frame_rate / seq_cfg.latent_downsample_rate))
            if args.enable_speech_prompt: 
                prompt = f"A person says: '{prompt}'"
            log.info(f'Prompt: {prompt}, Duration: {duration}, Latent seq len: {latent_seq_len}')
            audios = generate_fm([prompt],
                                  feature_utils=feature_utils,
                                  net=net,
                                  fm=fm,
                                  rng=rng,
                                  cfg_strength=cfg_strength,
                                  latent_seq_len=latent_seq_len)
            audio = audios.float().cpu()[0]
            save_path = output_dir / f'{file_name}.wav'
            torchaudio.save(save_path, audio, seq_cfg.sampling_rate)
            log.info(f'Audio saved to {save_path}')
        
        log.info('Memory usage: %.2f GB', torch.cuda.max_memory_allocated() / (2**30))
    elif args.eval_dataset.lower() == 'audiocaps':
        metadata_file = '/inspire/hdd/project/embodied-multimodality/public/xqli/TTA/resonate/sets/test-audiocaps.tsv'
        import pandas as pd
        data = pd.read_csv(metadata_file, sep="\t").to_dict('records')
        # for d in tqdm(data): 
        #     audio_id = d['id']
        #     prompt = d['caption']
            
        #     log.info(f'Audio id: {audio_id} Prompt: {prompt}, ')
        #     audios = generate_fm([prompt],
        #                           feature_utils=feature_utils,
        #                           net=net,
        #                           fm=fm,
        #                           rng=rng,
        #                           cfg_strength=cfg_strength)
        #     audio = audios.float().cpu()[0]
        #     save_path = output_dir / f'{audio_id}.wav'
        #     torchaudio.save(save_path, audio, seq_cfg.sampling_rate)
        #     log.info(f'Audio saved to {save_path}')
            
        bsz = 16
        for i in tqdm(range(0, len(data), bsz)):
            batch = data[i:i + bsz]
            audio_ids = [d['id'] for d in batch]
            prompts = [d['caption'] for d in batch]
            for audio_id, prompt in zip(audio_ids, prompts):
                log.info(f'Audio id: {audio_id} Prompt: {prompt}')

            # batch generate
            audios = generate_fm(
                prompts,
                feature_utils=feature_utils,
                net=net,
                fm=fm,
                rng=rng,
                cfg_strength=cfg_strength
            ) 

            audios = audios.float().cpu()
            for audio_id, audio in zip(audio_ids, audios):
                save_path = output_dir / f'{audio_id}.wav'
                audio = audio.detach().cpu()
                if audio.ndim == 1:
                    audio = audio.unsqueeze(0)          # [1, T]
                elif audio.ndim == 2:
                    pass                                # already [C, T]
                elif audio.ndim == 3:
                    audio = audio.squeeze(0)            # [1, T] or [C, T]
                else:
                    raise RuntimeError(f"Unexpected audio shape: {audio.shape}")
                torchaudio.save(save_path, audio, seq_cfg.sampling_rate)
                
    elif args.eval_dataset.lower() == 'tta-bench-acc':
        metadata_file = '/inspire/hdd/project/embodied-multimodality/public/xqli/TTA/TTA-Bench-tools/prompts/acc_prompt.json'
        import json
        data = json.load(open(metadata_file, 'r'))
            
        bsz = 16
        for i in tqdm(range(0, len(data), bsz)):
            batch = data[i:i + bsz]
            audio_ids = [d['id'] for d in batch]
            prompts = [d['prompt_text'] for d in batch]
            for audio_id, prompt in zip(audio_ids, prompts):
                log.info(f'Audio id: {audio_id} Prompt: {prompt}')

            # batch generate
            audios = generate_fm(
                prompts,
                feature_utils=feature_utils,
                net=net,
                fm=fm,
                rng=rng,
                cfg_strength=cfg_strength
            ) 

            audios = audios.float().cpu()
            for audio_id, audio in zip(audio_ids, audios):
                save_path = output_dir / f'{audio_id}.wav'
                audio = audio.detach().cpu()
                if audio.ndim == 1:
                    audio = audio.unsqueeze(0)          # [1, T]
                elif audio.ndim == 2:
                    pass                                # already [C, T]
                elif audio.ndim == 3:
                    audio = audio.squeeze(0)            # [1, T] or [C, T]
                else:
                    raise RuntimeError(f"Unexpected audio shape: {audio.shape}")
                torchaudio.save(save_path, audio, seq_cfg.sampling_rate)
                
                
    elif  "audioset-sl" in args.eval_dataset.lower(): 
        if args.eval_dataset.lower() == "audioset-sl-1k": 
            metadata_file = '/inspire/hdd/project/embodied-multimodality/public/xqli/TTA/resonate/data/AudioSet-SL/GRPO_Meta/test_metadata.jsonl'
        elif args.eval_dataset.lower() == "audioset-sl-200": 
            metadata_file = '/inspire/hdd/project/embodied-multimodality/public/xqli/TTA/resonate/data/AudioSet-SL/GRPO_Meta/test_metadata_200.jsonl'
        import json
        from resonate.data.online_audio import format_variant1

        bsz = 16
        data = [json.loads(d) for d in open(metadata_file, 'r')]

        for i in tqdm(range(0, len(data), bsz)):
            batch = data[i:i + bsz]
            audio_ids = [d['audio_id'] for d in batch]
            prompts = [format_variant1(d['phrases']) for d in batch]
            for audio_id, prompt in zip(audio_ids, prompts):
                log.info(f'Audio id: {audio_id} Prompt: {prompt}')

            # batch generate
            audios = generate_fm(
                prompts,
                feature_utils=feature_utils,
                net=net,
                fm=fm,
                rng=rng,
                cfg_strength=cfg_strength
            ) 

            audios = audios.float().cpu()
            for audio_id, audio in zip(audio_ids, audios):
                save_path = output_dir / f'{audio_id}.wav'
                audio = audio.detach().cpu()
                if audio.ndim == 1:
                    audio = audio.unsqueeze(0)          # [1, T]
                elif audio.ndim == 2:
                    pass                                # already [C, T]
                elif audio.ndim == 3:
                    audio = audio.squeeze(0)            # [1, T] or [C, T]
                else:
                    raise RuntimeError(f"Unexpected audio shape: {audio.shape}")
                torchaudio.save(save_path, audio, seq_cfg.sampling_rate)

    else:
        raise ValueError(f'Invalid eval dataset: {args.eval_dataset}')


if __name__ == '__main__':
    main()