from PIL import Image
import io
import numpy as np
import torch
from collections import defaultdict
import torchaudio
DEBUG = False

def clap_score(device): 
    def int16_to_float32(x):
        return (x / 32767.0).astype('float32')
    def float32_to_int16(x):
        x = np.clip(x, a_min=-1., a_max=1.)
        return (x * 32767.).astype('int16')
    import laion_clap
    laion_clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base').to(device).eval()
    _clap_ckpt_path = "./weights/music_speech_audioset_epoch_15_esc_89.98.pt"  
    laion_clap_model.load_ckpt(_clap_ckpt_path, verbose=False)

    @torch.no_grad()
    def _fn(audios, prompts, vae_sr):
        target_sr = 48000
        if vae_sr != target_sr:
            audios = audios.float().cpu()
            audios = torchaudio.functional.resample(audios, vae_sr, target_sr)
        else:
            audios = audios.float().cpu()
        text_embeddings = laion_clap_model.get_text_embedding(prompts, use_tensor=True)
        audios = audios.numpy()
        audios = torch.from_numpy(
            int16_to_float32(float32_to_int16(audios))
        ).float()
        audio_embeddings = laion_clap_model.get_audio_embedding_from_data(
            x=audios, use_tensor=True
        )
        scores = text_embeddings @ audio_embeddings.T
        scores = scores.diagonal()
        return scores, {}
    return _fn


def qwen25_omni_semantic_align_score(device):
    import torch
    import torch.nn.functional as F
    import torchaudio
    from transformers import (
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniProcessor,
    )
    from qwen_omni_utils import process_mm_info
    # QWEN_25_OMNI_MODEL = "../../models/Qwen25-omni"
    QWEN_25_OMNI_MODEL = "Qwen/Qwen2.5-Omni-7B"
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        QWEN_25_OMNI_MODEL,
        torch_dtype="auto",
        device_map=device,
    ).eval()
    model.disable_talker()

    processor = Qwen2_5OmniProcessor.from_pretrained(QWEN_25_OMNI_MODEL)
    tokenizer = processor.tokenizer

    yes_id = tokenizer.encode("Yes", add_special_tokens=False)
    no_id  = tokenizer.encode("No",  add_special_tokens=False)
    assert len(yes_id) == 1 and len(no_id) == 1
    yes_id, no_id = yes_id[0], no_id[0]

    TARGET_SR = 16000  # Qwen Omni audio encoder expects 16k

    @torch.no_grad()
    def _fn(audios, prompts, vae_sr):
        """
        audios: Tensor[B, T]
        prompts: List[str]
        vae_sr: int
        """
        B = audios.shape[0]
        scores = []

        if vae_sr != TARGET_SR:
            audios = torchaudio.functional.resample(
                audios.float().cpu(), vae_sr, TARGET_SR
            )
            audios = audios.float().cpu()
            audios = audios.numpy()
        else:
            audios = audios.float().cpu()
            audios = audios.numpy()

        for i in range(B):
            audio_caption = prompts[i]
            SEMANTIC_ALIGNMENT_PROMPT = f"Does this audio contain the sound events described by the text: {audio_caption}? Please only answer yes or no, without other explanations."
            # SEMANTIC_ALIGNMENT_PROMPT = f"Does this audio contain the sound events described by the text: {audio_caption}? Please answer yes or no."

            conversation = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SEMANTIC_ALIGNMENT_PROMPT},
                        {"type": "audio", "audio": audios[i]},
                    ],
                },
            ]

            text = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )

            audios_mm, images, videos = process_mm_info(
                conversation, use_audio_in_video=False
            )

            inputs = processor(
                text=text,
                audio=audios_mm,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False,
            ).to(model.device).to(model.dtype)

            gen_out = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

            sequences = gen_out.sequences          # [B, T]
            input_len = inputs["input_ids"].shape[1]
            gen_tokens = sequences[:, input_len:]  # [B, T_new]
            hit_pos = None
            hit_token_id = None
            hit_score_logits = None
            
            gen_scores = gen_out.scores                # list length = T_new, each [B, V]
            for i in range(gen_tokens.shape[1] - 1, -1, -1):
                tid = gen_tokens[0, i].item()
                if tid == yes_id or tid == no_id:
                    hit_pos = i
                    hit_token_id = tid
                    hit_score_logits = gen_scores[i]
                    break
                
            if hit_pos is None:
                print("[Warning] No yes/no token found in generation.")
                full_text = processor.batch_decode(
                    gen_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                print(f"Full text: {full_text}")
                aqa_score = 0
            else:
                log_probs = F.log_softmax(hit_score_logits, dim=-1)
                s_yes = log_probs[0, yes_id]
                s_no  = log_probs[0, no_id]
                aqa_score = torch.sigmoid(s_yes - s_no)

            scores.append(aqa_score)

        scores = torch.stack(scores)  # [B]
        info = {}
        return scores, info

    return _fn


def qwen3_omni_semantic_align_score(device):
    import torch
    import torch.nn.functional as F
    import torchaudio
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from qwen_omni_utils import process_mm_info
    MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="auto",
    # attn_implementation="flash_attention_2",
)
    model.disable_talker()
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
    tokenizer = processor.tokenizer

    yes_id = tokenizer.encode("yes", add_special_tokens=False)
    no_id  = tokenizer.encode("no",  add_special_tokens=False)
    assert len(yes_id) == 1 and len(no_id) == 1
    yes_id, no_id = yes_id[0], no_id[0]

    TARGET_SR = 16000  # Qwen Omni audio encoder expects 16k

    @torch.no_grad()
    def _fn(audios, prompts, vae_sr):
        """
        audios: Tensor[B, T]
        prompts: List[str]
        vae_sr: int
        """
        B = audios.shape[0]
        scores = []

        if vae_sr != TARGET_SR:
            audios = torchaudio.functional.resample(
                audios.float().cpu(), vae_sr, TARGET_SR
            )
            audios = audios.float().cpu()
            audios = audios.numpy()
        else:
            audios = audios.float().cpu()
            audios = audios.numpy()

        for i in range(B):
            audio_caption = prompts[i]
            SEMANTIC_ALIGNMENT_PROMPT = f"Does this audio contain the sound events described by the text: {audio_caption}? Please only answer yes or no, without other explanations."

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SEMANTIC_ALIGNMENT_PROMPT}, 
                        {"type": "audio", "audio": audios[i]},
                    ],
                },
            ]

            text = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )

            audios_mm, images, videos = process_mm_info(
                conversation, use_audio_in_video=False
            )

            inputs = processor(
                text=text,
                audio=audios_mm,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False,
            ).to(model.device).to(model.dtype)

            gen_out = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )[0]

            sequences = gen_out.sequences          # [B, T]
            input_len = inputs["input_ids"].shape[1]
            gen_tokens = sequences[:, input_len:]  # [B, T_new]
            hit_pos = None
            hit_token_id = None
            hit_score_logits = None
            
            gen_scores = gen_out.scores                # list length = T_new, each [B, V]
            for i in range(gen_tokens.shape[1] - 1, -1, -1):
                tid = gen_tokens[0, i].item()
                if tid == yes_id or tid == no_id:
                    hit_pos = i
                    hit_token_id = tid
                    hit_score_logits = gen_scores[i]
                    break
                
            if hit_pos is None:
                print("[Warning] No yes/no token found in generation.")
                aqa_score = 0
            else:
                log_probs = F.log_softmax(hit_score_logits, dim=-1)
                s_yes = log_probs[0, yes_id]
                s_no  = log_probs[0, no_id]
                aqa_score = torch.sigmoid(s_yes - s_no)

            scores.append(aqa_score)
        scores = [torch.as_tensor(s, dtype=torch.float32) for s in scores]
        scores = torch.stack(scores)  # [B]
        info = {}
        return scores, info

    return _fn


def qwen3_omni_thinking_semantic_align_score(device):
    import torch
    import torch.nn.functional as F
    import torchaudio
    from transformers import (
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoeProcessor,
    )
    from qwen_omni_utils import process_mm_info

    MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype="auto",
        device_map=device,
    ).eval()
    model.disable_talker()

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
    tokenizer = processor.tokenizer

    yes_id = tokenizer.encode("yes", add_special_tokens=False)
    no_id  = tokenizer.encode("no",  add_special_tokens=False)
    assert len(yes_id) == 1 and len(no_id) == 1
    yes_id, no_id = yes_id[0], no_id[0]

    TARGET_SR = 16000

    @torch.no_grad()
    def _fn(audios, prompts, vae_sr):
        B = audios.shape[0]
        rewards = []

        if vae_sr != TARGET_SR:
            audios = torchaudio.functional.resample(
                audios.float().cpu(), vae_sr, TARGET_SR
            ).cpu().numpy()
        else:
            audios = audios.float().cpu().numpy()

        for b in range(B):
            audio_caption = prompts[b]
            prompt = (
                f"Does this audio contain the sound events described by the text: "
                f"{audio_caption}? Please only answer yes or no."
            )

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "audio", "audio": audios[b]},
                    ],
                },
            ]

            text = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )

            audios_mm, images, videos = process_mm_info(
                conversation, use_audio_in_video=True
            )

            inputs = processor(
                text=text,
                audio=audios_mm,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=True,
            ).to(model.device).to(model.dtype)

            gen_out = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )[0]

            sequences = gen_out.sequences
            scores = gen_out.scores

            input_len = inputs["input_ids"].shape[1]
            gen_tokens = sequences[:, input_len:]

            hit_pos = None
            hit_score_logits = None

            for i in range(gen_tokens.shape[1] - 1, -1, -1):
                tid = gen_tokens[0, i].item()
                if tid == yes_id or tid == no_id:
                    hit_pos = i
                    hit_score_logits = scores[i]
                    break

            if hit_pos is None:
                aqa_score = torch.tensor(0.0, device=model.device)
            else:
                log_probs = F.log_softmax(hit_score_logits, dim=-1)
                s_yes = log_probs[0, yes_id]
                s_no  = log_probs[0, no_id]
                aqa_score = torch.sigmoid(s_yes - s_no)

            rewards.append(aqa_score)

        rewards = torch.stack(rewards)
        info = {}
        return rewards, info

    return _fn


def multi_score(device, score_dict):
    score_functions = {
        "clapscore": clap_score,
        'qwen25_omni_semantic_align_score': qwen25_omni_semantic_align_score,
        'qwen3_omni_semantic_align_score': qwen3_omni_semantic_align_score, 
        'qwen3_omni_thinking_semantic_align_score': qwen3_omni_thinking_semantic_align_score
    }
    score_fns={}
    for score_name, weight in score_dict.items():
        score_fns[score_name] = score_functions[score_name](device) if 'device' in score_functions[score_name].__code__.co_varnames else score_functions[score_name]()

    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(audios, prompts, metadata, vae_sr=44100, only_strict=True):
        total_scores = []
        score_details = {}
        
        for score_name, weight in score_dict.items():
            if score_name == "clapscore":
                scores, rewards = score_fns[score_name](audios, prompts, vae_sr)
            elif "flexsed_score" in score_name: 
                scores, rewards = score_fns[score_name](audios, prompts, metadata, vae_sr)
            elif "qwen25_omni" in score_name: 
                scores, rewards = score_fns[score_name](audios, prompts, vae_sr)
            else: 
                raise NotImplementedError(f'{score_name} not implemented')
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]
            
            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]
        
        score_details['avg'] = total_scores
        return score_details, {}
    return _fn