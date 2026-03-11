import json
import torch
from torch.utils.data import Dataset, Sampler
import torchaudio
from torchaudio.transforms import Resample
from resonate.data.utils import pad_sequence
import random
from tqdm import tqdm
import logging
from typing import List, Iterator, Optional, Sized
import torch.distributed as dist


logger = logging.getLogger(__name__)

def format_variant1(phrases):
    """
    Variant 1:
    Sound: from x s to y s, from x2 s to y2 s. Sound2: ...
    """
    chunks = []
    for item in phrases:
        sound = item["phrase"]
        segs = item["segments"]
        times = [
            f"from {s:.2f}s to {e:.2f}s"
            for s, e in segs
        ]
        chunk = f"{sound}: " + ", ".join(times)
        chunks.append(chunk)
    return ". ".join(chunks) + "."
def format_variant2(phrases):
    """
    Variant 2:
    Sound (x–y s, x2–y2 s). Sound2 (...)
    """
    chunks = []
    for item in phrases:
        sound = item["phrase"]
        segs = item["segments"]
        times = [
            f"{s:.2f}–{e:.2f}s"
            for s, e in segs
        ]
        chunk = f"{sound} (" + ", ".join(times) + ")"
        chunks.append(chunk)
    return ". ".join(chunks) + "."
def format_variant3(phrases):
    """
    Variant 3:
    Sound: starts at x s and end at y s, starts at x2 s and end at y2 s. Sound2: ...
    """
    chunks = []
    for item in phrases:
        sound = item["phrase"]
        segs = item["segments"]
        times = [
            f"starts at {s:.2f}s and ends at {e:.2f}s"
            for s, e in segs
        ]
        chunk = f"{sound}: " + ", ".join(times)
        chunks.append(chunk)
    return ". ".join(chunks) + "."
            
class AudioCaptionDataset(Dataset):
    """
    Audio caption dataset for retrieval and classification
    Data format: {"id": id, "audio_path": audio_path, "caption": [captions] or caption}
    """

    def __init__(self,
                 datasets,
                 sample_rate: int = 16000,
                 max_length: int = 10, 
                 use_speech_prompt: bool = False, 
                 audio_min_duration: int = 2, 
                 use_dynamic_batch: bool = False,
                 use_temporal_template: bool = False):
    
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.use_speech_prompt = use_speech_prompt
        if self.use_speech_prompt:
            self.speech_prompt = ['Somebody speaks', 'A person says', 'Someone said']
        # Load data and expand multi-caption entries
        self.use_temporal_template = use_temporal_template
        logger.info(f"Using temporal template: {use_temporal_template}")
        if use_temporal_template: 
            self.use_temporal_caption_ratio = 0.75
            logger.info(f"Temporal caption ratio {self.use_temporal_caption_ratio}")
            # self.format_fn = [format_variant1, format_variant2, format_variant3]
            self.format_fn = [format_variant1]
            
        self.data = []
        for dataset in tqdm(datasets, desc="Loading datasets"): 
            data_list = []
            meta_list = dataset['meta']
            logger.info(f"Loading {dataset['name']} ... ")
            weight = dataset['weight']
            for meta in meta_list: 
                with open(meta, 'r') as f:
                    for line in f:
                        item = json.loads(line)
                        id = item["audio_id"]
                        audio_path = item.get("audio_path", "")
                        captions = item.get("caption", [])
                        duration = item.get("duration", 10)
                        if duration is None and use_dynamic_batch:
                            logger.warning(f"Duration is not found for {id}")
                            continue
                        else: 
                            if duration < audio_min_duration:
                                # logger.info(f"Duration is less than {audio_min_duration} seconds for {id}, skipping ...")
                                continue

                        if isinstance(captions, str):
                            captions = [captions]
                        elif isinstance(captions, list):
                            pass
                        else: 
                            continue

                        for caption in captions:
                            if self.use_temporal_template and "phrases" in item: 
                                data_list.append({
                                    "id": id,
                                    "audio_path": audio_path,
                                    "caption": caption,
                                    "duration": duration,  # only for dynamic sampler
                                    "phrases": item['phrases']
                                })
                            else: 
                                data_list.append({
                                    "id": id,
                                    "audio_path": audio_path,
                                    "caption": caption,
                                    "duration": duration,  # only for dynamic sampler
                                })
            self.data.extend(data_list * weight)
    
    def __getitem__(self, index):
        item = self.data[index]
        id = item["id"]
        audio_path = item["audio_path"]
        caption = item["caption"]
        if self.use_temporal_template: 
            if "phrases" in item: 
                phrases = item["phrases"]
                rnd = random.random()
                if rnd < self.use_temporal_caption_ratio: 
                    format_fn = random.choice(self.format_fn)
                    caption = format_fn(phrases)
                # if index < 3:  # for debug
                # logger.info(f"Caption with temporally trong annotation: {caption}")
            
        if self.use_speech_prompt and 'Emilia' in audio_path:  # FIXME: hardcode speech prompt for Emilia set
            speech_prompt = random.choice(self.speech_prompt)
            caption = f"{speech_prompt}: '{caption}'"

        # Load audio
        try:
            wav_info = torchaudio.info(audio_path)
            waveform, sr = torchaudio.load(
                audio_path, 
                num_frames=self.max_length * wav_info.sample_rate
            )
        except Exception as e:
            logger.warning(f"[WARNING] Failed to load audio: {audio_path} error={e}")
            sr = 16000
            waveform = torch.zeros((1, 2 * sr)) # min length
        
        waveform = waveform[0]
        resampler = Resample(orig_freq=sr, new_freq=self.sample_rate)
        waveform = resampler(waveform)

        return {
            "id": id,
            "waveform": waveform,
            "caption": caption
        }
    
    def __len__(self):
        return len(self.data)

    def get_frame_len(self, index):
        item = self.data[index]
        duration = item["duration"]
        return duration
    
    def collate_fn(self, data_batch):
        ids = []
        waveforms = []
        captions = []
        
        for data in data_batch:
            ids.append(data["id"])
            waveforms.append(data["waveform"])
            captions.append(data["caption"])
        
        # Pad waveforms
        padded_waveforms, waveform_lengths = pad_sequence(waveforms)
        
        return {
            "id": ids,
            "waveforms": padded_waveforms,
            "waveform_lengths": waveform_lengths,
            "caption": captions
        }


# Dynamic Batch Sampler
class DynamicBatchSampler(Sampler[list[int]]):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    3.  Shuffle batches each epoch while maintaining reproducibility.
    """

    def __init__(
        self, sampler: Sampler[int], 
        frames_threshold: int, 
        max_samples=0, 
        random_seed=None, 
        drop_residual: bool = False
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.epoch = 0

        indices, batches = [], []
        data_source = self.sampler.data_source

        for idx in tqdm(
            self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        ):
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} seconds of audio per gpu"
        ):
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:  # put it into a new batch
                    batch = [idx]
                    batch_frames = frame_len
                else:  # frame_len is too long, delete it 
                    batch = []
                    batch_frames = 0

        if not drop_residual and len(batch) > 0:
            batches.append(batch)

        del indices
        self.batches = batches

        # Ensure even batches with accelerate BatchSamplerShard cls under frame_per_batch setting
        self.drop_last = True

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch

    def __iter__(self):
        # Use both random_seed and epoch for deterministic but different shuffling per epoch
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            # Use PyTorch's random permutation for better reproducibility across PyTorch versions
            indices = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        return iter(batches)

    def __len__(self):
        return len(self.batches)


class DistributedDynamicBatchSampler(DynamicBatchSampler):
    """
    Extends DynamicBatchSampler to handle distributed data parallelism (DDP).
    The global batches are computed, shuffled consistently across all ranks,
    and then partitioned based on rank and world size.
    """

    def __init__(
        self, sampler: Sampler[int], 
        frames_threshold: int, 
        max_samples: int = 0, 
        random_seed: Optional[int] = None, 
        drop_residual: bool = False,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last_ddp = drop_last

        super().__init__(
            sampler=sampler,
            frames_threshold=frames_threshold,
            max_samples=max_samples,
            random_seed=random_seed,
            drop_residual=drop_residual
        )
        
        self.total_batches = len(self.batches)
        if self.drop_last_ddp:
            self.num_local_batches = self.total_batches // self.num_replicas
        else:
            self.num_local_batches = (self.total_batches + self.num_replicas - 1) // self.num_replicas
            
    def __iter__(self) -> Iterator[List[int]]:
        # global shuffle
        if self.shuffle and self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch) 
            indices = torch.randperm(self.total_batches, generator=g).tolist()
            global_batches = [self.batches[i] for i in indices]
        else:
            global_batches = self.batches

        start = self.rank
        local_batches = global_batches[start::self.num_replicas]

        if self.drop_last_ddp and len(local_batches) > self.num_local_batches:
             local_batches = local_batches[:self.num_local_batches]

        return iter(local_batches)

    def __len__(self) -> int:
        return self.num_local_batches