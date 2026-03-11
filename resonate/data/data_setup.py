import logging
import random
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler

from resonate.utils.dist_utils import local_rank
from resonate.data.online_audio import AudioCaptionDataset, DistributedDynamicBatchSampler

log = logging.getLogger()


# Re-seed randomness every time we start a worker
def worker_init_fn(worker_id: int):
    worker_seed = torch.initial_seed() % (2**31) + worker_id + local_rank * 1000
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    log.debug(f'Worker {worker_id} re-seeded with seed {worker_seed} in rank {local_rank}')


def setup_training_datasets(cfg: DictConfig) -> tuple[Dataset, DistributedSampler, DataLoader]:
    
    if cfg.online_feature_extraction: 
        dataset = AudioCaptionDataset(datasets=cfg.data.datasets,
                                    sample_rate=cfg.audio_sample_rate,
                                    max_length=cfg.audio_max_length, 
                                    use_dynamic_batch=cfg.use_dynamic_batch,
                                    use_temporal_template=cfg.get("use_temporal_template", False))
        collate_fn = dataset.collate_fn
            
    else: 
        raise NotImplementedError("Please use the online audio dataset.")
        
    sampler, loader = construct_loader(dataset,
                                       cfg.batch_size,
                                       cfg.num_workers,
                                       shuffle=True,
                                       drop_last=True,
                                       pin_memory=cfg.pin_memory,
                                       collate_fn=collate_fn, 
                                       use_dynamic_batch=cfg.get('use_dynamic_batch', False),
                                       frames_threshold=cfg.get('frames_threshold', 0),
                                       max_samples=cfg.get('max_samples', 0),
                                       random_seed=cfg.get('seed', 14159265),
                                       drop_residual=cfg.get('drop_residual', False))

    return dataset, sampler, loader


def error_avoidance_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))   # batch = [x for x in batch if x is not None]
    return default_collate(batch)


def construct_loader(dataset: Dataset,
                     batch_size: int,
                     num_workers: int,
                     *,
                     shuffle: bool = True,
                     drop_last: bool = True,
                     pin_memory: bool = False,
                     collate_fn=None,
                     use_dynamic_batch: bool = False,
                     frames_threshold: int = 0,
                     max_samples: int = 0,
                     random_seed: Optional[int] = None,
                     drop_residual: bool = False) -> tuple[DistributedSampler, DataLoader]:
    
    if use_dynamic_batch: 
        sampler = SequentialSampler(dataset)
        train_sampler = DistributedDynamicBatchSampler(
            sampler=sampler,
            frames_threshold=frames_threshold,
            max_samples=max_samples, 
            random_seed=random_seed,
            drop_residual=drop_residual,
            drop_last=True
        )
        if collate_fn is None:
            collate_fn = error_avoidance_collate
        train_loader = DataLoader(dataset,
                                  batch_size=1,  # batch_size=1 because sampler returns batches
                                  batch_sampler=train_sampler,
                                  num_workers=num_workers,
                                  worker_init_fn=worker_init_fn,
                                  drop_last=False,  # drop_last is handled by DynamicBatchSampler
                                  persistent_workers=num_workers > 0,
                                  pin_memory=pin_memory,
                                  collate_fn=collate_fn)
        return train_sampler, train_loader
    else:
        train_sampler = DistributedSampler(dataset, rank=local_rank, shuffle=shuffle)
        if collate_fn is None:
            collate_fn = error_avoidance_collate
        train_loader = DataLoader(dataset,
                                  batch_size,
                                  sampler=train_sampler,
                                  num_workers=num_workers,
                                  worker_init_fn=worker_init_fn,
                                  drop_last=drop_last,
                                  persistent_workers=num_workers > 0,
                                  pin_memory=pin_memory,
                                  collate_fn=collate_fn)
                                  
        return train_sampler, train_loader