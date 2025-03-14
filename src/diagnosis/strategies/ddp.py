

from typing import Optional

import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .naive import NaiveStrategy
# from .sampler import DistributedSampler
from torch.utils.data.distributed import DistributedSampler

import socket
def get_open_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    # return random.randint(9000,1000)

class DDPStrategy(NaiveStrategy):
    """
        Strategy for distributed training using torch.distributed.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.node_rank = 0
        self.nproc_per_node = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        self.world_size = self.nproc_per_node
        
        self.setup_distributed()
        super().__init__()

    def setup_distributed(self) -> None:
        try:
            # node_rank * nproc_per_node + local_rank
            rank = self.node_rank * self.nproc_per_node + self.local_rank
            local_rank = self.local_rank
            # world_size = nproc_per_node * nnodes
            world_size = self.world_size

            # os.environ['MASTER_ADDR'] = 'localhost'
            if 'MASTER_PORT' not in os.environ:
                os.environ['MASTER_PORT'] = get_open_port()
                # os.environ['MASTER_PORT'] = 9501
            host = 'localhost'
            port = os.environ['MASTER_PORT']
            
        except Exception as e:
            print(e)
            raise RuntimeError(
                f"Could not find {e} in the torch environment, visit https://www.colossalai.org/ for more information on launching with torch"
            )
        self.set_seed(self.seed)
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl', init_method=f'tcp://[{host}]:{port}', world_size=world_size, rank=rank)

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # 如果你使用 CUDA 设备，还需要设置以下代码
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def setup_model(self, model: nn.Module) -> nn.Module:
        device = torch.cuda.current_device()
        model = model.to(device)
        # return DDP(model, device_ids=[device],broadcast_buffers=False)
        return DDP(model, device_ids=[device],broadcast_buffers=False,find_unused_parameters=True)

    def setup_dataloader(self, dataset, pin_memory: bool = False) -> DataLoader:
        # DDP only mode, replay buffers on each rank are different.
        
        if 'train' in dataset.data_dir:
            sampler = DistributedSampler(dataset,
                                        num_replicas=dist.get_world_size(),
                                        rank=dist.get_rank(),
                                        shuffle=True)
            return DataLoader(
                dataset,
                batch_size=dataset.batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=6,
                # drop_last=True,
                pin_memory=pin_memory,
                collate_fn=dataset.collate_fn)
        else:
            return DataLoader(
                dataset,
                batch_size=dataset.batch_size,
                # sampler=sampler,
                num_workers=6,
                shuffle=False,
                pin_memory=pin_memory,
                collate_fn=dataset.collate_fn)
