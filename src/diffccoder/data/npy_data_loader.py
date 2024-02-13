from bisect import bisect
from dataclasses import dataclass
from operator import attrgetter
import os
from pathlib import Path

from loguru import logger
from lightning.pytorch import LightningDataModule
import numpy as np
import torch as t
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, random_split

from diffccoder.data.utils import get_dir_list_from_file, partition_dataset


class NPYDataModule(LightningDataModule):
    def __init__(self, 
                 in_dir: Path,
                 dir_list_txt: Path,
                 split_val_ratio: float = 0.2,
                 use_pinned_memory=False,
                 num_workers: int | None = None,
                 batch_size: int = 1,
                 val_batch_size: int = 1) -> None:
        super().__init__()
        self.root = in_dir
        self.npy_to_select = dir_list_txt
        self.val_ratio = split_val_ratio
        self.use_pinned_mem = use_pinned_memory
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.prepare_data_per_node = False
        self.allow_zero_length_dataloader_with_multiple_devices = True

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.npy_train,
                   shuffle=True,
                   num_workers=self.num_workers,
                   batch_size=self.batch_size,
                   pin_memory=self.use_pinned_mem)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.npy_val,
                   shuffle=False,
                   num_workers=self.num_workers,
                   batch_size=self.val_batch_size,
                   pin_memory=self.use_pinned_mem)

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            npy_all = NPYCLMDataset(self.root, self.npy_to_select, precision=self.trainer.precision)
            generator = t.Generator().manual_seed(int(os.environ['DIFFCCODER_SEED']))
            self.npy_train, self.npy_val = random_split(npy_all, 
                                                        [1 - self.val_ratio, self.val_ratio],
                                                        generator=generator)
            
        if stage == "test":
            assert False, 'Currently no test data'


@dataclass
class MMapLimit:
    lower: int
    upper: int
    index: int


class NPYCLMDataset(Dataset):
    def __init__(self, in_dir: Path, sub_dir_list_file: Path, precision: t.dtype=t.float32) -> None:
        assert in_dir.is_dir(), f'Provided directory does not exist: {in_dir}'
        assert sub_dir_list_file.is_file(), f'Provided file does not exist: {sub_dir_list_file}'

        dir_list = get_dir_list_from_file(list_dir_path=sub_dir_list_file)

        self.mmaps: list[np.memmap] = [np.load(in_dir / npy_dir / 'data.npy', mmap_mode='r+') for npy_dir in dir_list]
        
        self.upper_limits:list[MMapLimit] = []
        running_sum = 0
        
        self.__cols = 0
        self.__dtype: np.dtype = None
        
        for i, mmap in enumerate(self.mmaps):
            if not self.__cols:
                self.__cols = mmap.shape[1]
            if not self.__dtype:
                self.__dtype: np.dtype = mmap.dtype
            rows = mmap.shape[0]
            _lower = running_sum
            running_sum += rows
            self.upper_limits.append(MMapLimit(_lower, running_sum-1, i))
        
        self.precision = precision
        self.__rows = running_sum #shape is cached-property, so as a result it is 'tuple' not 'callable' in runtime
        
        logger.info(f"Dataset consists of {self.__rows * self.__cols * self.__dtype.itemsize} bytes, num of lines: {self.__rows}, num of cols: {self.__cols}, dtype: {self.__dtype}")
        
        if dist.is_initialized() and dist.get_world_size() > 1:
            self._part_indices = partition_dataset(data_len=self.__rows,
                                                   num_partitions=os.environ['EXP_DEVICES'],
                                                   shuffle=True,
                                                   seed=int(os.environ['DIFFCCODER_SEED']),
                                                   drop_last=True,
                                                   even_divisible=True)[dist.get_rank()]
        
        
    def __len__(self):
        return self.__rows if int(os.environ.get('EXP_DEVICES', '1')) == 1 else len(self._part_indices)
    
    def __getitem__(self, index) -> tuple[int, t.Tensor, t.Tensor]:
        
        if dist.is_initialized() and dist.get_world_size() > 1:
            index = self._part_indices[index]
        
        logger.debug(f"Index obj {index} of type: {index.__class__}")
        
        by_upper = attrgetter('upper')
        obj = self.upper_limits[bisect(self.upper_limits, index, key=by_upper)]
        logger.debug(f"For index: {index} found: {obj}")
        
        arr: np.ndarray = self.mmaps[obj.index][index-obj.lower]
        data: t.Tensor = t.from_numpy(arr.astype(np.uint16).astype(np.int32))

        x, y = data, data.clone()

        return index, x, y
