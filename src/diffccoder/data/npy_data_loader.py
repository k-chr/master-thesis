from bisect import bisect
from dataclasses import dataclass
from operator import attrgetter
import os
from pathlib import Path

from loguru import logger
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import numpy as np
import torch as t
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
                 val_batch_size: int = 1,
                 prefix_lm: bool = False,
                 pad_id: int | None = None) -> None:
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
        self.prefix_lm = prefix_lm
        self.pad_id = pad_id
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.npy_train,
                   shuffle=True,
                   num_workers=self.num_workers,
                   batch_size=self.batch_size,
                   pin_memory=self.use_pinned_mem,
                   persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.npy_val,
                   shuffle=False,
                   num_workers=self.num_workers,
                   batch_size=self.val_batch_size,
                   pin_memory=self.use_pinned_mem,
                   persistent_workers=True)

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            npy_all = NPYCLMDataset(self.root, self.npy_to_select, prefix_lm=self.prefix_lm, pad_id=self.pad_id)
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
    mmaps: list[np.memmap]
    
    def __init__(self, 
                 in_dir: Path, 
                 sub_dir_list_file: Path,
                 prefix_lm: bool = False,
                 pad_id: int | None = None) -> None:
        assert in_dir.is_dir(), f'Provided directory does not exist: {in_dir}'
        assert sub_dir_list_file.is_file(), f'Provided file does not exist: {sub_dir_list_file}'

        self.dir_list = get_dir_list_from_file(list_dir_path=sub_dir_list_file)
        self.in_dir = in_dir
        self.prefic_lm = prefix_lm
        self.pad_id = pad_id
        
        self.__load_mmaps()
        
        self.upper_limits:list[MMapLimit] = []
        
        self.__cols = 0
        self.__dtype: np.dtype = None
        
        running_sum = 0
        
        for i, mmap in enumerate(self.mmaps):
            if not self.__cols:
                self.__cols = mmap.shape[1]
            if not self.__dtype:
                self.__dtype: np.dtype = mmap.dtype
            rows = mmap.shape[0]
            _lower = running_sum
            running_sum += rows
            self.upper_limits.append(MMapLimit(_lower, running_sum, i))
        
        self.__rows = running_sum #shape is cached-property, so as a result it is 'tuple' not 'callable' in runtime
        
        logger.info(f"Dataset consists of {self.__rows * self.__cols * self.__dtype.itemsize} bytes, num of lines: {self.__rows}, num of cols: {self.__cols}, dtype: {self.__dtype}")
        
        if int(os.environ['EXP_DEVICES']) > 1:
            self._part_indices = partition_dataset(data_len=self.__rows,
                                                   num_partitions=int(os.environ['EXP_DEVICES']),
                                                   shuffle=True,
                                                   seed=int(os.environ['DIFFCCODER_SEED']),
                                                   drop_last=True)
        
            logger.info(f'RANK {getattr(rank_zero_only, "rank", 0)}, num of unique indices in partitions {set(sum([vec for vec in self._part_indices],start=[])).__len__()}')
        del self.mmaps
        self.mmaps = []
            
    def __load_mmaps(self):
        self.mmaps: list[np.memmap] = [np.load(self.in_dir / npy_dir / 'data.npy', mmap_mode='r') for npy_dir in self.dir_list]
            
    def __len__(self):
        return len(self._part_indices[getattr(rank_zero_only, 'rank', 0)]) if int(os.environ.get('EXP_DEVICES', '1')) > 1 else self.__rows
    
    def __getitem__(self, index) -> tuple[int, t.Tensor, t.Tensor]:
        
        if not len(self.mmaps):
            self.__load_mmaps()
            
        if int(os.environ['EXP_DEVICES']) > 1:
            _index = self._part_indices[rank_zero_only.rank][index]
        else:
            _index = index
        by_upper = attrgetter('upper')
        try:
            _id = bisect(self.upper_limits, _index, key=by_upper)

            obj = self.upper_limits[_id]
        except IndexError as _:
            logger.info(f'RANK: {rank_zero_only.rank} Worker: For _index: {_index}, index: {index} found _id: {_id}')
            raise IndexError(f'RANK: {rank_zero_only.rank} Num partitions = {len(self._part_indices)}: For _index: {_index}, index: {index} found _id: {_id}')
        try:
            arr: np.ndarray = self.mmaps[obj.index][_index-obj.lower]
        except IndexError as _:
            logger.info(f'RANK: {rank_zero_only.rank} For _index: {_index}, index: {index} found obj: {obj}')
            raise IndexError(f'RANK: {rank_zero_only.rank} For _index: {_index}, index: {index} found obj: {obj}')
        data: t.Tensor = t.from_numpy(arr.astype(np.uint16))

        if self.prefic_lm:
            l = data.shape[-1]
            x, y = data[:l//2], data[l//2:]
            assert x.shape == y.shape
        else:    
            x, y = data[:-1], data[1:].clone()

        return index, x, y
