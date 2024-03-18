from bisect import bisect
from operator import attrgetter
import os
from pathlib import Path

from lightning.pytorch.utilities.rank_zero import rank_zero_only
from loguru import logger
import numpy as np
import torch as t
from torch.utils.data import Dataset

from diffccoder.data.utils import MMapLimit, get_dir_list_from_file, partition_dataset


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
        self.prefix_lm = prefix_lm
        self.pad_id = pad_id
        
        self.__load_mmaps()
          
        self.upper_limits, self.__rows, self.__cols, self.__dtype = self.__compute_upper_limits()
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

    def __compute_upper_limits(self) -> tuple[list[MMapLimit], int, int, np.dtype]:
        running_sum = 0
        upper_limits = []
        dtype: np.dtype = None
        cols: int = 0
        
        for i, mmap in enumerate(self.mmaps):
            if not cols:
                cols = mmap.shape[1]
            if not dtype:
                dtype: np.dtype = mmap.dtype
            rows = mmap.shape[0]
            _lower = running_sum
            running_sum += rows
            upper_limits.append(MMapLimit(_lower, running_sum, i))
        return upper_limits, running_sum, cols, dtype
            
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
        arr = arr.astype(np.uint16)
        if self.prefix_lm:
            arr = process_arr(arr)
        else:
            arr = clear_pad(arr)
        data: t.Tensor = t.from_numpy(arr)

        if self.prefix_lm:
            l = data.shape[-1]
            x, y = data[:l//2], data[l//2:]
            assert x.shape == y.shape
        else:    
            x, y = data[:-1], data[1:].clone()

        return index, x, y

def process_arr(arr:np.ndarray):
    pad_id = 1
    eos_id = 0
    orig_len = arr.shape[-1]
    mask = ((arr == pad_id) | (arr == eos_id))
    good = arr[~mask]
    l = good.shape[-1]
    good_1, good_2 = good[:l//2], good[l//2:]
    
    pad = np.ones(orig_len//2 - good_1.shape[-1])
    eos = np.zeros(orig_len//2 - good_2.shape[-1])
    
    return np.concatenate((good_1, pad, good_2, eos))

def clear_pad(arr:np.ndarray):
    pad_id = 1
    eos_id = 0
    orig_len = arr.shape[-1]
    mask = ((arr == pad_id) | (arr == eos_id))
    good = arr[~mask]
    
    eos = np.zeros(orig_len - good.shape[-1])
    return np.concatenate((good, eos))