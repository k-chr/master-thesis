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


class NPZFineTuneDataset(Dataset):
    mmaps: list[np.memmap]
    
    def __init__(self, 
                 in_dir: Path, 
                 sub_dir_list_file: Path,
                 pad_id: int | None = None) -> None:
        assert in_dir.is_dir(), f'Provided directory does not exist: {in_dir}'
        assert sub_dir_list_file.is_file(), f'Provided file does not exist: {sub_dir_list_file}'

        self.dir_list = get_dir_list_from_file(list_dir_path=sub_dir_list_file)
        self.in_dir = in_dir
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
                cols = mmap['y'].shape[1]
            if not dtype:
                dtype: np.dtype = mmap['y'].dtype
            rows = mmap['y'].shape[0]
            _lower = running_sum
            running_sum += rows
            upper_limits.append(MMapLimit(_lower, running_sum, i))
        return upper_limits, running_sum, cols, dtype
  
    def __load_mmaps(self):
        self.mmaps: list[np.memmap] = [np.load(self.in_dir / npz_dir / 'data.npz', mmap_mode='r') for npz_dir in self.dir_list]
            
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
            mmap: np.memmap = self.mmaps[obj.index]
            y: np.ndarray = mmap['y'][_index-obj.lower]
            x: np.ndarray = mmap['x']
        except IndexError as _:
            logger.info(f'RANK: {rank_zero_only.rank} For _index: {_index}, index: {index} found obj: {obj}')
            raise IndexError(f'RANK: {rank_zero_only.rank} For _index: {_index}, index: {index} found obj: {obj}')
        
        x_t: t.Tensor = t.from_numpy(x.astype(np.uint16))
        y_t: t.Tensor = t.from_numpy(y.astype(np.uint16))
        return index, x_t, y_t