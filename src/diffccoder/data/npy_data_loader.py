from pathlib import Path

import dask.array as da
from dask.array.core import Array
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import numpy as np
import torch as t
from torch.utils.data import DataLoader, Dataset, random_split

from diffccoder.data.utils import get_dir_list_from_file


class NPYDataModule(LightningDataModule):
    def __init__(self, 
                 in_dir: Path,
                 dir_list_txt: Path,
                 split_val_ratio: float = 0.2,
                 use_pinned_memory=False,
                 num_workers: int | None = None,
                 batch_size: int = 1,
                 val_batch_size: int = 1) -> None:
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
    
    def test_dataloader(self) -> DataLoader:
        return super().test_dataloader()
    
    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            npy_all = NPYCLMDataset(self.root, self.npy_to_select)
            self.npy_train, self.npy_val = random_split(npy_all, [1 - self.val_ratio, self.val_ratio])

        if stage == "test":
            assert False, 'Currently no test data'
    
    def prepare_data(self) -> None:
        return super().prepare_data()
    
class NPYCLMDataset(Dataset):
    def __init__(self, in_dir: Path, sub_dir_list_file: Path) -> None:
        assert in_dir.is_dir(), f'Provided directory does not exist: {in_dir}'
        assert sub_dir_list_file.is_file(), f'Provided file does not exist: {sub_dir_list_file}'
        
        dir_list = get_dir_list_from_file(list_dir_path=sub_dir_list_file)
        
        mmaps: list[np.memmap] = [np.load(in_dir / npy_dir / 'data.npy', mmap_mode='r') for npy_dir in dir_list]
        
        self.mmap: Array = da.concatenate(mmaps)        
            
        self.__len = self.mmap.shape[0] #shape is cached-property, so as a result it is 'tuple' not 'callable' in runtime
        
    def __len__(self):
        return self.__len
    
    def __getitem__(self, index) -> tuple[int, t.Tensor, t.Tensor]:
        
        data: t.Tensor = t.from_numpy(self.mmap[index].compute())
        
        x, y = data, data.clone()
        
        return index, x, y
        
