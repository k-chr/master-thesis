from pathlib import Path

import dask.array as da
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import numpy as np
import torch as t
from torch.utils.data import DataLoader, Dataset, random_split

from diffccoder.data.utils import get_dir_list_from_file


class NPYDataModule(LightningDataModule):
    def __init__(self, in_dir: Path, dir_list_txt: Path) -> None:
        self.root = in_dir
        self.npy_to_select = dir_list_txt
        
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return super().train_dataloader()
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        DataLoader(self.npy_val, )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return super().test_dataloader()
    
    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            npy_all = NPYCLMDataset(self.root, self.npy_to_select)
            self.npy_train, self.npy_val = random_split(npy_all, [0.8, 0.2])

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
        
        self.mmap = da.concatenate(mmaps)        
            
        self.__len = self.mmap.shape()[0]
        
    def __len__(self):
        return self.__len
    
    def __getitem__(self, index) -> tuple[int, t.Tensor, t.Tensor]:
        
        data: t.Tensor = t.from_numpy(self.mmap[index])
        
        x, y = data, data.clone()
        
        return index, x, y
        
