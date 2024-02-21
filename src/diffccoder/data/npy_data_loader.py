import os
from pathlib import Path
from typing import Literal

from lightning.pytorch import LightningDataModule
import torch as t
from torch.utils.data import DataLoader, random_split

from diffccoder.data.npy_clm_dataset import NPYCLMDataset
from diffccoder.data.npz_fine_tune_dataset import NPZFineTuneDataset


class NPYZDataModule(LightningDataModule):
    def __init__(self, 
                 in_dir: Path,
                 dir_list_txt: Path,
                 split_val_ratio: float = 0.2,
                 use_pinned_memory=False,
                 num_workers: int | None = None,
                 batch_size: int = 1,
                 val_batch_size: int = 1,
                 mode: Literal['npz', 'npy'] = 'npy',
                 prefix_lm: bool = False,
                 pad_id: int | None = None) -> None:
        super().__init__()
        self.root = in_dir
        self.npyz_to_select = dir_list_txt
        self.val_ratio = split_val_ratio
        self.use_pinned_mem = use_pinned_memory
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.prepare_data_per_node = False
        self.allow_zero_length_dataloader_with_multiple_devices = True
        self.prefix_lm = prefix_lm
        self.pad_id = pad_id
        self.mode = mode
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.npyz_train,
                   shuffle=True,
                   num_workers=self.num_workers,
                   batch_size=self.batch_size,
                   pin_memory=self.use_pinned_mem,
                   persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.npyz_val,
                   shuffle=False,
                   num_workers=self.num_workers,
                   batch_size=self.val_batch_size,
                   pin_memory=self.use_pinned_mem,
                   persistent_workers=True)

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            match (mode:=self.mode):
                case 'npz':
                    npyz_all = NPZFineTuneDataset(self.root, self.npyz_to_select, pad_id=self.pad_id)
                case 'npy':
                    npyz_all = NPYCLMDataset(self.root, self.npyz_to_select, prefix_lm=self.prefix_lm, pad_id=self.pad_id)
                case _:
                    raise ValueError(f'Unknown mode: {mode}')
            generator = t.Generator().manual_seed(int(os.environ['DIFFCCODER_SEED']))
            self.npyz_train, self.npyz_val = random_split(npyz_all, 
                                                          [1 - self.val_ratio, self.val_ratio],
                                                          generator=generator)
            
        if stage == "test":
            assert False, 'Currently no test data'