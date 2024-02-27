from pathlib import Path
from loguru import logger
from tokenizers import Tokenizer

import torch as t
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import BasePredictionWriter

class PredictionStringWriter(BasePredictionWriter):

    def __init__(self, 
                 output_dir: Path,
                 tokenizer: Tokenizer,
                 label: str =""):
        super().__init__('epoch')
        self.output_dir = output_dir
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        self.label = label
        
    def write_on_epoch_end(self,
                           trainer,
                           pl_module,
                           predictions: list[list[list[str]]],
                           batch_indices: list[list[list[int]]]):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        file_path = self.output_dir / f'predictions_{trainer.global_rank}{("_"+self.label if self.label else "")}.log'
        indices_path = self.output_dir / f'batch_indices_{trainer.global_rank}{("_"+self.label if self.label else "")}.pt'
        dataloader: DataLoader = trainer.predict_dataloaders
        dataset = dataloader.dataset
        
        with file_path.open('w', encoding='utf-8') as file:
            for dataloader_index in batch_indices:
                for batch_index, batch_sample in zip(dataloader_index, predictions):
                
                    file.write(f'{"="*20} Batch id(s): {batch_index} {"="*20}\n')
                    for samples, idx in zip(batch_sample, batch_index):
                        tup = dataset[idx]
                        src:t.Tensor = tup[1]
                        src_text = self.tokenizer.decode(src.tolist())
                        file.write(f'{"-"*23}Input No. {idx}{"-"*23}\n')
                        file.write(src_text)
                        file.write('\n')
                        for i, sample in enumerate(samples):
                            file.write(f'{"*"*15}For Input No. {idx}, Sample No. {i}{"*"*15}\n')
                            file.write(sample)
                            file.write('\n')
        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        t.save(batch_indices, indices_path)