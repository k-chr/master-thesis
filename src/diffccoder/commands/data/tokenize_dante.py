from pathlib import Path
from typing import Literal

from cleo.commands.command import Command
from cleo.helpers import argument, option
from loguru import logger
import numpy as np
from tokenizers import Tokenizer, Encoding

from diffccoder.data.utils import get_dir_list_from_file, try_read


def save_encoded(tokenizer: Tokenizer,
                 max_seq_len: int,
                 out_dir: Path,
                 dir_list: list[Path]):
    for _dir in dir_list:
        for task in _dir.iterdir():
            codes = task / 'preprocessed'
            task_content = task / 'task_content.txt'
            
            if (task_text := try_read(task_content)) is None:
                continue
            
            task_encoded: Encoding = tokenizer.encode(task_text)
            task_ids = task_encoded.ids
            
            task_ids = pad_ids(tokenizer.token_to_id('<|padding|>'), task_ids, max_seq_len, 'right')
            task_array = np.asarray(task_ids, dtype=np.int16)
            
            codes_array = get_codes_ndarray(tokenizer, max_seq_len, codes)

            npz_dir = out_dir / _dir.parent.stem / _dir.stem / task.stem

            if not npz_dir.exists():
                npz_dir.mkdir(parents=True, exist_ok=True)
            
            npz_path = npz_dir / 'data.npz'
            
            np.savez(npz_path, x=task_array, y=codes_array)
            logger.success(f'Successfully written {codes_array.size + task_array.size} elements to {npz_path}')  

def pad_ids(pad_token_id: int, ids: list[int], seq_len: int, mode: Literal['right', 'left', 'both']):
    padding = seq_len - len(ids)
    if not padding > 0: return ids
    
    pad_value = [pad_token_id]
    
    match mode:
        case 'right':
            return ids + pad_value * padding
        case 'left':
            return pad_value * padding + ids
        case 'both':
            padding_left = padding // 2
            padding_right = padding - padding_left
            
            return pad_value * padding_left + ids + pad_value * padding_right
        case _: 
            raise        

def get_codes_ndarray(tokenizer: Tokenizer,
                      max_seq_len: int,
                      codes_dir: Path) -> np.ndarray:
    tensors: list[list[int]] = []
    assert max_seq_len > 0
    
    for sample_path in codes_dir.iterdir():
        if (code := try_read(sample_path)) == None:
            continue
        
        encoded_code: Encoding = tokenizer.encode(code)
        tensor = encoded_code.ids + [tokenizer.token_to_id('<|endoftext|>')]
        pad_id = tokenizer.token_to_id('<|padding|>')
        if len(tensor) > max_seq_len * 2:
            continue
        elif len(tensor) <= max_seq_len:
            tensor = pad_ids(pad_id, tensor, 2 * max_seq_len, 'right')
        else:
            tensor = pad_ids(pad_id, tensor, 2 * max_seq_len, 'both')
            
        assert tensor.__len__() == 2 * max_seq_len
        tensors.append(tensor)
    
    arr = np.asarray(tensors, dtype=np.int16)
    return arr


class TokenizeDanteCommand(Command):
    name = 'tokenize-dante'
    description = 'tokenize_dante.py - Try to tokenize provided Dante dataset and save it to numpy format.'
    arguments = [argument('in-dir',
                          description='Path to directory with Dante dataset.'),
                 argument('vocab-json',
                          description='Path to input vocab json file.'),
                 argument('list-dir-txt',
                          description='Path to file that contains directory names list to process.'),
                 argument('out-dir',
                          description='Path to the output directory.')]
    options = [option('length', 'l',
                      description='Max length of sequence [default: 1024].',
                      default=1024,
                      flag=False)]
    
    def handle(self) -> int:
        in_dir = Path(self.argument('in-dir'))
        assert in_dir.is_dir(), 'Not a directory'
        list_dir_path = Path(self.argument('list-dir-txt'))
        assert list_dir_path.is_file(), 'Not a file'
        out_dir = Path(self.argument('out-dir')) 
        dir_list =[in_dir/d for d in get_dir_list_from_file(list_dir_path)]
        
        tokenizer = Tokenizer.from_file(self.argument('vocab-json'))

        save_encoded(tokenizer=tokenizer,
                     max_seq_len=int(self.option('length')), 
                     out_dir=out_dir,
                     dir_list=dir_list)