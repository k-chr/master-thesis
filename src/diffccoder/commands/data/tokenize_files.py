from functools import partial
import gc
from itertools import islice
from typing import Any, Generator
from pathlib import Path

from cleo.commands.command import Command
from cleo.helpers import argument, option
from loguru import logger
import numpy as np
from tokenizers import Tokenizer, Encoding
from tqdm import tqdm

from diffccoder.data.utils import get_dir_list_from_file, lazy_load


def save_encoded(tokenizer: Tokenizer,
                 max_seq_len: int,
                 out_dir: Path,
                 encoded_batches : Generator[tuple[str, list[Encoding]], Any, None]):
    for file, encoded_content in encoded_batches:
        logger.info(f'Processing encoding of: {file}')
        
        file_path = Path(file)
        npz_path = file_path.with_suffix('.npy')
        f_name = npz_path.name
        npz_dir = (out_dir / npz_path.parent.parent.stem / npz_path.parent.stem)
        npz_dir.mkdir(parents=True, exist_ok=True)
        
        npz_path = npz_dir / f_name
        
        arr = get_ndarray_from_encoding(tokenizer, max_seq_len, encoded_content)
        np.save(npz_path, arr)
        logger.success(f'Successfully written {arr.size} elements to {npz_path}')
        del arr
        gc.collect()

def get_ndarray_from_encoding(tokenizer: Tokenizer,
                              max_seq_len: int,
                              encoded_content: list[Encoding]) -> np.ndarray:
    chunk = []
    tensors: list[list[int]] = []
    assert max_seq_len > 0
    collect_max = 1000
    line_id = 0
    total = len(encoded_content)
    with tqdm(total=total) as pbar:
        while encoded_content:
            line = encoded_content.pop()
            tensor: list[int] = line.ids + [tokenizer.token_to_id('<|endoftext|>')]
            chunk += tensor
            pbar.update()
            del line
            if (line_id + 1) % (min(line_id + collect_max, total)) == 0:
                gc.collect()
            line_id += 1
            

    if len(chunk) > max_seq_len:
        tensors, chunk = split_chunk(max_seq_len, chunk)
        
    if len(chunk) > 0:
        
        logger.info(f'Adding padding to sequence: {(max_seq_len - len(chunk))} padding tokens')
        chunk += tuple([tokenizer.token_to_id('<|padding|>')]) * (max_seq_len - len(chunk))
        assert len(chunk) == max_seq_len
        tensors += [chunk]
        
    arr = np.asarray(tensors, dtype=np.int16)
    return arr

def split_chunk(max_seq_len: int, chunk: list[int]) -> tuple[list[list[int]], list[int]]:
    split = list(iter(partial(lambda it: tuple(islice(it, max_seq_len)), iter(chunk)), ()))
    tensors, chunk = split[:-1], split[-1]
    
    return tensors, chunk

def batch_encode(files: list[Path], tokenizer: Tokenizer):
    for file, content in zip(files, lazy_load(files, True)):
        yield file, tokenizer.encode_batch(content)


class TokenizeFilesCommand(Command):
    name = 'tokenize-files'
    description = 'tokenize_files.py - Try to tokenize provided txt files and save them to numpy format.'
    arguments = [argument('in-dir',
                          description='Path to directory with dataset.'),
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
        dirs = get_dir_list_from_file(list_dir_path)

        tokenizer = Tokenizer.from_file(self.argument('vocab-json'))

        files = [in_dir / _dir / 'data.txt' for _dir in dirs]
        logger.info(f'Tokenizing {len(files)} files.')  
        save_encoded(tokenizer=tokenizer,
                     max_seq_len=int(self.option('length')), 
                     out_dir=out_dir,
                     encoded_batches=batch_encode(files=files,
                                                  tokenizer=tokenizer))
