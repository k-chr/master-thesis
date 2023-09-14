''' tokenize_files.py - Try to tokenize provided txt files and save them to numpy format.
Usage:
    tokenize_files.py [options] <in_dir> <vocab.json> <list_dir.txt> <out_dir>
    tokenize_files.py (-h | --help)
    tokenize_files.py (-v | --version)
    
Arguments:
    <in_dir>                Path to directory with dataset.
    <vocab.json>            Path to input vocab json file.
    <list_dir.txt>          Path to file that contains directory names list to process.
    <out_dir>               Path to the output directory.
    
Options:
    -h --help               Shows this screen.
    -v --version            Displays version.
    -l --length <length>    Max length of sequence [default: 1024].

'''
import gc
from typing import Any, Generator
from pathlib import Path

from docopt import docopt
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
        npz_path = file_path.with_suffix('.npz')
        f_name = npz_path.name
        npz_dir = (out_dir / npz_path.parent.parent.stem / npz_path.parent.stem)
        npz_dir.mkdir(mode=777, parents=True, exist_ok=True)
        
        npz_path = npz_dir / f_name
        
        arr = get_ndarray_from_encoding(tokenizer, max_seq_len, encoded_content)
        np.save(npz_path, arr)
        logger.success(f'Successfully written {arr.size} elements to {npz_path}')
        del arr
        del encoded_content
        gc.collect()

def get_ndarray_from_encoding(tokenizer: Tokenizer,
                              max_seq_len: int,
                              encoded_content: list[Encoding]) -> np.ndarray:
    chunk = []
    tensors: list[list[int]] = []
    assert max_seq_len > 0
    for line in tqdm(encoded_content):
        tensor: list[int] = line.ids + [tokenizer.token_to_id('<|endoftext|>')]
        chunk += tensor
        if len(chunk) > max_seq_len:
            tensor, chunk = split_chunk(max_seq_len, chunk)
            tensors += tensor
        
    if len(chunk) > 0:
        if len(chunk) > max_seq_len:
            last_tensors, chunk = split_chunk(max_seq_len, chunk)
            tensors += last_tensors
            
        if len(chunk) > 0:
            
            logger.info(f'Adding padding to sequence: {(max_seq_len - len(chunk))} padding tokens')
            chunk += [tokenizer.token_to_id('<|padding|>')] * (max_seq_len - len(chunk))
            assert len(chunk) == max_seq_len
            tensors += [chunk]
            
    arr = np.asarray(tensors)
    return arr

def split_chunk(max_seq_len: int, chunk: list[int]) -> tuple[list[list[int]], list[int]]:
    tensors = []
    while len(chunk) > max_seq_len:
        tensor, chunk = (chunk[:max_seq_len], chunk[max_seq_len:])
        tensors += [tensor]
    return tensors, chunk

def batch_encode(files: list[Path], tokenizer: Tokenizer):
    for file, content in zip(files, lazy_load(files, True, logger)):
        yield file, tokenizer.encode_batch(content)

def main(args: dict[str, Any]):
    in_dir = Path(args['<in_dir>'])
    assert in_dir.is_dir(), 'Not a directory'
    list_dir_path = Path(args['<list_dir.txt>'])
    assert list_dir_path.is_file(), 'Not a file'
    out_dir = Path(args['<out_dir>']) 
    dirs = get_dir_list_from_file(list_dir_path)
    
    tokenizer = Tokenizer.from_file(args['<vocab.json>'])
    
    files = [in_dir / _dir / 'data.txt' for _dir in dirs]
    logger.info(f'Tokenizing {len(files)} files.')  
    save_encoded(tokenizer=tokenizer,
                 max_seq_len=int(args['--length']), 
                 out_dir=out_dir,
                 encoded_batches=batch_encode(files=files,
                                              tokenizer=tokenizer))
    
if __name__ == '__main__':
    main(docopt(__doc__, version='tokenize_files.py 0.0.1'))
     