from dataclasses import dataclass
import math
from pathlib import Path

from loguru import logger
import numpy as np


@dataclass
class MMapLimit:
    lower: int
    upper: int
    index: int

def save_chunk(stem: str,
               out_dir: Path,
               out_fn: str,
               lines_read: list[str],
               chunk_counter: int):
    new_dir = out_dir / (stem + f'_part_{chunk_counter}')
 
    if not new_dir.is_dir():
        new_dir.mkdir(parents=True, exist_ok=True)
                
    out_path = (new_dir / out_fn).with_suffix('.txt')
            
    with open(out_path, 'w') as out_file:
        out_file.writelines(lines_read)
        logger.info(f'Written {len(lines_read)} lines to {out_path}')
        
def get_dir_list_from_file(list_dir_path: Path):
    with open(list_dir_path, 'r') as list_dir_file:
        dirs = [_dir.rstrip() for _dir in list_dir_file.readlines()]
    return dirs

def get_version_from_filename(filename: str, version_sep: str) -> int:
    rV = 0
    if filename.find(version_sep) != -1:
        rV = int(filename.split(version_sep)[1])
    return rV

def get_last_ckpt_name(ckpt_dir: Path):
    last_version = max([get_version_from_filename(path.stem, '-v') for path in ckpt_dir.glob('./last*.ckpt')], default=0)
    last_ckpt_fname = f'last{("" if not last_version else "-v" + str(last_version))}.ckpt'
    return last_ckpt_fname

def lazy_load(paths: list[Path], read_whole_file: bool = False):
    line: str
    for path in paths:
        with open(path, 'r') as f:
            logger.info(f'Yielding content from: {path}')

            if read_whole_file:
                yield [line.rstrip() for line in f.readlines()]
            else:
                while line := f.readline():
                    yield line.rstrip()

def try_read(path: Path) -> str | None:
    text = None
    for enc in ['utf-8', 'windows-1250', 'windows-1252']:
            try:
                text = read_with_encoding(path, enc)
                break
            except:
                pass
    else:
        logger.info(f'No more encodings; Current file: {path};')
    return text

def read_with_encoding(path: Path, enc: str ='utf-8'):
    with path.open('r', encoding=enc) as f:
        return f.read()
                    
def partition_dataset(data_len: int,
                      num_partitions: int | None = None,
                      shuffle: bool = False,
                      seed: int = 0,
                      drop_last: bool = False) -> list[list[int]]:
   
    indices = []

    root_indices = list(range(data_len))
    
    if shuffle:
        # deterministically shuffle based on fixed seed for every process
        rs = np.random.RandomState(seed)
        
        rs.shuffle(root_indices)

    if not num_partitions:
        raise ValueError("must specify number of partitions or ratios.")
    
    if data_len < num_partitions:
        raise RuntimeError(f"there is no enough data to be split into {num_partitions} partitions.")

    if drop_last and data_len % num_partitions != 0:
        # split to nearest available length that is evenly divisible
        num_samples = math.ceil((data_len - num_partitions) / num_partitions)
    else:
        num_samples = math.ceil(data_len / num_partitions)
    # use original data length if not even divisible
    total_size = num_samples * num_partitions

    if not drop_last and total_size - data_len > 0:
        # add extra samples to make it evenly divisible
        root_indices += root_indices[: (total_size - data_len)]
    else:
        # remove tail of data to make it evenly divisible
        root_indices = root_indices[:total_size]

    for i in range(num_partitions):
        _indices = root_indices[i:total_size:num_partitions]
        indices.append(_indices)

    return indices