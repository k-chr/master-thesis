""" extract_parquet.py - extracts parquet data to txt
Usage:
    extract_parquet.py [options] <in_dir> <out_dir>
    extract_parquet.py (-h | --help)
    extract_parquet.py (-v | --version)
    
Arguments:
    <in_dir>                Path to directory that contains .parquet files
    <out_dir>               Path to directory that will contain extracted data.
    
Options:
    -h --help               Shows this screen.
    -v --version            Displays version.
    -s --scalar <scalar>    Column / scalar name, where text is stored in parquet file [default: text].
    -n --name <name>        Name of output txt file [default: data].       
    
"""
from pathlib import Path
from typing import Any

from loguru import logger
from docopt import docopt
import pandas as pd
from tqdm import tqdm


def extract_and_dump_text_scalar(path: Path,
                                 out_dir: Path,
                                 scalar_name: str,
                                 out_fn: str):
    
    d = pd.read_parquet(path, 'fastparquet')
    assert scalar_name in d, 'Provided key is not valid column name in parquet file'
    def filter_new_line(s: str):
        return s.replace('\n', ' ').replace('\r', ' ')
    text = d[scalar_name].transform(filter_new_line)
    _dir = out_dir / path.stem
    _dir.mkdir(mode=777, exist_ok=True)
    out_without_ext = _dir / out_fn
    
    with open(out_without_ext.with_suffix('.txt'), 'w') as f:
        df_string = '\n'.join(text[:])
        f.write(df_string)
        logger.success(f'Written: {out_without_ext.__str__()}.txt')    
    

def find_parquet_files_in_dir(in_dir: Path):
    paths: list[Path] = []
    
    for i in (in_dir.iterdir()):
        logger.info(f'Traversing: {i}')
        if i.is_dir():
            paths += find_parquet_files_in_dir(i)
        
        elif i.suffix == '.parquet':
            paths += [i]
    
    return paths

def main(args: dict[str, Any]):

    in_dir = Path(args['<in_dir>'])
    assert in_dir.is_dir(), 'Not a directory'
    
    out_dir = Path(args['<out_dir>']) / in_dir.stem
    
    if not out_dir.is_dir():
        out_dir.mkdir(mode=777, parents=True, exist_ok=True)
            
    parquets = find_parquet_files_in_dir(in_dir)
    logger.info(f'Found {len(parquets)} files')
    for parquet in tqdm(parquets):
        extract_and_dump_text_scalar(path=parquet,
                                     out_dir=out_dir,
                                     scalar_name=args['--scalar'],
                                     out_fn=args['--name'])
    
if __name__ == '__main__':
    main(docopt(__doc__, version='extract_parquet.py 0.0.1'))
