''' chunkify_large_txt.py - extracts parquet data to txt
Usage:
    chunkify_large_txt.py [options] <in_file.txt> <out_dir>
    chunkify_large_txt.py (-h | --help)
    chunkify_large_txt.py (-v | --version)
    
Arguments:
    <in_file.txt>           Path to large txt file.
    <out_dir>               Path to directory that will contain extracted data.
    
Options:
    -h --help               Shows this screen.
    -v --version            Displays version.
    -n --name <name>        Name of output txt file [default: data].
    -l --lines <lines>      Number of lines to split [default: 16000].
'''

from io import TextIOWrapper
from pathlib import Path
from typing import Any

from docopt import docopt
from loguru import logger

from diffccoder.data.utils import save_chunk

def process_file(file: TextIOWrapper, out_dir: Path, out_fn: str, lines: int):
    lines_read = []
    counter = 0
    chunk_counter = 0
    stem = Path(file.name).stem
    
    while line := file.readline():
        counter += 1
        lines_read += [line.rstrip()]
        
        if counter >= lines:
            chunk_counter += 1
            save_chunk(stem, out_dir, out_fn, lines_read, chunk_counter, logger)
            lines_read.clear()
            counter = 0
    
    if counter > 0:
        chunk_counter += 1
        save_chunk(stem, out_dir, out_fn, lines_read, chunk_counter, logger)
        lines_read.clear() 
        counter = 0
    

def main(args: dict[str, Any]):
    in_file = Path(args['<in_file.txt>'])
    assert in_file.is_file(), 'Not a file'
        
    with open(in_file, 'r') as f:
        process_file(f, Path(args['<out_dir>']), args['--name'], int(args['--lines']))
        logger.success(f'Finished: {in_file}')
        
if __name__ == '__main__':
    main(docopt(__doc__, version='chunkify_large_txt.py 0.0.1'))
