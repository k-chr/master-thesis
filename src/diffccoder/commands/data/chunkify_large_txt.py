from io import TextIOWrapper
from pathlib import Path

from cleo.commands.command import Command
from cleo.helpers import argument, option
from loguru import logger

from diffccoder.data.utils import save_chunk


class ChunkifyLargeTextCommand(Command):
    name = 'chunkify-large-txt'
    description = 'chunkify_large_txt.py - splits large txt to smaller files'
    arguments = [argument('in-file-txt',
                          description='Path to large txt file.'),
                 argument('out-dir',
                          description='Path to directory that will contain extracted data.')]
    options = [option('lines', 'l',
                      description='Number of lines to split [default: 16000].',
                      default='text',
                      flag=False),
               option('name', 'n',
                      description='Name of output txt file [default: data].',
                      default='data',
                      flag=False)]
    
    def handle(self):
        in_file = Path(self.argument('in-file-txt'))
        assert in_file.is_file(), 'Not a file'

        with open(in_file, 'r') as f:
            process_file(f, Path(self.argument('out-dir')), self.option('name'), int(self.option('lines')))
            logger.success(f'Finished: {in_file}')
            
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
            save_chunk(stem, out_dir, out_fn, lines_read, chunk_counter)
            lines_read.clear()
            counter = 0
    
    if counter > 0:
        chunk_counter += 1
        save_chunk(stem, out_dir, out_fn, lines_read, chunk_counter)
        lines_read.clear() 
        counter = 0
