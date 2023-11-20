from pathlib import Path

from cleo.commands.command import Command
from cleo.helpers import argument, option
from loguru import logger
import pandas as pd
from tqdm import tqdm


class ExtractParquetCommand(Command):
    name = 'extract-parquet'
    description = 'extract_parquet.py - extracts parquet data to txt'
    arguments = [argument('in-dir',
                          description='Path to directory that contains .parquet files.'),
                 argument('out-dir',
                          description='Path to directory that will contain extracted data.')]
    options = [option('scalar', 's',
                      description='Column / scalar name, where text is stored in parquet file [default: text].',
                      default='text',
                      flag=False),
               option('name', 'N',
                      description='Name of output txt file [default: data].',
                      default='data',
                      flag=False)]

    def handle(self) -> int:
        in_dir = Path(self.argument('in-dir'))
        assert in_dir.is_dir(), 'Not a directory'

        out_dir = Path(self.argument('out-dir')) / in_dir.stem

        if not out_dir.is_dir():
            out_dir.mkdir(parents=True, exist_ok=True)

        parquets = find_parquet_files_in_dir(in_dir)
        logger.info(f'Found {len(parquets)} files')
        for parquet in tqdm(parquets):
            extract_and_dump_text_scalar(path=parquet,
                                         out_dir=out_dir,
                                         scalar_name=self.option('scalar'),
                                         out_fn=self.option('name'))

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
    _dir.mkdir(exist_ok=True)
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
