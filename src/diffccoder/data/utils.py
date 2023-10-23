from pathlib import Path

from loguru import logger


def save_chunk(stem: str,
               out_dir: Path,
               out_fn: str,
               lines_read: int,
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
                    