''' process_small_xml.py - Extracts and concatenates text from xmls (National Corpus of Polish)
Usage:
    process_small_xml.py [options] <in_dir> <out_dir> <list_dir.txt>
    process_small_xml.py (-h | --help)
    process_small_xml.py (-v | --version)
    
Arguments:
    <in_dir>                Path to directory with xmls.
    <out_dir>               Path to directory that will contain extracted data.
    <list_dir.txt>          Path to file that contains directory names list to process.
    
Options:
    -h --help               Shows this screen.
    -v --version            Displays version.
    -f --file <file>        Which xml contains target data? [default: text]
    -n --name <name>        Name of output txt file [default: data].
    -s --subdir <subdir>    Name of subdirectory in `out_dir` [default: xml_mixed_corpus]
    -l --lines <lines>      Number of lines to split [default: 16000].
    -t --tag <tag>          Tag of xml content with text [default: ab]
'''
from pathlib import Path
from typing import Any

from docopt import docopt
from loguru import logger
from lxml.etree import parse, _ElementTree, QName, _Element
from tqdm import tqdm

from diffccoder.data.utils import get_dir_list_from_file, save_chunk


def process_dir(path: Path, in_fn: str, tag: str):
    file_path = (path / in_fn).with_suffix('.xml')
    logger.info(f'Processing: {file_path}')
    tree: _ElementTree = parse(file_path)
    
    mapped = map(lambda node: (node, QName(node).localname), tree.iter())
    text_containers: list[tuple[_Element, str]] = list(filter(lambda tup: tup[1] == tag, mapped))
    
    return [node.text for node, _ in text_containers if node.text]

def process_dirs(dirs: list[str],
                 in_dir: Path,
                 out_dir: Path,
                 sub_dir: str,
                 lines_limit: int,
                 in_fn: str,
                 out_fn: str,
                 tag: str):
    lines_read = []
    counter = 0
    chunk_counter = 0
    
    for _dir in tqdm(dirs):
        path = in_dir / _dir
        new_lines = process_dir(path, in_fn, tag)
        
        l = len(new_lines)
        logger.info(f'Found {l} lines in xml')
        
        if counter + l > lines_limit:
            save_chunk(sub_dir, out_dir, out_fn, lines_read, chunk_counter)
            lines_read.clear()
            counter = 0
            
        lines_read += new_lines
        counter += l
        
    if counter > 0:
        chunk_counter += 1
        save_chunk(sub_dir, out_dir, out_fn, lines_read, chunk_counter)
        lines_read.clear() 
        counter = 0    

def main(args: dict[str, Any]):
    in_dir = Path(args['<in_dir>'])
    assert in_dir.is_dir(), 'Not a directory'
    
    list_dir_path = Path(args['<list_dir.txt>'])
    assert list_dir_path.is_file(), 'Not a file'
    
    dirs = get_dir_list_from_file(list_dir_path)
    
    logger.info(f'Found {dirs.__len__()} directories to process')   
    
    process_dirs(dirs=dirs,
                 in_dir=in_dir,
                 out_dir=Path(args['<out_dir>']),
                 sub_dir=args['--subdir'],
                 lines_limit=int(args['--lines']),
                 in_fn=args['--file'],
                 out_fn=args['--name'],
                 tag=args['--tag'])
        
    logger.success(f'Finished all in {in_dir}')

if __name__ == '__main__':
    main(docopt(__doc__, version='process_small_xml.py 0.0.1'))
    