from pathlib import Path
from typing import Any

from cleo.commands.command import Command
from cleo.helpers import argument, option
from loguru import logger
from lxml.etree import parse, _ElementTree, QName, _Element
from tqdm import tqdm

from diffccoder.data.utils import get_dir_list_from_file, save_chunk


class ProcessSmallXMLCommand(Command):
    name = 'process-small-xml'
    description = 'process_small_xml.py - Extracts and concatenates text from xmls (generally speaking this script was made for National Corpus of Polish)'
    arguments = [argument('in-dir',
                          description='Path to directory with xmls.'),
                 argument('out-dir',
                          description='Path to directory that will contain extracted data.'),
                 argument('list-dir-txt',
                          description='Path to file that contains directory names list to process.')]
    options = [option('subdir', 's',
                      description='Name of subdirectory in `out_dir` [default: xml_mixed_corpus]',
                      default='xml_mixed_corpus',
                      flag=False),
               option('name', 'n',
                      description='Name of output txt file [default: data].',
                      default='data',
                      flag=False),
               option('file', 'f',
                      description='Which xml contains target data? [default: text]',
                      default='text',
                      flag=False),
               option('lines', 'l',
                      description='Number of lines to split [default: 16000].',
                      default=16000,
                      flag=False),
               option('tag', 't',
                      description='Tag of xml content with text [default: ab]',
                      default='ab',
                      flag=False)]
    
    def handle(self) -> int:
        
        in_dir = Path(self.argument('in-dir'))
        assert in_dir.is_dir(), 'Not a directory'

        list_dir_path = Path(self.argument('list-dir-txt'))
        assert list_dir_path.is_file(), 'Not a file'

        dirs = get_dir_list_from_file(list_dir_path)

        logger.info(f'Found {dirs.__len__()} directories to process')

        process_dirs(dirs=dirs,
                     in_dir=in_dir,
                     out_dir=Path(self.argument('out-dir')),
                     sub_dir=self.option('subdir'),
                     lines_limit=int(self.option('lines')),
                     in_fn=self.option('file'),
                     out_fn=self.option('name'),
                     tag=self.option('tag'))

        logger.success(f'Finished all in {in_dir}')

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
    