from pathlib import Path

from cleo.commands.command import Command
from cleo.helpers import argument
from loguru import logger
from tokenizers import AddedToken
from tokenizers.implementations import ByteLevelBPETokenizer

from diffccoder.configs.base import load_config
from diffccoder.configs.tokenizer_config import TokenizerConfig
from diffccoder.data.utils import get_dir_list_from_file, lazy_load


class TrainTokenizerCommand(Command):
    name = 'train-tokenizer'
    description = 'train_tokenizer.py - Trains ByteLevel-BPE tokenizer on provided raw txt data.'
    arguments = [argument('in-dir',
                          description='Path to directory with txts.'),
                 argument('out-file',
                          description='Path to output json file.'),
                 argument('list-dir-txt',
                          description='Path to file that contains directory names list to process.'),
                 argument('tokenizer-yaml',
                          description='Path to tokenizer config.')]
    
    def handle(self) -> int:
        in_dir = Path(self.argument('in-dir'))
        assert in_dir.is_dir(), 'Not a directory'

        out_file = Path(self.argument('out-file'))

        list_dir_path = Path(self.argument('list-dir-txt'))
        assert list_dir_path.is_file(), 'Not a file'

        dirs = get_dir_list_from_file(list_dir_path)

        input_files = [(in_dir / _dir / 'data.txt') for _dir in dirs]
        
        logger.info(f'{len(input_files)} files will be processed to perform ByteLevelBPE.')
        
        yaml_config_path = Path(self.argument('tokenizer-yaml'))     
        yaml_config: TokenizerConfig = load_config(yaml_config_path)

        tokenizer = ByteLevelBPETokenizer(merges=yaml_config.merges,
                                          add_prefix_space=yaml_config.add_prefix_space,
                                          continuing_subword_prefix=yaml_config.continuing_subword_prefix,
                                          dropout=yaml_config.dropout,
                                          lowercase=yaml_config.lowercase,
                                          unicode_normalizer=yaml_config.unicode_normalizer,
                                          end_of_word_suffix=yaml_config.end_of_word_suffix,
                                          trim_offsets=yaml_config.trim_offsets)
        
        tokenizer.train_from_iterator(iterator=lazy_load(input_files),
                                      special_tokens=yaml_config.special_tokens,
                                      vocab_size=yaml_config.vocab_size,
                                      min_frequency=yaml_config.min_frequency,
                                      length=len(input_files))
        tokenizer.add_tokens([AddedToken(" " * n) for n in range(2, 25)])
        logger.info(f'Saving result of BPE to: {out_file}')
        tokenizer.save(out_file.__str__())
        logger.success('All done!')
