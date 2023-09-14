''' train_tokenizer.py - Extracts and concatenates text from xmls (National Corpus of Polish)
Usage:
    train_tokenizer.py [options] <in_dir> <out_file> <list_dir.txt> <tokenizer.yaml>
    train_tokenizer.py (-h | --help)
    train_tokenizer.py (-v | --version)
    
Arguments:
    <in_dir>                Path to directory with xmls.
    <out_file>              Path to output json file.
    <list_dir.txt>          Path to file that contains directory names list to process.
    <tokenizer.yaml>        Path to tokenizer config.
    
Options:
    -h --help               Shows this screen.
    -v --version            Displays version.

'''
from docopt import docopt
from pathlib import Path
from loguru import logger
from tokenizers.implementations import ByteLevelBPETokenizer
# from tokenizers import Tokenizer
# from tokenizers.models import BPE
# from tokenizers.pre_tokenizers import Whitespace
# from tokenizers.trainers import BpeTrainer

from diffccoder.configs.base import load_config
from diffccoder.configs.tokenizer_config import TokenizerConfig
from diffccoder.data.utils import get_dir_list_from_file, lazy_load


def main(args):  
    in_dir = Path(args['<in_dir>'])
    assert in_dir.is_dir(), 'Not a directory'
    
    out_file = Path(args['<out_file>'])
    
    list_dir_path = Path(args['<list_dir.txt>'])
    assert list_dir_path.is_file(), 'Not a file'
    
    dirs = get_dir_list_from_file(list_dir_path)
    
    input_files = [(in_dir / _dir / 'data.txt') for _dir in dirs]
    logger.info(f'{len(input_files)} files will be processed to perform ByteLevelBPE.')
    yaml_config_path = Path(args['<tokenizer.yaml>'])     
    yaml_config: TokenizerConfig = load_config(yaml_config_path)
    # tokenizer = Tokenizer(BPE())
    # tokenizer.pre_tokenizer = Whitespace()
    # trainer = BpeTrainer(
    #     special_tokens=yaml_config.special_tokens,
    #     vocab_size=yaml_config.vocab_size,
    #     continuing_subword_prefix="\u0120",
    #     min_frequency=1,
    # )
    
    tokenizer = ByteLevelBPETokenizer(merges=yaml_config.merges,
                                      add_prefix_space=yaml_config.add_prefix_space,
                                      continuing_subword_prefix=yaml_config.continuing_subword_prefix,
                                      dropout=yaml_config.dropout,
                                      lowercase=yaml_config.lowercase,
                                      unicode_normalizer=yaml_config.unicode_normalizer,
                                      end_of_word_suffix=yaml_config.end_of_word_suffix,
                                      trim_offsets=yaml_config.trim_offsets)
    
    tokenizer.train_from_iterator(iterator=lazy_load(input_files, logger=logger),
                                  special_tokens=yaml_config.special_tokens,
                                  vocab_size=yaml_config.vocab_size,
                                  min_frequency=yaml_config.min_frequency,
                                  length=len(input_files))
    
    logger.info(f'Saving result of BPE to: {out_file}')
    tokenizer.save(out_file.__str__())
    logger.success('All done!')
    
if __name__ == '__main__':
    main(docopt(__doc__, version='train_tokenizer.py 0.0.1'))
    