''' generate_template_yamls.py
Usage:
    generate_template_yamls.py [options] <out_dir>
    generate_template_yamls.py (-h | --help)
    generate_template_yamls.py (-v | --version)
    
Arguments:
    <out_dir>              Path to output json file.
    
Options:
    -h --help               Shows this screen.
    -v --version            Displays version.
    -s --sanity             Check if yamls loads properly
'''

from pathlib import Path
from typing import Any

from docopt import docopt
from loguru import logger

from diffccoder.configs.base import dump_config, load_config
from diffccoder.configs.diffusion_config import DiffusionConfig
from diffccoder.configs.optimization_config import OptimizationConfig
from diffccoder.configs.rwkv_config import RWKVConfig
from diffccoder.configs.tokenizer_config import TokenizerConfig
from diffccoder.configs.trainer_config import DebugTrainerConfig, TrainerConfig


def main(args: dict[str, Any]):
    diffusion = DiffusionConfig()
    optimization = OptimizationConfig()
    rwkv = RWKVConfig()
    tokenizer = TokenizerConfig()
    trainer = TrainerConfig()
    debug = DebugTrainerConfig()
    
    out_dir = Path(args['<out_dir>'])
    
    if not out_dir.is_dir():
        out_dir.mkdir(mode=777, parents=True, exist_ok=True)
      
    dump_config(diffusion, out_dir)
    dump_config(optimization, out_dir)
    dump_config(rwkv, out_dir)
    dump_config(tokenizer, out_dir)
    dump_config(trainer, out_dir)
    dump_config(debug, out_dir)
    
    if args['--sanity']:
        for file in out_dir.iterdir():
            if file.suffix == '.yaml':
                logger.debug(f'Checking: {file}')
                load_config(file)
                
if __name__ == '__main__':
    main(docopt(__doc__, version='generate_template_yamls.py 0.0.2'))