from pathlib import Path

from cleo.commands.command import Command
from cleo.helpers import argument, option
from loguru import logger

from diffccoder.configs.base import dump_config, load_config
from diffccoder.configs.diffusion_config import DiffusionConfig
from diffccoder.configs.experiment_config import ExperimentConfig
from diffccoder.configs.optimization_config import OptimizationConfig
from diffccoder.configs.rwkv_config import RWKVConfig
from diffccoder.configs.tokenizer_config import TokenizerConfig
from diffccoder.configs.trainer_config import DebugTrainerConfig, TrainerConfig


class GenerateTemplateYamlsCommand(Command):
    name = 'generate-template-yamls'
    description = 'generate_template_yamls.py Generates template yamls from configuration classes.'
    arguments = [argument('out-dir',
                          description='Path to directory that will contain extracted data.')]
    options = [option('sanity', 's',
                      description='Check if yamls loads properly.')]
    
    def handle(self) -> int:
        diffusion = DiffusionConfig()
        optimization = OptimizationConfig()
        rwkv = RWKVConfig()
        tokenizer = TokenizerConfig()
        trainer = TrainerConfig()
        debug = DebugTrainerConfig()
        exp = ExperimentConfig()

        out_dir = Path(self.argument('out-dir'))

        if not out_dir.is_dir():
            out_dir.mkdir(mode=777, parents=True, exist_ok=True)

        dump_config(diffusion, out_dir)
        dump_config(optimization, out_dir)
        dump_config(rwkv, out_dir)
        dump_config(tokenizer, out_dir)
        dump_config(trainer, out_dir)
        dump_config(debug, out_dir)
        dump_config(exp, out_dir)

        if self.option('sanity'):
            for file in out_dir.iterdir():
                if file.suffix == '.yaml':
                    logger.debug(f'Checking: {file}')
                    load_config(file)
