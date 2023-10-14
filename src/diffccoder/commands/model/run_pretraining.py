from pathlib import Path

from cleo.commands.command import Command
from cleo.helpers import argument

from diffccoder.configs.base import load_config
from diffccoder.configs.experiment_config import ExperimentConfig
from diffccoder.configs.optimization_config import OptimizationConfig
from diffccoder.configs.rwkv_config import RWKVConfig
from diffccoder.configs.trainer_config import DebugTrainerConfig
from diffccoder.data.npy_data_loader import NPYDataModule
from diffccoder.workflow.model_runner import ModelRunner
from diffccoder.workflow.pretraining.module import PretrainingModule


class PreTrainingCommand(Command):
    name = 'run-pretraining'
    description = 'run_pretraining.py - Runs RWKV model on pre-training scenario.'
    arguments = [argument('pretraining-yaml',
                          description='Path to pretraining configuration.')]
    
    def handle(self) -> int:
        exp_config: ExperimentConfig = load_config(Path(self.argument('pretraining-yaml')))
        config_dir = exp_config.work_dir / 'config'
        
        data_module = NPYDataModule(in_dir=exp_config.data_dir,
                                    dir_list_txt=config_dir / 'pre-train_dir_list.txt',
                                    split_val_ratio=exp_config.split_val_ratio,
                                    use_pinned_memory=exp_config.pin_memory,
                                    num_workers=exp_config.number_of_workers,
                                    batch_size=exp_config.batch_size,
                                    val_batch_size=exp_config.val_batch_size)
        
        trainer_cfg = load_config(config_dir / 'trainerconfig.yaml')
        debug_cfg: DebugTrainerConfig = load_config(config_dir / 'debugtrainerconfig.yaml')
        optim_cfg: OptimizationConfig = load_config(config_dir / 'optimizationconfig.yaml')
        rwkv_cfg: RWKVConfig = load_config(config_dir / 'rwkvconfig.yaml')
        
        
        
        model_runner = ModelRunner(trainer_config=trainer_cfg,
                                debug_config=debug_cfg,
                                logger=[

                                ],
                                callbacks=[
                                    
                                ])
                                
        net_module = PretrainingModule(optim_cfg, rwkv_cfg)
        
        model_runner.fit(net_module, datamodule=data_module)
    