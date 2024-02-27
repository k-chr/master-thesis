from datetime import timedelta
import os
from pathlib import Path

from cleo.commands.command import Command
from cleo.helpers import argument, option
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.strategies import DDPStrategy
from loguru import logger
import mlflow as mlflow_client
import numpy.random as np_rand
from tokenizers import Tokenizer

from diffccoder.configs.base import dump_config, load_config
from diffccoder.configs.diffusion_config import DiffusionConfig 
from diffccoder.configs.experiment_config import ExperimentConfig
from diffccoder.configs.rwkv_config import RWKVConfig
from diffccoder.configs.trainer_config import DebugTrainerConfig, TrainerConfig, get_auto_devices
from diffccoder.data.npy_data_loader import NPYZDataModule
from diffccoder.data.utils import get_last_ckpt_name
from diffccoder.lightning_modules.inference.module import DiffusionInferModule
from diffccoder.lightning_modules.model_runner import ModelRunner
from diffccoder.lightning_modules.prediction_writer import PredictionStringWriter

DEFAULT_RUN_NAME = 'Diff-Inference'

def update_mlflow(command: str, name='mlflow-updater'):
    return os.system(f'poetry run app {name} {command}')


class InferCommand(Command):
    
    name = 'run-inference'
    description = 'run_inference.py - Runs DIFFRWKV model inference.'
    arguments = [argument('experiment-yaml',
                          description='Path to training configuration.'),
                 argument('infer-dirlist.txt',
                          description='Path to file with directory list to include during training'),
                 argument('vocab.json',
                          description='Path to file with tokenizer vocabulary')]
    options = [option('batch-size', 'b',
                      description='Batch size for inference',
                      default=1,
                      flag=False),
               option('target-len', 'l',
                      description='Len of generated samples',
                      default=1024,
                      flag=False)]
    
    def handle(self) -> int:
        exp_config_path = Path(self.argument('experiment-yaml'))
        exp_config: ExperimentConfig = load_config(exp_config_path)
        config_dir = exp_config.work_dir / 'configs'
        
        trainer_cfg: TrainerConfig = load_config(config_dir / 'trainerconfig.yaml')     
        debug_cfg: DebugTrainerConfig = load_config(p) if (p := Path(config_dir / 'debugtrainerconfig.yaml')).is_file() else None
        rwkv_cfg: RWKVConfig = load_config(config_dir / 'rwkvconfig.yaml')
        diff_cfg: DiffusionConfig = load_config(config_dir / 'diffusionconfig.yaml')
        
        if exp_config.seed is None:
            seed = np_rand.randint(int(2**31))
            logger.info(f'Setting experiment random seed to: {seed}')
            exp_config.seed = seed
            
            dump_config(exp_config, config_dir)
        
        if os.environ.get('DIFFCCODER_SEED', None) is None:
            os.environ['DIFFCCODER_SEED'] = str(exp_config.seed)
        
        if os.environ.get('EXP_DEVICES', None) is None:
            os.environ['EXP_DEVICES'] = str(trainer_cfg.devices if isinstance(trainer_cfg.devices, int) else get_auto_devices(trainer_cfg))
            
        if os.environ.get('DTYPE', None) is None:
            os.environ['DTYPE'] = str(trainer_cfg.precision)
            
        dirlist_txt = Path(self.argument('infer-dirlist.txt'))
        
        data_module = NPYZDataModule(in_dir=exp_config.data_dir,
                                     dir_list_txt=dirlist_txt,
                                     dir_list_for_infer=dirlist_txt,
                                     split_val_ratio=exp_config.split_val_ratio,
                                     use_pinned_memory=exp_config.pin_memory,
                                     num_workers=exp_config.number_of_workers,
                                     batch_size=exp_config.batch_size,
                                     val_batch_size=exp_config.val_batch_size,
                                     test_batch_size=self.option('batch-size'),
                                     prefix_lm=exp_config.prefix_lm,
                                     pad_id=rwkv_cfg.pad_token_id,
                                     mode='npz')
        
        _logger = []
        _callbacks = []
        tokenizer = Tokenizer.from_file(self.argument('vocab.json'))
        _summary = ModelSummary(max_depth=-1)
        _predictor_writer = PredictionStringWriter(exp_config.out_dir, tokenizer, label=diff_cfg.inference_sampler.name.lower())
        
        _callbacks.append(_summary)
        _callbacks.append(_predictor_writer)
        
        if exp_config.mlflow_enabled and exp_config.experiment_name:
            
            os.environ.pop('MLFLOW_HTTP_REQUEST_TIMEOUT', '')
            if exp_config.mlflow_http_timeout != 1200:
                os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = str(exp_config.mlflow_http_timeout)
            else:
                os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = os.environ.get('DEFAULT_MLFLOW_HTTP_REQUEST_TIMEOUT', '1200')
            
            if not exp_config.mlflow_continue_run or exp_config.mlflow_run_id is None:
            
                with mlflow_client.start_run(run_name=exp_config.mlflow_run_name or DEFAULT_RUN_NAME) as run:
                    exp_config.mlflow_run_name = run.info.run_name
                    exp_config.mlflow_run_id = run.info.run_id
                    dump_config(exp_config, exp_config_path)
        
        model_runner = ModelRunner(trainer_config=trainer_cfg,
                                   debug_config=debug_cfg,
                                   logger=_logger,
                                   callbacks=_callbacks,
                                #    strategy=DDPStrategy(process_group_backend='gloo',
                                #                         timeout=timedelta(days=1.0),
                                #                         start_method='popen',
                                #                         gradient_as_bucket_view=True),
                                   use_distributed_sampler = False)
             
        ckpt_dir: Path = exp_config.work_dir / 'artifacts'
        last_ckpt_fname = get_last_ckpt_name(ckpt_dir)    
        
        ckpt_path: Path = ckpt_dir / last_ckpt_fname
        ema_path = ckpt_dir / 'ema.pt' if ckpt_path.is_file() else None
        kwargs = {'ckpt_path':ckpt_path} if ckpt_path.is_file() else {}
        if not exp_config.from_pretrained:
            net_module = DiffusionInferModule(rwkv_cfg, diff_cfg, tokenizer=tokenizer, ema_local_path=ema_path)
        else:
            net_module = DiffusionInferModule(rwkv_cfg,
                                              diff_cfg,
                                              from_pretrained=exp_config.from_pretrained,
                                              tokenizer=tokenizer)           

        logger.info(f'Running on: {model_runner.accelerator}; Skipping initialization?: {bool(kwargs)}')
                    
        model_runner.predict(net_module, datamodule=data_module, **kwargs)