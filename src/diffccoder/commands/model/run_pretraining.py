from datetime import timedelta
from functools import partial
import os
from pathlib import Path

from cleo.commands.command import Command
from cleo.helpers import argument
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from loguru import logger
import mlflow as mlflow_client
import numpy.random as np_rand
import torch as t
from torchinfo import summary

from diffccoder.configs.base import dump_config, load_config 
from diffccoder.configs.experiment_config import EXCL_KEYS as EXP_EXCL_KEYS, ExperimentConfig
from diffccoder.configs.optimization_config import OptimizationConfig
from diffccoder.configs.rwkv_config import RWKVConfig
from diffccoder.configs.trainer_config import DebugTrainerConfig, TrainerConfig, get_auto_devices
from diffccoder.data.npy_data_loader import NPYDataModule
from diffccoder.data.utils import get_last_ckpt_name
from diffccoder.lightning_modules.mlflow_distinct_logger import MLFlowDistinctLogger
from diffccoder.utils.mlflow_utils import log_config
from diffccoder.utils.task_scheduler import RepeatingScheduler
from diffccoder.lightning_modules.model_runner import ModelRunner
from diffccoder.lightning_modules.pretraining.module import PretrainingModule

DEFAULT_RUN_NAME = 'Pre-Training'

def update_mlflow(command: str, name='mlflow-updater'):
    return os.system(f'poetry run app {name} {command}')


class PreTrainingCommand(Command):
    
    name = 'run-pretraining'
    description = 'run_pretraining.py - Runs RWKV model on pre-training scenario.'
    arguments = [argument('pretraining-yaml',
                          description='Path to pretraining configuration.'),
                 argument('pre-train-dirlist.txt',
                          description='Path to file with directory list to include during training')]
    
    def handle(self) -> int:
        exp_config_path = Path(self.argument('pretraining-yaml'))
        exp_config: ExperimentConfig = load_config(exp_config_path)
        config_dir = exp_config.work_dir / 'configs'
        
        trainer_cfg: TrainerConfig = load_config(config_dir / 'trainerconfig.yaml')     
        debug_cfg: DebugTrainerConfig = load_config(p) if (p := Path(config_dir / 'debugtrainerconfig.yaml')).is_file() else None
        optim_cfg: OptimizationConfig = load_config(config_dir / 'optimizationconfig.yaml')
        rwkv_cfg: RWKVConfig = load_config(config_dir / 'rwkvconfig.yaml')
        
        if exp_config.seed is None:
            seed = np_rand.randint(int(2**32 - 1))
            logger.info(f'Setting experiment random seed to: {seed}')
            exp_config.seed = seed
            
            dump_config(exp_config, config_dir)
        
        if os.environ.get('DIFFCCODER_SEED', None) is None:
            os.environ['DIFFCCODER_SEED'] = str(exp_config.seed)
        
        if os.environ.get('EXP_DEVICES', None) is None:
            os.environ['EXP_DEVICES'] = str(trainer_cfg.devices if isinstance(trainer_cfg.devices, int) else get_auto_devices(trainer_cfg))
            
        if os.environ.get('DTYPE', None) is None:
            os.environ['DTYPE'] = str(trainer_cfg.precision)
        
        dirlist_txt = Path(self.argument('pre-train-dirlist.txt'))
        data_module = NPYDataModule(in_dir=exp_config.data_dir,
                                    dir_list_txt=dirlist_txt,
                                    split_val_ratio=exp_config.split_val_ratio,
                                    use_pinned_memory=exp_config.pin_memory,
                                    num_workers=exp_config.number_of_workers,
                                    batch_size=exp_config.batch_size,
                                    val_batch_size=exp_config.val_batch_size)
                
        _logger = []
        _callbacks = []
        
        last = ModelCheckpoint(dirpath=exp_config.work_dir / 'artifacts',
                               save_top_k=0,
                               save_last=True,
                               every_n_train_steps=trainer_cfg.log_every_n_steps)
            
        _callbacks.append(last)
            
        for metric in exp_config.metrics_to_save_cp:
            monitor = ModelCheckpoint(dirpath=exp_config.work_dir / 'artifacts',
                                      filename=f'best_on_val_{metric}',
                                      save_top_k=1,
                                      save_on_train_epoch_end=False,
                                      monitor=f'validation_{metric}',
                                      save_last=False)
            _callbacks.append(monitor)
            monitor = ModelCheckpoint(dirpath=exp_config.work_dir / 'artifacts',
                                      filename=f'best_on_train_{metric}',
                                      save_top_k=1,
                                      save_on_train_epoch_end=True,
                                      monitor=f'train_{metric}',
                                      save_last=False)
            _callbacks.append(monitor)

        lr_monitor = LearningRateMonitor(logging_interval='step')
        _summary = ModelSummary(max_depth=-1)
        _callbacks.append(lr_monitor)
        _callbacks.append(_summary)
        
        if exp_config.mlflow_enabled and exp_config.experiment_name:
            
            if exp_config.mlflow_http_timeout != 120:
                os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = str(exp_config.mlflow_http_timeout)
            
            if not exp_config.mlflow_continue_run or exp_config.mlflow_run_id is None:
            
                with mlflow_client.start_run(run_name=exp_config.mlflow_run_name or DEFAULT_RUN_NAME) as run:
                    exp_config.mlflow_run_name = run.info.run_name
                    exp_config.mlflow_run_id = run.info.run_id
                    dump_config(exp_config, config_dir)
                
            mlflow = MLFlowDistinctLogger(experiment_name=exp_config.experiment_name,
                                  run_name=exp_config.mlflow_run_name,
                                  run_id=exp_config.mlflow_run_id,
                                  tracking_uri=exp_config.mlflow_server,
                                  artifact_location=exp_config.work_dir / 'artifacts')
            
            _logger.append(mlflow)
            
        elif exp_config.mlflow_enabled:
            logger.error(f'MLFlow is set to be enabled in experiment, but experiment is not set-up. Closing...')
            return -1
        
        if exp_config.use_tensorboard:
            tensorboard = TensorBoardLogger(save_dir=exp_config.work_dir / 'tboard',
                                            name=exp_config.experiment_name)
            _logger.append(tensorboard)

        if not _logger:
            csv_logger = CSVLogger(save_dir=exp_config.work_dir / 'csvlogs',
                                   name=exp_config.experiment_name,
                                   flush_logs_every_n_steps=trainer_cfg.log_every_n_steps**2)
            _logger.append(csv_logger)
        
        model_runner = ModelRunner(trainer_config=trainer_cfg,
                                debug_config=debug_cfg,
                                logger=_logger,
                                callbacks=_callbacks,
                                strategy=DDPStrategy(process_group_backend='gloo',
                                                     timeout=timedelta(days=1.0),
                                                     start_method='popen'))
        if exp_config.mlflow_enabled and rank_zero_only.rank == 0:

            command = f'{os.environ["REMOTE_TRACKING_URI"]} {exp_config.experiment_name} {exp_config.mlflow_run_name} -vvv'

            r = RepeatingScheduler(function=partial(update_mlflow, *(command, 'mlflow-updater')),
                                   interval=exp_config.mlflow_log_to_remote_freq)
            r.daemon = True
            r.start()
            
        try:
            
            ckpt_dir: Path = exp_config.work_dir / 'artifacts'
            last_ckpt_fname = get_last_ckpt_name(ckpt_dir)
            
            ckpt_path: Path = ckpt_dir / last_ckpt_fname if not exp_config.from_pretrained else exp_config.from_pretrained
            kwargs = {'ckpt_path':ckpt_path} if ckpt_path.is_file() else {}
            if not kwargs:
                if not ckpt_dir.exists():
                    ckpt_dir.mkdir(exist_ok=True)
                init_path = ckpt_dir / 'init.pt'
                if model_runner.global_rank == 0:
                    net_module = PretrainingModule(optim_cfg, rwkv_cfg, skip_init=False)
                    t.save(net_module.model.state_dict(), init_path)
                else:
                    while not  init_path.is_file():...
                    net_module = PretrainingModule(optim_cfg, rwkv_cfg, skip_init=True)
                    net_module.model.load_state_dict(t.load(init_path, map_location='cpu'))
            else:
                net_module = PretrainingModule(optim_cfg, rwkv_cfg, skip_init=True)


            if rank_zero_only.rank == 0 and exp_config.mlflow_enabled:
                log_config(mlflow.experiment,
                           exp_config.mlflow_run_id,
                           exp_config,
                           excl_keys=EXP_EXCL_KEYS)
                log_config(mlflow.experiment,
                           exp_config.mlflow_run_id,
                           rwkv_cfg)
                log_config(mlflow.experiment,
                           exp_config.mlflow_run_id,
                           optim_cfg)
            
            logger.info(f"Summary:\n{summary(net_module.model, (exp_config.batch_size, rwkv_cfg.context_length), dtypes=[t.int64])}")

            logger.info(f'Running on: {model_runner.accelerator}; Skipping initialisation?: {bool(kwargs)}')
            model_runner.fit(net_module, datamodule=data_module, **kwargs)
            
        finally:
            if exp_config.mlflow_enabled and rank_zero_only.rank == 0:
                r.cancel()