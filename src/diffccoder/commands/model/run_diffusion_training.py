import os
from pathlib import Path

from cleo.commands.command import Command
from cleo.helpers import argument
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from loguru import logger
import mlflow as mlflow_client
import torch as t
from torchinfo import summary

from diffccoder.configs.base import dump_config, load_config
from diffccoder.configs.diffusion_config import DiffusionConfig 
from diffccoder.configs.experiment_config import EXCL_KEYS as EXP_EXCL_KEYS, ExperimentConfig
from diffccoder.configs.optimization_config import OptimizationConfig
from diffccoder.configs.rwkv_config import RWKVConfig
from diffccoder.configs.trainer_config import DebugTrainerConfig, TrainerConfig
from diffccoder.data.npy_data_loader import NPYDataModule
from diffccoder.data.utils import get_last_ckpt_name
from diffccoder.model.diffusion.ema import EMA
from diffccoder.utils.mlflow_utils import log_config
from diffccoder.utils.task_scheduler import RepeatingScheduler
from diffccoder.lightning_modules.model_runner import EMAModelRunner
from diffccoder.lightning_modules.training.module import DiffusionTrainingModule

DEFAULT_RUN_NAME = 'Diff-Training'


class DiffTrainingCommand(Command):
    
    name = 'run-diff-training'
    description = 'run_diffusion_training.py - Runs RWKV model on pre-training scenario.'
    arguments = [argument('training-yaml',
                          description='Path to training configuration.'),
                 argument('train-dirlist.txt',
                          description='Path to file with directory list to include during training')]
    
    def handle(self) -> int:
        exp_config_path = Path(self.argument('training-yaml'))
        exp_config: ExperimentConfig = load_config(exp_config_path)
        config_dir = exp_config.work_dir / 'configs'
        dirlist_txt = Path(self.argument('train-dirlist.txt'))
        data_module = NPYDataModule(in_dir=exp_config.data_dir,
                                    dir_list_txt=dirlist_txt,
                                    split_val_ratio=exp_config.split_val_ratio,
                                    use_pinned_memory=exp_config.pin_memory,
                                    num_workers=exp_config.number_of_workers,
                                    batch_size=exp_config.batch_size,
                                    val_batch_size=exp_config.val_batch_size)
        
        trainer_cfg: TrainerConfig = load_config(config_dir / 'trainerconfig.yaml')      
        debug_cfg: DebugTrainerConfig = load_config(p) if (p := Path(config_dir / 'debugtrainerconfig.yaml')).is_file() else None
        optim_cfg: OptimizationConfig = load_config(config_dir / 'optimizationconfig.yaml')
        rwkv_cfg: RWKVConfig = load_config(config_dir / 'rwkvconfig.yaml')
        diff_cfg: DiffusionConfig = load_config(config_dir / 'diffusionconfig.yaml')
        
        _logger = []
        _callbacks = []
        
        os.environ['DTYPE'] = str(trainer_cfg.precision)
        
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
                                      save_on_train_epoch_end=False,
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
                    dump_config(exp_config, exp_config_path)
                
            mlflow = MLFlowLogger(experiment_name=exp_config.experiment_name,
                                  run_name=exp_config.mlflow_run_name,
                                  run_id=exp_config.mlflow_run_id,
                                  tracking_uri=exp_config.mlflow_server,
                                  artifact_location=exp_config.work_dir / 'artifacts')
            
            _logger.append(mlflow)
            
        elif exp_config.mlflow_enabled:
            logger.error(f'MlFlow is set to be enabled in experiment, but experiment is not set-up. Closing...')
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
        
        model_runner = EMAModelRunner(trainer_config=trainer_cfg,
                                debug_config=debug_cfg,
                                logger=_logger,
                                callbacks=_callbacks)
        command = f'PLACEHOLDER {os.environ["REMOTE_TRACKING_URI"]} {exp_config.experiment_name} {exp_config.mlflow_run_name} -vvv'
        r = RepeatingScheduler(function=lambda *_: self.call('mlflow-updater', command),
                               interval=exp_config.mlflow_log_to_remote_freq)
        r.daemon = True
        r.start()
        
        try:
            ckpt_dir: Path = exp_config.work_dir / 'artifacts'
            last_ckpt_fname = get_last_ckpt_name(ckpt_dir)    
            
            ckpt_path: Path = ckpt_dir / last_ckpt_fname if not exp_config.from_pretrained else exp_config.from_pretrained
            kwargs = {'ckpt_path':ckpt_path} if ckpt_path.is_file() else {}
            net_module = DiffusionTrainingModule(optim_cfg, rwkv_cfg, diff_cfg, skip_init=bool(kwargs))
            logger.info(f"Summary:\n{summary(net_module.model, [(exp_config.batch_size, rwkv_cfg.context_length), (exp_config.batch_size, rwkv_cfg.context_length)], dtypes=[t.int64, t.int64])}")
            logger.info(f'Running on: {model_runner.accelerator}; Skipping initialization?: {bool(kwargs)}')
            
            if rank_zero_only.rank == 0:
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
                log_config(mlflow.experiment,
                           exp_config.mlflow_run_id,
                           diff_cfg)
                        
            model_runner.fit(net_module, datamodule=data_module, **kwargs)
            
        finally:
            r.cancel()