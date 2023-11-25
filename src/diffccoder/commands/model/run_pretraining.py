from pathlib import Path

from cleo.commands.command import Command
from cleo.helpers import argument
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from loguru import logger
import mlflow as mlflow_client

from diffccoder.configs.base import dump_config, load_config 
from diffccoder.configs.experiment_config import ExperimentConfig
from diffccoder.configs.optimization_config import OptimizationConfig
from diffccoder.configs.rwkv_config import RWKVConfig
from diffccoder.configs.trainer_config import DebugTrainerConfig, TrainerConfig
from diffccoder.data.npy_data_loader import NPYDataModule
from diffccoder.workflow.model_runner import ModelRunner
from diffccoder.workflow.pretraining.module import PretrainingModule

DEFAULT_RUN_NAME = 'Pre-Training'


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
        dirlist_txt = Path(self.argument('pre-train-dirlist.txt'))
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
        
        _logger = []
        _callbacks = []
        
        last = ModelCheckpoint(dirpath=exp_config.work_dir / 'artifacts',
                               save_top_k=0,
                               save_last=True,
                               every_n_train_steps=trainer_cfg.log_every_n_steps)
            
        _callbacks.append(last)
            
        for metric in exp_config.metrics_to_log:
            monitor = ModelCheckpoint(dirpath=exp_config.work_dir / 'artifacts',
                                      filename=f'best_on_{metric}',
                                      save_top_k=1,
                                      save_on_train_epoch_end=False,
                                      monitor=f'validation_{metric}',
                                      save_last=False)
            _callbacks.append(monitor)

        lr_monitor = LearningRateMonitor(logging_interval='step')
        _callbacks.append(lr_monitor)
        
        if exp_config.mlflow_enabled and exp_config.experiment_name:
            
            if exp_config.mlflow_http_timeout != 120:
                import os
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

        model_runner = ModelRunner(trainer_config=trainer_cfg,
                                debug_config=debug_cfg,
                                logger=_logger,
                                callbacks=_callbacks)
                                
        ckpt_path: Path = exp_config.work_dir / 'artifacts' / 'last.ckpt' if not exp_config.from_pretrained else exp_config.from_pretrained
        kwargs = {'ckpt_path':ckpt_path} if ckpt_path.is_file() else {}
        net_module = PretrainingModule(optim_cfg, rwkv_cfg, skip_init=bool(kwargs))
        net_module.skip_validation_step = bool(kwargs)
        logger.info(f'Running on: {model_runner.accelerator}; Skipping initialization?: {bool(kwargs)}')
        model_runner.fit(net_module, datamodule=data_module, **kwargs)
    