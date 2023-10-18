from datetime import timedelta
from typing import Optional
from lightning import Trainer
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from mlflow import MlflowClient
from torch import Tensor


class MLFlowModelCheckpoint(ModelCheckpoint):
    
    def __init__(self,
                 mlflow_logger: MLFlowLogger,
                 dirpath: _PATH | None = None,
                 filename: str | None = None,
                 monitor: str | None = None,
                 verbose: bool = False,
                 save_last: bool | None = None,
                 save_top_k: int = 1,
                 save_weights_only: bool = False,
                 mode: str = "min",
                 auto_insert_metric_name: bool = True,
                 every_n_train_steps: int | None = None,
                 train_time_interval: timedelta | None = None,
                 every_n_epochs: int | None = None,
                 save_on_train_epoch_end: bool | None = None):
        super().__init__(dirpath, filename, monitor, verbose, save_last, save_top_k, save_weights_only, mode, auto_insert_metric_name, every_n_train_steps, train_time_interval, every_n_epochs, save_on_train_epoch_end)
        self.mlflow_logger = mlflow_logger

    @property
    def experiment(self) -> MlflowClient:
        return self.mlflow_logger.experiment    
        
    def _save_last_checkpoint(self, trainer: Trainer, monitor_candidates: dict[str, Tensor]) -> None:
        super()._save_last_checkpoint(trainer, monitor_candidates)

        if self.save_last: self.experiment.log_artifact(self.mlflow_logger.run_id, self.last_model_path)
    
    def _update_best_and_save(self,
                              current: Tensor,
                              trainer: Trainer,
                              monitor_candidates: dict[str, Tensor]) -> None:
        super()._update_best_and_save(current, trainer, monitor_candidates)
        self.experiment.log_artifact(self.mlflow_logger.run_id, self.best_model_path)
        
