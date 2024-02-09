import re
from time import time
from typing import Any, Literal, Mapping

from lightning.pytorch.loggers.mlflow import MLFlowLogger
from lightning.fabric.utilities.logger import _add_prefix
from lightning.pytorch.utilities.rank_zero import rank_zero_only, rank_zero_warn
from loguru import logger
from mlflow import MlflowClient

from diffccoder.utils.mlflow_utils import get_last_logged_step, get_last_logged_step_for_each_metric


class MLFlowDistinctLogger(MLFlowLogger):
    
    def __init__(self,
                 experiment_name: str = "lightning_logs",
                 run_name: str | None = None,
                 tracking_uri: str | None = ...,
                 tags: dict[str, Any] | None = None,
                 save_dir: str | None = "./mlruns",
                 log_model: bool | Literal['all'] = False,
                 prefix: str = "",
                 artifact_location: str | None = None,
                 run_id: str | None = None):
        super().__init__(experiment_name, run_name, tracking_uri, tags, save_dir, log_model, prefix, artifact_location, run_id)
           
    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        from mlflow.entities import Metric

        assert step is not None
        
        client: MlflowClient = self.experiment
        
        run = client.get_run(self.run_id)
        
        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        last_steps = get_last_logged_step_for_each_metric(run, client, list(metrics.keys()))
        
        metrics_list: list[Metric] = []

        timestamp_ms = int(time() * 1000)
        for k, v in metrics.items():
            if isinstance(v, str):
                logger.warning(f"Discarding metric with string value {k}={v}.")
                continue

            new_k = re.sub("[^a-zA-Z0-9_/. -]+", "", k)
            if k != new_k:
                rank_zero_warn(
                    "MLFlow only allows '_', '/', '.' and ' ' special characters in metric name."
                    f" Replacing {k} with {new_k}.",
                    category=RuntimeWarning,
                )
                k = new_k
                
            if last_steps[k] is None or last_steps[k] < step:
                metrics_list.append(Metric(key=k, value=v, timestamp=timestamp_ms, step=step))
            # else:
            #     logger.debug(f'last_steps[{k}]={last_steps[k]}, step={step}')   
        if metrics_list:
            client.log_batch(run_id=self.run_id, metrics=metrics_list)
            logger.debug(f'Updated: {len(metrics_list)} metric(s)')
        else:
            logger.debug('Nothing to update...')