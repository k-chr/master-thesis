from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pytorch_lightning.trainer.connectors.accelerator_connector import _PRECISION_INPUT

from diffccoder.configs.base import BaseConfig


@dataclass
class TrainerConfig(BaseConfig):
    default_root_dir: Path | None = None
    accelerator: str  = "auto"
    devices: list[int] | str | int = "auto"
    num_nodes: int = 1
    precision: _PRECISION_INPUT = "32-true"
    val_check_interval: int | float | None = None
    check_val_every_n_epoch: int | None = 1
    log_every_n_steps: int | None = None
    enable_checkpointing: bool | None = None
    enable_progress_bar: bool | None = None
    enable_model_summary: bool | None = None
    accumulate_grad_batches: int = 1
    gradient_clip_val: int | float | None = None
    gradient_clip_algorithm: str | None = None
    deterministic: bool | Literal['warn'] | None = None
    inference_mode: bool = True
    max_steps: int = -1
    
    
@dataclass    
class DebugTrainerConfig(BaseConfig):
    fast_dev_run: int | bool = False
    detect_anomaly: bool = False
    limit_train_batches: int | float | None = None
    limit_val_batches: int | float | None = None
    limit_test_batches: int | float | None = None
    limit_predict_batches: int | float | None = None
    num_sanity_val_steps: int | None = None
    