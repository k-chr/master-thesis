from dataclasses import dataclass, field
from pathlib import Path

from diffccoder.configs.base import BaseConfig


@dataclass
class ExperimentConfig(BaseConfig):
    experiment_name: str | None = 'DIFFRWKV'
    
    from_pretrained: Path | None = None
    
    batch_size: int = 128
    val_batch_size: int = 32
    pin_memory: bool = True
    number_of_workers: int = 1
    split_val_ratio: float = 0.2
    
    exp_root: Path = Path.home() / 'exp'
    work_dir: Path | None = None
    data_dir: Path | None = None
    checkpoint_dir: Path | None = None
    out_dir: Path | None = None
    
    mlflow_enabled: bool = True
    mlflow_server: str | None = 'https://united-namely-macaw.ngrok-free.app'
    mlflow_run_id: str | None = None #it will be set during the first run
    mlflow_continue_run: bool = True 
    mlflow_run_name: str | None = None
    mlflow_http_timeout: int = 120
    mlflow_log_to_remote_freq: int = 120
    
    use_tensorboard: bool = False
    tensorboard: Path | None = None
    
    metrics_to_save_cp: list[str] = field(default_factory=lambda:['loss', 'perplexity'])
    metrics_to_log: list[str] = field(default_factory=lambda:['loss', 'perplexity'])
    