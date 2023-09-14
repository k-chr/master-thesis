from dataclasses import dataclass
from typing import Literal

from diffccoder.configs.base import BaseConfig

@dataclass
class OptimizationConfig(BaseConfig):
    optimizer: Literal['adam'] | Literal['sgd'] = 'adam'
    lr_scheduler: Literal['step'] | Literal['cosine'] = 'cosine'
    layerwise_lr: bool = False
    weight_decay: float = 0.9985
    lr_base: float = 0.01
    
    sgd_momentum: float = 0.95
    
    adam_eps: float = 1e-8
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_fused: bool = False
    
    lr_decay_step_size: int = 500
    lr_decay_rate: int = 0.985
    
    cos_t0: int = 500
    cos_eta_min: float | None = None
    cos_t_decay: int | None = 1.15
    cos_lb_ratio: float | None = 0.25
    cos_warm_restarts: bool = True
    
    warmup_tokens: int = 5000

    
    