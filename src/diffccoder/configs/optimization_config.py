from dataclasses import dataclass
import math
from typing import Literal

from diffccoder.configs.base import BaseConfig
from diffccoder.configs.enums import (LRSchedulerType,
                                      WarmUpMetric,
                                      WarmUpRoutine,
                                      OptimizerType)


@dataclass
class OptimizationConfig(BaseConfig):
    optimizer: OptimizerType = OptimizerType.ADAM
    lr_scheduler: LRSchedulerType = LRSchedulerType.COSINE
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
    
    warmup_max: int = 5000
    warmup_scheduler: LRSchedulerType = LRSchedulerType.COSINE
    warmup_routine: WarmUpRoutine = WarmUpRoutine.SIN
    warmup_metric: WarmUpMetric = WarmUpMetric.GLOBAL_STEP
    warmup_k: float = math.e
    warmup_lr_0: float = 0.0
    