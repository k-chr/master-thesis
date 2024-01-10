from enum import Enum, auto
from typing import Final


class LRSchedulerType(Enum):
    COSINE = auto()
    STEP = auto()
    WARMUP = auto()

class WarmUpRoutine(Enum):
    LINEAR = auto()
    LOGARITHMIC = auto()
    EXPONENTIAL = auto()
    SIN = auto()
    
class WarmUpMetric(Enum):
    GLOBAL_STEP = auto()
    TOKENS = auto()   
    
class OptimizerType(Enum):
    ADAM = auto()
    ADAM_8 = auto()
    SGD = auto()
    ADAM_W_8 = auto()
    ADAM_W = auto()
    ADA_GRAD = auto()
    ADA_GRAD_8 = auto()
    
class BetaSchedule(Enum):
    COSINE = auto()
    SIGMOID = auto()
    SQRT = auto()
    LINEAR = auto()
    PW_LINEAR = auto()

__enums__: Final[Enum]= [
    BetaSchedule,
    LRSchedulerType,
    OptimizerType,
    WarmUpMetric,
    WarmUpRoutine
]