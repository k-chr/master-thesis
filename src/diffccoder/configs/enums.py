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
    SGD = auto()
    
__enums__: Final[Enum]= [
    LRSchedulerType,
    WarmUpMetric,
    OptimizerType,
    WarmUpRoutine
]