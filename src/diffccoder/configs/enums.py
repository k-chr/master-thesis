from enum import Enum, auto
from typing import Final


class LRSchedulerType(Enum):
    COSINE = auto()
    STEP = auto()
    WARMUP = auto()
    NONE = auto()

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

class InferenceSamplerType(Enum):
    DDIM = auto()
    SKIP = auto()
    DEFAULT = auto()

class LossType(Enum):
    MSE = auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = auto() # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = auto()  # use the variational lower-bound
    RESCALED_KL = auto()  # like KL, but rescale to estimate the full VLB
    E2E_KL = auto()
    E2E_MSE = auto()
    E2E_Simple_MSE = auto()
    E2E_Simple_KL = auto()

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL

class DiffusionModelType(Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = auto()  # the model predicts x_{t-1}
    START_X = auto()  # the model predicts x_0
    NOISE = auto()  # the model predicts epsilon


__enums__: Final[Enum]= [BetaSchedule,
                         DiffusionModelType,
                         InferenceSamplerType,
                         LossType,
                         LRSchedulerType,
                         OptimizerType,
                         WarmUpMetric,
                         WarmUpRoutine]