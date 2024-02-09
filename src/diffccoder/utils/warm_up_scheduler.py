from copy import deepcopy
import math
from typing import Any, Callable, Final
import warnings

from torch.optim.lr_scheduler import LRScheduler

from diffccoder.configs.enums import WarmUpMetric, WarmUpRoutine


LIMIT_K: Final[int] = 10
EPS: Final[float] = 1e-8


class WarmUpScheduler(LRScheduler):
    _warm_up_routine: Callable[[int, float], float] = lambda *_:0.0
    
    def __init__(self, 
                 scheduler_to_wrap: LRScheduler,
                 get_metric: Callable[..., int | float],
                 warm_up_steps: int = 3e3,
                 warm_up_metric: WarmUpMetric = WarmUpMetric.GLOBAL_STEP,
                 warm_up_routine: WarmUpRoutine = WarmUpRoutine.LINEAR,
                 lr_0: float = 0.0,
                 k: float = math.e,
                 last_epoch: int = -1,
                 verbose: bool = False) -> None:
        
        self.wrapped_scheduler = scheduler_to_wrap
        self.warm_up_metric_max = warm_up_steps
        self.warm_up_metric = warm_up_metric
        self.lr_0 = lr_0
        self.curr_metric_value = 0
        self.get_metric = get_metric
        assert k - LIMIT_K < EPS 
        
        self.k = k
        self.warm_up_routine = warm_up_routine
        super().__init__(scheduler_to_wrap.optimizer, last_epoch, verbose)
        self.initial_lrs = deepcopy(self.base_lrs)
        self.base_lrs = [group['initial_lr'] * group.get('my_lr_scale', 1) for group in self.optimizer.param_groups]
        self.setup_warm_up()
        
    def __linear(self) -> Callable[[int, float], float]:
        a = {lr_base : (lr_base - self.lr_0) / self.warm_up_metric_max for lr_base in self.base_lrs}
        return lambda step, lr_base: step * a[lr_base] + self.lr_0
        
    def __sin(self) -> Callable[[int, float], float]:
        return lambda step, lr_base: self.lr_0 + (lr_base - self.lr_0) * (1 - math.cos( math.pi * ((step or 0) / self.warm_up_metric_max))) / 2
    
    def __log(self) -> Callable[[int, float], float]:
        a = {lr_base: (lr_base - self.lr_0) / math.log10(self.warm_up_metric_max + 1) for lr_base in self.base_lrs}
        return lambda step, lr_base: a[lr_base] * math.log10((step or 0) + 1) + self.lr_0
    
    def __exp(self) -> Callable[[int, float], float]:
        return lambda step, lr_base: ((lr_base - self.lr_0) * (math.exp(self.k * (step or 0) / self.warm_up_metric_max) - 1) / (math.exp(self.k) - 1)) + self.lr_0
    
    def setup_warm_up(self):
        match self.warm_up_routine:
            case WarmUpRoutine.LINEAR:
                self._warm_up_routine = self.__linear()            
            case WarmUpRoutine.EXPONENTIAL:
                self._warm_up_routine = self.__exp()
            case WarmUpRoutine.LOGARITHMIC:
                self._warm_up_routine = self.__log()
            case WarmUpRoutine.SIN:
                self._warm_up_routine = self.__sin()
    
    def step(self, epoch: int | None = None) -> None:
        metric = self.__check_if_metric_exists(self.get_metric(), None)
        self.curr_metric_value = metric
        if metric and self.curr_metric_value >= self.warm_up_metric_max:
            self.wrapped_scheduler.step(epoch)
        
        return super().step(epoch)
    
    def get_lr(self) -> float:
        if not getattr(self, '_get_lr_called_within_step', None):
            warnings.warn('To get the last learning rate computed by the scheduler, '
                          'please use `get_last_lr()`.', UserWarning)

        if self.curr_metric_value is not None and self.curr_metric_value >= self.warm_up_metric_max:
            return self.wrapped_scheduler.get_last_lr()
        
        return [self._warm_up_routine(self.curr_metric_value, base_lr) for base_lr in self.base_lrs]

    def __check_if_metric_exists(self, global_steps: int | None, tokens: int | None) -> int:
        match self.warm_up_metric:
            case WarmUpMetric.GLOBAL_STEP:
                return global_steps
            case WarmUpMetric.TOKENS:
                return tokens
            case _:
                assert False
    
    def state_dict(self) -> dict[str, Any]:
        excl_keys = {'_warm_up_routine', 'optimizer', 'wrapped_scheduler', 'get_metric'}
        incl_keys = set(self.__dict__) - excl_keys
        state = {k: self.__dict__[k] for k in incl_keys}
        state['wrapped_scheduler'] = self.wrapped_scheduler.state_dict()
        return state
    
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        wrapped_scheduler_config = state_dict.pop('wrapped_scheduler')
        self.wrapped_scheduler.load_state_dict(wrapped_scheduler_config)
        super().load_state_dict(state_dict)
        return self.setup_warm_up()
    