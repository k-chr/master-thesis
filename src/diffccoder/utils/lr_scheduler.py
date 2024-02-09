from copy import deepcopy
import math
from typing import Any, Callable
import warnings

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


class EhnancedCosineSchedulerLR(LRScheduler):
    
    eta_min_functor_name: str
    _eta_min: Callable[..., float]
    
    def __init__(self,
                 optimizer: Optimizer,
                 t_0: int,
                 eta_min: float | None = None,
                 lr_decay: float | None = None,
                 t_decay: float | None = None,
                 lr_lowest_bound_ratio: float | None = None,
                 warm_restarts: bool = False,
                 last_epoch: int | None = -1,
                 verbose: bool = False) -> None:
        
        t_0 *= int(warm_restarts) + 1
        self.t_0 = t_0
        self.t_decay = t_decay
        self.lr_decay = lr_decay
        self.t_cur = 0
        self.t_i = t_0
        self.lb_ratio = lr_lowest_bound_ratio
        self.eta_min = eta_min or 0
        self.warm_restarts = warm_restarts
        name = ['const', 'ratio'][int(bool(lr_lowest_bound_ratio))]
        self._set_eta_min_functor(name)
        super().__init__(optimizer, last_epoch, verbose)
        self.initial_lrs = deepcopy(self.base_lrs)
        self.base_lrs = [group['initial_lr'] * group.get('my_lr_scale', 1) for group in optimizer.param_groups]
    
    def _set_eta_min_functor(self, name: str) -> None:
        if name != 'const':
            assert self.lb_ratio
            self._eta_min = lambda lr: lr * self.lb_ratio
            self.eta_min = 0
        else:
            self._eta_min = lambda _: self.eta_min
        
        self.eta_min_functor_name = name
        
    def step(self, epoch: int | None = None) -> None:
        self.t_cur = self.t_cur + 1
        self._update_lr_base()
        
        if self.warm_restarts:
            self._update_time(epoch)
        
        return super().step(epoch)

    def _update_lr_base(self):
        if (not self.warm_restarts and (self.last_epoch - 1 - self.t_0) % (2 * self.t_0) == 0) \
           or (self.warm_restarts and self.t_cur >= self.t_i):
            self.base_lrs = [base_lr * self.lr_decay for base_lr in self.base_lrs]

    def _update_time(self, epoch: int | None):
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            
            if self.t_cur >= self.t_i:
                self.t_cur = self.t_cur - self.t_i
                self.t_i = self.t_i * self.t_decay
        else:
            if epoch < 0:
                raise ValueError(f'Expected non-negative epoch, but got {epoch}')
            if epoch >= self.t_0:
                if self.t_decay == 1:
                    self.t_cur = epoch % self.t_0
                else:
                    n = int(math.log((epoch / self.t_0 * (self.t_decay - 1) + 1), self.t_decay))
                    self.t_cur = epoch - self.t_0 * (self.t_decay ** n - 1) / (self.t_decay - 1)
                    self.t_i = self.t_0 * self.t_decay ** (n)
            else:
                self.t_i = self.t_0
                self.t_cur = epoch
        
    
    def get_lr(self) -> float:
        if not getattr(self, '_get_lr_called_within_step', None):
            warnings.warn('To get the last learning rate computed by the scheduler, '
                          'please use `get_last_lr()`.', UserWarning)
            
        return [self._eta_min(base_lr) + (base_lr - self._eta_min(base_lr)) * (1 + math.cos(math.pi * self.t_cur / self.t_i)) / 2
                for base_lr in self.base_lrs]
    
    def state_dict(self) -> dict[str, Any]:
        excl_keys = {'_eta_min', 'optimizer'}
        incl_keys = set(self.__dict__) - excl_keys
        state = {k: self.__dict__[k] for k in incl_keys}
        
        return state
    
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        return self._set_eta_min_functor(self.eta_min_functor_name)
    