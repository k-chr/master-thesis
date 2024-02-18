from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Iterator, Set, Optional

import torch
from torch import Tensor
from torch.nn import Module, Parameter


def exists(val):
    return val is not None

def get_module_device(m: Module):
    return next(m.parameters()).device

def inplace_copy(tgt: Tensor, src: Tensor, *, auto_move_device = False):
    if auto_move_device:
        src = src.to(tgt.device)

    tgt.copy_(src)

def inplace_lerp(tgt: Tensor, src: Tensor, weight, *, auto_move_device = False):
    if auto_move_device:
        src = src.to(tgt.device)

    tgt.lerp_(src, weight)

class EMA(Module):
    """
    Implements exponential moving average shadowing for your model.

    Utilizes an inverse decay schedule to manage longer term training runs.
    By adjusting the power, you can control how fast EMA will ramp up to your specified beta.

    @crowsonkb's notes on EMA Warmup:

    If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).

    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 2/3.
        min_value (float): The minimum EMA decay rate. Default: 0.
    """
    initted: Tensor
    step: Tensor
    ema_path: Tensor
    
    def __init__(self,
                 model: Module,
                 ema_model: Optional[Module] = None,           # if your model has lazylinears or other types of non-deepcopyable modules, you can pass in your own ema model
                 beta = 0.9999,
                 update_after_step = 100,
                 update_every = 10,
                 inv_gamma = 1.0,
                 power = 2 / 3,
                 min_value = 0.0,
                 param_or_buffer_names_no_ema: Set[str] = set(),
                 ignore_names: Set[str] = set(),
                 ignore_startswith_names: Set[str] = set(),
                 include_online_model = True,                  # set this to False if you do not wish for the online model to be saved along with the ema model (managed externally)
                 allow_different_devices = False,
                 ema_path: Path= None):              # if the EMA model is on a different device (say CPU), automatically move the tensor
        
        super().__init__()
        self.beta = beta

        self.is_frozen = beta == 1.

        # whether to include the online model within the module tree, so that state_dict also saves it

        self.include_online_model = include_online_model

        if include_online_model:
            self.online_model = model
        else:
            self.online_model = [model] # hack

        # ema model

        self.ema_model = ema_model


        if not exists(self.ema_model):
            try:
                self.ema_model = deepcopy(model)
            except Exception as e:
                print(f'Error: While trying to deepcopy model: {e}')
                print('Your model was not copyable. Please make sure you are not using any LazyLinear')
                exit()

        for p in self.ema_model.parameters():
            if not p.is_leaf:
                p.detach_()
            p.requires_grad_(False)

        # parameter and buffer names

        self.parameter_names = {name for name, param in self.ema_model.named_parameters() if param.dtype in [torch.float, torch.float16, torch.bfloat16]}
        self.buffer_names = {name for name, buffer in self.ema_model.named_buffers() if buffer.dtype in [torch.float, torch.float16, torch.bfloat16]}

        # tensor update functions

        self.inplace_copy = partial(inplace_copy, auto_move_device = allow_different_devices)
        self.inplace_lerp = partial(inplace_lerp, auto_move_device = allow_different_devices)

        # updating hyperparameters

        self.update_every = update_every
        self.update_after_step = update_after_step

        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

        assert isinstance(param_or_buffer_names_no_ema, (set, list))
        self.param_or_buffer_names_no_ema = param_or_buffer_names_no_ema # parameter or buffer

        self.ignore_names = ignore_names
        self.ignore_startswith_names = ignore_startswith_names

        # whether to manage if EMA model is kept on a different device

        self.allow_different_devices = allow_different_devices

        # init and step states

        self.register_buffer('initted', torch.tensor(False))
        self.register_buffer('ema_path', torch.tensor([0], dtype=torch.int16))
        self.register_buffer('step', torch.tensor(0))
        self.set_path(ema_path)

    def load(self):
        if (p := self.get_path()) is not None and p.exists():
            self.load_state_dict(torch.load(p, map_location='cpu'))

    def save(self):
        if (p := self.get_path()) is not None:
            state_dict = self.state_dict()
            torch.save(state_dict, p)

    def set_path(self, path: Path | None = None):
        if path is not None:
            ords = list(map(ord, path.__str__()))
            self.ema_path = torch.tensor(ords, dtype=torch.int16)

    def get_path(self) -> Path | None:
        ords = self.ema_path.tolist()
        if len(ords) > 1:
            return Path(''.join(map(chr, ords)))
        return None

    @property
    def model(self):
        return self.online_model if self.include_online_model else self.online_model[0]

    def eval(self):
        return self.ema_model.eval()
    
    def restore_ema_model_device(self):
        device = self.initted.device
        self.ema_model.to(device)

    def get_params_iter(self, model: Module) -> Iterator[tuple[str, Parameter]]:
        for name, param in model.named_parameters():
            if name not in self.parameter_names:
                continue
            yield name, param

    def get_buffers_iter(self, model: Module) -> Iterator[tuple[str, Tensor]]:
        for name, buffer in model.named_buffers():
            if name not in self.buffer_names:
                continue
            yield name, buffer

    def copy_params_from_model_to_ema(self):
        copy = self.inplace_copy

        for (_, ma_params), (_, current_params) in self.iter_zip_params():
            copy(ma_params.data, current_params.data)

        for (_, ma_buffers), (_, current_buffers) in self.iter_zip_buffers():
            copy(ma_buffers.data, current_buffers.data)

    def iter_zip_buffers(self) -> Iterator[tuple[tuple[str, Tensor], tuple[str, Tensor]]]:
        return zip(self.get_buffers_iter(self.ema_model), self.get_buffers_iter(self.model))

    def iter_zip_params(self) -> Iterator[tuple[tuple[str, Parameter], tuple[str, Parameter]]]:
        return zip(self.get_params_iter(self.ema_model), self.get_params_iter(self.model))

    def copy_params_from_ema_to_model(self):
        copy = self.inplace_copy

        for (_, ma_params), (_, current_params) in self.iter_zip_params():
            copy(current_params.data, ma_params.data)

        for (_, ma_buffers), (_, current_buffers) in self.iter_zip_buffers():
            copy(current_buffers.data, ma_buffers.data)

    def get_current_decay(self):
        epoch = (self.step - self.update_after_step - 1).clamp(min = 0.)
        value = 1 - (1 + epoch / self.inv_gamma) ** - self.power

        if epoch.item() <= 0:
            return 0.

        return value.clamp(min = self.min_value, max = self.beta).item()

    def update(self):
        
        step = self.step.item()
        self.step += 1

        if (step % self.update_every) != 0:
            return

        if step <= self.update_after_step:
            self.copy_params_from_model_to_ema()
            return

        if not self.initted.item():
            self.copy_params_from_model_to_ema()
            self.initted.data.copy_(torch.tensor(True))

        self.update_moving_average(self.ema_model, self.model)

    @torch.no_grad()
    def update_moving_average(self, ma_model: Module, current_model: Module):
        if self.is_frozen:
            return

        copy, lerp = self.inplace_copy, self.inplace_lerp
        current_decay = self.get_current_decay()

        for (name, current_params), (_, ma_params) in zip(self.get_params_iter(current_model), self.get_params_iter(ma_model)):
            if name in self.ignore_names:
                continue

            if any([name.startswith(prefix) for prefix in self.ignore_startswith_names]):
                continue

            if name in self.param_or_buffer_names_no_ema:
                copy(ma_params.data, current_params.data)
                continue

            lerp(ma_params.data, current_params.data, 1. - current_decay)

        for (name, current_buffer), (_, ma_buffer) in zip(self.get_buffers_iter(current_model), self.get_buffers_iter(ma_model)):
            if name in self.ignore_names:
                continue

            if any([name.startswith(prefix) for prefix in self.ignore_startswith_names]):
                continue

            if name in self.param_or_buffer_names_no_ema:
                copy(ma_buffer.data, current_buffer.data)
                continue

            lerp(ma_buffer.data, current_buffer.data, 1. - current_decay)

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)