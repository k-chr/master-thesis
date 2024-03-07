from loguru import logger as logging
import torch
from torch import nn
import numpy as np

from diffccoder.utils.outputs import ChannelMixState, TimeMixState

def process(out: torch.Tensor|TimeMixState|ChannelMixState, nan_mask: torch.Tensor):
    if isinstance(out, torch.Tensor):
        return out[nan_mask.nonzero()[:, 0].unique(sorted=True)]
    if isinstance(out, ChannelMixState):
        return out.shift_state[nan_mask.nonzero()[:, 0].unique(sorted=True)]
    if isinstance(out, TimeMixState):
        tensor: torch.Tensor = np.asarray([out.shift_state, out.wkv_state], dtype=object)[nan_mask.tolist()][0]
        t_nan_mask = tensor.isnan()
        return tensor[t_nan_mask.nonzero()[:, 0].unique(sorted=True)]

def get_max(out: torch.Tensor|TimeMixState|ChannelMixState):
    if isinstance(out, torch.Tensor):
        return out.max()
    if isinstance(out, ChannelMixState):
        return out.shift_state.max()
    if isinstance(out, TimeMixState):
        return [out.shift_state.max(), out.wkv_state.max()]


class DetectNaNModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        def nan_hook(self, inp, output):
            if not isinstance(output, tuple):
                outputs = [output]
            else:
                outputs = output

            current_max = []
            for i, out in enumerate(outputs):
                if not isinstance(out, (torch.Tensor, ChannelMixState, TimeMixState)):
                    continue
                elif isinstance(out, ChannelMixState):
                    nan_mask = out.shift_state.isnan()
                elif isinstance(out, TimeMixState):
                    nan_mask = torch.as_tensor([out.shift_state.isnan().any(), out.wkv_state.isnan().any()], dtype=torch.bool)
                else:
                    nan_mask = torch.isnan(out)
                    
                if nan_mask.any():
                    logging.error(f'Checking {self.__class__.__name__}, failed')
                    logging.info(inp)
                    
                    raise RuntimeError(f"In {self.__class__.__name__}, found NAN in output {i} of type: {out.__class__.__name__} at indices: ",
                                       nan_mask.nonzero(), "where:", process(out, nan_mask))
                else:
                    current_max.append(get_max(out))
            #logging.success(f'Passed: {self.__class__.__name__}, current maximimums: {current_max}')
        for submodule in self.modules():
            submodule.register_forward_hook(nan_hook)