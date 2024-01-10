from dataclasses import dataclass

from diffccoder.configs.base import BaseConfig
from diffccoder.configs.enums import BetaSchedule

@dataclass
class DiffusionConfig(BaseConfig):
    schedule_sampler: BetaSchedule
    mode: str = 's2s'
    rescale_timesteps: bool = False
    skip_sample: bool = False
    infer_self_condition: bool = False
    gen_timesteps: int = 3000
    skip_timestep: int = 1000
    ddim_sample: bool = False
    pred_len: int = -1
    predict_xstart: bool = True
    tgt_len: int = 1024
    loss_aware: bool = False