from dataclasses import dataclass

from diffccoder.configs.base import BaseConfig

@dataclass
class DiffusionConfig(BaseConfig):
    schedule_sampler: str = 'cosine'
    mode: str = 's2s'
    rescale_timesteps: bool = False
    skip_sample: bool = False
    infer_self_condition: bool = False
    gen_timesteps: int = 3000
    skip_timestep: int = 1000
    ddim_sample: bool = False
    pred_len: int = -1
    predict_xstart: bool = True