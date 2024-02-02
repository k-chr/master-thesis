from dataclasses import dataclass

from diffccoder.configs.base import BaseConfig
from diffccoder.configs.enums import BetaSchedule, DiffusionModelType, InferenceSamplerType


@dataclass
class DiffusionConfig(BaseConfig):
    beta_sampler: BetaSchedule = BetaSchedule.COSINE
    objective: DiffusionModelType = DiffusionModelType.NOISE
    rescale_timesteps: bool = False
    self_condition: bool = False
    timesteps: int = 3000
    gen_timesteps: int = 3000
    skip_timesteps: int = 1000
    ddim_eta: float = 0
    tgt_len: int = 1024
    time_att: bool = True
    time_dropout: float = 0.2
    time_channels: int = 512
    offset_noise_strength: float = 0.1  # offset noise strength - in blogpost, they claimed 0.1 was ideal
    min_snr_loss_weight: bool = False # snr - signal noise ratio
    min_snr_gamma: float = 5 # https://arxiv.org/abs/2303.09556
    end_point_scale: float = 2.0
    scale_embedding: bool = False
    inference_sampler: InferenceSamplerType = InferenceSamplerType.DEFAULT