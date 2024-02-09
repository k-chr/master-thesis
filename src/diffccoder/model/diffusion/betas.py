import math

import torch as t

from diffccoder.configs.enums import BetaSchedule

"""Based on original AR-Diffusion implementation: https://github.com/wutong4012/AR-Diffusion
    & module: https://github.com/lucidrains/denoising-diffusion-pytorch
"""

def get_named_beta_schedule(schedule_type: BetaSchedule, num_diffusion_timesteps: int) -> t.Tensor:
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    
    match schedule_type:
        case BetaSchedule.LINEAR:
            return _linear_schedule(num_diffusion_timesteps)
        case BetaSchedule.COSINE:
            return _cosine_schedule(num_diffusion_timesteps)
        case BetaSchedule.SIGMOID:
            return _sigmoid_schedule(num_diffusion_timesteps)
        case BetaSchedule.SQRT:
            return _sqrt_schedule(num_diffusion_timesteps)
        case BetaSchedule.PW_LINEAR:
            return _pw_lin_schedule(num_diffusion_timesteps)
        case _:
            raise NotImplementedError(f"unknown beta schedule: {schedule_type}")

def _get_betas_from_alphas_cumprod(alphas_cumprod: t.Tensor):
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    
    return t.clip(betas, 0, 0.999)

def _sqrt_schedule(num_diffusion_timesteps: int, shift: float =1e-4):
    """
        \bar{α}_t = 1 - \sqrt{t/T + s} (sqrt schedule)
        (s is a small constant that corresponds to the starting noise level)
        β_t = 1 - α_t = 1 - \bar{α}_t / \bar{α}_{t-1}
        x_t = \sqrt{1-β_t} * x_{t-1}
        """
    steps = num_diffusion_timesteps + 1
    
    timesteps = t.linspace(0,
                           num_diffusion_timesteps,
                           steps,
                           dtype = t.float64) / num_diffusion_timesteps
    
    alphas_cumprod = t.sqrt(timesteps + shift)
    
    return _get_betas_from_alphas_cumprod(alphas_cumprod) 

def _sigmoid_schedule(num_diffusion_timesteps: int,
                          start: int =-3,
                          end: int =3,
                          tau: float =1.0,
                          clamp_min: float =1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = num_diffusion_timesteps + 1
    timesteps: t.Tensor = t.linspace(0, num_diffusion_timesteps, steps, dtype = t.float64) / num_diffusion_timesteps
    
    v_start = t.tensor(start / tau).sigmoid()
    v_end = t.tensor(end / tau).sigmoid()
    
    alphas_cumprod = (-((timesteps * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return _get_betas_from_alphas_cumprod(alphas_cumprod)

def _cosine_schedule(num_diffusion_timesteps, shift: float = 0.008):
    steps = num_diffusion_timesteps + 1
    
    timesteps = t.linspace(0,
                           num_diffusion_timesteps,
                           steps,
                           dtype = t.float64) / num_diffusion_timesteps
    
    alphas_cumprod = t.cos((timesteps + shift) / (1 + shift) * math.pi * 0.5) ** 2
    return _get_betas_from_alphas_cumprod(alphas_cumprod)

def _linear_schedule(num_diffusion_timesteps: int):
    # Linear schedule from Ho et al, extended to work for any number of
    # diffusion steps.
    
    scale = 1000 / num_diffusion_timesteps
    
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    
    return t.linspace(beta_start,
                      beta_end,
                      num_diffusion_timesteps,
                      dtype=t.float64)

def _pw_lin_schedule(num_diffusion_timesteps: int):
    scale = 1000 / num_diffusion_timesteps

    beta_start = scale * 0.0001 + 0.01
    beta_mid = scale * 0.0001
    beta_end = scale * 0.02

    first_part = t.linspace(beta_start, beta_mid, 10, dtype=t.float64)
    second_part = t.linspace(beta_mid, 
                             beta_end, 
                             num_diffusion_timesteps - 10,
                             dtype=t.float64)

    return t.concatenate([first_part, second_part])