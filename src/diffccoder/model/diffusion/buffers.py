import torch as t
from torch import nn
import torch.nn.functional as F


def register_buffer_to_f32(module: nn.Module, name: str, buffer: t.Tensor):
    module.register_buffer(f"_{module._get_name()}__{name}", buffer.to(t.float32))


class PosteriorBuffers(nn.Module):
    __variance: t.Tensor
    __log_variance_clipped: t.Tensor
    __mean_coef1: t.Tensor
    __mean_coef2: t.Tensor
    
    def __init__(self, betas: t.Tensor, alphas_cumprod: t.Tensor, alphas_cumprod_prev: t.Tensor) -> None:
        super().__init__()

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer_to_f32(self, 'variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer_to_f32(self, 'log_variance_clipped', t.log(posterior_variance.clamp(min =1e-20)))
        register_buffer_to_f32(self, 'mean_coef1', betas * t.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer_to_f32(self, 'mean_coef2', (1. - alphas_cumprod_prev) * t.sqrt(1. - betas) / (1. - alphas_cumprod))
        
    @property
    def variance(self) -> t.Tensor:
        return self.__variance
    
    @property
    def log_variance_clipped(self) -> t.Tensor:
        return self.__log_variance_clipped
    
    @property
    def mean_coef1(self) -> t.Tensor:
        return self.__mean_coef1
    
    @property
    def mean_coef2(self) -> t.Tensor:
        return self.__mean_coef2
    
        
class AlphaBuffers(nn.Module):
    __cumprod: t.Tensor
    __cumprod_prev: t.Tensor
    __sqrt_cumprod: t.Tensor
    __sqrt_one_minus_cumprod: t.Tensor
    __log_one_minus_cumprod: t.Tensor
    __sqrt_recip_cumprod: t.Tensor
    __sqrt_recip_minus_one_cumprod: t.Tensor
    
    
    def __init__(self, alphas: t.Tensor) -> None:
        super().__init__()
        alphas_cumprod = t.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
        
        register_buffer_to_f32(self, 'cumprod', alphas_cumprod)
        register_buffer_to_f32(self, 'cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer_to_f32(self, 'sqrt_cumprod', t.sqrt(alphas_cumprod))
        register_buffer_to_f32(self, 'sqrt_one_minus_cumprod', t.sqrt(1. - alphas_cumprod))
        register_buffer_to_f32(self, 'log_one_minus_cumprod', t.log(1. - alphas_cumprod))
        register_buffer_to_f32(self, 'sqrt_recip_cumprod', t.sqrt(1. / alphas_cumprod))
        register_buffer_to_f32(self, 'sqrt_recip_minus_one_cumprod', t.sqrt(1. / alphas_cumprod - 1))

    @property
    def cumprod(self) -> t.Tensor:
        return self.__cumprod
    
    @property
    def cumprod_prev(self) -> t.Tensor:
        return self.__cumprod_prev
    
    @property
    def sqrt_cumprod(self) -> t.Tensor:
        return self.__sqrt_cumprod
    
    @property
    def sqrt_one_minus_cumprod(self) -> t.Tensor:
        return self.__sqrt_one_minus_cumprod
    
    @property
    def log_one_minus_cumprod(self) -> t.Tensor:
        return self.__log_one_minus_cumprod
    
    @property
    def sqrt_recip_cumprod(self) -> t.Tensor: 
        return self.__sqrt_recip_cumprod
    
    @property
    def sqrt_recip_minus_one_cumprod(self) -> t.Tensor:
        return self.__sqrt_recip_minus_one_cumprod