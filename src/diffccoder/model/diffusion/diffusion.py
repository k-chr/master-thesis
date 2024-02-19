from functools import partial
from random import random
from typing import Optional

from einops import rearrange, reduce
import torch as t
from torch import nn
from torch.cuda.amp import autocast
import torch.nn.functional as F

from diffccoder.configs.diffusion_config import DiffusionConfig
from diffccoder.configs.enums import DiffusionModelType
from diffccoder.model.diffusion.DIFFRWKV import DIFF_RWKV
from diffccoder.model.diffusion.betas import get_named_beta_schedule
from diffccoder.model.diffusion.buffers import AlphaBuffers, PosteriorBuffers, register_buffer_to_type
from diffccoder.model.diffusion.sampler import XYUniformSampler
from diffccoder.model.rwkv.RWKVCM import RWKV
from diffccoder.utils.l2wrap import L2Wrap
from diffccoder.utils.outputs import BlockStateList, DiffusionLosses, DiffusionPrediction, RWKVOutput

def identity(x: t.Tensor):
    return x


class GaussianDiffusion(nn.Module):
    __betas: t.Tensor
    __loss_weight: t.Tensor
    
    def __init__(self,
                 encoder: RWKV,
                 model: DIFF_RWKV, 
                 *,
                 config: DiffusionConfig,
                 schedule_fn_kwargs = dict()):
        super().__init__()
        
        self.encoder = encoder
        self.model = model
        self.timesteps_sampler = XYUniformSampler(config)
        self.config = config
        
        betas = get_named_beta_schedule(config.beta_sampler, config.timesteps, **schedule_fn_kwargs)
        alphas = 1. - betas
        
        self.alpha_buffers = AlphaBuffers(alphas) 
        self.posterior_buffers = PosteriorBuffers(betas, self.alpha_buffers.cumprod, self.alpha_buffers.cumprod_prev)

        register_buffer_to_type(self, 'betas', betas)

        # derive loss weight
       
        snr: t.Tensor = self.alpha_buffers.cumprod / (1 - self.alpha_buffers.cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if self.config.min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = self.config.min_snr_gamma)

        match self.config.objective:
            case DiffusionModelType.PREVIOUS_X | DiffusionModelType.START_X:
                loss_weight = t.ones_like(snr)
            case DiffusionModelType.NOISE:
                loss_weight = snr
            case _:
                raise RuntimeError()
        
        register_buffer_to_type(self, 'loss_weight', loss_weight)

    @property
    def betas(self) -> t.Tensor: 
        return self.__betas
    
    @property
    def loss_weight(self) -> t.Tensor:
        return self.__loss_weight
    
    @property
    def device(self):
        return self.__betas.device

    def predict_start_from_noise(self, x_t: t.Tensor, timestep: t.Tensor, noise: t.Tensor):
        return (
            extract(self.alpha_buffers.sqrt_recip_cumprod, timestep, x_t.shape) * x_t -
            extract(self.alpha_buffers.sqrt_recip_minus_one_cumprod, timestep, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t: t.Tensor, timestep: t.Tensor, x_start: t.Tensor):
        return (
            (extract(self.alpha_buffers.sqrt_recip_cumprod, timestep, x_t.shape) * x_t - x_start) / \
            extract(self.alpha_buffers.sqrt_recip_minus_one_cumprod, timestep, x_t.shape)
        )
        
    def predict_start_from_prev(self, x_t: t.Tensor, timestep: t.Tensor, x_prev: t.Tensor):
        return (
            extract(1.0 / self.posterior_buffers.mean_coef1, timestep, x_t.shape) * x_prev -
            extract(self.posterior_buffers.mean_coef2 / self.posterior_buffers.mean_coef1, timestep, x_t.shape) * x_t
        )

    def q_posterior(self, x_start: t.Tensor, x_t: t.Tensor, timestep: t.Tensor):
        posterior_mean = (
            extract(self.posterior_buffers.mean_coef1, timestep, x_t.shape) * x_start +
            extract(self.posterior_buffers.mean_coef2, timestep, x_t.shape) * x_t
        )
        
        posterior_variance = extract(self.posterior_buffers.variance, timestep, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_buffers.log_variance_clipped, timestep, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_mean_variance(self, x_start: t.Tensor, timestep: t.Tensor):
        mean = extract(self.alpha_buffers.sqrt_cumprod, timestep, x_shape=x_start.shape) * x_start
        variance = extract(1.0 - self.alpha_buffers.cumprod, timestep, x_shape=x_start.shape)
        log_variance = extract(self.alpha_buffers.log_one_minus_cumprod, timestep, x_shape=x_start.shape)
        
        return mean, variance, log_variance

    def model_predictions(self,
                          x: t.Tensor,
                          timestep: t.Tensor,
                          encoder_hidden_states: t.Tensor,
                          encoder_wkv_states: Optional[BlockStateList] =None,
                          x_self_cond: Optional[t.Tensor] =None,
                          diff_state: Optional[BlockStateList] = None,
                          clip_x_start: bool =False,
                          rederive_pred_noise: bool =False,
                          model_output: Optional[t.Tensor] = None):
        
        if model_output is None:
            model_output, _ = self.model(x, self._scale_timesteps(timestep), encoder_hidden_states, encoder_wkv_states, diff_state, x_self_cond)
            
        maybe_clip = partial(t.clamp, min = -1., max = 1.) if clip_x_start else identity
        x_prev = None
        
        match self.config.objective:
            case DiffusionModelType.PREVIOUS_X:
                x_prev = model_output
                x_start = self.predict_start_from_prev(x, timestep, x_prev)
                x_start = maybe_clip(x_start)
                pred_noise = self.predict_noise_from_start(x, timestep, x_start)
            case DiffusionModelType.START_X:
                x_start = model_output
                x_start = maybe_clip(x_start)
                pred_noise = self.predict_noise_from_start(x, timestep, x_start)
                x_prev = self.q_posterior(x_start, x, timestep)[0]
            case DiffusionModelType.NOISE:
                pred_noise = model_output
                x_start = self.predict_start_from_noise(x, timestep, pred_noise)
                x_start = maybe_clip(x_start)
                x_prev = self.q_posterior(x_start, x, timestep)[0]

                if clip_x_start and rederive_pred_noise:
                    pred_noise = self.predict_noise_from_start(x, timestep, x_start)

        return DiffusionPrediction(pred_noise=pred_noise, pred_x_start=x_start, pred_x_prev=x_prev)

    def p_mean_variance(self,
                        x: t.Tensor,
                        timestep: t.Tensor,
                        encoder_hidden_states: t.Tensor,
                        encoder_state: BlockStateList,
                        x_self_cond: Optional[t.Tensor] = None,
                        clip_denoised: bool = True,
                        diff_state: Optional[BlockStateList] =None):
        
        preds = self.model_predictions(x, 
                                       timestep,
                                       encoder_hidden_states,
                                       encoder_wkv_states=encoder_state,
                                       x_self_cond=x_self_cond,
                                       diff_state=diff_state)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, timestep = timestep)
        model_mean = model_mean if self.config.objective is not DiffusionModelType.PREVIOUS_X else preds.pred_x_prev
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @autocast(enabled = False)
    def q_sample(self, x_start: t.Tensor, timestep: t.Tensor, noise: t.Tensor | None = None):
        if noise is None:
            noise = t.randn_like(x_start)

        return (
            extract(self.alpha_buffers.sqrt_cumprod, timestep, x_start.shape) * x_start +
            extract(self.alpha_buffers.sqrt_one_minus_cumprod, timestep, x_start.shape) * noise
        )

    def x_start(self, B: int, C: int, noise: t.Tensor, x_start_mean: t.Tensor):
        t_0 = t.zeros(B, C, device=x_start_mean.device, dtype=t.int64)
        std = extract(self.alpha_buffers.sqrt_one_minus_cumprod, t_0, x_start_mean.shape)
        print(std.shape, x_start_mean.shape, noise.shape)
        x_start = x_start_mean + std * noise
        return x_start   

    def p_losses(self, 
                 src_indices: t.Tensor, 
                 tgt_indices: t.Tensor, 
                 timesteps: t.Tensor, 
                 noise: Optional[t.Tensor] =None, 
                 offset_noise_strength: float =None):
        
        if noise is None:
            noise = t.randn_like(x_start)
        
        x_start_mean = self.model.get_embeds(tgt_indices)
            
        x_start = self.x_start(tgt_indices.shape[0], x_start_mean.shape[1], noise, x_start_mean)
        
        # offset noise - https://www.crosslabs.org/ blog/diffusion-with-offset-noise
        
        if offset_noise_strength is None:
            offset_noise_strength = self.config.offset_noise_strength

        if offset_noise_strength > 0.:
            offset_noise = t.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 ')

        # noise sample

        x = self.q_sample(x_start=x_start, timestep=timesteps, noise=noise)
        e_cfg = self.encoder.config
        
        encoder_output: RWKVOutput = self.encoder(src_indices, 
                                                  BlockStateList.create(e_cfg.num_hidden_layers,
                                                                        src_indices.shape[0],
                                                                        e_cfg.embedding_size,
                                                                        src_indices.device,
                                                                        t.float32))
        encoder_wkv_state = encoder_output.state
        encoder_hidden_states = encoder_output.last_hidden_state
        
        x_self_cond = None
        if self.config.self_condition and random() < 0.5:
            with t.inference_mode():
                preds: DiffusionPrediction =self.model_predictions(x, timesteps, encoder_hidden_states, encoder_wkv_state)
                x_self_cond = preds.pred_x_start
                x_self_cond.detach_()

        # predict

        preds: DiffusionPrediction =self.model_predictions(x, timesteps, encoder_hidden_states, encoder_wkv_state, x_self_cond)
        
        match self.config.objective:
            case DiffusionModelType.NOISE:            
                target = noise
                pred = preds.pred_noise
            case DiffusionModelType.START_X:
                target = x_start
                pred = preds.pred_x_start
            case DiffusionModelType.PREVIOUS_X:         
                target = self.q_posterior(x_start=x_start, x_t=x, timestep=timesteps)[0]
                pred = preds.pred_x_prev

        mse_loss = F.mse_loss(pred, target, reduction = 'none')
        mse_loss = reduce(mse_loss, 'b s ... -> b s', 'mean')

        t0_mask = (timesteps == 0)
        t0_loss = ((x_start_mean - preds.pred_x_start) ** 2)
        t0_loss = reduce(t0_loss, 'b s ... -> b s', 'mean')
        
        mse_pre = reduce(mse_loss, 'b s ... -> b s', 'mean')

        mse_loss = reduce(t.where(t0_mask, t0_loss, mse_loss), 'b s ... -> b s', 'mean')
        
        x_output = preds.pred_x_start if self.config.objective is DiffusionModelType.START_X else x_start
        
        out_mean, _, _ = self.q_mean_variance(x_output, t.full((tgt_indices.shape[0], x_start_mean.shape[1]), fill_value=self.config.timesteps - 1, device=x_start_mean.device, dtype=t.int64))
        tT_loss = (out_mean ** 2)
        tT_loss = reduce(tT_loss, 'b s ... -> b s', 'mean')
        
        logits = self.model.get_logits(x_output)
        shift_x_hat = logits.contiguous()
        shift_y = tgt_indices.contiguous()
        
        decoder_nll = F.cross_entropy(shift_x_hat.view(-1, shift_x_hat.size(-1)),
                                      shift_y.view(-1))
        
        loss = mse_loss + L2Wrap.apply(decoder_nll, tgt_indices.to(decoder_nll.dtype)) + tT_loss
        
        loss = reduce(loss, 'b s ... -> b', 'mean')
        return DiffusionLosses(loss=loss,
                               mse_loss=mse_loss,
                               mse_pre=mse_pre,
                               t0_loss=t0_loss,
                               tT_loss=tT_loss,
                               decoder_nll=decoder_nll)

    def _scale_timesteps(self, timesteps: t.Tensor):
        return timesteps if not self.config.rescale_timesteps else timesteps.float() * (1e3 / self.config.timesteps)
    
    def forward(self, src_indices: t.Tensor, tgt_indices: t.Tensor):
        b = src_indices.shape[0]
        timesteps, _ = self.timesteps_sampler.sample(b, src_indices.device, src_indices.shape[1])

        losses = self.p_losses(src_indices, tgt_indices, timesteps)
        
        return losses

def extract(x_0: t.Tensor, timesteps: t.Tensor, x_shape: tuple[int, ...] | t.Size):
    b, *_ = timesteps.shape
    out = x_0.expand((b, x_0.shape[-1])).gather(-1, timesteps)

    if len(x_shape) > 2:
        out = out.reshape(*out.shape, *([1] * (len(x_shape) - 2)))
    return out