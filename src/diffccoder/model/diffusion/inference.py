from abc import ABC, abstractmethod
import math
from typing import Callable
from loguru import logger

import torch as t
from torch import nn
from tqdm import tqdm

from diffccoder.configs.diffusion_config import DiffusionConfig
from diffccoder.configs.enums import InferenceSamplerType
from diffccoder.model.diffusion.diffusion import GaussianDiffusion, extract
from diffccoder.utils.outputs import BlockStateList


def denoised_fn_round(emb_model: nn.Embedding, text_emb: t.Tensor):
    down_proj_emb = emb_model.weight  # (vocab_size, embed_dim)

    old_shape = text_emb.shape
    old_device = text_emb.device

    def get_efficient_knn(down_proj_emb: t.Tensor, text_emb: t.Tensor, dist='l2'):
        if dist == 'l2':
            emb_norm = (down_proj_emb ** 2).sum(-1).view(-1, 1)  # (vocab, 1)
            text_emb_t = t.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # (emb_dim, bs*seqlen)
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # (bs*seqlen, 1)
            # down_proj_emb: (vocab, emb_dim), text_emb_t:(emb_dim, bs*seqlen)
            # a+b automatically broadcasts to the same dimension i.e. (vocab, bs*seqlen)
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * t.mm(down_proj_emb, text_emb_t) 
            dist = t.clamp(dist, 0.0, t.inf)  # Limit the value of input to [min, max].
        # Select the smallest distance in the vocab dimension, 
        # that is, select bs*seq_len most likely words from all vocabs.
        topk_out = t.topk(-dist, k=1, dim=0)

        return topk_out.values, topk_out.indices  # logits, token_id (1, bs*seq_len)

    dist = 'l2'
    if len(text_emb.shape) > 2:
        text_emb = text_emb.reshape(-1, text_emb.size(-1))
    else:
        text_emb = text_emb

    _, indices = get_efficient_knn(down_proj_emb,
                                     text_emb.to(down_proj_emb.device), dist=dist)
    rounded_tokens = indices[0]  # (bs*seq_len,)
    new_embeds = emb_model(rounded_tokens).view(old_shape).to(old_device)

    return new_embeds


class InferenceSampler(ABC):
    
    def __init__(self, config: DiffusionConfig, diffusion: GaussianDiffusion) -> None:
        super().__init__()
        self.config = config
        self.diffusion = diffusion
    
    @abstractmethod
    def loop(self,
             shape: tuple[int, ...] | t.Size,
             encoder_hidden_state: t.Tensor,
             encoder_layer_state: BlockStateList,
             return_all_timesteps = False,
             denoised_fn: Callable[[t.Tensor], t.Tensor] = None)-> t.Tensor:...
    
    @abstractmethod
    def sample(self, *args, **kwargs):...
    
    def xy_timestep(self,
                    shape: tuple[int, ...] | t.Size,
                    device: t.device, 
                    end_point: t.Tensor, 
                    x_cord_start: t.Tensor,
                    timestep: int) -> t.Tensor:
        timestep: t.Tensor = t.tensor([timestep] * shape[0], device=device)

        middle_point = t.stack([t.clamp((shape[1] - 1) - timestep, 0), t.clamp(timestep - (shape[1] - 1), 0)], dim=-1)
        middle_point = middle_point.unsqueeze(-1)

        timestep = (x_cord_start.float() - end_point[:, 0].float()
                 ).div(middle_point[:, 0].float() - end_point[:, 0].float()
                       ).mul(middle_point[:, 1].float() - end_point[:, 1].float()
                             ).add(end_point[:, 1].float())
        timestep = timestep.round().clamp(0, self.config.timesteps - 1).long()
        return timestep
    
    @staticmethod
    def create(config: DiffusionConfig, diffusion: GaussianDiffusion):
        match config.inference_sampler:
            case InferenceSamplerType.DDIM:
                return DDIM(config, diffusion)
            case InferenceSamplerType.SKIP:
                return PSkipSampler(config, diffusion)
            case InferenceSamplerType.DEFAULT:
                return PSampler(config, diffusion)
            

class DDIM(InferenceSampler):
    
    def __init__(self, config: DiffusionConfig, diffusion: GaussianDiffusion) -> None:
        super().__init__(config, diffusion)
        
    @t.inference_mode()
    def sample(self,  x: t.Tensor,
               timestep: int | t.Tensor,
               next_timestep: int | t.Tensor,
               encoder_hidden_state: t.Tensor,
               encoder_state: BlockStateList,
               x_self_cond: t.Tensor | None = None,
               diff_state: BlockStateList | None = None,
               denoised_fn: Callable[[t.Tensor], t.Tensor] = None):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        eta = self.config.ddim_eta
        b, *_, device = *x.shape, x.device
        batched_times = t.full((b,), timestep, device = device, dtype = t.long) if not isinstance(timestep, t.Tensor) else timestep
        out = self.diffusion.p_mean_variance(x,
                                             batched_times,
                                             encoder_hidden_state=encoder_hidden_state,
                                             encoder_state=encoder_state,
                                             x_self_cond=x_self_cond, 
                                             clip_denoised=self.config.clip_denoised,
                                             diff_state=diff_state,
                                             denoised_fn=denoised_fn)

        _, _, _, x_start, new_state_list = out
        
        eps = self.diffusion.predict_noise_from_start(x, timestep, x_start)
        alpha_bar = extract(self.diffusion.alpha_buffers.cumprod, timestep, x.shape)
        alpha_bar_prev = extract(self.diffusion.alpha_buffers.cumprod, next_timestep, x.shape)
        
        sigma = (eta * t.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar+1e-7) + 1e-7) * t.sqrt(1 - alpha_bar / (alpha_bar_prev + 1e-7)))

        # Equation 12.
        mean_pred = (x_start * t.sqrt(alpha_bar_prev + 1e-7) + t.sqrt(1 - alpha_bar_prev - sigma ** 2 + 1e-7) * eps)
        noise = t.randn_like(x) 
        mask = t.full((b,), int(next_timestep != 0), device=device, dtype=x.dtype) if isinstance(next_timestep, int) else (next_timestep != 0).to(x.dtype).unsqueeze(-1)
        pred_latent = mean_pred + sigma * noise * mask
        return pred_latent, mean_pred, x_start, new_state_list
    
    @t.inference_mode()
    def loop(self,
             shape: tuple[int, ...] | t.Size,
             encoder_hidden_state: t.Tensor,
             encoder_layer_state: BlockStateList,
             return_all_timesteps = False,
             denoised_fn: Callable[[t.Tensor], t.Tensor] = None):
        
        _, device = shape[0], self.diffusion.device
        scale = self.config.end_point_scale
  
        indices = list(range(self.config.timesteps + shape[1]))[::-1]
        skip_timestep = (self.config.timesteps + shape[1]) // self.config.gen_timesteps
        
        indices = indices[::skip_timestep][:self.config.gen_timesteps]
            
        if indices[-1] != 0:
            indices.append(0)
            
        src_indices = indices[:-1]
        tgt_indices = indices[1:]

        scale, batch_size = self.config.end_point_scale, shape[0]

        end_point = t.tensor([[int(scale * shape[1]), self.config.timesteps - 1]], device=device).repeat(batch_size, 1)

        xs = t.arange(shape[1]).unsqueeze(0).repeat(batch_size, 1).to(device)
        end_point = end_point.unsqueeze(-1)

        latent = t.randn(*shape, device = device)
        latent_samples = [latent]

        x_start = None
        fixed_len = self.config.tgt_len
        times_forward = max(math.ceil(shape[1] / fixed_len), 1)
        for src_i, tgt_i in zip(tqdm(src_indices, desc = 'DDIM sampling loop time step'), tgt_indices):
            src_i = t.tensor([src_i] * batch_size, device=device)
            src_t = self.xy_timestep(shape, device, end_point, xs, src_i)

            tgt_i = t.tensor([tgt_i] * batch_size, device=device)
            tgt_t = self.xy_timestep(shape, device, end_point, xs, tgt_i)
            diff_state = BlockStateList.create(self.diffusion.model.rwkv_config.num_hidden_layers,
                                               shape[0],
                                               self.diffusion.model.rwkv_config.embedding_size,
                                               self.diffusion.device,
                                               t.float32)
            for i in range(times_forward):

                self_cond = x_start if self.config.self_condition else None
                new_latent, _, x_start, state  = self.sample(latent[:, i*fixed_len:min(shape[1]-i*fixed_len, fixed_len), :],
                                                             src_t[:, i*fixed_len:min(shape[1]-i*fixed_len, fixed_len)],
                                                             tgt_t[:, i*fixed_len:min(shape[1]-i*fixed_len, fixed_len)],
                                                             encoder_hidden_state=encoder_hidden_state,
                                                             encoder_state=encoder_layer_state,
                                                             x_self_cond=self_cond,
                                                             diff_state=diff_state,
                                                             denoised_fn=denoised_fn)
                diff_state = state
                latent[:, i*fixed_len:min(shape[1]-i*fixed_len, fixed_len), :] = new_latent
            latent_samples.append(latent)

        ret = latent if not return_all_timesteps else t.stack(latent_samples, dim = 1)

        return ret 
    
    
class PSampler(InferenceSampler):
    
    def __init__(self, config: DiffusionConfig, diffusion: GaussianDiffusion) -> None:
        super().__init__(config, diffusion)
    
    @t.inference_mode()   
    def loop(self,
             shape: tuple[int, ...] | t.Size,
             encoder_hidden_state: t.Tensor,
             encoder_layer_state: BlockStateList,
             return_all_timesteps = False,
             denoised_fn: Callable[[t.Tensor], t.Tensor] = None):
        
        _, device = shape[0], self.diffusion.device
        scale = self.config.end_point_scale
        
        indices = list(range(self.config.timesteps + shape[1]))[::-1]
        
        end_point = t.tensor([[int(scale * shape[1]), self.config.timesteps - 1]], device=device).repeat(shape[0], 1)
        xs = t.arange(shape[1], device=device).unsqueeze(0).repeat(shape[0], 1)
        end_point = end_point.unsqueeze(-1)
        
        latent = t.randn(*shape, device = device)
        latent_samples = [latent]

        x_start = None
        
        fixed_len = self.config.tgt_len
        times_forward = max(math.ceil(shape[1] / fixed_len), 1)
        for timestep in tqdm(indices, desc = 'sampling loop time step', total = self.config.timesteps + shape[1]):
            self_cond = x_start if self.config.self_condition else None
            
            timestep = self.xy_timestep(shape, device, end_point, xs, timestep)
            
            diff_state = BlockStateList.create(self.diffusion.model.rwkv_config.num_hidden_layers,
                                               shape[0],
                                               self.diffusion.model.rwkv_config.embedding_size,
                                               self.diffusion.device,
                                               t.float32)
            for i in range(times_forward):
                
                new_latent, _, x_start, state = self.sample(latent[:, i*fixed_len:min(shape[1]-i*fixed_len, fixed_len), :],
                                                            timestep[:, i*fixed_len:min(shape[1]-i*fixed_len, fixed_len)], 
                                                            encoder_hidden_state, 
                                                            encoder_layer_state, 
                                                            self_cond,
                                                            diff_state,
                                                            denoised_fn=denoised_fn)
                
                diff_state = state

                latent[:, i*fixed_len:min(shape[1]-i*fixed_len, fixed_len), :] = new_latent
                          
            latent_samples.append(latent)

        ret = latent if not return_all_timesteps else t.stack(latent_samples, dim = 1)

        return ret
    
    @t.inference_mode()
    def sample(self, x: t.Tensor,
               timestep: int | t.Tensor,
               encoder_hidden_state: t.Tensor,
               encoder_state: BlockStateList,
               x_self_cond: t.Tensor | None = None,
               diff_state: BlockStateList | None = None,
               denoised_fn: Callable[[t.Tensor], t.Tensor] = None):
        b, *_, device = *x.shape, x.device
        
        batched_times = t.full((b,), timestep, device = device, dtype = t.long) if not isinstance(timestep, t.Tensor) else timestep
        
        out = self.diffusion.p_mean_variance(x = x,
                                             timestep = batched_times,
                                             encoder_hidden_state=encoder_hidden_state,
                                             encoder_state=encoder_state,
                                             x_self_cond = x_self_cond,
                                             clip_denoised = self.config.clip_denoised,
                                             diff_state=diff_state,
                                             denoised_fn=denoised_fn)
        
        model_mean, _, model_log_variance, x_start, new_state_list = out
        
        mask = t.full((b,), int(timestep != 0), device=device, dtype=x.dtype) if isinstance(timestep, int) else (timestep != 0).to(x.dtype).unsqueeze(-1)
        noise = t.randn_like(x) 
        pred_latent = model_mean + (0.5 * model_log_variance).exp() * noise * mask
        
        return pred_latent, model_mean, x_start, new_state_list


class PSkipSampler(InferenceSampler):

    def __init__(self, config: DiffusionConfig, diffusion: GaussianDiffusion) -> None:
        super().__init__(config, diffusion)
        
    @t.inference_mode()
    def sample(self, x: t.Tensor,
               timestep: int | t.Tensor,
               next_timestep: int | t.Tensor,
               encoder_hidden_state: t.Tensor,
               encoder_state: BlockStateList,
               x_self_cond: t.Tensor | None = None,
               diff_state: BlockStateList | None = None,
               denoised_fn: Callable[[t.Tensor], t.Tensor] = None):
        
        b, *_, device = *x.shape, self.diffusion.device
        batched_times = t.full((b,), timestep, device = device, dtype = t.long) if not isinstance(timestep, t.Tensor) else timestep

        out = self.diffusion.p_mean_variance(x=x,
                                             timestep = batched_times,
                                             encoder_hidden_state=encoder_hidden_state,
                                             encoder_state=encoder_state,
                                             x_self_cond = x_self_cond,
                                             clip_denoised = self.config.clip_denoised,
                                             diff_state=diff_state,
                                             denoised_fn=denoised_fn)
        
        model_mean, _, model_log_variance, x_start, new_state_list = out

        alpha_nt = extract(self.diffusion.alpha_buffers.cumprod, next_timestep, x.shape)
        alpha_t = extract(self.diffusion.alpha_buffers.cumprod, timestep, x.shape)
        t_div_nt = alpha_t / (alpha_nt + 1e-7)
        one_minus_nt_div_t = (1 - alpha_nt) / (1 - alpha_t + 1e-7)

        model_mean = (t_div_nt + 1e-7).sqrt() * one_minus_nt_div_t * x + (alpha_nt+ 1e-7).sqrt() * (1 - alpha_t / (alpha_nt + + 1e-7)) / (1 - alpha_t + 1e-7) * x_start
        model_variance = (1 - t_div_nt) * one_minus_nt_div_t
        
        model_log_variance = (model_variance + 1e-7).log()
        noise = t.randn_like(x)
        mask = t.full((b,), int(next_timestep != 0), device=device, dtype=x.dtype) if isinstance(next_timestep, int) else (next_timestep != 0).to(x.dtype).unsqueeze(-1)
        
        pred_latent = model_mean + (0.5 * model_log_variance).exp() * noise * mask
        return pred_latent, model_mean, x_start, new_state_list
    
    def loop(self,
             shape: tuple[int, ...] | t.Size,
             encoder_hidden_state: t.Tensor,
             encoder_layer_state: BlockStateList,
             return_all_timesteps = False,
             denoised_fn: Callable[[t.Tensor], t.Tensor] = None):
        
        _, device = shape[0], self.diffusion.device
        scale = self.config.end_point_scale
  
        indices = list(range(self.config.timesteps + shape[1]))[::-1]
        
        skip_timesteps = self.config.skip_timesteps if self.config.skip_timesteps > 0 and self.config.skip_timesteps < len(indices) // 2 else 1
        
        indices = indices[::skip_timesteps]
    
        if indices[-1] != 0:
            indices.append(0)
            
        src_indices = indices[:-1]
        tgt_indices = indices[1:]

        scale, batch_size = self.config.end_point_scale, shape[0]

        end_point = t.tensor([[int(scale * shape[1]), self.config.timesteps - 1]], device=device).repeat(batch_size, 1)

        xs = t.arange(shape[1]).unsqueeze(0).repeat(batch_size, 1).to(device)
        end_point = end_point.unsqueeze(-1)

        latent = t.randn(*shape, device = device)
        latent_samples = [latent]

        x_start = None
        fixed_len = self.config.tgt_len
        times_forward = max(math.ceil(shape[1] / fixed_len), 1)
        for src_i, tgt_i in zip(tqdm(src_indices, desc = 'SKIP sampling loop time step'), tgt_indices):
            src_i = t.tensor([src_i] * batch_size, device=device)
            src_t = self.xy_timestep(shape, device, end_point, xs, src_i)

            tgt_i = t.tensor([tgt_i] * batch_size, device=device)
            tgt_t = self.xy_timestep(shape, device, end_point, xs, tgt_i)
            diff_state = BlockStateList.create(self.diffusion.model.rwkv_config.num_hidden_layers,
                                               shape[0],
                                               self.diffusion.model.rwkv_config.embedding_size,
                                               self.diffusion.device,
                                               t.float32)
            for i in range(times_forward):
                logger.info(f"Latent at {i} before forward and {src_i}, {tgt_i} timesteps, value range: [{latent.min()}, {latent.max()}]")
                diff_state = BlockStateList.create(self.diffusion.model.rwkv_config.num_hidden_layers,
                                               shape[0],
                                               self.diffusion.model.rwkv_config.embedding_size,
                                               self.diffusion.device,
                                               t.float32)
                self_cond = x_start if self.config.self_condition else None
                new_latent, _, x_start, state  = self.sample(latent[:, i*fixed_len:min(shape[1]-i*fixed_len, fixed_len), :],
                                                             src_t[:, i*fixed_len:min(shape[1]-i*fixed_len, fixed_len)],
                                                             tgt_t[:, i*fixed_len:min(shape[1]-i*fixed_len, fixed_len)],
                                                             encoder_hidden_state=encoder_hidden_state,
                                                             encoder_state=encoder_layer_state,
                                                             x_self_cond=self_cond,
                                                             diff_state=diff_state,
                                                             denoised_fn=denoised_fn)
                del diff_state, state
                latent[:, i*fixed_len:min(shape[1]-i*fixed_len, fixed_len), :] = new_latent
                logger.info(f"Latent at {i} forward and {src_i}, {tgt_i} timesteps, value range: [{latent.min()}, {latent.max()}]")
            latent_samples.append(latent)

        ret = latent if not return_all_timesteps else t.stack(latent_samples, dim = 1)

        return ret 