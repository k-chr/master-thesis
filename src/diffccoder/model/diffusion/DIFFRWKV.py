import math
from typing import Optional

import torch as t
from torch import nn

from diffccoder.configs.diffusion_config import DiffusionConfig
from diffccoder.configs.enums import DiffusionModelType
from diffccoder.configs.rwkv_config import RWKVConfig
from diffccoder.model.rwkv.layers import RWKVChannelMix, RWKVTimeMix
from diffccoder.utils.outputs import BlockState, BlockStateList


class DIFF_RWKVSequential(nn.Sequential):
    def forward(self, 
                x: t.Tensor,
                encoder_state: BlockStateList,
                state: Optional[BlockStateList] =None,
                time_emb: Optional[t.Tensor] =None) -> tuple[t.Tensor, Optional[BlockStateList]]:
        
        new_state = None
        if not state:
            for module in self:
                layer_id = getattr(module,'layer_id')
                x, _ = module(x, encoder_state[layer_id], None, time_emb)
        else:
            new_state = BlockStateList.empty_like(state)
            for module in self:
                layer_id = getattr(module,'layer_id')
                x, layer_state = module(x, encoder_state[layer_id], state[layer_id], time_emb)
                new_state[layer_id] = layer_state
        return x, new_state
    

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, theta: int =10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: t.Tensor):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = t.exp(t.arange(half_dim, device=device) * -emb)
        x_new_axis = x[:,:, None] if x.dim() > 1 else x[:,None]
        emb = x_new_axis * emb[None, :]
        emb = t.cat((emb.cos(), emb.sin()), dim=-1)
        return emb


class DIFF_RWKVBlock(nn.Module):
    
    def __init__(self, diff_config: DiffusionConfig, rwkv_config: RWKVConfig, layer_id: int) -> None:
        super().__init__()

        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(rwkv_config.embedding_size)
        self.ln2 = nn.LayerNorm(rwkv_config.embedding_size)
        self.ln3 = nn.LayerNorm(rwkv_config.embedding_size)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(rwkv_config.embedding_size)
            
        self.ffn = RWKVChannelMix(rwkv_config, layer_id)
        self.att = RWKVTimeMix(rwkv_config, layer_id)
        self.cross_att = RWKVTimeMix(rwkv_config, layer_id, cross_att=True)

        if diff_config.time_att:
            self.diff_time_projection = nn.Sequential(nn.Dropout(diff_config.time_dropout),
                                                      nn.Linear(diff_config.time_channels, diff_config.time_channels * 4),
                                                      nn.SiLU(),
                                                      nn.Linear(diff_config.time_channels * 4, rwkv_config.embedding_size))
            
    def forward(self, hidden_states: t.Tensor, encoder_state: BlockState, self_state: Optional[BlockState] =None, time_emb: Optional[t.Tensor] =None):
        if time_emb is not None:
            time_emb = self.diff_time_projection(time_emb)
            
        if self.layer_id == 0:
            hidden_states = self.ln0(hidden_states)
        
        # Self-WKV attention + optional diffusion timestep embeddings
        att, att_state = self.att(self.ln1((hidden_states + time_emb) if time_emb is not None else hidden_states),
                                  self_state.time_mix_state if self_state else None)
        hidden_states = hidden_states + att
        
        # 'Cross'-WKV attention
        att, _ = self.cross_att(self.ln2(hidden_states), encoder_state.time_mix_state)
        hidden_states = hidden_states + att
        
        # FFN RKV
        ffn_out, ffn_state = self.ffn(self.ln3(hidden_states),
                                      self_state.channel_mix_state if self_state else None)
        hidden_states = hidden_states + ffn_out
        
        return hidden_states, BlockState(att_state, ffn_state) if self_state else None
        

class DIFF_RWKV(nn.Module):
    def __init__(self, diff_config: DiffusionConfig, rwkv_config: RWKVConfig):
        super().__init__()
        
        self.diff_config = diff_config
        self.rwkv_config = rwkv_config
        
        self.context_length = rwkv_config.context_length
        self.emb_scale = math.sqrt(rwkv_config.embedding_size) if self.diff_config.scale_embedding else 1.0
        
        self.timestep_embedding = SinusoidalPosEmb(diff_config.time_channels)

        if self.diff_config.objective is DiffusionModelType.START_X:
            self.lm_transform = nn.Sequential(nn.Linear(rwkv_config.embedding_size,
                                                        rwkv_config.embedding_size),
                                              nn.GELU(),
                                              nn.LayerNorm(rwkv_config.embedding_size))
        
        if not diff_config.time_att:
            # time embedding layer
            self.time_trans = nn.Sequential(nn.Linear(diff_config.time_channels,
                                                      diff_config.time_channels * 4),
                                            nn.SiLU(),
                                            nn.Linear(diff_config.time_channels * 4,
                                                      rwkv_config.embedding_size))
            
        self.blocks = DIFF_RWKVSequential(*[DIFF_RWKVBlock(diff_config, rwkv_config, i)
                                    for i in range(rwkv_config.num_hidden_layers)])
            
        self.ln_out = nn.LayerNorm(rwkv_config.embedding_size)
        
        self.dropout = nn.Dropout(diff_config.time_dropout)
        
        self.emb = nn.Embedding(rwkv_config.vocab_size, rwkv_config.embedding_size)
        self.head = nn.Linear(rwkv_config.embedding_size, rwkv_config.vocab_size, bias=False)
        
        with t.inference_mode():
            self.head.weight - self.emb.weight
        
        if diff_config.self_condition:
            self.input_up_proj = nn.Sequential(
                nn.Linear(rwkv_config.embedding_size, rwkv_config.embedding_size * 2),
                nn.Tanh(),
                nn.Linear(rwkv_config.embedding_size * 2, rwkv_config.embedding_size)
            )     
        
    def forward(self,
                x_t: t.Tensor,
                timesteps: t.Tensor,
                encoder_state: BlockStateList,
                state: Optional[BlockStateList] =None,
                x_self_cond: Optional[t.Tensor] =None) -> tuple[t.Tensor, Optional[BlockStateList]]:
        
        time_emb = self.timestep_embedding(timesteps)
        
        if self.diff_config.self_condition:
            if x_self_cond is None:
                x_self_cond = t.zeros_like(x_t)
            x_t = t.cat((x_self_cond, x_t), dim=-1)
            x_t = self.input_up_proj(x_t)
        
        decoder_input = x_t
           
        if not self.diff_config.time_att:
            time_proj = self.time_trans(time_emb)
            decoder_input = decoder_input + time_proj
            time_emb = None
        
        hidden_states = self.dropout(decoder_input)
        
        hidden_states, new_state = self.blocks(hidden_states, encoder_state, state, time_emb=time_emb)   
            
        return self.ln_out(hidden_states), new_state
    
    def get_embeds(self, input_ids: t.Tensor) -> t.Tensor:
        return self.emb(input_ids) * self.emb_scale

    def get_logits(self, hidden_repr: t.Tensor) -> t.Tensor:
        if self.diff_config.objective is DiffusionModelType.START_X:
            hidden_repr = self.lm_transform(hidden_repr)
        return self.head(hidden_repr)