import abc
import math

import torch as t
from torch import nn

from diffccoder.configs.rwkv_config import RWKVConfig
from diffccoder.model.kernels.wkv import wkv_cuda as wkv


class RWKVTimeMix(t.jit.ScriptModule):
    def __init__(self, config: RWKVConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.context_length = config.context_length
        self.embedding_size = config.embedding_size

        attn_size = config.embedding_size

        self.init_positional_weight_decay_vectors(config, layer_id, attn_size)
            
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = nn.Linear(config.embedding_size, attn_size, bias=False)
        self.value = nn.Linear(config.embedding_size, attn_size, bias=False)
        self.receptance = nn.Linear(config.embedding_size, attn_size, bias=False)

        self.output = nn.Linear(attn_size, config.embedding_size, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def init_positional_weight_decay_vectors(self, config: RWKVConfig, layer_id: int, attn_size: int):
        with t.no_grad(): # fancy init
            ratio_0_to_1 = (layer_id / (config.num_hidden_layers - 1)) # 0 to 1
            ratio_1_to_almost0 = (1.0 - (layer_id / config.num_hidden_layers)) # 1 to ~0
            
            # fancy time_decay
            decay_speed = t.ones(attn_size)
            for h in range(attn_size):
                decay_speed[h] = -5 + 8 * (h / (attn_size-1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)

            # fancy time_first
            zigzag = (t.tensor([(i+1)%3 - 1 for i in range(attn_size)]) * 0.5)
            self.time_first = nn.Parameter(t.ones(attn_size) * math.log(0.3) + zigzag)
            
            # fancy time_mix
            x = t.ones(1, 1, config.embedding_size)
            for i in range(config.embedding_size):
                x[0, 0, i] = i / config.embedding_size
            self.time_mix_k = nn.Parameter(t.pow(x, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(t.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(t.pow(x, 0.5 * ratio_1_to_almost0))

    @t.jit.script_method
    def attention(self, x):
        # Mix x with the previous timestep to produce xk, xv, xr
        xk, xv, xr = self.time_mix(x)

        # Use xk, xv, xr to produce k, v, r
        k, v, r = self.rkv(xk, xv, xr)
        sr = t.sigmoid(r)

        return sr, k, v

    def rkv(self, 
            xk: t.Tensor,
            xv: t.Tensor,
            xr: t.Tensor) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        
        return k, v, r

    def time_mix(self, x: t.Tensor) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
        xx: t.Tensor = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        
        return xk, xv, xr

    def forward(self, x: t.Tensor) -> t.Tensor:
        B, T, C = x.size() # x = (Batch,Time,Channel)
        sr, k, v = self.attention(x)

        rwkv = sr * wkv(B, T, C, self.time_decay, self.time_first, k, v)
        rwkv = self.output(rwkv)
        
        return rwkv


class RWKVChannelMix(t.jit.ScriptModule):
    def __init__(self, config: RWKVConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.init_positional_weight_decay_vectors(config, layer_id)

        hidden_size = 4 * config.embedding_size
        self.key = nn.Linear(config.embedding_size, hidden_size, bias=False)
        self.receptance = nn.Linear(config.embedding_size, config.embedding_size, bias=False)
        self.value = nn.Linear(hidden_size, config.embedding_size, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def init_positional_weight_decay_vectors(self, config: RWKVConfig, layer_id: int):
        with t.no_grad(): # fancy init of time_mix
            ratio_1_to_almost0 = (1.0 - (layer_id / config.num_hidden_layers)) # 1 to ~0

            x = t.ones(1, 1, config.embedding_size)
            for i in range(config.embedding_size):
                x[0, 0, i] = i / config.embedding_size

            self.channel_mix_k = nn.Parameter(t.pow(x, ratio_1_to_almost0))
            self.channel_mix_r = nn.Parameter(t.pow(x, ratio_1_to_almost0))

    @t.jit.script_method
    def forward(self, x: t.Tensor) -> t.Tensor:
        
        xx = self.time_shift(x)
        xk = x * self.channel_mix_k + xx * (1 - self.channel_mix_k)
        xr = x * self.channel_mix_r + xx * (1 - self.channel_mix_r)

        k = self.key(xk)
        k = t.square(t.relu(k))
        kv = self.value(k)

        rkv = t.sigmoid(self.receptance(xr)) * kv
        return rkv

class RWKVBlockBase(nn.Module, abc.ABC):
    def __init__(self, config: RWKVConfig, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(config.embedding_size)
        self.ln2 = nn.LayerNorm(config.embedding_size)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(config.embedding_size)
            
        self.ffn = RWKVChannelMix(config, layer_id)

    @abc.abstractmethod
    def _attention(self, x: t.Tensor) -> t.Tensor: ...
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        if self.layer_id == 0:
            x = self.ln0(x)        
            
        x = x + self._attention(x)
        x = x + self.ffn(self.ln2(x))
        return x
    
class RWKVBlock(RWKVBlockBase):
    def __init__(self, config: RWKVConfig, layer_id: int):

        super().__init__(config=config, layer_id=layer_id)
        self.att = RWKVTimeMix(config, layer_id)

    def _attention(self, x: t.Tensor) -> t.Tensor:
        
        return self.att(self.ln1(x))
    
class RWKVFfnPreBlock(RWKVBlockBase):
    def __init__(self, config: RWKVConfig, layer_id: int) -> None:
        super().__init__(config, layer_id)
        
        self.ffn_pre = RWKVChannelMix(config=config, layer_id=0)
        
    def _attention(self, x: t.Tensor) -> t.Tensor:
        return self.ffn_pre(self.ln1(x))