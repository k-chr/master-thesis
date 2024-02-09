import abc
import math
from typing import Optional

from loguru import logger
import torch as t
from torch import Tensor, nn

from diffccoder.configs.rwkv_config import RWKVConfig
from diffccoder.model.kernels.wkv import wkv_cuda as wkv
from diffccoder.utils.outputs import BlockState, BlockStateList, ChannelMixState, TimeMixState


class RWKVTimeMix(t.jit.ScriptModule):
    def __init__(self, config: RWKVConfig, layer_id: int, cross_att: bool = False):
        super().__init__()
        self.cross_att = cross_att
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
    def rkv(self, x: t.Tensor, state: Optional[t.Tensor]) -> tuple[t.Tensor, t.Tensor, t.Tensor, Optional[t.Tensor]]:
        # Mix x with the previous timestep to produce xk, xv, xr
        xk, xv, xr, state = self.time_mix(x, state)

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        
        sr = t.sigmoid(r)

        return sr, k, v, state

    def time_mix(self, x: t.Tensor, state: Optional[t.Tensor] = None) -> tuple[t.Tensor, t.Tensor, t.Tensor, Optional[t.Tensor]]:
        
        if x.size(1) == 1 and state is not None:
            xx = state
        else:
            xx: t.Tensor = self.time_shift(x)
            if state is not None:
                xx[:, 0] = state
    
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        
        return xk, xv, xr, x[:, -1] if state is not None else None
    
    def forward(self, x: t.Tensor, state: Optional[TimeMixState]) -> tuple[t.Tensor, Optional[TimeMixState]]:
        B, T, C = x.size() # x = (Batch,Time,Channel)
        
        sr, k, v, shift_state = self.rkv(x, state.shift_state if state is not None else None)
        
        
        att, wkv_state = wkv(B, T, C, self.time_decay, self.time_first, k, v, state.wkv_state if state is not None else None, self.cross_att)
        
        rwkv = sr * att
        rwkv = self.output(rwkv)
        
        return rwkv, TimeMixState(shift_state, wkv_state) if wkv_state is not None else None


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

    def forward(self, x: t.Tensor, state: Optional[ChannelMixState] = None) -> tuple[t.Tensor, Optional[ChannelMixState]]:
        
        if x.size(1) == 1 and state:
            xx = state.shift_state
        else:
            xx: t.Tensor = self.time_shift(x)
            if state:
                xx[:, 0] = state.shift_state

        xk = x * self.channel_mix_k + xx * (1 - self.channel_mix_k)
        xr = x * self.channel_mix_r + xx * (1 - self.channel_mix_r)

        k = self.key(xk)
        k = t.square(t.relu(k))
        kv = self.value(k)
            
        rkv = t.sigmoid(self.receptance(xr)) * kv
        
        return rkv, ChannelMixState(x[:, -1]) if state is not None else None


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
    def _attention(self, x: t.Tensor, state: Optional[list[t.Tensor]] = None) -> tuple[t.Tensor, Optional[list[t.Tensor]]]: ...
    
    def forward(self, x: t.Tensor, state: Optional[BlockState] = None) -> tuple[t.Tensor, Optional[BlockState]]:
        
        if self.layer_id == 0:
            x = self.ln0(x)        
        att, attention_state = self._attention(x, state)
        x = x + att
        ffn_out, channel_mix_state = self.ffn(self.ln2(x), state.channel_mix_state if state is not None else None)
        x = x + ffn_out
        return x, BlockState(attention_state, channel_mix_state) if state is not None else None


class RWKVBlock(RWKVBlockBase):
    def __init__(self, config: RWKVConfig, layer_id: int):

        super().__init__(config=config, layer_id=layer_id)
        self.att = RWKVTimeMix(config, layer_id)

    def _attention(self, x: t.Tensor, state: Optional[BlockState] = None) -> tuple[t.Tensor, Optional[TimeMixState]]:
        return self.att(self.ln1(x), state.time_mix_state if state is not None else None)


class RWKVFfnPreBlock(RWKVBlockBase):
    def __init__(self, config: RWKVConfig, layer_id: int) -> None:
        super().__init__(config, layer_id)
        
        self.ffn_pre = RWKVChannelMix(config=config, layer_id=0)
        
    def _attention(self, x: t.Tensor, state: Optional[BlockState] = None) -> tuple[t.Tensor, Optional[ChannelMixState]]:
        return self.ffn_pre(self.ln1(x), state.channel_mix_state if state else None)
    
    def forward(self, x: Tensor, state: Optional[BlockState] = None) -> tuple[Tensor, Optional[BlockState]]:
        if self.layer_id == 0:
            x = self.ln0(x)        
        att, channel_mix_state = self._attention(x, state)
        x = x + att
        ffn_out, channel_mix_state = self.ffn(self.ln2(x), channel_mix_state)
        x = x + ffn_out
        return x, BlockState(state.time_mix_state, channel_mix_state) if state is not None else None
    

class RWKVSequential(nn.Sequential):
    def forward(self, x: t.Tensor, state: Optional[BlockStateList]) -> tuple[t.Tensor, Optional[BlockStateList]]:
        new_state = None
        if state is None:
            for module in self:
                x, _ = module(x, None)
        else:
            new_state = BlockStateList.empty_like(state)
            for module in self:
                layer_id = getattr(module,'layer_id')
                x, layer_state = module(x, state[layer_id])
                new_state[layer_id] = layer_state
        return x, new_state