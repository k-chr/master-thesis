from functools import lru_cache
import os
from pathlib import Path
from typing import Optional

import einops
from loguru import logger
import torch as t
from torch.autograd.function import Function
from torch.utils.cpp_extension import load
from diffccoder.utils.generic import get_dtype

from diffccoder.utils.rwkv_kernel_context import RWKVContext


# it's possible to go beyond CUDA limitations if you slice the ctx and pass the hidden state in each slice
_kernels = Path(__file__).resolve().parent

def cast(tensor: t.Tensor, dtype: t.dtype):
    if tensor.dtype is dtype: return tensor
    match dtype:
        case t.bfloat16: return tensor.bfloat16()
        case t.float16: return tensor.half()
        case t.float32: return tensor.float()
        case _: return tensor
        
class WKVKernel:
    _kernel: Function = None
    
    @classmethod
    @property
    def T_MAX(cls):
        # increase this if your ctx_len is long [NOTE: TAKES LOTS OF VRAM!]
        return WKVKernel._get_ctx_len()

    @staticmethod
    @lru_cache
    def _get_ctx_len():
        return int(os.environ.get('CTX_LEN', '1024'))
    
    @classmethod
    @property
    def _dtype(cls):
        return WKVKernel._get_dtype()

    @staticmethod
    @lru_cache
    def _get_dtype():
        str_type = os.environ.get('DTYPE', '32-true')
        return get_dtype(str_type)
    
    @staticmethod
    def state(): 
        return WKVKernel._kernel is not None

    @staticmethod
    def load():
        logger.info(f'Loading WKV cuda-kernel for {WKVKernel._dtype} and ctx len: {WKVKernel.T_MAX}')
        import sys
        
        flags = ['-res-usage',
                 '--use_fast_math',
                 '-O3',
                 '--extra-device-vectorization',
                 f'-DTmax={WKVKernel.T_MAX}']
        
        if sys.platform != 'win32':
            flags += ['--maxrregcount 60',
                      '-Xptxas -O3']
        extra_flags = [] if WKVKernel._dtype is not t.bfloat16 else ['-t 4', '-std=c++17']
        op_fname = 'cuda/wkv_op' + ('' if WKVKernel._dtype is not t.bfloat16 else '_bf16')
        cuda_fname = 'cuda/wkv_cuda' + ('' if WKVKernel._dtype is not t.bfloat16 else '_bf16')
        WKVKernel._kernel = load(name='wkv_cuda', 
                                 sources=[ _kernels / (op_fname + '.cpp'), _kernels /  (cuda_fname + '.cu')],
                                 verbose=True, 
                                 extra_cuda_cflags=extra_flags + flags)
class StateWKV(Function):
    @staticmethod
    def forward(ctx: RWKVContext,
                B: int,
                T: int,
                C: int,
                w: t.Tensor,
                u: t.Tensor,
                k: t.Tensor,
                v: t.Tensor,
                state: Optional[t.Tensor]):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= WKVKernel.T_MAX
        assert B * C % min(C, 32) == 0
        
        w = -t.exp(cast(w, t.float32).contiguous())
        tensor_target_dtype = t.float32 if WKVKernel._dtype != t.bfloat16 else t.bfloat16
        u = cast(u, tensor_target_dtype).contiguous()
        k = cast(k, tensor_target_dtype).contiguous()
        v = cast(v, tensor_target_dtype).contiguous()
        y = t.empty_like(k, dtype=tensor_target_dtype).contiguous()
        
        if state is None:
            state = t.zeros(B, C, 3, dtype=t.float32, device=k.device).contiguous()
            state[:, :, 2] -= 1e38
            
        new_state = t.empty_like(state).contiguous()
                
        getattr(WKVKernel._kernel, 'forward_state')(B, T, C, w, u, k, v, state, y, new_state)
        
        ctx.save_for_backward(w, u, k, v, state)
        
        return cast(y, WKVKernel._dtype), new_state

    @staticmethod
    def backward(ctx: RWKVContext, gy: t.Tensor, gnew_state: Optional[t.Tensor]):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= WKVKernel.T_MAX
        assert B * C % min(C, 32) == 0
        w, u, k, v, state = ctx.saved_tensors
        
        tensor_target_dtype = t.float32 if WKVKernel._dtype != t.bfloat16 else t.bfloat16
        
        gw = einops.repeat(t.empty_like(w).to(tensor_target_dtype), 'c -> b c', b=B).contiguous()
        gu = einops.repeat(t.empty_like(u), 'c -> b c', b=B).contiguous()
        gk = t.empty_like(k).contiguous()
        gv = t.empty_like(v).contiguous()
        g_state = t.empty_like(state).contiguous()
        
        getattr(WKVKernel._kernel, 'backward_state')(B, T, C, w, u, k, v, state, cast(gy, tensor_target_dtype).contiguous(), gnew_state, gw, gu, gk, gv, g_state)
        
        gw = t.sum(gw, dim=0)
        gu = t.sum(gu, dim=0)
        
        return (None, None, None, cast(gw, WKVKernel._dtype), cast(gu, WKVKernel._dtype), cast(gk, WKVKernel._dtype), cast(gv, WKVKernel._dtype), g_state)    

class WKV(Function):

    @classmethod
    @property
    def return_state(cls):
        return WKV._should_return_state()

    @staticmethod
    @lru_cache
    def _should_return_state():
        return bool(int(os.environ.get('USE_CACHE', '0')))        

    @staticmethod
    def forward(ctx: RWKVContext,
                B: int,
                T: int,
                C: int,
                w: t.Tensor,
                u: t.Tensor,
                k: t.Tensor,
                v: t.Tensor,
                s: Optional[t.Tensor]):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= WKVKernel.T_MAX
        assert B * C % min(C, 32) == 0
        
        w = -t.exp(cast(w, t.float32).contiguous())
        tensor_target_dtype = t.float32 if WKVKernel._dtype != t.bfloat16 else t.bfloat16
        u = cast(u, tensor_target_dtype).contiguous()
        k = cast(k, tensor_target_dtype).contiguous()
        v = cast(v, tensor_target_dtype).contiguous()
        y = t.empty_like(k).contiguous()
        
        if WKV.return_state or s is not None:
            if s is None:
                
                state = t.zeros(B, C, 3, dtype=t.float32, device=k.device).contiguous()
                state[:, :, 2] -= 1e38
            else:
                state = s
                
        WKVKernel._kernel.forward(B, T, C, w, u, k, v, y) if not WKV.return_state else getattr(WKVKernel._kernel, 'forward_with_state')(B, T, C, w, u, k, v, y, state)
        
        ctx.save_for_backward(w, u, k, v, y)
        
        return cast(y, WKVKernel._dtype), state if WKV.return_state or s is not None else None

    @staticmethod
    def backward(ctx: RWKVContext, gy: t.Tensor, g_state: Optional[t.Tensor]):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= WKVKernel.T_MAX
        assert B * C % min(C, 32) == 0
        w, u, k, v, y = ctx.saved_tensors
        tensor_target_dtype = t.float32 if WKVKernel._dtype != t.bfloat16 else t.bfloat16

        gw = einops.repeat(t.empty_like(w).to(tensor_target_dtype), 'c -> b c', b=B).contiguous()
        gu = einops.repeat(t.empty_like(u), 'c -> b c', b=B).contiguous()
        gk = t.empty_like(k).contiguous()
        gv = t.empty_like(v).contiguous()
        
        WKVKernel._kernel.backward(B, T, C, w, u, k, v, y, cast(gy, tensor_target_dtype).contiguous(), gw, gu, gk, gv)
        
        gw = t.sum(gw, dim=0)
        gu = t.sum(gu, dim=0)
        return (None, None, None, cast(gw, WKVKernel._dtype), cast(gu, WKVKernel._dtype), cast(gk, WKVKernel._dtype), cast(gv, WKVKernel._dtype), None) 

def wkv_cuda(B: int,
             T: int,
             C: int,
             w: t.Tensor,
             u: t.Tensor,
             k: t.Tensor,
             v: t.Tensor,
             s: t.Tensor = None, 
             cross_att = False):
    if not WKVKernel.state():
       WKVKernel.load() 
    func_cls: Function = StateWKV if cross_att else WKV
    return func_cls.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda(), s.cuda() if s is not None else s)