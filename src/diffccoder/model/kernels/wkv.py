from functools import lru_cache
import os
from pathlib import Path

from loguru import logger
import torch as t
from torch.autograd.function import Function
from torch.utils.cpp_extension import load

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


class WKV(Function):
    _wkv_cuda: Function = None
    
    @classmethod
    @property
    def T_MAX(cls):
        # increase this if your ctx_len is long [NOTE: TAKES LOTS OF VRAM!]
        return int(os.environ.get('CTX_LEN', '1024'))
    
    @classmethod
    @property
    def _dtype(cls):
        precision = os.environ.get('DTYPE', '32-true')
        return WKV._get_dtype(precision)
     
    @staticmethod
    @lru_cache
    def _get_dtype(str_type: str):
        if '32' in str_type: return t.float32
        elif 'bf16' in str_type: return t.bfloat16
        else: return t.float16        
        
    @staticmethod
    def state(): 
        return WKV._wkv_cuda is not None
    
    @staticmethod
    def load():
        logger.info(f'Loading WKV cuda-kernel for {WKV._dtype} and ctx len: {WKV.T_MAX}')
        flags = ['-res-usage',
                 '--maxrregcount 60',
                 '--use_fast_math',
                 '-O3',
                 '-Xptxas -O3',
                 '--extra-device-vectorization',
                 f'-DTmax={WKV.T_MAX}']
        extra_flags = [] if WKV._dtype is not t.bfloat16 else ['-t 4', '-std=c++17']
        op_fname = 'cuda/wkv_op' + ('' if WKV._dtype is not t.bfloat16 else 'bf16')
        cuda_fname = 'cuda/wkv_cuda' + ('' if WKV._dtype is not t.bfloat16 else 'bf16')
        WKV._wkv_cuda = load(name='wkv_cuda', 
                             sources=[ _kernels / (op_fname + '.cpp'), _kernels /  (cuda_fname + '.cu')],
                             verbose=True, 
                             extra_cuda_cflags=extra_flags + flags)
 
    @staticmethod
    def forward(ctx: RWKVContext,
                B: int,
                T: int,
                C: int,
                w: t.Tensor,
                u: t.Tensor,
                k: t.Tensor,
                v: t.Tensor):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        
        assert T <= WKV.T_MAX
        assert B * C % min(C, 32) == 0
        
        w = -t.exp(cast(w, t.float32).contiguous())
        tensor_target_dtype = t.float32 if WKV._dtype != t.bfloat16 else t.bfloat16
        u = cast(u, tensor_target_dtype).contiguous()
        k = cast(k, tensor_target_dtype).contiguous()
        v = cast(v, tensor_target_dtype).contiguous()
        y = t.empty((B, T, C), device=w.device, memory_format=t.contiguous_format, dtype=tensor_target_dtype)
        
        WKV._wkv_cuda.forward(B, T, C, w, u, k, v, y)
        
        ctx.save_for_backward(w, u, k, v, y)
        
        return cast(y, WKV._dtype)

    @staticmethod
    def backward(ctx: RWKVContext, gy: t.Tensor):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= WKV.T_MAX
        assert B * C % min(C, 32) == 0
        w, u, k, v, y = ctx.saved_tensors
        dev = gy.device
        tensor_target_dtype = t.float32 if WKV._dtype != t.bfloat16 else t.bfloat16
        gw = t.empty((B, C), device=dev, dtype=tensor_target_dtype).contiguous()
        gu = t.empty((B, C), device=dev, dtype=tensor_target_dtype).contiguous()
        gk = t.empty((B, T, C), device=dev, dtype=tensor_target_dtype).contiguous()
        gv = t.empty((B, T, C), device=dev, dtype=tensor_target_dtype).contiguous()
        
        WKV._wkv_cuda.backward(B, T, C, w, u, k, v, y, cast(gy, tensor_target_dtype).contiguous(), gw, gu, gk, gv)
        
        gw = t.sum(gw, dim=0)
        gu = t.sum(gu, dim=0)
        
        return (None, None, None, cast(gw, WKV._dtype), cast(gu, WKV._dtype), cast(gk, WKV._dtype), cast(gv, WKV._dtype))

def wkv_cuda(B: int,
             T: int,
             C: int,
             w: t.Tensor,
             u: t.Tensor,
             k: t.Tensor,
             v: t.Tensor):
    if not WKV.state():
       WKV.load() 
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())