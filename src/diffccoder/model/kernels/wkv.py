from pathlib import Path

import torch as t
from torch.autograd.function import Function
from torch.utils.cpp_extension import load

from diffccoder.utils.rwkv_kernel_context import RWKVContext


T_MAX = 1024 # increase this if your ctx_len is long [NOTE: TAKES LOTS OF VRAM!]
# it's possible to go beyond CUDA limitations if you slice the ctx and pass the hidden state in each slice
_kernels = Path(__file__).resolve().parent
_wkv_cuda: Function = load(name="wkv_cuda", sources=[ _kernels / "cuda/wkv_op.cpp", _kernels / "cuda/wkv_cuda.cu"],
                verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}'])


class WKV(Function):
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
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w = -t.exp(w.contiguous())
        u = u.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        ctx.save_for_backward(w, u, k, v)
        y = t.empty((B, T, C), device='cuda', memory_format=t.contiguous_format)
        _wkv_cuda.forward(B, T, C, w, u, k, v, y)
        return y

    @staticmethod
    def backward(ctx: RWKVContext, gy: t.Tensor):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = t.zeros((B, C), device='cuda').contiguous()
        gu = t.zeros((B, C), device='cuda').contiguous()
        gk = t.zeros((B, T, C), device='cuda').contiguous()
        gv = t.zeros((B, T, C), device='cuda').contiguous()
        
        _wkv_cuda.backward(B, T, C, w, u, k, v, gy.contiguous(), gw, gu, gk, gv)
        
        gw = t.sum(gw, dim=0)
        gu = t.sum(gu, dim=0)
        
        return (None, None, None, gw, gu, gk, gv)

def WKV_CUDA(B: int,
             T: int,
             C: int,
             w: t.Tensor,
             u: t.Tensor,
             k: t.Tensor,
             v: t.Tensor):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())