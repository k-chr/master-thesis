import torch as t
from torch.autograd import Function

from diffccoder.utils.rwkv_kernel_context import RWKVContext


class L2Wrap(Function):
    @staticmethod
    def forward(ctx: RWKVContext, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx: RWKVContext, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = t.max(y, -1, keepdim=True)
        gy = t.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)
    