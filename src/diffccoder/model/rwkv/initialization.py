import gc
import math

from loguru import logger
import torch as t
from torch import nn

from diffccoder.configs.rwkv_config import RWKVConfig


def RWKV_Init(model: nn.Module, args: RWKVConfig):  # fancy initialization of all lin & emb layer in the model
    logger.info("\n[--> first run, init model params (very slow for large models) <--]")
    logger.info("[so you shall only do it for 1 single GPU and save the checkpt and load it when using multiple GPU]\n")

    for mm in model.modules():
        if "RecursiveScriptModule" in str(type(mm)):
            if mm.original_name not in ["Linear"]:
                continue
            ww = None
            for name, param in mm.named_parameters():
                if name == "weight":
                    ww = param
        else:
            m = mm
            if not isinstance(m, (nn.Linear, nn.Embedding)):
                continue
            ww = m.weight
        with t.no_grad():
            name = "[unknown weight]"
            for name, parameter in model.named_parameters():  # find the name of the weight
                if id(ww) == id(parameter):
                    break

            shape = ww.shape
            gain = 1.0
            scale = 1.0  # extra scale for gain
            match m.__class__:
                case nn.Embedding:
                    gain = math.sqrt(max(shape[0], shape[1]))
                    if shape[0] == args.vocab_size and shape[1] == args.embedding_size:  # token emb?
                        scale = 1e-4
                    else:
                        scale = 0        
                case nn.Linear:
                    if shape[0] > shape[1]:
                        gain = math.sqrt(shape[0] / shape[1])
                    if shape[0] == args.vocab_size and shape[1] == args.embedding_size:  # final projection?
                        scale = 0.5

            if hasattr(m, "scale_init"):
                scale = m.scale_init

            gain *= scale
            if gain == 0:
                # zero init is great for some RWKV matrices
                nn.init.zeros_(ww)
            elif gain > 0:
                nn.init.orthogonal_(ww, gain=gain)
            else:
                nn.init.normal_(ww, mean=0.0, std=-scale)
    gc.collect()
    t.cuda.empty_cache()