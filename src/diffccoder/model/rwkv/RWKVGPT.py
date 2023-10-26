from loguru import logger
import torch as t
from torch import nn
from torch.functional import F

from diffccoder.configs.rwkv_config import RWKVConfig
from diffccoder.model.rwkv.initialization import RWKV_Init
from diffccoder.model.rwkv.layers import RWKVBlock, RWKVFfnPreBlock
from diffccoder.model.rwkv.outputs import RWKVOutput


class RWKV(nn.Module):
    def __init__(self, config: RWKVConfig) -> None:
        super().__init__()
        self.step = 0
        self.config = config

        self.emb = nn.Embedding(config.vocab_size, config.embedding_size)

        self.blocks = nn.Sequential(*[(RWKVFfnPreBlock if config.use_ffn_pre and not i else RWKVBlock)(config, i)
                                    for i in range(config.num_hidden_layers)])

        self.ln_out = nn.LayerNorm(config.embedding_size)
        self.context_length = config.context_length
        
    def forward(self, idx: t.Tensor) -> RWKVOutput:
        idx = idx.to(self.emb.weight.device)

        self.step += 1
        
        x: t.Tensor = self.emb(idx)
        x = self.blocks(x)
        x = self.ln_out(x)
        
        return RWKVOutput(last_hidden_state=x)


class GPT(nn.Module):
    def __init__(self, config: RWKVConfig, skip_init: bool =False):
        super().__init__()
        self.rwkv = RWKV(config)
        self.head = nn.Linear(config.embedding_size, config.vocab_size, bias=False)

        QK_att = config.qk_attention
        if  QK_att > 0:
            self.head_q = nn.Linear(config.embedding_size, QK_att, bias=False)
            self.head_q.scale_init = 0
            self.head_k = nn.Linear(config.embedding_size, QK_att, bias=False)
            self.head_k.scale_init = 0.1
            self.register_buffer("copy_mask", t.tril(
                t.ones(config.context_length, config.context_length)))
        self.config = config
        
        if not skip_init:
            RWKV_Init(self, config) 

        logger.info(f"Number of parameters: {sum(p.numel() for p in self.parameters())}")
    
    def get_ctx_len(self):
        return self.rwkv.context_length

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.01)
        if isinstance(module, (nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=1e-5)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, idx: t.Tensor) -> RWKVOutput:
        B, T = idx.size()
        assert T <= self.get_ctx_len(), "Cannot forward, because len(input) > model ctx_len."
        
        rwkv_out: RWKVOutput = self.rwkv(idx)
        x = rwkv_out.last_hidden_state
        
        if self.config.qk_attention > 0:
            q: t.Tensor = self.head_q(x)[:, :T, :]
            k: t.Tensor = self.head_k(x)[:, :T, :]
            c: t.Tensor = (q @ k.transpose(-2, -1)) * (1.0 / self.config.qk_attention)
            c = c.masked_fill(self.get_buffer('copy_mask')[:T, :T] == 0, 0)
            
            c = c @ F.one_hot(idx, num_classes=self.config.vocab_size)

            x = self.head(x) + c
        else:
            x = self.head(x)

        return RWKVOutput(logits=x, last_hidden_state=rwkv_out.last_hidden_state)        
