import os

import torch as t
from torch.functional import F

from diffccoder.configs.optimization_config import OptimizationConfig
from diffccoder.configs.rwkv_config import RWKVConfig
from diffccoder.lightning_modules.training_base import TrainingBase
from diffccoder.model.rwkv.RWKVCM import RWKVCM
from diffccoder.utils.outputs import RWKVOutput
from diffccoder.utils.l2wrap import L2Wrap


class PretrainingModule(TrainingBase):
    model: RWKVCM
    
    def __init__(self, optimization_config: OptimizationConfig, config: RWKVConfig, skip_init: bool = False) -> None:
        super().__init__(optimization_config, not skip_init)
        
        self.model_config = config
        os.environ['CTX_LEN'] = str(config.context_length)
        os.environ['USE_CACHE'] = str(int(config.use_cache and not self.training))
        self.model = RWKVCM(config, not skip_init)
 
    def training_step(self, batch: t.Tensor, batch_idx: int) -> t.Tensor:
        loss, _, y = self._process_batch(batch)
        
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        self.log('train_perplexity', t.exp(loss.mean()), on_step=True, prog_bar=True)
        
        return L2Wrap.apply(loss, y.to(self.dtype))

    def validation_step(self, batch: t.Tensor, batch_idx: int) -> t.Tensor:
        loss, _, _ = self._process_batch(batch)
        
        self.log('validation_loss', loss)
        self.log('validation_perplexity', t.exp(loss.mean()))

        return loss
    
    def _process_batch(self, batch: t.Tensor):
        _, x, y = batch
        y = y.to(t.int64)
        rwkv_out: RWKVOutput = self.model(x.int())

        x_hat = rwkv_out.logits.contiguous()
        _y = y.contiguous()
        
        loss = F.cross_entropy(x_hat.view(-1, x_hat.size(-1)), _y.view(-1))
                               
        return loss, rwkv_out, y