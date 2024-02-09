import os
from typing import Any

from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch.utilities.types import STEP_OUTPUT
from loguru import logger
import torch as t

from diffccoder.configs.diffusion_config import DiffusionConfig
from diffccoder.configs.optimization_config import OptimizationConfig
from diffccoder.configs.rwkv_config import RWKVConfig
from diffccoder.lightning_modules.model_runner import EMAModelRunner
from diffccoder.lightning_modules.pretraining.module import PretrainingModule
from diffccoder.lightning_modules.training_base import TrainingBase
from diffccoder.model.diffusion.DIFFRWKV import DIFF_RWKV
from diffccoder.model.diffusion.diffusion import GaussianDiffusion
from diffccoder.model.rwkv.RWKVCM import RWKV
from diffccoder.model.rwkv.initialization import RWKV_Init
from diffccoder.utils.outputs import DiffusionLosses
from diffccoder.utils.l2wrap import L2Wrap


class DiffusionTrainingModule(TrainingBase):
    
    def __init__(self, 
                 optimization_config: OptimizationConfig,
                 rwkv_config: RWKVConfig,
                 diff_config: DiffusionConfig,
                 skip_init: bool = False) -> None:
        super().__init__(optimization_config, not skip_init)
        self.rwkv_config = rwkv_config
        self.diff_config = diff_config
        
        os.environ['CTX_LEN'] = str(rwkv_config.context_length)
        os.environ['USE_CACHE'] = str(int(rwkv_config.use_cache and not self.training))
        rwkv_path = diff_config.encoder_path
        
        if rwkv_path is not None:
            obj: PretrainingModule = t.load(rwkv_path)
            encoder: RWKV = obj.model.rwkv
        else:
            encoder = RWKV(rwkv_config)
            if not skip_init:
                RWKV_Init(encoder, rwkv_config)
        
        decoder = DIFF_RWKV(diff_config=diff_config, rwkv_config=rwkv_config)
        
        if not skip_init:
            RWKV_Init(decoder, rwkv_config)
        
        self.model = GaussianDiffusion(encoder=encoder, model=decoder, config=diff_config)
       
    def training_step(self, batch: t.Tensor, batch_idx: int) -> t.Tensor:
        losses, y = self._process_batch(batch)
        
        self.log('train_loss', losses.loss.mean(-1), on_step=True, prog_bar=True)
        self.log('train_mse', losses.mse_loss.mean(-1), on_step=True, prog_bar=True)
        self.log('train_t0_loss', losses.t0_loss.mean(-1), on_step=True, prog_bar=True)
        self.log('train_tT_loss', losses.tT_loss.mean(-1), on_step=True, prog_bar=True)
        self.log('train_decoder_nll', losses.decoder_nll.mean(-1), on_step=True, prog_bar=True)
        self.log('train_perplexity', t.exp(losses.decoder_nll.mean(-1)), on_step=True, prog_bar=True)
        
        return L2Wrap.apply(losses.loss.mean(-1), y.to(self.dtype))

    def validation_step(self, batch: t.Tensor, batch_idx: int) -> t.Tensor:
        losses, _ = self._process_batch(batch)
        
        self.log('validation_loss', losses.loss.mean())
        self.log('validation_mse', losses.mse_loss.mean())
        self.log('validation_t0_loss', losses.t0_loss.mean())
        self.log('validation_tT_loss', losses.tT_loss.mean())
        self.log('validation_decoder_nll', losses.decoder_nll.mean())
        self.log('validation_perplexity', t.exp(losses.decoder_nll.mean()))

        return losses.loss.mean()

    def __trainer(self) -> EMAModelRunner:
        return self.trainer
    
    def _process_batch(self, batch: t.Tensor):
        _, x, y = batch
        y = y.to(t.int64)
        
        if self.training == False and self.diff_config.use_ema_at_infer and self.__trainer().ema is not None:
            logger.info('Using EMA')
            model: GaussianDiffusion = self.__trainer().ema.ema_model
        else:
            model = self.model
        
        diff_out: DiffusionLosses = model(x, y)
        
        return diff_out, y
    
    def on_train_start(self) -> None:
        logger.info('Initialization of EMA parameters...')
        self.__trainer().init_ema(self, self.diff_config)
        
        return super().on_train_start()
    
    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        if rank_zero_only.rank == 0:
            self.__trainer().ema.update()        

        return super().on_train_batch_end(outputs, batch, batch_idx)