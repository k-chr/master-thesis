import os
from pathlib import Path
from typing import Any

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch.utilities.types import STEP_OUTPUT
from loguru import logger
import torch as t

from diffccoder.configs.diffusion_config import DiffusionConfig
from diffccoder.configs.optimization_config import OptimizationConfig
from diffccoder.configs.rwkv_config import RWKVConfig
from diffccoder.lightning_modules.pretraining.module import PretrainingModule
from diffccoder.lightning_modules.training_base import TrainingBase
from diffccoder.model.diffusion.DIFFRWKV import DIFF_RWKV
from diffccoder.model.diffusion.diffusion import GaussianDiffusion
from diffccoder.model.diffusion.ema import EMA
from diffccoder.model.rwkv.RWKVCM import RWKV
from diffccoder.model.rwkv.initialization import RWKV_Init
from diffccoder.utils.outputs import DiffusionLosses
from diffccoder.utils.l2wrap import L2Wrap


class DiffusionTrainingModule(TrainingBase):
    
    def __init__(self, 
                 optimization_config: OptimizationConfig,
                 rwkv_config: RWKVConfig,
                 diff_config: DiffusionConfig,
                 skip_init: bool = False,
                 init_path: Path = None,
                 from_pretrained: Path = None) -> None:
        super().__init__(optimization_config, not skip_init)
        self.rwkv_config = rwkv_config
        self.diff_config = diff_config
        self.init_path = init_path
        self.from_pretrained = from_pretrained

        os.environ['CTX_LEN'] = str(rwkv_config.context_length)
        os.environ['USE_CACHE'] = str(int(rwkv_config.use_cache and not self.training))

        rwkv_path = self.diff_config.encoder_path
        
        if rwkv_path is not None:
            obj: PretrainingModule = PretrainingModule(self.config,
                                                       self.rwkv_config,
                                                       skip_init=True)
            obj.load_state_dict(t.load(rwkv_path)['state_dict'])
            if self.diff_config.freeze_encoder:
                obj.freeze()
            encoder: RWKV = obj.model.rwkv
        else:
            encoder = RWKV(self.rwkv_config)
            if self.save_before_training:
                RWKV_Init(encoder, self.rwkv_config)
        
        decoder = DIFF_RWKV(diff_config=self.diff_config, rwkv_config=self.rwkv_config)
        
        if self.save_before_training:
            RWKV_Init(decoder, self.rwkv_config)
                   
        self.model = GaussianDiffusion(encoder=encoder, model=decoder, config=self.diff_config)

    def configure_model(self) -> None:            
        if self.save_before_training:
            t.save(self.model.state_dict(), self.init_path)
            logger.debug(f'[rank: {rank_zero_only.rank}] Initialized weights saved successfully...')
            self.trainer.strategy.barrier('Wait for initialized checkpoint')
        elif self.init_path is not None:
            self.trainer.strategy.barrier('Wait for initialized checkpoint')
            self.model.load_state_dict(t.load(self.init_path, map_location='cpu'))
            logger.debug(f'[rank: {rank_zero_only.rank}] Initialized weights loaded successfully...')

        if self.from_pretrained is not None:
            self.load_state_dict(t.load(self.from_pretrained, map_location='cpu')['state_dict'], strict=False)
        
        self.init_ema()

        if rank_zero_only.rank == 0:
            self._ddp_params_and_buffers_to_ignore = [f'ema.{key}' for key in self.ema.state_dict().keys()]

    def training_step(self, batch: t.Tensor, batch_idx: int) -> t.Tensor:
        losses, y = self._process_batch(batch)
        
        self.log_losses(losses, 'train')
        
        return L2Wrap.apply(losses.loss.mean(-1), y.to(self.dtype))

    def validation_step(self, batch: t.Tensor, batch_idx: int) -> t.Tensor:
        losses, _ = self._process_batch(batch)
        
        self.log_losses(losses, 'validation')

        if rank_zero_only.rank == 0 and self.diff_config.use_ema_at_infer:
            losses, _ = self._process_batch(batch, use_ema=True)
            self.log_losses(losses, 'ema')
        
        return losses.loss.mean()

    def log_losses(self, losses: DiffusionLosses, prefix: str):
        kwargs = {'on_step':True, 'prog_bar':True} if prefix == 'train' else {}
        
        self.log(f'{prefix}_loss', losses.loss.mean(), **kwargs)
        self.log(f'{prefix}_mse', losses.mse_loss.mean(), **kwargs)
        self.log(f'{prefix}_t0_loss', losses.t0_loss.mean(), **kwargs)
        self.log(f'{prefix}_tT_loss', losses.tT_loss.mean(), **kwargs)
        self.log(f'{prefix}_decoder_nll', losses.decoder_nll.mean(), **kwargs)
        self.log(f'{prefix}_perplexity', t.exp(losses.decoder_nll.mean()), **kwargs)
    
    def _process_batch(self, batch: t.Tensor, use_ema: bool =False):
        _, x, y = batch
        y = y.long()
        x = x.int()
        
        model: GaussianDiffusion = self.model if not use_ema else self.ema
        
        diff_out: DiffusionLosses = model(x, y)
        
        return diff_out, y
    
    def on_train_start(self) -> None:                        
        return super().on_train_start()
    
    @rank_zero_only
    def init_ema(self):
        if self.init_path is not None:
            ema_path = self.init_path.parent
        else:
            checkpoint_callback: ModelCheckpoint = self.trainer.checkpoint_callback
            ema_path = Path(checkpoint_callback.last_model_path).parent
        ema_path /= 'ema.pt'
        self.ema = EMA(beta=self.diff_config.ema_beta,
                           update_after_step=self.diff_config.update_ema_every,
                           include_online_model=False,
                           model=self.model,
                           ema_path=ema_path)
            
        if self.from_pretrained is not None:
            ema_pretrained = self.from_pretrained.parent / 'ema.pt'
            self.ema.set_path(ema_pretrained)
            self.ema.load()
            self.ema.set_path(ema_path)
        else:
            self.ema.load()
    
    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        if rank_zero_only.rank == 0:
            self.ema.update()        

        return super().on_train_batch_end(outputs, batch, batch_idx)
    
    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if rank_zero_only.rank == 0:
            state_dict: dict[str, t.Tensor| t.nn.Parameter] = checkpoint['state_dict']
            ema_keys = [key for key in state_dict.keys() if 'ema.' in key]
            if ema_keys:
                [state_dict.pop(key) for key in ema_keys]
                checkpoint['state_dict'] = state_dict
            if self.trainer.state.stage is not None and self.trainer.state.stage.evaluating:
                self.ema.save()