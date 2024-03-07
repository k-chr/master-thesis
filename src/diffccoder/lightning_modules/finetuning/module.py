import os
from pathlib import Path
from typing import Any, Optional

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch.utilities.types import STEP_OUTPUT
from loguru import logger
import torch as t

from diffccoder.configs.diffusion_config import DiffusionConfig
from diffccoder.configs.optimization_config import OptimizationConfig
from diffccoder.configs.rwkv_config import RWKVConfig
from diffccoder.lightning_modules.training_base import TrainingBase
from diffccoder.model.diffusion.DIFFRWKV import DIFF_RWKV
from diffccoder.model.diffusion.diffusion import GaussianDiffusion
from diffccoder.model.diffusion.ema import EMA
from diffccoder.model.rwkv.RWKVCM import RWKV
from diffccoder.utils.outputs import BlockStateList, DiffusionLosses
from diffccoder.utils.l2wrap import L2Wrap


class DiffusionFineTuningModule(TrainingBase):
    
    def __init__(self, 
                 optimization_config: OptimizationConfig,
                 rwkv_config: RWKVConfig,
                 diff_config: DiffusionConfig,
                 from_pretrained: Path,
                 ema_dir: Path = None) -> None:
        super().__init__(optimization_config)
        self.rwkv_config = rwkv_config
        self.diff_config = diff_config
        self.from_pretrained = from_pretrained
        self.ema_dir = ema_dir
        
        os.environ['CTX_LEN'] = str(rwkv_config.context_length)
        os.environ['USE_CACHE'] = str(int(rwkv_config.use_cache and not self.training))
        
        encoder = RWKV(self.rwkv_config)
        decoder = DIFF_RWKV(diff_config=self.diff_config, rwkv_config=self.rwkv_config)
                   
        self.model = GaussianDiffusion(encoder=encoder, model=decoder, config=self.diff_config)

    def configure_model(self) -> None:            
        if self.from_pretrained is not None and self.from_pretrained.exists():
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
        
        match prefix:
            case 'train':
                kwargs = {'on_step':True, 'prog_bar':True}
            case 'ema':
                kwargs = {'rank_zero_only':True}
            case _:
                kwargs = {}
        
        self.log(f'{prefix}_loss', losses.loss.mean(), **kwargs)
        self.log(f'{prefix}_mse', losses.mse_loss.mean(), **kwargs)
        self.log(f'{prefix}_t0_loss', losses.t0_loss.mean(), **kwargs)
        self.log(f'{prefix}_tT_loss', losses.tT_loss.mean(), **kwargs)
        self.log(f'{prefix}_decoder_nll', losses.decoder_nll.mean(), **kwargs)
        self.log(f'{prefix}_perplexity', t.exp(losses.decoder_nll.mean()), **kwargs)
    
    def _process_batch(self, batch: t.Tensor, use_ema: bool =False):
        _, x, y = batch
    
        y = y.long()
        ctx_len = self.rwkv_config.context_length
        y1, y2, y2_mask = y[:, :ctx_len], y[:, ctx_len:], ~(y[:, ctx_len:] == 1).all(-1)
        y1_mask = t.ones_like(y2_mask)
        
        state = BlockStateList.create(self.rwkv_config.num_hidden_layers,
                                      x.shape[0],
                                      self.rwkv_config.embedding_size,
                                      x.device,
                                      t.float32)
        x = x.int()
        
        model: GaussianDiffusion = self.model if not use_ema else self.ema
        noise: t.Tensor = None
        timesteps: t.Tensor = None
        total_diff_losses: DiffusionLosses = None
        
        for _y, mask, offset_noise_strength in [(y1, y1_mask, None), (y2, y2_mask, 0.0)]:
            if not t.any(mask): continue
            
            diff_out: tuple[DiffusionLosses, Optional[BlockStateList], t.Tensor, t.Tensor] = model(x[mask],
                                                                                                   _y[mask],
                                                                                                   None,#noise[mask] if noise is not None else None,
                                                                                                   None, #timesteps[mask] if timesteps is not None else None,
                                                                                                   state.subset(mask),
                                                                                                   offset_noise_strength=offset_noise_strength)
            diff_losses, state, timesteps, noise = diff_out
        
            if total_diff_losses is None:
                total_diff_losses = diff_losses
            else:
                total_diff_losses.loss[mask] += diff_losses.loss
                total_diff_losses.mse_loss[mask] += diff_losses.mse_loss
                total_diff_losses.decoder_nll += diff_losses.decoder_nll
                total_diff_losses.t0_loss[mask] += diff_losses.t0_loss
                total_diff_losses.tT_loss[mask] += diff_losses.tT_loss
                total_diff_losses.mse_pre[mask] += diff_losses.mse_pre
        
        return diff_losses, y
    
    @rank_zero_only
    def init_ema(self):
        if self.ema_dir is not None:
            ema_path = self.ema_dir
        else:
            checkpoint_callback: ModelCheckpoint = self.trainer.checkpoint_callback
            ema_path = Path(checkpoint_callback.last_model_path).parent
        ema_path /= 'ema.pt'
        self.ema = EMA(beta=self.diff_config.ema_beta,
                       update_after_step=self.diff_config.update_ema_every,
                       include_online_model=False,
                       model=self.model,
                       ema_path=ema_path)
        
        if ema_path.is_file() and ema_path.exists() and not (len(ema_path.parents) == 1 and ema_path.parents[0].__str__() == '.'):
            self.ema.load()
        elif self.from_pretrained is not None:
            ema_pretrained = self.from_pretrained.parent / 'ema.pt'
            self.ema.set_path(ema_pretrained)
            self.ema.load()
            self.ema.set_path(ema_path)
        else:
            raise ValueError('Directory of loaded online model checkpoint should have EMA checkpoint too')
    
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