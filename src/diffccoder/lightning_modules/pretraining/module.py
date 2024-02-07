import os

from bitsandbytes.optim import AdamW8bit, Adagrad8bit, Adam8bit
from lightning import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
import torch as t
from torch.functional import F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, StepLR

from diffccoder.configs.enums import LRSchedulerType, OptimizerType
from diffccoder.configs.optimization_config import OptimizationConfig
from diffccoder.configs.rwkv_config import RWKVConfig
from diffccoder.model.rwkv.RWKVCM import RWKVCM
from diffccoder.utils.outputs import RWKVOutput
from diffccoder.utils.l2wrap import L2Wrap
from diffccoder.utils.lr_scheduler import EhnancedCosineSchedulerLR
from diffccoder.utils.warm_up_scheduler import WarmUpScheduler


class PretrainingModule(LightningModule):
    skip_validation_step: bool = False
    q: dict[str,list[tuple[ModelCheckpoint, int]]] = {'training':[], 'validation':[]}
    def __init__(self, optimization_config: OptimizationConfig, config: RWKVConfig, skip_init: bool = False) -> None:
        super().__init__()
        self.config = optimization_config
        self.model_config = config
        os.environ['CTX_LEN'] = str(config.context_length)
        os.environ['USE_CACHE'] = str(int(config.use_cache and not self.training))
        self.__skip_one_step = skip_init
        self.model = RWKVCM(config, skip_init)
        
    def training_step(self, batch: t.Tensor, batch_idx: int) -> t.Tensor:
        loss, _, y = self._process_batch(batch)
        
        if self.__skip_one_step:
            self.__skip_one_step = False
            self.__stop_checkpointers('training')
        else:
            self.__restore_checkpointers('training')
            self.log('train_loss', loss, on_step=True, prog_bar=True)
            self.log('train_perplexity', t.exp(loss.mean()), on_step=True, prog_bar=True)
        
        return L2Wrap.apply(loss, y.to(self.dtype))

    def validation_step(self, batch: t.Tensor, batch_idx: int) -> t.Tensor:
        loss, _, _ = self._process_batch(batch)
        
        if not self.trainer.sanity_checking and not self.skip_validation_step:
            self.__restore_checkpointers()
            self.log('validation_loss', loss)
            self.log('validation_perplexity', t.exp(loss.mean()))
        elif self.skip_validation_step:
            self.__stop_checkpointers()
            
            self.skip_validation_step = False
        return loss

    def __restore_checkpointers(self, mode='validation'):
        while self.q[mode]:
            c, k = self.q[mode].pop()
            c.save_top_k = k

    def __stop_checkpointers(self, mode='validation'):
        model_checkpoints: list[ModelCheckpoint] = self.trainer.checkpoint_callbacks
        for model_checkpointer in model_checkpoints:
            if model_checkpointer.monitor in [f'{mode}_loss', f'{mode}_perplexity']:
                self.q[mode].append((model_checkpointer, model_checkpointer.save_top_k))
                model_checkpointer.save_top_k = 0
    
    def _process_batch(self, batch: t.Tensor):
        _, x, y = batch
        y = y.to(t.int64)
        rwkv_out: RWKVOutput = self.model(x)

        shift_x_hat = rwkv_out.logits[..., :-1, :].contiguous()
        shift_y = y[..., 1:].contiguous()
        
        loss = F.cross_entropy(shift_x_hat.view(-1, shift_x_hat.size(-1)),
                               shift_y.view(-1))
                               
        return loss, rwkv_out, y
 
    def configure_optimizers(self) -> dict[str, Optimizer | dict[str, LRScheduler | int | str]] | Optimizer:
        optim_groups = self._compute_optim_groups()
            
        optimizer: t.optim.Optimizer = self.__configure_optimizer(optim_groups)
            
        scheduler_aux_config, scheduler = self.__configure_scheduler(optimizer)
            
        if scheduler:
            scheduler_config = {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            } 
            
            scheduler_config = scheduler_config | scheduler_aux_config
            
            return {'optimizer': optimizer,
                    'lr_scheduler': scheduler_config}
            
        return optimizer

    def __configure_scheduler(self, optimizer) -> tuple[dict[str, str], LRScheduler | None]:
        scheduler_aux_config = {}
        
        match self.config.lr_scheduler:
            case LRSchedulerType.STEP:
                scheduler = StepLR(optimizer=optimizer, 
                                   step_size=self.config.lr_decay_step_size,
                                   gamma=self.config.lr_decay_rate)
                
                scheduler_aux_config = {'name': 'step'}
            case LRSchedulerType.COSINE:
                scheduler = EhnancedCosineSchedulerLR(optimizer=optimizer,
                                                      t_0=self.config.cos_t0,
                                                      eta_min=self.config.cos_eta_min,
                                                      lr_decay=self.config.lr_decay_rate,
                                                      t_decay=self.config.cos_t_decay,
                                                      lr_lowest_bound_ratio=self.config.cos_lb_ratio,
                                                      warm_restarts=self.config.cos_warm_restarts)
                
                scheduler_aux_config = {'name': 'cosine'}
            case LRSchedulerType.WARMUP:
                scheduler = self._configure_warmup(optimizer)
                
                scheduler_aux_config = {'monitor': self.config.warmup_metric.name.lower(),
                                        'name': f'warmup_{self.config.warmup_scheduler.name.lower()}'}
            case _:
                scheduler = None
        return scheduler_aux_config,scheduler

    def _configure_warmup(self, optimizer):
        match self.config.warmup_scheduler:
            case LRSchedulerType.STEP:
                to_wrap = StepLR(optimizer=optimizer, 
                                 step_size=self.config.lr_decay_step_size,
                                 gamma=self.config.lr_decay_rate)
            case LRSchedulerType.COSINE:
                to_wrap = EhnancedCosineSchedulerLR(optimizer=optimizer,
                                                    t_0=self.config.cos_t0,
                                                    eta_min=self.config.cos_eta_min,
                                                    lr_decay=self.config.lr_decay_rate,
                                                    t_decay=self.config.cos_t_decay,
                                                    lr_lowest_bound_ratio=self.config.cos_lb_ratio,
                                                    warm_restarts=self.config.cos_warm_restarts)
            case _:
                raise ValueError(f'Unknown scheduler: {self.config.warmup_scheduler}')

        return WarmUpScheduler(scheduler_to_wrap=to_wrap,
                               warm_up_steps=self.config.warmup_max,
                               warm_up_metric=self.config.warmup_metric,
                               warm_up_routine=self.config.warmup_routine,
                               k=self.config.warmup_k,
                               lr_0=self.config.warmup_lr_0,
                               get_metric=lambda: self.trainer.global_step)

    def __configure_optimizer(self, optim_groups) -> Optimizer:
        match self.config.optimizer:
            case OptimizerType.ADAM:
                optimizer = t.optim.Adam(params=optim_groups,
                                         lr=self.config.lr_base,
                                         eps=self.config.adam_eps,
                                         betas=self.config.adam_betas,
                                         fused=self.config.adam_fused)
            case OptimizerType.ADAM_8:
                optimizer = Adam8bit(params=optim_groups,
                                     lr=self.config.lr_base,
                                     eps=self.config.adam_eps,
                                     betas=self.config.adam_betas)
            case OptimizerType.SGD:
                optimizer = t.optim.SGD(params=optim_groups,
                                        lr=self.config.lr_base,
                                        momentum=self.config.sgd_momentum)
            case OptimizerType.ADAM_W:
                optimizer = t.optim.AdamW(params=optim_groups,
                                         lr=self.config.lr_base,
                                         eps=self.config.adam_eps,
                                         betas=self.config.adam_betas,
                                         fused=self.config.adam_fused)
            case OptimizerType.ADAM_W_8:
                optimizer = AdamW8bit(params=optim_groups,
                                      lr=self.config.lr_base,
                                      eps=self.config.adam_eps,
                                      betas=self.config.adam_betas)     
            case OptimizerType.ADA_GRAD:
                optimizer = t.optim.Adagrad(params=optim_groups,
                                            lr=self.config.lr_base,
                                            eps=self.config.adam_eps)
            case OptimizerType.ADA_GRAD_8:
                optimizer = Adagrad8bit(params=optim_groups,
                                        lr=self.config.lr_base,
                                        eps=self.config.adam_eps)
            case _:
                raise RuntimeError(f'Not implemented optimizer: {self.config.optimizer}')
        return optimizer

    def _compute_optim_groups(self):
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        
        for n, p in self.model.named_parameters():
            if ("time_mix" in n) and (self.config.layerwise_lr):
                lr_1x.add(n)
            elif ("time_decay" in n) and (self.config.layerwise_lr):
                lr_2x.add(n)
            elif ("time_first" in n) and (self.config.layerwise_lr):
                lr_3x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (self.config.weight_decay):
                lr_decay.add(n)
            else:
                lr_1x.add(n)


        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))
        
        params = {n: p for n, p in self.model.named_parameters()}
        
        if self.config.layerwise_lr:
            optim_groups = [
                    {"params": [params[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [params[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                    {"params": [params[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                ]
        else:
            optim_groups = [{"params": [params[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]
        
        if self.config.weight_decay > 0:
            optim_groups += [{"params": [params[n] for n in lr_decay],
                              "weight_decay": self.config.weight_decay,
                              "my_lr_scale": 1.0}]
        return optim_groups
    