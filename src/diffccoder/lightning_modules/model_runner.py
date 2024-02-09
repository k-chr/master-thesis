from dataclasses import asdict
from typing import Iterable

from lightning import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from torch import nn

from diffccoder.configs.diffusion_config import DiffusionConfig
from diffccoder.configs.trainer_config import TrainerConfig, DebugTrainerConfig
from diffccoder.model.diffusion.ema import EMA


class ModelRunner(Trainer):
    def __init__(self,
                 trainer_config: TrainerConfig,
                 callbacks: list[Callback] | Callback | None = None,
                 logger: Logger | Iterable[Logger] | bool | None = None,
                 debug_config: DebugTrainerConfig | None = None,
                 **kwargs) -> None:
        
        super().__init__(**asdict((debug_config if debug_config else DebugTrainerConfig())),
                         **asdict(trainer_config),
                         callbacks=callbacks,
                         logger=logger,
                         **kwargs)
        
class EMAModelRunner(ModelRunner):
    ema: EMA
    def __init__(self,
                 trainer_config: TrainerConfig,
                 callbacks: list[Callback] | Callback | None = None,
                 logger: Logger | Iterable[Logger] | bool | None = None,
                 debug_config: DebugTrainerConfig | None = None, **kwargs) -> None:
        super().__init__(trainer_config, callbacks, logger, debug_config, **kwargs)
        
    def init_ema(self, module: nn.Module, diff_cfg: DiffusionConfig):
        self.ema = EMA(module,
                       beta=diff_cfg.ema_beta,
                       update_every=diff_cfg.update_ema_every,
                       include_online_model=False)