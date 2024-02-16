from dataclasses import asdict
from typing import Iterable

from lightning import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger

from diffccoder.configs.trainer_config import TrainerConfig, DebugTrainerConfig


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