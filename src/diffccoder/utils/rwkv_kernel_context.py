from dataclasses import dataclass

from torch.autograd.function import FunctionCtx


@dataclass
class RWKVContext(FunctionCtx):
    B: int
    T: int
    C: int
    
    @property
    def saved_tensors(self):
        return self.to_save