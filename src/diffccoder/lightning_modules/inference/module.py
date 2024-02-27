from functools import partial
import os
from pathlib import Path

from lightning import LightningModule
from tokenizers import Tokenizer
import torch as t
import torch.nn.functional as F

from diffccoder.configs.diffusion_config import DiffusionConfig
from diffccoder.configs.rwkv_config import RWKVConfig
from diffccoder.model.diffusion.DIFFRWKV import DIFF_RWKV
from diffccoder.model.diffusion.diffusion import GaussianDiffusion
from diffccoder.model.diffusion.ema import EMA
from diffccoder.model.diffusion.inference import InferenceSampler, denoised_fn_round
from diffccoder.model.rwkv.RWKVCM import RWKV
from diffccoder.utils.outputs import BlockStateList, RWKVOutput


class DiffusionInferModule(LightningModule):
    sampler: InferenceSampler = None
    
    def __init__(self, 
                 rwkv_config: RWKVConfig,
                 diff_config: DiffusionConfig,
                 from_pretrained: Path,
                 tokenizer: Tokenizer,
                 ema_local_path: Path = None,
                 target_length: int = 1024) -> None:
        super().__init__()
        self.rwkv_config = rwkv_config
        self.diff_config = diff_config
        self.from_pretrained = from_pretrained
        self.ema_local_path = ema_local_path
        self.tokenizer = tokenizer
        self.target_length = target_length
        os.environ['CTX_LEN'] = str(rwkv_config.context_length)
        os.environ['USE_CACHE'] = str(int(rwkv_config.use_cache and not self.training))
        
        encoder = RWKV(self.rwkv_config)
        decoder = DIFF_RWKV(diff_config=self.diff_config, rwkv_config=self.rwkv_config)
                   
        self.model = GaussianDiffusion(encoder=encoder, model=decoder, config=self.diff_config)

    def configure_model(self) -> None:            
        if self.from_pretrained is not None and self.from_pretrained.exists():
            self.load_state_dict(t.load(self.from_pretrained, map_location='cpu')['state_dict'], strict=False)

        self.init_ema()

        self._ddp_params_and_buffers_to_ignore = [f'ema.{key}' for key in self.ema.state_dict().keys()]
        
        self.sampler = InferenceSampler.create(self.diff_config, self.model if not self.diff_config.use_ema_at_infer else self.ema.ema_model)

    def predict_step(self, batch: t.Tensor, batch_idx: int) -> list[str] | str:
        maybe_samples = self._process_batch(batch)
        decoded = []
        if not self.diff_config.return_all_timesteps:
            maybe_samples = [maybe_samples]
        for sample in maybe_samples:
            logits = t.nan_to_num(self.sampler.diffusion.model.get_logits(sample))
            #logits[:, :, 0] = -9999
            
            #print(logits)
            probs = F.softmax(logits, -1)
            probs = probs[0].pow(1)
            sample_id_tensor = t.multinomial(probs/probs.sum(-1, keepdim=True), num_samples=1).flatten()#   t.argmax(logits, dim=-1) #
            decoded.append(self.tokenizer.decode_batch([sample_id_tensor.cpu().tolist()]))
        return decoded
    
    def _process_batch(self, batch: t.Tensor):
        _, x, _ = batch
        x = x.int()
        
        encoder = self.sampler.diffusion.encoder
        
        ctx, hidden_states = self.__build_ctx(encoder, x)
        
        sample = self.sampler.loop(shape=(x.shape[0], self.target_length, encoder.config.embedding_size),
                                   encoder_hidden_state=hidden_states,
                                   encoder_layer_state=ctx,
                                   return_all_timesteps=self.diff_config.return_all_timesteps,
                                   denoised_fn=partial(denoised_fn_round, self.sampler.diffusion.model.emb))
        
        return sample
    
    @t.inference_mode()
    def __build_ctx(self, encoder: RWKV, indices: t.Tensor):
        ctx = BlockStateList.create(encoder.config.num_hidden_layers,
                                    indices.shape[0],
                                    encoder.config.embedding_size,
                                    indices.device,
                                    next(encoder.parameters()).dtype)
        # for i in range(indices.shape[1]):
        #     output: RWKVOutput = encoder(indices[:, :i+1], ctx)
        output: RWKVOutput = encoder(indices, ctx)
        ctx = output.state
        return ctx, output.last_hidden_state
    
    def init_ema(self):
        ema_path = self.ema_local_path
        
        self.ema = EMA(beta=self.diff_config.ema_beta,
                       update_after_step=self.diff_config.update_ema_every,
                       include_online_model=False,
                       model=self.model,
                       ema_path=ema_path)
        
        if ema_path is not None and ema_path.is_file():
            self.ema.load()
        elif self.from_pretrained is not None:
            ema_pretrained = self.from_pretrained.parent / 'ema.pt'
            self.ema.set_path(ema_pretrained)
            self.ema.load()
            self.ema.set_path(ema_path)
        else:
            raise ValueError('Directory of loaded online model checkpoint should have EMA checkpoint too')