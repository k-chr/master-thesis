from functools import partial
import os
import pathlib
from pathlib import Path
import platform

from lightning import LightningModule
from tokenizers import Tokenizer
import torch as t
from loguru import logger

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
                 out: Path = None,
                 target_length: int = 1024) -> None:
        super().__init__()
        # self.infer_logger = logger.opt()
        # self.infer_logger.remove()
        # self.infer_logger.add(f'~/{diff_config.inference_sampler.name.lower()}_indices.log')
        self.rwkv_config = rwkv_config
        
        self.diff_config = diff_config
        self.from_pretrained = from_pretrained
        self.ema_local_path = ema_local_path
        self.tokenizer = tokenizer
        self.target_length = target_length
        os.environ['CTX_LEN'] = str(rwkv_config.context_length)
        os.environ['USE_CACHE'] = str(int(rwkv_config.use_cache and not self.training))
        
        if platform.system() != "Windows":
            pathlib.WindowsPath = pathlib.PosixPath
        else:
            pathlib.PosixPath = pathlib.WindowsPath
            
        encoder = RWKV(self.rwkv_config)
        decoder = DIFF_RWKV(diff_config=self.diff_config, rwkv_config=self.rwkv_config)
                   
        self.model = GaussianDiffusion(encoder=encoder, model=decoder, config=self.diff_config)
        
        if not (logs := out / 'logs').is_dir():
            logs.mkdir(exist_ok=True)
        self.log_dir = logs
        
    def configure_model(self) -> None:            
        if self.from_pretrained is not None and self.from_pretrained.exists():
            logger.info(f'Loading from: {self.from_pretrained}')
            self.load_state_dict(t.load(self.from_pretrained, map_location='cpu')['state_dict'], strict=False)

        self.init_ema()

        self._ddp_params_and_buffers_to_ignore = [f'ema.{key}' for key in self.ema.state_dict().keys()]
        
        self.sampler = InferenceSampler.create(self.diff_config, self.model if not self.diff_config.use_ema_at_infer else self.ema.ema_model)

    def predict_step(self, batch: t.Tensor, batch_idx: int) -> list[str] | str:
        maybe_samples = self._process_batch(batch)
        decoded = []
        if not self.diff_config.return_all_timesteps:
            maybe_samples = [maybe_samples]
        for i, sample in enumerate(maybe_samples):
            logits = self.sampler.diffusion.model.get_logits(sample).cpu()
            # logits[:, :, 0] = -9999
            # logits[:, :, 1] = -9999
            
            logger.info(f'{logits[logits.isnan()].shape, logits.shape}')
            # probs = F.softmax(logits, -1)
            # probs = probs[0].pow(1)
            #ids = t.argmax(logits, dim=-1)
            #print(ids.shape)
            with (self.log_dir / f'sample_{i}_batch_{batch_idx}_top_k_{self.diff_config.inference_sampler.name.lower()}.log').open('w', encoding='utf-8') as log_file:
                top_k = t.topk(logits, 100, -1)
                for tok in range(top_k.indices.shape[1]):
                    log_file.write(f'Token No. {tok}: candidates: \n')
                    for candidate_idx, candidate_value in zip(top_k.indices[:, tok, :].squeeze().tolist(), top_k.values[:, tok, :].squeeze().tolist()):
                        log_file.write(f'\tIDX: {candidate_idx}; LOGIT: {candidate_value}; VALUE: {self.tokenizer.decode([candidate_idx])}||\n')
                    else:
                        log_file.write('\n')
                    
                    
            sample_id_tensor = t.argmax(logits, dim=-1) # t.multinomial(probs/probs.sum(-1, keepdim=True), num_samples=1).flatten()#   
            logger.info((sample_id_tensor.sum()))
            
            decoded.append(self.tokenizer.decode_batch(sample_id_tensor.tolist()))
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
                                   denoised_fn=partial(denoised_fn_round, self.sampler.diffusion.model.emb))#None)#
        
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
        #     ctx = output.state
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