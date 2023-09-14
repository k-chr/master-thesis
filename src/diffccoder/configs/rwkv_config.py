from dataclasses import dataclass

from diffccoder.configs.base import BaseConfig

@dataclass
class RWKVConfig(BaseConfig):
    use_ffn_pre: bool = False
    qk_attention: int = 0
    vocab_size: int = 50277
    context_length: int = 1024
    embedding_size: int = 4096
    num_hidden_layers: int = 32
    attention_hidden_size: int | None = None
    intermediate_size: int | None = None
    layer_norm_epsilon: float = 1e-5
    bos_token_id: int = 0
    eos_token_id: int = 0
    rescale_every: int = 6
    tie_word_embeddings: bool = False
    use_cache: bool = True
