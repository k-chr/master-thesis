
from dataclasses import dataclass, field

from diffccoder.configs.base import BaseConfig


@dataclass
class TokenizerConfig(BaseConfig):
    vocab: str | dict[str, int] | None = None
    merges: str | dict[tuple[int, int], tuple[int, int]] | None = None
    add_prefix_space: bool = False
    lowercase: bool = False
    dropout: float | None = None
    unicode_normalizer: str | None = None
    continuing_subword_prefix: str | None = None
    end_of_word_suffix: str | None = None
    trim_offsets: bool = False
    vocab_size: int = 50000
    min_frequency: int = 2
    special_tokens: list[str] = field(default_factory=list)