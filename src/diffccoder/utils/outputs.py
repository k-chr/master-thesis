from dataclasses import dataclass

import torch as t

from diffccoder.utils.generic import ModelOutput


@dataclass  
class TimeMixState:
    shift_state: t.Tensor
    wkv_state: t.Tensor


@dataclass
class ChannelMixState:
    shift_state: t.Tensor


@dataclass
class BlockState:
    time_mix_state: TimeMixState
    channel_mix_state: ChannelMixState


class BlockStateList:

    def __init__(self, shift_states, wkv_states):
        self.wkv_states = wkv_states
        self.shift_states = shift_states

    @staticmethod
    def create(N, B, C, device, dtype):
        result = BlockStateList.empty(N, B, C, device, dtype)
        result.wkv_states[:] = 0
        result.wkv_states[:, :, :, -1] = -1e38
        result.shift_states[:] = 0
        return result
    
    @staticmethod
    def empty_like(state: 'BlockStateList'):
        wkv_states = t.empty_like(state.wkv_states)
        shift_states = t.empty_like(state.shift_states)
        return BlockStateList(shift_states, wkv_states)
    
    @staticmethod
    def empty(N, B, C, device, dtype):
        wkv_states = t.empty((N, B, C, 3),
                                 device=device,
                                 dtype=t.float)
        shift_states = t.empty((N, 2, B, C), device=device, dtype=dtype)
        return BlockStateList(shift_states, wkv_states)

    def __getitem__(self, layer: int):
        return BlockState(
            TimeMixState(self.shift_states[layer, 0], self.wkv_states[layer]),
            ChannelMixState(self.shift_states[layer, 1]))

    def __setitem__(self, layer: int, state: BlockState):
        self.shift_states[layer, 0] = state.time_mix_state.shift_state
        self.wkv_states[layer] = state.time_mix_state.wkv_state
        self.shift_states[layer, 1] = state.channel_mix_state.shift_state
    

@dataclass
class RWKVOutput(ModelOutput):
    """
    Class for the RWKV model outputs.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        state (list of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hiddenum_hidden_layerss)`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    logits: t.FloatTensor = None
    last_hidden_state: t.FloatTensor = None
    state: BlockStateList | None = None
    hidden_states: tuple[t.FloatTensor] | None = None
    attentions: tuple[t.FloatTensor] | None = None
    
    
@dataclass
class DiffusionLosses:
    loss: t.Tensor = None
    mse_loss: t.Tensor = None
    t0_loss: t.Tensor = None
    tT_loss: t.Tensor = None
    decoder_nll: t.Tensor = None
    mse_pre: t.Tensor = None
    
@dataclass
class DiffusionPrediction:
    pred_x_start: t.Tensor
    pred_noise: t.Tensor
    pred_x_prev: t. Tensor
