from abc import ABC, abstractmethod

import numpy as np
import torch as t
from torch import nn

from diffccoder.configs.diffusion_config import DiffusionConfig


class ScheduleSampler(nn.Module, ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = t.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = t.from_numpy(weights_np).float().to(device)
        return indices, weights


class XYUniformSampler(ScheduleSampler):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config 

    @property
    def timesteps(self):
        return self.config.timesteps if self.training else self.config.gen_timesteps


    def weights(self):
        return np.ones([self.config.timesteps + self.config.tgt_len], dtype=np.float64)
    
    def sample(self, batch_size: int, device: t.device, seq_len: t.Tensor | int):
        """
        example:
        num_timesteps = 3, seq_len = 5
        2 3 4 5 6
        1
        0         (0, 7[num_time_steps+seq_len-1])
        """
        
        w = self.weights()

        p = w / np.sum(w)
 
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)  # [bs, ]->(0, time_step + seq_len)
        indices = t.from_numpy(indices_np).long().to(device)

        middle_point = t.stack([
            t.clamp((seq_len - 1) - indices, 0),
            t.clamp(indices - (seq_len - 1), 0)
        ], dim=-1)

        end_point = t.tensor(
            [[int(self.config.end_point_scale * max(seq_len)), self.timesteps - 1]]
        ).repeat(batch_size, 1).type_as(middle_point)

        # the part of padding will be mask
        xs: t.Tensor = t.arange(max(seq_len))
        xs = xs.unsqueeze(0).repeat(batch_size, 1).type_as(middle_point)
        """
        (y - end_y) / (middle_y - end_y) = (x - end_x) / (middle_x - end_x)
        => y = (x - end_x) / (middle_x - end_x) * (middle_y - end_y) + end_y
        """
        end_point = end_point.unsqueeze(-1)
        middle_point = middle_point.unsqueeze(-1)
        ys = (xs.float() - end_point[:, 0].float()
              ).div(middle_point[:, 0].float() - end_point[:, 0].float()
                    ).mul(middle_point[:, 1].float() - end_point[:, 1].float()
                          ).add(end_point[:, 1].float())
        ys = ys.round().clamp(0, self.timesteps - 1).long().to(device)
        
        weights_np = 1 / (len(p) * p[indices_np])
        weights = t.from_numpy(weights_np).float().to(device)
        weights = weights.unsqueeze(-1).repeat(1, max(seq_len))
        return ys, weights