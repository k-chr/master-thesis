from abc import ABC, abstractmethod

import numpy as np
import torch as t
import torch.distributed as dist

from diffccoder.configs.diffusion_config import DiffusionConfig

class ScheduleSampler(ABC):
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

    def sample(self, batch_size, device, step_ratio=None):
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
    def __init__(self, config: DiffusionConfig, num_timesteps, history_per_term=10, uniform_prob=0.001):
        self.config = config
        self.num_timesteps = num_timesteps
        self.scale = getattr(self.config, "end_point_scale", 2.0)

        if config.loss_aware:
            self.history_per_term = history_per_term
            self.uniform_prob = uniform_prob
            self._loss_history = np.zeros(
                [self.num_timesteps, history_per_term], dtype=np.float64
            )
            self._loss_counts: np.ndarray = np.zeros([num_timesteps], dtype=np.int)
            

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.num_timesteps + self.config.tgt_len], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights
    
    def update_with_local_losses(self, local_ts: t.Tensor, local_losses: t.Tensor):
        batch_sizes = [
            t.tensor([0], dtype=t.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            t.tensor([len(local_ts)], dtype=t.int32, device=local_ts.device),
        )

        # Pad all_gather batches to be the maximum batch size.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [t.zeros(max_bs, self.config.tgt_len).to(local_ts) for _ in batch_sizes]
        loss_batches = [t.zeros(max_bs, self.config.tgt_len).to(local_losses) for _ in batch_sizes]
        
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        
        timesteps = [
            x.float().max().round().long().item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        
        losses = [x.float().max().item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)  # List -> len=bs*world_size, element=tensor[tgt_len]
    
    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return np.all(self._loss_counts == self.history_per_term)

    def sample(self, batch_size: int, device: t.device, seq_len: t.Tensor | int, step_ratio: float | None =None):
        """
        example:
        num_timesteps = 3, seq_len = 5
        2 3 4 5 6
        1
        0         (0, 7[num_time_steps+seq_len-1])
        """
        
        if not self.config.pred_len:
            assert t.all((seq_len == self.config.tgt_len))

        if self.config.loss_aware:
            w = self.weights()
        else:
            w = np.ones([max(seq_len) + self.num_timesteps])

        p = w / np.sum(w)
 
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)  # [bs, ]->(0, time_step + seq_len)
        indices = t.from_numpy(indices_np).long().to(device)

        middle_point = t.stack([
            t.clamp((seq_len - 1) - indices, 0),
            t.clamp(indices - (seq_len - 1), 0)
        ], dim=-1)

        if self.config.pred_len:
            end_point_x = (self.scale * seq_len).type_as(middle_point)
            end_point_y = t.tensor(self.num_timesteps - 1).repeat(batch_size).type_as(middle_point)
            end_point = t.stack([end_point_x, end_point_y], dim=-1)
        else:
            end_point = t.tensor(
                [[int(self.scale * max(seq_len)), self.num_timesteps - 1]]
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
        ys = ys.round().clamp(0, self.num_timesteps - 1).long().to(device)
        
        weights_np = 1 / (len(p) * p[indices_np])
        weights = t.from_numpy(weights_np).float().to(device)
        weights = weights.unsqueeze(-1).repeat(1, max(seq_len))
        return ys, weights