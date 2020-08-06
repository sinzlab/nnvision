import torch
import numpy as np


def mean_of_max(output, max_dim=0, mean_dim=1, gamma=1, normalizing_responses=None):
    if normalizing_responses is not None:
        output = output / torch.from_numpy(normalizing_responses).cuda()
    return torch.max(output, dim=max_dim).values.mean() + gamma * torch.mean(output, dim=mean_dim).min()


def mean(output, normalizing_responses=None):
    if normalizing_responses is not None:
        output = output / torch.from_numpy(normalizing_responses).cuda()
    return output.mean()


def mean_of_random_output_sample(output, n=26):
    output_units = output.shape[1]
    sampled_units = np.random.choice(np.arange(output_units), n, replace=False)
    return torch.mean(output[:, sampled_units])