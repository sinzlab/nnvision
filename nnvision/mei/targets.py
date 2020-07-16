import torch
from functools import partial


def sum_over_max(output, dim=0):
    return torch.max(output, dim=dim).values.sum()
