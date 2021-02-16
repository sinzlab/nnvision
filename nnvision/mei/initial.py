from abc import ABC, abstractmethod
from collections import Iterable

import torch
from torch import Tensor, randn
import numpy as np


from mei.initial import InitialGuessCreator
from nnfabrik.main import Dataset
from ..tables.main import Recording


def natural_image_initial(*args, key, img_tier="train", top_index=0, device="cuda"):
    print("finding best natural stimulus as initial image ...")
    dataloaders = (Dataset & key).get_dataloader()
    unit_index = (Recording.Units & key).fetch1("unit_index")
    if img_tier == "test":
        taget_means, unique_images = [],[]
        for batch in dataloaders[img_tier][key["data_key"]]:
            images, responses = batch[:2]
            if len(images.shape) == 5:
                images = images.squeeze(dim=0)
                responses = responses.squeeze(dim=0)
            assert torch.all(torch.eq(images[-1,], images[0,],)), "All images in the batch should be equal"
            unique_images.append(images[0])
            taget_means.append(responses.detach().cpu().numpy()[:, unit_index].mean())
        initial_image = unique_images[np.flipud(np.argsort(taget_means))[top_index]][None, ...]
    else:
        img_idx = torch.argsort(dataloaders["train"][key["data_key"]].dataset[:].targets[:, unit_index]).flipud()[top_index]
        initial_image = dataloaders["train"][key["data_key"]].dataset[img_idx].inputs.to(device)[None, ...]
    return initial_image


class BestNaturalImageInitial(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    _create_initial = staticmethod(natural_image_initial)

    def __init__(self, key, img_tier="train", top_index=1):
        self.key = key
        self.top_index = top_index
        self.img_tier = img_tier

    def __call__(self, *shape, img_tier=None, top_index=None):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        img_tier = self.img_tier if img_tier is None else img_tier
        top_index = self.top_index if top_index is None else top_index
        return self._create_initial(*shape, key=self.key, img_tier=img_tier, top_index=top_index)

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"





