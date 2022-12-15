from abc import ABC, abstractmethod
from collections import Iterable

import os
import numpy as np
import torch
from torch import Tensor, randn

from ..tables.main import Recording
from ..tables.from_mei import MEI
from nnfabrik.main import Dataset
from nnfabrik import builder
from mei.initial import InitialGuessCreator, RandomNormal

fetch_download_path = os.environ.get("FETCH_DOWNLOAD_PATH", "/data/fetched_from_attach")


def AllGray(RandomNormal):
    _create_random_tensor = torch.zeros


def natural_image_initial(*args, key, img_tier="train", top_index=0, device="cuda"):
    print("finding best natural stimulus as initial image ...")
    dataloaders = (Dataset & key).get_dataloader()
    unit_index = (Recording.Units & key).fetch1("unit_index")
    if img_tier == "test":
        taget_means, unique_images = [], []
        for batch in dataloaders[img_tier][key["data_key"]]:
            images, responses = batch[:2]
            if len(images.shape) == 5:
                images = images.squeeze(dim=0)
                responses = responses.squeeze(dim=0)
            assert torch.all(
                torch.eq(
                    images[
                        -1,
                    ],
                    images[
                        0,
                    ],
                )
            ), "All images in the batch should be equal"
            unique_images.append(images[0])
            taget_means.append(responses.detach().cpu().numpy()[:, unit_index].mean())
        initial_image = unique_images[np.flipud(np.argsort(taget_means))[top_index]][
            None, ...
        ]
    else:
        img_idx = torch.argsort(
            dataloaders["train"][key["data_key"]].dataset[:].targets[:, unit_index]
        ).flipud()[top_index]
        initial_image = (
            dataloaders["train"][key["data_key"]]
            .dataset[img_idx]
            .inputs.to(device)[None, ...]
        )
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
        return self._create_initial(
            *shape, key=self.key, img_tier=img_tier, top_index=top_index
        )

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"


class RandomNormalCenterRing(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    _create_random_tensor = randn

    def __init__(self, key, mask_thres_for_ring=0.3):
        src_method_fn = key["src_method_fn"]
        unit_id = key["unit_id"]

        inner_ensemble_hash = key["inner_ensemble_hash"]
        outer_ensemble_hash = key["outer_ensemble_hash"]
        inner_method_hash = key["inner_method_hash"]
        outer_method_hash = key["outer_method_hash"]
        unit_id = key["unit_id"]

        outer_mei_path = (
            MEI
            & dict(method_fn=src_method_fn)
            & dict(ensemble_hash=outer_ensemble_hash)
            & dict(method_hash=outer_method_hash)
            & dict(unit_id=unit_id)
        ).fetch1("mei", download_path=fetch_download_path)
        inner_mei_path = (
            MEI
            & dict(method_fn=src_method_fn)
            & dict(ensemble_hash=inner_ensemble_hash)
            & dict(method_hash=inner_method_hash)
            & dict(unit_id=unit_id)
        ).fetch1("mei", download_path=fetch_download_path)

        outer_mei = torch.load(outer_mei_path)
        inner_mei = torch.load(inner_mei_path)

        self.centerimg = inner_mei[0][0]
        self.ring_mask = (outer_mei[0][1] - inner_mei[0][1] > mask_thres_for_ring) * 1

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        randinitial = self._create_random_tensor(*shape)
        initial = randinitial * self.ring_mask + self.centerimg
        return initial

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"


class RandomNormalNullChannel(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    _create_random_tensor = randn

    def __init__(self, null_channel, null_value=0):
        self.null_channel = null_channel
        self.null_value = null_value

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        inital = self._create_random_tensor(*shape)
        inital[:, self.null_channel, ...] = self.null_value
        return inital

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"


class RandomNormalNonlinearCenterRing(InitialGuessCreator):

    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    ### put cropped nonlinear MEI in center
    _create_random_tensor = randn

    def __init__(self, key, mask_thres_for_ring=0.3):
        src_method_fn = key["src_method_fn"]
        unit_id = key["unit_id"]

        inner_ensemble_hash = key["inner_ensemble_hash"]
        outer_ensemble_hash = key["outer_ensemble_hash"]
        inner_method_hash = key["inner_method_hash"]
        outer_method_hash = key["outer_method_hash"]
        unit_id = key["unit_id"]

        outer_mei_path = (
            MEI
            & dict(method_fn=src_method_fn)
            & dict(ensemble_hash=outer_ensemble_hash)
            & dict(method_hash=outer_method_hash)
            & dict(unit_id=unit_id)
        ).fetch1("mei", download_path=fetch_download_path)
        inner_mei_path = (
            MEI
            & dict(method_fn=src_method_fn)
            & dict(ensemble_hash=inner_ensemble_hash)
            & dict(method_hash=inner_method_hash)
            & dict(unit_id=unit_id)
        ).fetch1("mei", download_path=fetch_download_path)

        outer_mei = torch.load(outer_mei_path)
        inner_mei = torch.load(inner_mei_path)

        self.centerimg = outer_mei[0][0]
        self.ring_mask = (outer_mei[0][1] - inner_mei[0][1] > mask_thres_for_ring) * 1

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        randinitial = self._create_random_tensor(*shape)
        initial = randinitial * self.ring_mask + self.centerimg
        return initial

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"


class RandomNormalRing(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    """center_type could be 'noise', 'grey','light','null'，'naturalimg' """
    _create_random_tensor = randn

    def __init__(
        self, key, mask_thres_for_ring=0.3, center_type="null", center_norm=20, img_id=0
    ):
        src_method_fn = key["src_method_fn"]
        unit_id = key["unit_id"]

        inner_ensemble_hash = key["inner_ensemble_hash"]
        outer_ensemble_hash = key["outer_ensemble_hash"]
        inner_method_hash = key["inner_method_hash"]
        outer_method_hash = key["outer_method_hash"]
        unit_id = key["unit_id"]

        outer_mei_path = (
            MEI
            & dict(method_fn=src_method_fn)
            & dict(ensemble_hash=outer_ensemble_hash)
            & dict(method_hash=outer_method_hash)
            & dict(unit_id=unit_id)
        ).fetch1("mei", download_path=fetch_download_path)
        inner_mei_path = (
            MEI
            & dict(method_fn=src_method_fn)
            & dict(ensemble_hash=inner_ensemble_hash)
            & dict(method_hash=inner_method_hash)
            & dict(unit_id=unit_id)
        ).fetch1("mei", download_path=fetch_download_path)

        outer_mei = torch.load(outer_mei_path)
        inner_mei = torch.load(inner_mei_path)

        self.ring_mask = (outer_mei[0][1] - inner_mei[0][1] > mask_thres_for_ring) * 1
        self.inner_mask = (inner_mei[0][1] > 0.3) * 1
        self.center_type = center_type
        self.center_norm = center_norm
        self.img_id = img_id  # for the case with natural image in center

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        randinitial = self._create_random_tensor(*shape)
        if self.center_type == "noise":
            centerimg = randinitial * self.inner_mask
        if self.center_type == "grey":
            centerimg = torch.ones(*shape) * -3 * self.inner_mask
        if self.center_type == "light":
            centerimg = torch.ones(*shape) * 3 * self.inner_mask
        if self.center_type == "null":
            centerimg = torch.ones(*shape) * 0
        if self.center_type == "naturalimg":
            dataset_fn = "nndichromacy.datasets.static_loaders"
            dataset_config = {
                "paths": [
                    "/data/mouse/toliaslab/static/static22564-3-12-GrayImageNet-50ba42e98651ac33562ad96040d88513.zip"
                ],
                "normalize": True,
                "include_behavior": False,
                "batch_size": 128,
                "exclude": None,
                "file_tree": True,
                "scale": 1,
            }
            dataloaders = builder.get_data(dataset_fn, dataset_config)
            images = []
            for i, j in dataloaders["train"]["22564-3-12"]:
                images.append(i.squeeze().data)
            centerimg = torch.vstack(images)[self.img_id].cpu() * self.inner_mask

        if self.center_type != "null":
            centerimg = centerimg * (
                self.center_norm / torch.norm(centerimg)
            )  # control centerimg contrast

        initial = centerimg + randinitial * self.ring_mask

        return initial

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"


class RandomNormalSurround(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    """center_type could be 'noise', 'grey','light','null' """
    _create_random_tensor = randn

    def __init__(self, key, mask_thres=0.3):
        src_method_fn = key["src_method_fn"]
        unit_id = key["unit_id"]

        inner_ensemble_hash = key["inner_ensemble_hash"]
        inner_method_hash = key["inner_method_hash"]
        unit_id = key["unit_id"]

        inner_mei_path = (
            MEI
            & dict(method_fn=src_method_fn)
            & dict(ensemble_hash=inner_ensemble_hash)
            & dict(method_hash=inner_method_hash)
            & dict(unit_id=unit_id)
        ).fetch1("mei", download_path=fetch_download_path)

        # outer_mei=torch.load(outer_mei_path)
        inner_mei = torch.load(inner_mei_path)

        self.center_mask = (inner_mei[0][1] > mask_thres) * 1
        # self.center_type = center_type
        # self.center_norm = center_norm
        self.centerimg = inner_mei[0][0]

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        randinitial = self._create_random_tensor(*shape)
        initial = self.centerimg + randinitial * (1 - self.center_mask)

        return initial

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"


class RandomNormalSelectChannels(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    _create_random_tensor = randn

    def __init__(self, selected_channels, selected_values):
        if not isinstance(selected_channels, Iterable):
            selected_channels = selected_channels

        if not isinstance(selected_values, Iterable):
            selected_values = selected_values

        self.selected_channels = selected_channels
        self.selected_values = selected_values

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        inital = self._create_random_tensor(*shape)
        for channel, value in zip(self.selected_channels, self.selected_values):
            inital[:, channel, ...] = value

        return inital

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"
