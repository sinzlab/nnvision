from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.ndimage import center_of_mass, shift
import torch
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler
from torchvision import transforms

from neuralpredictors.data.samplers import RepeatsBatchSampler

from ..tables.from_mei import MEI


def mei_tsmincne_loader(
    mei_key,
    size=75,
    center=True,
    other_seed_prob=0,
    crop_scale_lo=0.75,
    crop_scale_hi=1,
    crop_ratio=(0.9, 1.1),
    rotation=None,
    batch_size=100,
    num_workers=1,
    shuffle=True,
    initial_center_crop=None,
    positive_only=False,
    toss_mei_threshold=None,
    seed=None,
):
    dataloaders = {}

    if isinstance(mei_key, list) and ("seed" in mei_key[0]):
        print(" ... correcting ...")
        corrected_keys = []
        for k in mei_key:
            k["mei_seed"] = k["seed"]
            corrected_keys.append(k)

        n_meis = len(MEI & corrected_keys)
        print(f" ... {n_meis} MEIs found ...")
    else:
        corrected_keys = mei_key

    print("... fetching MEIs ...")
    meis, units, seeds, scores = (MEI & corrected_keys).fetch(
        "mei",
        "unit_id",
        "mei_seed",
        "score",
        download_path="/data/fetched_from_attach/",
    )
    print("... fetching complete ...")
    meis = torch.stack([torch.load(i).detach().cpu().squeeze() for i in meis])

    print("all meis: ", meis.shape)
    if toss_mei_threshold is not None:
        print("... tossing MEIs ...")
        u_ids = np.unique(units)
        all_rel_scores = []
        for u in u_ids:
            my_filter = units == u
            u_scores = scores[my_filter]
            all_rel_scores.extend(u_scores / u_scores.max())
        all_rel_scores = np.array(all_rel_scores)
        idx_selected = all_rel_scores > toss_mei_threshold
        meis = meis[idx_selected]
        units = units[idx_selected]
        seeds = seeds[idx_selected]
        print("... tossing complete ...", meis.shape)


    train_dataloader = mei_contrastive_loader(
        mei_key=None,
        meis=meis,
        units=units,
        seeds=seeds,
        size=size,
        center=center,
        other_seed_prob=other_seed_prob,
        crop_scale_lo=crop_scale_lo,
        crop_scale_hi=crop_scale_hi,
        crop_ratio=crop_ratio,
        initial_center_crop=initial_center_crop,
        rotation=rotation,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        positive_only=positive_only,
    )
    dataloaders["train_contrastive"] = train_dataloader

    plain_loader = mei_plain_loader(
        mei_key=None,
        meis=meis,
        units=units,
        seeds=seeds,
        size=size,
        center=center,
        initial_center_crop=initial_center_crop,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    dataloaders["validation"] = plain_loader

    plain_dataset = plain_loader.dataset
    sampler = RepeatsBatchSampler(plain_dataset.units)
    dei_batch_loader = DataLoader(plain_dataset, batch_sampler=sampler, num_workers=num_workers)
    dataloaders["dei_batch_loader"] = dei_batch_loader

    return dataloaders


def mei_contrastive_loader(
    mei_key=None,
    meis: torch.Tensor = None,
    units: np.array = None,
    seeds: np.array = None,
    size=75,
    center=True,
    other_seed_prob=0,
    crop_scale_lo=0.75,
    crop_scale_hi=1,
    crop_ratio=(0.9, 1.1),
    rotation=None,
    initial_center_crop=None,
    batch_size=100,
    num_workers=1,
    shuffle=True,
    positive_only=False,
):

    if positive_only:
        dataset = PositiveContrastiveMEIDataset(
            key=mei_key,
            meis=meis,
            units=units,
            seeds=seeds,
            size=size,
            center=center,
            other_seed_prob=other_seed_prob,
            crop_scale_lo=crop_scale_lo,
            crop_scale_hi=crop_scale_hi,
            crop_ratio=crop_ratio,
            rotation=rotation,
            initial_center_crop=initial_center_crop,
        )
    else:
        dataset = ContrastiveMEIDataset(
            key=mei_key,
            meis=meis,
            units=units,
            seeds=seeds,
            size=size,
            center=center,
            other_seed_prob=other_seed_prob,
            crop_scale_lo=crop_scale_lo,
            crop_scale_hi=crop_scale_hi,
            crop_ratio=crop_ratio,
            rotation=rotation,
            initial_center_crop=initial_center_crop,
        )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataloader


def mei_plain_loader(
    mei_key=None,
    meis: torch.Tensor = None,
    units: np.array = None,
    seeds: np.array = None,
    size=75,
    center=True,
    initial_center_crop=None,
    batch_size=100,
    num_workers=1,
    shuffle=False,
):

    dataset = MEIDataset(
        key=mei_key,
        meis=meis,
        units=units,
        seeds=seeds,
        size=size,
        center=center,
        initial_center_crop=initial_center_crop,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataloader


def mei_repeat_loader(
    mei_key=None,
    meis: torch.Tensor = None,
    units: np.array = None,
    seeds: np.array = None,
    size=75,
    center=True,
    initial_center_crop=None,
    num_workers=1,
):

    dataset = MEIDataset(
        key=mei_key,
        meis=meis,
        units=units,
        seeds=seeds,
        size=size,
        center=center,
        initial_center_crop=initial_center_crop,
    )
    if units is None:
        raise ValueError("units must be provided to use repeat loader")
    sampler = RepeatsBatchSampler(units)
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
    )
    return dataloader


class MEIDataset(Dataset):
    """
    A PyTorch Dataset that fetches individual MEIs and returns them as tensors.
    """

    def __init__(
        self,
        key: Dict = None,
        meis: torch.Tensor = None,
        units: np.array = None,
        seeds: np.array = None,
        size=75,
        center=True,
        initial_center_crop=None,
    ):

        # fetch MEIs if they are not provided
        if key is not None:
            if isinstance(key, list) and ("seed" in key[0]):
                print(" ... correcting ...")
                corrected_keys = []
                for k in key:
                    k["mei_seed"] = k["seed"]
                    corrected_keys.append(k)

                n_meis = len(MEI & corrected_keys)
                print(f" ... {n_meis} MEIs found ...")
            else:
                corrected_keys = key

            print("... fetching MEIs ...")
            meis, units, seeds = (MEI & corrected_keys).fetch(
                "mei",
                "unit_id",
                "mei_seed",
                download_path="/data/fetched_from_attach/",
            )
            print("... fetching complete ...")
            meis = torch.stack([torch.load(i).detach().cpu().squeeze() for i in meis])

        n_channels, h, w = meis.shape[1:]

        if center and (n_channels != 2):
            raise ValueError(
                "Centering is only supported for 2-channel MEIs with transparency masks"
            )

        if center:
            centered_meis = []
            for mei in meis:
                img, mask = mei[0].numpy(), mei[1].numpy()
                h, w = center_of_mass(
                    mask,
                )
                centered_image = shift(img, (len(img) // 2 - h, len(img) // 2 - w))
                centered_meis.append(torch.from_numpy(centered_image))
            meis = torch.stack(centered_meis)[:, None, ...]
            print("... centering complete ...")
        else:
            meis = meis[:, [0], :, :]

        self.meis = meis
        self.seeds = seeds
        self.units = units
        assert self.meis.shape[0] == self.units.shape[0] == self.seeds.shape[0]

        transforms_list = []
        if initial_center_crop:
            transforms_list.append(transforms.CenterCrop(initial_center_crop))
        transforms_list.append(transforms.Resize(size))

        self.transform = transforms.Compose(transforms_list)

    def __len__(self):
        return len(self.meis)

    def __getitem__(self, idx):
        unit = self.units[idx]
        mei1 = self.transform(self.meis[idx])

        return mei1, unit


class ContrastiveMEIDataset(Dataset):
    """
    A PyTorch Dataset that fetches individual MEIs and returns two MEIs, one for each random transform
    """

    def __init__(
        self,
        key: Dict,
        meis: torch.Tensor = None,
        units: np.array = None,
        seeds: np.array = None,
        other_seed_prob=0.25,
        size=75,
        crop_scale_lo=0.7,
        crop_scale_hi=1,
        crop_ratio=(0.75, 1.25),
        rotation=None,
        center=False,
        initial_center_crop=None,
    ):
        # fetch MEIs if they are not provided
        if key is not None:
            if isinstance(key, list) and ("seed" in key[0]):
                print(" ... correcting ...")
                corrected_keys = []
                for k in key:
                    k["mei_seed"] = k["seed"]
                    corrected_keys.append(k)

                n_meis = len(MEI & corrected_keys)
                print(f" ... {n_meis} MEIs found ...")
            else:
                corrected_keys = key

            print("... fetching MEIs ...")
            meis, units, seeds = (MEI & corrected_keys).fetch(
                "mei",
                "unit_id",
                "mei_seed",
                download_path="/data/fetched_from_attach/",
            )
            print("... fetching complete ...")
            meis = torch.stack([torch.load(i).detach().cpu().squeeze() for i in meis])
        n_channels, h, w = meis.shape[1:]

        if center and (n_channels != 2):
            raise ValueError(
                "Centering is only supported for 2-channel MEIs with transparency masks"
            )

        if center:
            centered_meis = []
            for mei in meis:
                img, mask = mei[0].numpy(), mei[1].numpy()
                h, w = center_of_mass(
                    mask,
                )
                centered_image = shift(img, (len(img) // 2 - h, len(img) // 2 - w))
                centered_meis.append(torch.from_numpy(centered_image))
            meis = torch.stack(centered_meis)[:, None, ...]
            print("... centering complete ...")
        else:
            meis = meis[:, [0], :, :]

        self.meis = meis
        self.seeds = seeds
        self.units = units
        assert self.meis.shape[0] == self.units.shape[0] == self.seeds.shape[0]

        self.transform = get_transforms(
            size=size,
            crop_scale_lo=crop_scale_lo,
            crop_scale_hi=crop_scale_hi,
            crop_ratio=crop_ratio,
            rotation=rotation,
            initial_center_crop=initial_center_crop,
        )
        self.other_seed_prob = other_seed_prob

        seed_dict = {}
        for seed in np.unique(seeds):
            seed_dict[seed] = torch.from_numpy(seeds != seed).to(torch.bool)
        self.seed_dict = seed_dict

        unit_dict = {}
        for unit in np.unique(units):
            unit_dict[unit] = torch.from_numpy(units == unit).to(torch.bool)
        self.unit_dict = unit_dict

    def __len__(self):
        return len(self.meis)

    def __getitem__(self, idx):
        unit = self.units[idx]
        seed = self.seeds[idx]
        mei1 = self.transform(self.meis[idx])

        if np.random.rand() < self.other_seed_prob:
            other_index = np.random.choice(
                np.where(self.unit_dict[unit] & self.seed_dict[seed])[0]
            )
            mei2 = self.transform(self.meis[other_index])
        else:
            mei2 = self.transform(self.meis[idx])

        return (mei1, mei2), unit


class PositiveContrastiveMEIDataset(ContrastiveMEIDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.unique_units = np.unique(self.units)
        self.unit_where_dict = {}
        for i, unit in enumerate(self.unique_units):
            self.unit_where_dict[i] = np.where(self.unit_dict[unit])[0]
        # now we have a dictionary of len (units) and their corresponding indeces in the dataset to get MEIs later

    def __len__(self):
        return len(self.unique_units)

    def __getitem__(self, idx):
        mei_idx = np.random.choice(self.unit_where_dict[idx])
        mei = self.meis[mei_idx]
        mei1 = self.transform(mei)
        mei2 = self.transform(mei)

        return (mei1, mei2), self.unique_units[idx]


def get_transforms(
    size,
    crop_scale_lo=0.75,
    crop_scale_hi=1,
    crop_ratio=(0.9, 1.1),
    rotation=None,
    initial_center_crop=None,
):
    crop_scale = crop_scale_lo, crop_scale_hi

    transform_list = []
    if initial_center_crop is not None:
        transform_list.append(transforms.CenterCrop(initial_center_crop))

    if rotation is not None:
        transform_list.append(transforms.RandomRotation(rotation))

    transform_list.append(
        transforms.RandomResizedCrop(size=size, scale=crop_scale, ratio=crop_ratio)
    )

    return transforms.Compose(transform_list)
