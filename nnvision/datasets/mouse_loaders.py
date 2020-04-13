from collections import OrderedDict
from itertools import zip_longest
import warnings

import numpy as np
from mlutils.data.datasets import StaticImageSet, FileTreeDataset
from mlutils.data.transforms import Subsample, ToTensor, NeuroNormalizer, AddBehaviorAsChannels, SelectInputChannel
from mlutils.data.samplers import SubsetSequentialSampler
from nnfabrik.utility.nn_helpers import set_random_seed
from .utility import get_oracle_dataloader
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def mouse_static_loader(path=None,
                        batch_size=None,
                        seed=None,
                        areas=None,
                        layers=None,
                        tier=None,
                        neuron_ids=None,
                        get_key=False,
                        cuda=True,
                        normalize=True,
                        include_behavior=False,
                        exclude=None,
                        select_input_channel=None,
                        toy_data=False,
                        file_tree=False,
                        return_test_sampler=False,
                        oracle_condition=None,
                        **kwargs):
    """
    returns a single data

    Args:
        path (list): list of path(s) for the dataset(s)
        batch_size (int): batch size.
        seed (int, optional): random seed for images. Defaults to None.
        areas (list, optional): the visual area.
        layers (list, optional): the layer from visual area.
        tier (str, optional): tier is a placeholder to specify which set of images to pick for train, val, and test loader. Defaults to None.
        neuron_ids (list, optional): select neurons by their ids. neuron_ids and path should be of same length. Defaults to None.
        get_key (bool, optional): whether to retun the data key, along with the dataloaders. Defaults to False.
        cuda (bool, optional): whether to place the data on gpu or not. Defaults to True.

    Returns:
        if get_key is False returns a dictionary of dataloaders for one dataset, where the keys are 'train', 'validation', and 'test'.
        if get_key is True it also the data_key (as the first output) followed by the dalaoder dictionary.

    """
    if ("paths" in kwargs) and (path is None):
        paths = kwargs["paths"]
        path = paths[0] if len(paths) == 1 else None

    if file_tree:
        dat = FileTreeDataset(path, 'images', 'responses', 'behavior') if include_behavior else FileTreeDataset(path,
                                                                                                                'images',
                                                                                                                'responses')
    else:
        dat = StaticImageSet(path, 'images', 'responses', 'behavior') if include_behavior else StaticImageSet(path,
                                                                                                              'images',
                                                                                                              'responses')

    assert (include_behavior and select_input_channel) is False, \
        "Selecting an Input Channel and Adding Behavior can not both be true"

    if toy_data:
        dat.transforms = [ToTensor(cuda)]
    else:
        # The permutation MUST be added first and the conditions below MUST NOT be based on the original order

        # specify condition(s) for sampling neurons. If you want to sample specific neurons define conditions that would effect idx
        conds = np.ones(len(dat.neurons.area), dtype=bool)
        if areas is not None:
            conds &= (np.isin(dat.neurons.area, areas))
        if layers is not None:
            conds &= (np.isin(dat.neurons.layer, layers))
        if neuron_ids is not None:
            conds &= (np.isin(dat.neurons.unit_ids, neuron_ids))

        idx = np.where(conds)[0]
        more_transforms = [Subsample(idx), ToTensor(cuda)]
        if normalize:
            more_transforms.insert(0, NeuroNormalizer(dat, exclude=exclude))

        if include_behavior:
            more_transforms.insert(0, AddBehaviorAsChannels())

        if select_input_channel is not None:
            more_transforms.insert(0, SelectInputChannel(select_input_channel))

    dat.transforms.extend(more_transforms)

    if return_test_sampler:
        dataloader = get_oracle_dataloader(dat, toy_data=toy_data, oracle_condition=oracle_condition)
        return dataloader

    # subsample images
    dataloaders = {}
    keys = [tier] if tier else ['train', 'validation', 'test']
    for tier in keys:

        if seed is not None:
            set_random_seed(seed)
            # torch.manual_seed(img_seed)

        # sample images
        subset_idx = np.where(dat.trial_info.tiers == tier)[0] if file_tree else np.where(dat.tiers == tier)[0]
        sampler = SubsetRandomSampler(subset_idx) if tier == 'train' else SubsetSequentialSampler(subset_idx)
        dataloaders[tier] = DataLoader(dat, sampler=sampler, batch_size=batch_size)

    # create the data_key for a specific data path
    data_key = path.split('static')[-1].split('.')[0].replace('preproc', '')
    return (data_key, dataloaders) if get_key else dataloaders


def mouse_static_loaders(paths,
                         batch_size,
                         seed=None,
                         areas=None,
                         layers=None,
                         tier=None,
                         neuron_ids=None,
                         cuda=True,
                         normalize=False,
                         include_behavior=False,
                         exclude=None,
                         select_input_channel=None,
                         toy_data=False,
                         file_tree=False,
                         **kwargs):
    """
    Returns a dictionary of dataloaders (i.e., trainloaders, valloaders, and testloaders) for >= 1 dataset(s).

    Args:
        paths (list): list of path(s) for the dataset(s)
        batch_size (int): batch size.
        seed (int, optional): random seed for images. Defaults to None.
        areas (str, optional): the visual area. Defaults to 'V1'.
        layers (str, optional): the layer from visual area. Defaults to 'L2/3'.
        tier (str, optional): tier is a placeholder to specify which set of images to pick for train, val, and test loader. Defaults to None.
        neuron_ids ([type], optional): select neurons by their ids. Defaults to None.
        cuda (bool, optional): whether to place the data on gpu or not. Defaults to True.

    Returns:
        dict: dictionary of dictionaries where the first level keys are 'train', 'validation', and 'test', and second level keys are data_keys.
    """

    neuron_ids = neuron_ids if neuron_ids is not None else []
    layers = layers if layers is not None else ('L2/3',)
    areas = areas if areas is not None else ('V1',)
    dls = OrderedDict({})
    keys = [tier] if tier else ['train', 'validation', 'test']
    for key in keys:
        dls[key] = OrderedDict({})

    for path, neuron_id in zip_longest(paths, neuron_ids, fillvalue=None):
        data_key, loaders = mouse_static_loader(path, batch_size, seed=seed,
                                                areas=areas, layers=layers, cuda=cuda,
                                                tier=tier, get_key=True, neuron_ids=neuron_id,
                                                normalize=normalize, include_behavior=include_behavior,
                                                exclude=exclude, select_input_channel=select_input_channel,
                                                toy_data=toy_data, file_tree=file_tree)
        for k in dls:
            dls[k][data_key] = loaders[k]

    return dls


def mouse_shared_static_loaders(paths,
                                batch_size,
                                seed=None,
                                areas=None,
                                layers=None,
                                tier=None,
                                neuron_ids=None,
                                cuda=True,
                                normalize=False,
                                include_behavior=False,
                                exclude=None,
                                select_input_channel=None,
                                toy_data=False,
                                **kwargs):
    """
    Returns a dictionary of dataloaders (i.e., trainloaders, valloaders, and testloaders) for >= 1 dataset(s).

    Args:
        paths (list): list of path(s) for the dataset(s)
        batch_size (int): batch size.
        seed (int, optional): random seed for images. Defaults to None.
        areas (str, optional): the visual area. Defaults to 'V1'.
        layers (str, optional): the layer from visual area. Defaults to 'L2/3'.
        tier (str, optional): tier is a placeholder to specify which set of images to pick for train, val, and test loader. Defaults to None.
        neuron_ids ([type], optional): select neurons by their ids. Defaults to None.
        cuda (bool, optional): whether to place the data on gpu or not. Defaults to True.

    Returns:
        dict: dictionary of dictionaries where the first level keys are 'train', 'validation', and 'test', and second level keys are data_keys.
    """

    # Collect overlapping multi matches

    if len(paths) > 1:
        warnings.warn("Only one dataset was specified in 'paths'. When using the 'mouse_shared_loaders', more than one dataset"
                      "has to be passed. Returning Dataloaders as if the function 'mouse_static_loaders' has been called.")
        dls = mouse_static_loaders(paths=paths, batch_size=batch_size, seed=seed, areas=areas, layers=layers,
                                   tier=tier, neuron_ids=neuron_ids, cuda=cuda, normalize=normalize, include_behavior=include_behavior,
                                   exclude=exclude, select_input_channel=select_input_channel, toy_data=toy_data, **kwargs)
        return dls

    multi_unit_ids = []
    per_data_set_ids = []
    match_set = None
    for path in paths:
        dat = FileTreeDataset(path, 'responses')
        multi_unit_ids.append(dat.neurons.multi_match_id)
        per_data_set_ids.append(dat.neurons.unit_ids)
        if match_set is None:
            match_set = set(multi_unit_ids[-1])
        else:
            match_set &= set(multi_unit_ids[-1])
    match_set -= {-1}  # remove the unmatched neurons
    match_set = np.array(list(match_set))

    # get unit_ids of intersecting multi-unit ids
    all_set_neurons = [pdsi[np.isin(munit_ids, match_set)] for munit_ids, pdsi in zip(multi_unit_ids, per_data_set_ids)]

    if neuron_ids is None:
        neuron_ids = all_set_neurons
    else:
        neuron_ids = [list(set(unit_ids) & set(ni)) for unit_ids, ni in zip(all_set_neurons, neuron_ids)]

    # generate single dataloaders
    dls = OrderedDict({})
    keys = [tier] if tier else ['train', 'validation', 'test']
    for key in keys:
        dls[key] = OrderedDict({})

    for path, neuron_id in zip(paths, neuron_ids):
        data_key, loaders = mouse_static_loader(path, batch_size, seed=seed,
                                                areas=areas, layers=layers, cuda=cuda,
                                                tier=tier, get_key=True, neuron_ids=neuron_id,
                                                normalize=normalize, include_behavior=include_behavior,
                                                exclude=exclude, select_input_channel=select_input_channel,
                                                toy_data=toy_data, file_tree=True)
        for k in dls:
            dls[k][data_key] = loaders[k]

    return dls
