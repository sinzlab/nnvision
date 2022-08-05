import warnings
import torch
import torch.utils.data as utils
import numpy as np
import pickle
#from retina.retina import warp_image
from collections import namedtuple, Iterable
import os
from pathlib import Path
from neuralpredictors.data.samplers import RepeatsBatchSampler
from .utility import get_validation_split, ImageCache, get_cached_loader, get_cached_loader_extended, get_fraction_of_training_images, get_crop_from_stimulus_location
from nnfabrik.utility.nn_helpers import set_random_seed, get_dims_for_loader_dict
from neuralpredictors.utils import get_module_output
from nnfabrik.utility.dj_helpers import make_hash
import scipy

def monkey_static_loader(dataset,
                         neuronal_data_files,
                         image_cache_path,
                         batch_size=64,
                         seed=None,
                         train_frac=0.8,
                         subsample=1,
                         crop=((96, 96), (96, 96)),
                         scale=1.,
                         time_bins_sum=tuple(range(12)),
                         avg=False,
                         image_file=None,
                         return_data_info=False,
                         store_data_info=True,
                         image_frac=1.,
                         image_selection_seed=None,
                         randomize_image_selection=True,
                         img_mean=None,
                         img_std=None,
                         stimulus_location=None,
                         monitor_scaling_factor=4.57,
                         include_prev_image=False,
                         include_trial_id=False,
                         ):
    """
    Function that returns cached dataloaders for monkey ephys experiments.

     creates a nested dictionary of dataloaders in the format
            {'train' : dict_of_loaders,
             'validation'   : dict_of_loaders,
            'test'  : dict_of_loaders, }

        in each dict_of_loaders, there will be  one dataloader per data-key (refers to a unique session ID)
        with the format:
            {'data-key1': torch.utils.data.DataLoader,
             'data-key2': torch.utils.da.DataLoader, ... }

    requires the types of input files:
        - the neuronal data files. A list of pickle files, with one file per session
        - the image file. The pickle file that contains all images.
        - individual image files, stored as numpy array, in a subfolder

    Args:
        dataset: a string, identifying the Dataset:
            'PlosCB19_V1', 'CSRF19_V1', 'CSRF19_V4'
            This string will be parsed by a datajoint table

        neuronal_data_files: a list paths that point to neuronal data pickle files
        image_file: a path that points to the image file
        image_cache_path: The path to the cached images
        batch_size: int - batch size of the dataloaders
        seed: int - random seed, to calculate the random split
        train_frac: ratio of train/validation images
        subsample: int - downsampling factor
        crop: int or tuple - crops x pixels from each side. Example: Input image of 100x100, crop=10 => Resulting img = 80x80.
            if crop is tuple, the expected input is a list of tuples, the specify the exact cropping from all four sides
                i.e. [(crop_top, crop_bottom), (crop_left, crop_right)]
        scale: float or integer - up-scale or down-scale via interpolation hte input images (default= 1)
        time_bins_sum: sums the responses over x time bins.
        avg: Boolean - Sums oder Averages the responses across bins.

    Returns: nested dictionary of dataloaders
    """

    dataset_config = locals()

    # initialize dataloaders as empty dict
    dataloaders = {'train': {}, 'validation': {}, 'test': {}}

    if not isinstance(time_bins_sum, Iterable):
        time_bins_sum = tuple(range(time_bins_sum))

    if isinstance(crop, int):
        crop = [(crop, crop), (crop, crop)]

    if stimulus_location is not None:
        crop = get_crop_from_stimulus_location(stimulus_location, crop, monitor_scaling_factor=monitor_scaling_factor)

    if not isinstance(image_frac, Iterable):
        image_frac = [image_frac for i in neuronal_data_files]

    # clean up image path because of legacy folder structure
    image_cache_path = image_cache_path.split('individual')[0]

    # Load image statistics if present
    stats_filename = make_hash(dataset_config)
    stats_path = os.path.join(image_cache_path, 'statistics/', stats_filename)

    # Get mean and std
    if os.path.exists(stats_path):
        with open(stats_path, "rb") as pkl:
            data_info = pickle.load(pkl)
        if return_data_info:
            return data_info
        img_mean = list(data_info.values())[0]["img_mean"]
        img_std = list(data_info.values())[0]["img_std"]
        
        # Initialize cache
        cache = ImageCache(path=image_cache_path, subsample=subsample, crop=crop, scale=scale, img_mean= img_mean, img_std=img_std, transform=True, normalize=True)
    else:

        if img_mean is not None:
            cache = ImageCache(path=image_cache_path, subsample=subsample, crop=crop, scale=scale, img_mean=img_mean,
                               img_std=img_std, transform=True, normalize=True)
        else:
            # Initialize cache with no normalization
            cache = ImageCache(path=image_cache_path, subsample=subsample, crop=crop, scale=scale, transform=True, normalize=False)

            # Compute mean and std of transformed images and zscore data (the cache wil be filled so first epoch will be fast)
            cache.zscore_images(update_stats=True)
            img_mean = cache.img_mean
            img_std  = cache.img_std
    
    
    n_images = len(cache)
    data_info = {}

    # set up parameters for the different dataset types
    if dataset == 'PlosCB19_V1':
        # for the "Amadeus V1" Dataset, recorded by Santiago Cadena, there was no specified test set.
        # instead, the last 20% of the dataset were classified as test set. To make sure that the test set
        # of this dataset will always stay identical, the `train_test_split` value is hardcoded here.
        train_test_split = 0.8
        image_id_offset = 1
    else:
        train_test_split = 1
        image_id_offset = 0

    all_train_ids, all_validation_ids = get_validation_split(n_images=n_images * train_test_split,
                                                             train_frac=train_frac,
                                                             seed=seed)
    
    # cycling through all datafiles to fill the dataloaders with an entry per session
    for i, datapath in enumerate(neuronal_data_files):

        with open(datapath, "rb") as pkl:
            raw_data = pickle.load(pkl)

        subject_ids = raw_data["subject_id"]
        data_key = str(raw_data["session_id"])
        responses_train = raw_data["training_responses"].astype(np.float32)
        responses_test = raw_data["testing_responses"].astype(np.float32)
        training_image_ids = raw_data["training_image_ids"] - image_id_offset
        testing_image_ids = raw_data["testing_image_ids"] - image_id_offset

        if include_prev_image and not "training_prior_image_ids" in raw_data:
            raise ValueError("No previous image IDs present in the provided data files. Set 'include_prev_image' to False")

        if include_prev_image:
            prev_training_image_ids = raw_data["training_prior_image_ids"] - image_id_offset
            prev_testing_image_ids = raw_data["testing_prior_image_ids"] - image_id_offset

        if dataset != 'PlosCB19_V1':
            if len(responses_test.shape) != 3:
                responses_test = responses_test[None, ...]
                responses_train = responses_train[None, ...]
                # correct the shape of the responses for a session that was exported incorrectly
                if data_key != '3653663964522':
                    warnings.warn("Pickle file with invalid response shape detected")

            responses_test = responses_test.transpose((2, 0, 1))
            responses_train = responses_train.transpose((2, 0, 1))

            if time_bins_sum is not None:  # then average over given time bins
                responses_train = (np.mean if avg else np.sum)(responses_train[:, :, time_bins_sum], axis=-1)
                responses_test = (np.mean if avg else np.sum)(responses_test[:, :, time_bins_sum], axis=-1)

        if image_frac[i] < 1:
            if randomize_image_selection:
                image_selection_seed = int(image_selection_seed*image_frac[i])
            idx_out = get_fraction_of_training_images(image_ids=training_image_ids, fraction=image_frac[i], seed=image_selection_seed)
            training_image_ids = training_image_ids[idx_out]
            responses_train = responses_train[idx_out]

        train_idx = np.isin(training_image_ids, all_train_ids)
        val_idx = np.isin(training_image_ids, all_validation_ids)

        responses_val = responses_train[val_idx]
        responses_train = responses_train[train_idx]

        validation_image_ids = training_image_ids[val_idx]
        training_image_ids = training_image_ids[train_idx]

        if include_prev_image:
            prev_validation_image_ids = prev_training_image_ids[val_idx]
            prev_training_image_ids = prev_training_image_ids[train_idx]

        if include_trial_id:
            test_trial_ids = scipy.stats.zscore(range(len(testing_image_ids)))
            all_train_trial_ids = scipy.stats.zscore(range(len(training_image_ids) + len(validation_image_ids)))
            train_trial_ids = all_train_trial_ids[train_idx]
            val_trial_ids = all_train_trial_ids[val_idx]

        args_train = [training_image_ids, responses_train]
        args_val = [validation_image_ids, responses_val]
        args_test = [testing_image_ids, responses_test]

        if include_prev_image:
            args_train.insert(1, prev_training_image_ids)
            args_val.insert(1, prev_validation_image_ids)
            args_test.insert(1, prev_testing_image_ids)

        if include_trial_id:
            args_train.insert(1 + include_prev_image, train_trial_ids)
            args_val.insert(1 + include_prev_image, val_trial_ids)
            args_test.insert(1 + include_prev_image, test_trial_ids)

        train_loader = get_cached_loader(*args_train,
                                         batch_size=batch_size,
                                         image_cache=cache,
                                         include_trial_id=include_trial_id,
                                         include_prev_image=include_prev_image,)

        val_loader = get_cached_loader(*args_val,
                                       batch_size=batch_size,
                                       image_cache=cache,
                                       include_trial_id=include_trial_id,
                                       include_prev_image=include_prev_image,)

        test_loader = get_cached_loader(*args_test,
                                        batch_size=None,
                                        shuffle=None,
                                        image_cache=cache,
                                        repeat_condition=testing_image_ids,
                                        include_prev_image=include_prev_image,
                                        include_trial_id=include_trial_id)

        dataloaders["train"][data_key] = train_loader
        dataloaders["validation"][data_key] = val_loader
        dataloaders["test"][data_key] = test_loader


    if store_data_info and not os.path.exists(stats_path):

        in_name, out_name = next(iter(list(dataloaders["train"].values())[0]))._fields

        session_shape_dict = get_dims_for_loader_dict(dataloaders["train"])
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = {k: v[in_name][1] for k, v in session_shape_dict.items()}

        for data_key in session_shape_dict:
            data_info[data_key] = dict(input_dimensions=in_shapes_dict[data_key],
                                       input_channels=input_channels[data_key],
                                       output_dimension=n_neurons_dict[data_key],
                                       img_mean=img_mean,
                                       img_std=img_std)

        stats_path_base =  str(Path(stats_path).parent)
        if not os.path.exists(stats_path_base):
            os.mkdir(stats_path_base)
        with open(stats_path, "wb") as pkl:
            pickle.dump(data_info, pkl)

    return dataloaders if not return_data_info else data_info


def monkey_static_loader_extended(dataset,
                         neuronal_data_files,
                         image_cache_path,
                         batch_size=64,
                         seed=None,
                         train_frac=0.8,
                         subsample=1,
                         crop=((96, 96), (96, 96)),
                         scale=1.,
                         time_bins_sum=tuple(range(12)),
                         avg=False,
                         image_file=None,
                         return_data_info=False,
                         store_data_info=True,
                         image_frac=1.,
                         image_selection_seed=None,
                         randomize_image_selection=True,
                         img_mean=None,
                         img_std=None,
                         stimulus_location=None,
                         monitor_scaling_factor=4.57,
                         include_prev_image=False,
                         num_prev_images = 0,
                         include_trial_id=False,

                         ):
    """
    Function that returns cached dataloaders for monkey ephys experiments, extended with the number of prev images.

     creates a nested dictionary of dataloaders in the format
            {'train' : dict_of_loaders,
             'validation'   : dict_of_loaders,
            'test'  : dict_of_loaders, }

        in each dict_of_loaders, there will be  one dataloader per data-key (refers to a unique session ID)
        with the format:
            {'data-key1': torch.utils.data.DataLoader,
             'data-key2': torch.utils.da.DataLoader, ... }

    requires the types of input files:
        - the neuronal data files. A list of pickle files, with one file per session
        - the image file. The pickle file that contains all images.
        - individual image files, stored as numpy array, in a subfolder

    Args:
        dataset: a string, identifying the Dataset:
            'PlosCB19_V1', 'CSRF19_V1', 'CSRF19_V4'
            This string will be parsed by a datajoint table

        neuronal_data_files: a list paths that point to neuronal data pickle files
        image_file: a path that points to the image file
        image_cache_path: The path to the cached images
        batch_size: int - batch size of the dataloaders
        seed: int - random seed, to calculate the random split
        train_frac: ratio of train/validation images
        subsample: int - downsampling factor
        crop: int or tuple - crops x pixels from each side. Example: Input image of 100x100, crop=10 => Resulting img = 80x80.
            if crop is tuple, the expected input is a list of tuples, the specify the exact cropping from all four sides
                i.e. [(crop_top, crop_bottom), (crop_left, crop_right)]
        scale: float or integer - up-scale or down-scale via interpolation hte input images (default= 1)
        time_bins_sum: sums the responses over x time bins.
        avg: Boolean - Sums oder Averages the responses across bins.

    Returns: nested dictionary of dataloaders
    """

    dataset_config = locals()

    # initialize dataloaders as empty dict
    dataloaders = {'train': {}, 'validation': {}, 'test': {}}

    if not isinstance(time_bins_sum, Iterable):
        time_bins_sum = tuple(range(time_bins_sum))

    if isinstance(crop, int):
        crop = [(crop, crop), (crop, crop)]

    if stimulus_location is not None:
        crop = get_crop_from_stimulus_location(stimulus_location, crop, monitor_scaling_factor=monitor_scaling_factor)

    if not isinstance(image_frac, Iterable):
        image_frac = [image_frac for i in neuronal_data_files]

    # clean up image path because of legacy folder structure
    image_cache_path = image_cache_path.split('individual')[0]

    # Load image statistics if present
    stats_filename = make_hash(dataset_config)
    stats_path = os.path.join(image_cache_path, 'statistics/', stats_filename)

    # Get mean and std
    if os.path.exists(stats_path):
        with open(stats_path, "rb") as pkl:
            data_info = pickle.load(pkl)
        if return_data_info:
            return data_info
        img_mean = list(data_info.values())[0]["img_mean"]
        img_std = list(data_info.values())[0]["img_std"]

        # Initialize cache
        cache = ImageCache(path=image_cache_path, subsample=subsample, crop=crop, scale=scale, img_mean=img_mean,
                           img_std=img_std, transform=True, normalize=True)
    else:

        if img_mean is not None:
            cache = ImageCache(path=image_cache_path, subsample=subsample, crop=crop, scale=scale, img_mean=img_mean,
                               img_std=img_std, transform=True, normalize=True)
        else:
            # Initialize cache with no normalization
            cache = ImageCache(path=image_cache_path, subsample=subsample, crop=crop, scale=scale, transform=True,
                               normalize=False)

            # Compute mean and std of transformed images and zscore data (the cache wil be filled so first epoch will be fast)
            cache.zscore_images(update_stats=True)
            img_mean = cache.img_mean
            img_std = cache.img_std

    n_images = len(cache)
    data_info = {}

    # set up parameters for the different dataset types
    if dataset == 'PlosCB19_V1':
        # for the "Amadeus V1" Dataset, recorded by Santiago Cadena, there was no specified test set.
        # instead, the last 20% of the dataset were classified as test set. To make sure that the test set
        # of this dataset will always stay identical, the `train_test_split` value is hardcoded here.
        train_test_split = 0.8
        image_id_offset = 1
    else:
        train_test_split = 1
        image_id_offset = 0

    all_train_ids, all_validation_ids = get_validation_split(n_images=n_images * train_test_split,
                                                             train_frac=train_frac,
                                                             seed=seed)

    # cycling through all datafiles to fill the dataloaders with an entry per session
    for i, datapath in enumerate(neuronal_data_files):
        with open(datapath, "rb") as pkl:
            raw_data = pickle.load(pkl)

        subject_ids = raw_data["subject_id"]
        data_key = str(raw_data["session_id"])
        responses_train = raw_data["training_responses"].astype(np.float32)
        responses_test = raw_data["testing_responses"].astype(np.float32)
        training_image_ids = raw_data["training_image_ids"] - image_id_offset
        testing_image_ids = raw_data["testing_image_ids"] - image_id_offset
        if include_prev_image and not "training_prior_image_ids" in raw_data:
            raise ValueError(
                "No previous image IDs present in the provided data files. Set 'include_prev_image' to False")

        if include_prev_image and (num_prev_images == 0):
            raise ValueError(
                "Include_prev_images True, but num_prev_images is zero. Set 'num_prev_images' to a positive number")

        if not include_prev_image and (num_prev_images != 0):
            raise ValueError(
                "Include_prev_images False, but num_prev_images is zero. Set 'include_prev_images' to True")

        if include_prev_image:
            prev_training_image_ids = np.zeros((num_prev_images, len(training_image_ids)))
            prev_testing_image_ids = np.zeros((num_prev_images, len(testing_image_ids)))
            temp_prev_training = training_image_ids
            temp_prev_testing = testing_image_ids
            for j in range(num_prev_images):
                temp_prev_training = np.insert(temp_prev_training.copy(), 0, 0)[:-1]
                temp_prev_testing = np.insert(temp_prev_testing.copy(), 0, 0)[:-1]
                prev_training_image_ids[j] = temp_prev_training
                prev_testing_image_ids[j] = temp_prev_testing

        if dataset != 'PlosCB19_V1':
            if len(responses_test.shape) != 3:
                responses_test = responses_test[None, ...]
                responses_train = responses_train[None, ...]
                # correct the shape of the responses for a session that was exported incorrectly
                if data_key != '3653663964522':
                    warnings.warn("Pickle file with invalid response shape detected")

            responses_test = responses_test.transpose((2, 0, 1))
            responses_train = responses_train.transpose((2, 0, 1))

            if time_bins_sum is not None:  # then average over given time bins
                responses_train = (np.mean if avg else np.sum)(responses_train[:, :, time_bins_sum], axis=-1)
                responses_test = (np.mean if avg else np.sum)(responses_test[:, :, time_bins_sum], axis=-1)

        if image_frac[i] < 1:
            if randomize_image_selection:
                image_selection_seed = int(image_selection_seed * image_frac[i])
            idx_out = get_fraction_of_training_images(image_ids=training_image_ids, fraction=image_frac[i],
                                                      seed=image_selection_seed)
            training_image_ids = training_image_ids[idx_out]
            responses_train = responses_train[idx_out]

        train_idx = np.isin(training_image_ids, all_train_ids)
        val_idx = np.isin(training_image_ids, all_validation_ids)

        responses_val = responses_train[val_idx]
        responses_train = responses_train[train_idx]

        validation_image_ids = training_image_ids[val_idx]
        training_image_ids = training_image_ids[train_idx]

        if include_prev_image:
            all_prev_training_image_ids = prev_training_image_ids
            prev_validation_image_ids = np.zeros((num_prev_images, np.count_nonzero(val_idx)))
            prev_training_image_ids = np.zeros((num_prev_images, np.count_nonzero(train_idx)))
            for i in range(num_prev_images):
                prev_validation_image_ids[i] = all_prev_training_image_ids[i][val_idx]
                prev_training_image_ids[i] = all_prev_training_image_ids[i][train_idx]

        if include_trial_id:
            test_trial_ids = scipy.stats.zscore(range(len(testing_image_ids)))
            all_train_trial_ids = scipy.stats.zscore(range(len(training_image_ids) + len(validation_image_ids)))
            train_trial_ids = all_train_trial_ids[train_idx]
            val_trial_ids = all_train_trial_ids[val_idx]

        args_train = [training_image_ids, responses_train]
        args_val = [validation_image_ids, responses_val]
        args_test = [testing_image_ids, responses_test]

        if include_prev_image:
            args_train.insert(1, prev_training_image_ids)
            args_val.insert(1, prev_validation_image_ids)
            args_test.insert(1, prev_testing_image_ids)

        if include_trial_id:
            args_train.insert(1 + include_prev_image, train_trial_ids)
            args_val.insert(1 + include_prev_image, val_trial_ids)
            args_test.insert(1 + include_prev_image, test_trial_ids)

        train_loader = get_cached_loader_extended(*args_train,
                                         batch_size=batch_size,
                                         image_cache=cache,
                                         include_trial_id=include_trial_id,
                                         include_prev_image=include_prev_image,
                                         num_prev_images=num_prev_images,)

        val_loader = get_cached_loader_extended(*args_val,
                                       batch_size=batch_size,
                                       image_cache=cache,
                                       include_trial_id=include_trial_id,
                                       include_prev_image=include_prev_image,
                                       num_prev_images=num_prev_images,)

        test_loader = get_cached_loader_extended(*args_test,
                                        batch_size=None,
                                        shuffle=None,
                                        image_cache=cache,
                                        repeat_condition=testing_image_ids,
                                        include_prev_image=include_prev_image,
                                        include_trial_id=include_trial_id,
                                        num_prev_images=num_prev_images,)

        dataloaders["train"][data_key] = train_loader
        dataloaders["validation"][data_key] = val_loader
        dataloaders["test"][data_key] = test_loader

    if store_data_info and not os.path.exists(stats_path):

        in_name, out_name = next(iter(list(dataloaders["train"].values())[0]))._fields

        session_shape_dict = get_dims_for_loader_dict(dataloaders["train"])
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = {k: v[in_name][1] for k, v in session_shape_dict.items()}

        for data_key in session_shape_dict:
            data_info[data_key] = dict(input_dimensions=in_shapes_dict[data_key],
                                       input_channels=input_channels[data_key],
                                       output_dimension=n_neurons_dict[data_key],
                                       img_mean=img_mean,
                                       img_std=img_std)

        stats_path_base = str(Path(stats_path).parent)
        if not os.path.exists(stats_path_base):
            os.mkdir(stats_path_base)
        with open(stats_path, "wb") as pkl:
            pickle.dump(data_info, pkl)

    return dataloaders if not return_data_info else data_info


def monkey_mua_sua_loader(dataset,
                         neuronal_data_files,
                         mua_data_files,
                         image_cache_path,
                         batch_size=64,
                         seed=None,
                         train_frac=0.8,
                         subsample=1,
                         crop=((96, 96), (96, 96)),
                         scale=1.,
                         time_bins_sum=tuple(range(12)),
                         avg=False,
                         image_file=None,
                         return_data_info=False,
                         store_data_info=True,
                         mua_selector=None,
                         add_eye_movement=None):
    """
    Function that returns cached dataloaders for monkey ephys experiments.

     creates a nested dictionary of dataloaders in the format
            {'train' : dict_of_loaders,
             'validation'   : dict_of_loaders,
            'test'  : dict_of_loaders, }

        in each dict_of_loaders, there will be  one dataloader per data-key (refers to a unique session ID)
        with the format:
            {'data-key1': torch.utils.data.DataLoader,
             'data-key2': torch.utils.data.DataLoader, ... }

    requires the types of input files:
        - the neuronal data files. A list of pickle files, with one file per session
        - the image file. The pickle file that contains all images.
        - individual image files, stored as numpy array, in a subfolder

    Args:
        dataset: a string, identifying the Dataset:
            'PlosCB19_V1', 'CSRF19_V1', 'CSRF19_V4'
            This string will be parsed by a datajoint table

        neuronal_data_files: a list paths that point to neuronal data pickle files
        image_file: a path that points to the image file
        image_cache_path: The path to the cached images
        batch_size: int - batch size of the dataloaders
        seed: int - random seed, to calculate the random split
        train_frac: ratio of train/validation images
        subsample: int - downsampling factor
        crop: int or tuple - crops x pixels from each side. Example: Input image of 100x100, crop=10 => Resulting img = 80x80.
            if crop is tuple, the expected input is a list of tuples, the specify the exact cropping from all four sides
                i.e. [(crop_top, crop_bottom), (crop_left, crop_right)]
        scale: float or integer - up-scale or down-scale via interpolation hte input images (default= 1)
        time_bins_sum: sums the responses over x time bins.
        avg: Boolean - Sums oder Averages the responses across bins.

    Returns: nested dictionary of dataloaders
    """

    dataset_config = locals()

    # initialize dataloaders as empty dict
    dataloaders = {'train': {}, 'validation': {}, 'test': {}}

    if not isinstance(time_bins_sum, Iterable):
        time_bins_sum = tuple(range(time_bins_sum))

    if isinstance(crop, int):
        crop = [(crop, crop), (crop, crop)]

    # clean up image path because of legacy folder structure
    image_cache_path = image_cache_path.split('individual')[0]

    # Load image statistics if present
    stats_filename = make_hash(dataset_config)
    stats_path = os.path.join(image_cache_path, 'statistics/', stats_filename)

    # Get mean and std

    if os.path.exists(stats_path):
        with open(stats_path, "rb") as pkl:
            data_info = pickle.load(pkl)
        if return_data_info:
            return data_info
        img_mean = list(data_info.values())[0]["img_mean"]
        img_std = list(data_info.values())[0]["img_std"]

        # Initialize cache
        cache = ImageCache(path=image_cache_path, subsample=subsample, crop=crop, scale=scale, img_mean=img_mean,
                           img_std=img_std, transform=True, normalize=True)
    else:
        # Initialize cache with no normalization
        cache = ImageCache(path=image_cache_path, subsample=subsample, crop=crop, scale=scale, transform=True,
                           normalize=False)

        # Compute mean and std of transformed images and zscore data (the cache wil be filled so first epoch will be fast)
        cache.zscore_images(update_stats=True)
        img_mean = cache.img_mean
        img_std = cache.img_std

    n_images = len(cache)
    data_info = {}

    # set up parameters for the different dataset types
    if dataset == 'PlosCB19_V1':
        # for the "Amadeus V1" Dataset, recorded by Santiago Cadena, there was no specified test set.
        # instead, the last 20% of the dataset were classified as test set. To make sure that the test set
        # of this dataset will always stay identical, the `train_test_split` value is hardcoded here.
        train_test_split = 0.8
        image_id_offset = 1
    else:
        train_test_split = 1
        image_id_offset = 0

    all_train_ids, all_validation_ids = get_validation_split(n_images=n_images * train_test_split,
                                                             train_frac=train_frac,
                                                             seed=seed)

    # cycling through all datafiles to fill the dataloaders with an entry per session
    for i, datapath in enumerate(neuronal_data_files):

        with open(datapath, "rb") as pkl:
            raw_data = pickle.load(pkl)

        subject_ids = raw_data["subject_id"]
        data_key = str(raw_data["session_id"])
        responses_train = raw_data["training_responses"].astype(np.float32)
        responses_test = raw_data["testing_responses"].astype(np.float32)
        training_image_ids = raw_data["training_image_ids"] - image_id_offset
        testing_image_ids = raw_data["testing_image_ids"] - image_id_offset

        if add_eye_movement:
            if "avg_horizontal_eye_position_training_images" in raw_data:
                eye_pos_h_train = raw_data["avg_horizontal_eye_position_training_images"].astype(np.float32)
                eye_pos_v_train = raw_data["avg_vertical_eye_position_training_images"].astype(np.float32)
                eye_pos_h_test = raw_data["avg_horizontal_eye_position_testing_images"].astype(np.float32)
                eye_pos_v_test = raw_data["avg_vertical_eye_position_testing_images"].astype(np.float32)

                eye_pos_train = np.vstack([eye_pos_h_train, eye_pos_v_train]).T
                eye_pos_train = scipy.stats.zscore(eye_pos_train)
                eye_pos_test = np.vstack([eye_pos_h_test, eye_pos_v_test]).T
                eye_pos_test = scipy.stats.zscore(eye_pos_test)
            else:
                raise(FileNotFoundError, "Eye movement data is not found in the pickle file.")


        for mua_data_path in mua_data_files:
            with open(mua_data_path, "rb") as mua_pkl:
                mua_data = pickle.load(mua_pkl)

            if str(mua_data["session_id"]) == data_key:
                if mua_selector is not None:
                    selected_mua = mua_selector[data_key]
                else:
                    selected_mua = np.ones(len(mua_data["unit_ids"])).astype(bool)
                mua_responses_train = mua_data["training_responses"].astype(np.float32)[selected_mua]
                mua_responses_test = mua_data["testing_responses"].astype(np.float32)[selected_mua]
                mua_training_image_ids = mua_data["training_image_ids"] - image_id_offset
                mua_testing_image_ids = mua_data["testing_image_ids"] - image_id_offset
                break

        if not str(mua_data["session_id"]) == data_key:
            print("session {} does not exist in MUA. Skipping MUA".format(data_key))
        else:
            if not np.array_equal(training_image_ids, mua_training_image_ids):
                raise ValueError("Training image IDs do not match between the spike sorted data and mua data")
            if not np.array_equal(testing_image_ids, mua_testing_image_ids):
                raise ValueError("Testing image IDs do not match between the spike sorted data and mua data")

            if len(responses_train.shape) < 3:
                responses_train = responses_train[None, ...]
                responses_test = responses_test[None, ...]
            responses_train = np.concatenate([responses_train, mua_responses_train], axis=0)
            responses_test = np.concatenate([responses_test, mua_responses_test], axis=0)


        if dataset != 'PlosCB19_V1':
            responses_test = responses_test.transpose((2, 0, 1))
            responses_train = responses_train.transpose((2, 0, 1))

            if time_bins_sum is not None:  # then average over given time bins
                responses_train = (np.mean if avg else np.sum)(responses_train[:, :, time_bins_sum], axis=-1)
                responses_test = (np.mean if avg else np.sum)(responses_test[:, :, time_bins_sum], axis=-1)

        train_idx = np.isin(training_image_ids, all_train_ids)
        val_idx = np.isin(training_image_ids, all_validation_ids)

        responses_val = responses_train[val_idx]
        responses_train = responses_train[train_idx]
        if add_eye_movement:
            eye_pos_val = eye_pos_train[val_idx]
            eye_pos_train = eye_pos_train[train_idx]

        validation_image_ids = training_image_ids[val_idx]
        training_image_ids = training_image_ids[train_idx]

        if add_eye_movement:
            train_loader = get_cached_loader(training_image_ids, responses_train, eye_pos_train, batch_size=batch_size,
                                             image_cache=cache)
            val_loader = get_cached_loader(validation_image_ids, responses_val, eye_pos_val, batch_size=batch_size,
                                           image_cache=cache)
            test_loader = get_cached_loader(testing_image_ids,
                                            responses_test,
                                            eye_pos_test,
                                            batch_size=None,
                                            shuffle=None,
                                            image_cache=cache,
                                            repeat_condition=testing_image_ids)
        else:
            train_loader = get_cached_loader(training_image_ids, responses_train, batch_size=batch_size, image_cache=cache)
            val_loader = get_cached_loader(validation_image_ids, responses_val, batch_size=batch_size, image_cache=cache)
            test_loader = get_cached_loader(testing_image_ids,
                                            responses_test,
                                            batch_size=None,
                                            shuffle=None,
                                            image_cache=cache,
                                            repeat_condition=testing_image_ids)

        dataloaders["train"][data_key] = train_loader
        dataloaders["validation"][data_key] = val_loader
        dataloaders["test"][data_key] = test_loader

    if store_data_info and not os.path.exists(stats_path):

        in_name, out_name = next(iter(list(dataloaders["train"].values())[0]))._fields[:2]

        session_shape_dict = get_dims_for_loader_dict(dataloaders["train"])
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = {k: v[in_name][1] for k, v in session_shape_dict.items()}

        for data_key in session_shape_dict:
            data_info[data_key] = dict(input_dimensions=in_shapes_dict[data_key],
                                       input_channels=input_channels[data_key],
                                       output_dimension=n_neurons_dict[data_key],
                                       img_mean=img_mean,
                                       img_std=img_std)

        stats_path_base = str(Path(stats_path).parent)
        if not os.path.exists(stats_path_base):
            os.mkdir(stats_path_base)
        with open(stats_path, "wb") as pkl:
            pickle.dump(data_info, pkl)

    return dataloaders if not return_data_info else data_info


def monkey_static_loader_closed_loop(dataset,
                         neuronal_data_files,
                         image_cache_path,
                         batch_size=64,
                         seed=None,
                         train_frac=0.8,
                         subsample=1,
                         crop=((96, 96), (96, 96)),
                         scale=1.,
                         time_bins_sum=tuple(range(12)),
                         avg=False,
                         image_file=None,
                         return_data_info=False,
                         store_data_info=True,
                         image_frac=1.,
                         image_selection_seed=None,
                         randomize_image_selection=True,
                         stimulus_location=None,
                         img_mean=None,
                         img_std=None,
                         include_mei_training=False,
                         include_control_training=False):
    """
    Function that returns cached dataloaders for monkey ephys experiments.

     creates a nested dictionary of dataloaders in the format
            {'train' : dict_of_loaders,
             'validation'   : dict_of_loaders,
            'test'  : dict_of_loaders, }

        in each dict_of_loaders, there will be  one dataloader per data-key (refers to a unique session ID)
        with the format:
            {'data-key1': torch.utils.data.DataLoader,
             'data-key2': torch.utils.data.DataLoader, ... }

    requires the types of input files:
        - the neuronal data files. A list of pickle files, with one file per session
        - the image file. The pickle file that contains all images.
        - individual image files, stored as numpy array, in a subfolder

    Args:
        dataset: a string, identifying the Dataset:
            'PlosCB19_V1', 'CSRF19_V1', 'CSRF19_V4'
            This string will be parsed by a datajoint table

        neuronal_data_files: a list paths that point to neuronal data pickle files
        image_file: a path that points to the image file
        image_cache_path: The path to the cached images
        batch_size: int - batch size of the dataloaders
        seed: int - random seed, to calculate the random split
        train_frac: ratio of train/validation images
        subsample: int - downsampling factor
        crop: int or tuple - crops x pixels from each side. Example: Input image of 100x100, crop=10 => Resulting img = 80x80.
            if crop is tuple, the expected input is a list of tuples, the specify the exact cropping from all four sides
                i.e. [(crop_top, crop_bottom), (crop_left, crop_right)]
        scale: float or integer - up-scale or down-scale via interpolation hte input images (default= 1)
        time_bins_sum: sums the responses over x time bins.
        avg: Boolean - Sums oder Averages the responses across bins.

    Returns: nested dictionary of dataloaders
    """

    dataset_config = locals()

    if include_mei_training and include_control_training:
        raise ValueError("the entire test can not be included into the training set. Set either 'include_mei_training' "
                         "or 'include_control_training' to False.")

    # initialize dataloaders as empty dict
    dataloaders = {'train': {},
                   'validation': {},
                   'test': {},
                   'test_mei_uncropped': {},
                   'test_control_uncropped': {},
                   'test_mei_cropped': {},
                   'test_control_cropped': {}}

    if include_mei_training or include_control_training:
        dataloaders["validation_extended"] = {}

    if not isinstance(time_bins_sum, Iterable):
        time_bins_sum = tuple(range(time_bins_sum))

    if isinstance(crop, int):
        crop = [(crop, crop), (crop, crop)]

    if not isinstance(image_frac, Iterable):
        image_frac = [image_frac for i in neuronal_data_files]

    if not isinstance(stimulus_location, Iterable):
        stimulus_location = [stimulus_location for i in neuronal_data_files]

    # clean up image path because of legacy folder structure
    image_cache_path = image_cache_path.split('individual')[0]

    # Load image statistics if present
    stats_filename = make_hash(dataset_config)

    stats_path = os.path.join(image_cache_path, 'statistics/', stats_filename)

    # Get mean and std
    if os.path.exists(stats_path):
        with open(stats_path, "rb") as pkl:
            data_info = pickle.load(pkl)
        if return_data_info:
            return data_info
        img_mean = list(data_info.values())[0]["img_mean"]
        img_std = list(data_info.values())[0]["img_std"]

        # Initialize cache
        cache = ImageCache(path=image_cache_path, subsample=subsample, crop=crop, scale=scale, img_mean=img_mean,
                           img_std=img_std, transform=True, normalize=True)
    else:
        if img_mean is not None:
            cache = ImageCache(path=image_cache_path, subsample=subsample, crop=crop, scale=scale, img_mean=img_mean,
                               img_std=img_std, transform=True, normalize=True)
        else:

            # Initialize cache with no normalization
            cache = ImageCache(path=image_cache_path, subsample=subsample, crop=crop, scale=scale, transform=True,
                               normalize=False)

            # Compute mean and std of transformed images and zscore data (the cache wil be filled so first epoch will be fast)
            cache.zscore_images(update_stats=True)
            img_mean = cache.img_mean
            img_std = cache.img_std

    n_images = len(cache)
    data_info = {}
    if stimulus_location is not None:
        TrainImageCaches = {}
        TestImageCaches = {}

    # set up parameters for the different dataset types
    if dataset == 'PlosCB19_V1':
        # for the "Amadeus V1" Dataset, recorded by Santiago Cadena, there was no specified test set.
        # instead, the last 20% of the dataset were classified as test set. To make sure that the test set
        # of this dataset will always stay identical, the `train_test_split` value is hardcoded here.
        train_test_split = 0.8
        image_id_offset = 1
    else:
        train_test_split = 1
        image_id_offset = 0

    all_train_ids, all_validation_ids = get_validation_split(n_images=n_images * train_test_split,
                                                             train_frac=train_frac,
                                                             seed=seed)

    # cycling through all datafiles to fill the dataloaders with an entry per session
    for i, datapath in enumerate(neuronal_data_files):

        with open(datapath, "rb") as pkl:
            raw_data = pickle.load(pkl)

        subject_ids = raw_data["subject_id"]
        data_key = str(raw_data["session_id"])
        responses_train = raw_data["training_responses"].astype(np.float32)
        responses_test = raw_data["testing_responses"].astype(np.float32)

        mei_uncropped_responses = raw_data["mei_uncropped_responses"].astype(np.float32)
        control_uncropped_responses = raw_data["control_uncropped_responses"].astype(np.float32)
        mei_cropped_responses = raw_data["mei_cropped_responses"].astype(np.float32)
        control_cropped_responses = raw_data["control_cropped_responses"].astype(np.float32)

        training_image_ids = raw_data["training_image_ids"] - image_id_offset
        testing_image_ids = raw_data["testing_image_ids"] - image_id_offset

        mei_uncropped_ids = raw_data["mei_uncropped_ids"]
        mei_cropped_ids = raw_data["mei_cropped_ids"]
        control_uncropped_ids = raw_data["control_uncropped_ids"]
        control_cropped_ids = raw_data["control_cropped_ids"]


        if dataset != 'PlosCB19_V1':
            responses_test = responses_test.transpose((2, 0, 1))
            responses_train = responses_train.transpose((2, 0, 1))

            mei_uncropped_responses = mei_uncropped_responses.transpose((2, 0, 1))
            control_uncropped_responses = control_uncropped_responses.transpose((2, 0, 1))
            mei_cropped_responses = mei_cropped_responses.transpose((2, 0, 1))
            control_cropped_responses = control_cropped_responses.transpose((2, 0, 1))

            if time_bins_sum is not None:  # then average over given time bins
                responses_train = (np.mean if avg else np.sum)(responses_train[:, :, time_bins_sum], axis=-1)
                responses_test = (np.mean if avg else np.sum)(responses_test[:, :, time_bins_sum], axis=-1)
                mei_uncropped_responses = (np.mean if avg else np.sum)(mei_uncropped_responses[:, :, time_bins_sum], axis=-1)
                control_uncropped_responses = (np.mean if avg else np.sum)(control_uncropped_responses[:, :, time_bins_sum], axis=-1)
                mei_cropped_responses = (np.mean if avg else np.sum)(mei_cropped_responses[:, :, time_bins_sum], axis=-1)
                control_cropped_responses = (np.mean if avg else np.sum)(control_cropped_responses[:, :, time_bins_sum], axis=-1)

        if image_frac[i] < 1:
            if randomize_image_selection:
                image_selection_seed = int(image_selection_seed * image_frac)
            idx_out = get_fraction_of_training_images(image_ids=training_image_ids, fraction=image_frac,
                                                      seed=image_selection_seed)
            training_image_ids = training_image_ids[idx_out]
            responses_train = responses_train[idx_out]

        if include_mei_training or include_control_training:
            training_image_ids_original = training_image_ids
            responses_train_original = responses_train

        if include_control_training:
            training_image_ids = np.append(training_image_ids, control_cropped_ids)
            training_image_ids = np.append(training_image_ids, control_uncropped_ids)

            responses_train = np.vstack([responses_train, control_cropped_responses])
            responses_train = np.vstack([responses_train, control_uncropped_responses])

        if include_mei_training:
            training_image_ids = np.append(training_image_ids, mei_cropped_ids)
            training_image_ids = np.append(training_image_ids, mei_uncropped_ids)

            responses_train = np.vstack([responses_train, mei_cropped_responses])
            responses_train = np.vstack([responses_train, mei_uncropped_responses])

        train_idx = np.isin(training_image_ids, all_train_ids)
        val_idx = np.isin(training_image_ids, all_validation_ids)

        responses_val = responses_train[val_idx]
        responses_train = responses_train[train_idx]
        validation_image_ids = training_image_ids[val_idx]
        training_image_ids = training_image_ids[train_idx]

        if include_mei_training or include_control_training:
            train_idx = np.isin(training_image_ids_original, all_train_ids)
            val_idx = np.isin(training_image_ids_original, all_validation_ids)

            responses_val_original = responses_train_original[val_idx]
            responses_train_original = responses_train_original[train_idx]
            validation_image_ids_original = training_image_ids_original[val_idx]
            training_image_ids_original = training_image_ids_original[train_idx]

        if stimulus_location is not None:
            training_crop = get_crop_from_stimulus_location(stimulus_location[i], crop, monitor_scaling_factor=4.57)
            test_crop = crop - np.clip(training_crop, -999, 0)

            if img_mean is not None:
                TrainImageCaches[data_key] = ImageCache(path=image_cache_path, subsample=subsample, crop=training_crop, scale=scale,
                                   img_mean=img_mean, img_std=img_std, transform=True, normalize=True)

                TestImageCaches[data_key] = ImageCache(path=image_cache_path, subsample=subsample, crop=test_crop, scale=scale,
                                   img_mean=img_mean, img_std=img_std, transform=True, normalize=True)

            else:
                TrainImageCaches[data_key] = ImageCache(path=image_cache_path, subsample=subsample, crop=training_crop, scale=scale, transform=True,
                                   normalize=False)
                TrainImageCaches[data_key].zscore_images(update_stats=True)



            train_loader = get_cached_loader(training_image_ids, responses_train, batch_size=batch_size,
                                             image_cache=TrainImageCaches[data_key])
            if include_mei_training or include_control_training:
                val_loader = get_cached_loader(validation_image_ids_original, responses_val_original, batch_size=batch_size,
                                               image_cache=TrainImageCaches[data_key])
                val_loader_extended = get_cached_loader(validation_image_ids, responses_val, batch_size=batch_size,
                                               image_cache=TrainImageCaches[data_key])
            else:
                val_loader = get_cached_loader(validation_image_ids, responses_val, batch_size=batch_size,
                                               image_cache=TrainImageCaches[data_key])
        else:
            train_loader = get_cached_loader(training_image_ids, responses_train, batch_size=batch_size, image_cache=cache)
            if include_mei_training or include_control_training:
                val_loader_extended = get_cached_loader(validation_image_ids, responses_val, batch_size=batch_size, image_cache=cache)
                val_loader = get_cached_loader(validation_image_ids_original,
                                                responses_val_original,
                                                batch_size=batch_size,
                                                image_cache=cache)
            else:
                val_loader = get_cached_loader(validation_image_ids, responses_val, batch_size=batch_size,
                                               image_cache=cache)

            TestImageCaches[data_key] = cache

        test_loader = get_cached_loader(testing_image_ids,
                                        responses_test,
                                        batch_size=None,
                                        shuffle=None,
                                        image_cache=TestImageCaches[data_key],
                                        repeat_condition=testing_image_ids)

        mei_uncropped_loader = get_cached_loader(mei_uncropped_ids,
                                        mei_uncropped_responses,
                                        batch_size=None,
                                        shuffle=None,
                                        image_cache=TestImageCaches[data_key],
                                        repeat_condition=mei_uncropped_ids)

        control_uncropped_loader = get_cached_loader(control_uncropped_ids,
                                        control_uncropped_responses,
                                        batch_size=None,
                                        shuffle=None,
                                        image_cache=TestImageCaches[data_key],
                                        repeat_condition=control_uncropped_ids)

        mei_cropped_loader = get_cached_loader(mei_cropped_ids,
                                       mei_cropped_responses,
                                       batch_size=None,
                                       shuffle=None,
                                       image_cache=TestImageCaches[data_key],
                                       repeat_condition=mei_cropped_ids)

        control_cropped_loader = get_cached_loader(control_cropped_ids,
                                           control_cropped_responses,
                                           batch_size=None,
                                           shuffle=None,
                                           image_cache=TestImageCaches[data_key],
                                           repeat_condition=control_cropped_ids)

        dataloaders["train"][data_key] = train_loader
        dataloaders["validation"][data_key] = val_loader
        dataloaders["test"][data_key] = test_loader

        dataloaders["test_mei_uncropped"][data_key] = mei_uncropped_loader
        dataloaders["test_mei_cropped"][data_key] = mei_cropped_loader
        dataloaders["test_control_uncropped"][data_key] = control_uncropped_loader
        dataloaders["test_control_cropped"][data_key] = control_cropped_loader

        if include_mei_training or include_control_training:
            dataloaders["validation_extended"][data_key] = val_loader_extended

    if store_data_info and not os.path.exists(stats_path):

        in_name, out_name = next(iter(list(dataloaders["train"].values())[0]))._fields

        session_shape_dict = get_dims_for_loader_dict(dataloaders["train"])
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = {k: v[in_name][1] for k, v in session_shape_dict.items()}

        for data_key in session_shape_dict:
            data_info[data_key] = dict(input_dimensions=in_shapes_dict[data_key],
                                       input_channels=input_channels[data_key],
                                       output_dimension=n_neurons_dict[data_key],
                                       img_mean=img_mean,
                                       img_std=img_std)

        stats_path_base = str(Path(stats_path).parent)
        if not os.path.exists(stats_path_base):
            os.mkdir(stats_path_base)
        with open(stats_path, "wb") as pkl:
            pickle.dump(data_info, pkl)

    return dataloaders if not return_data_info else data_info


def monkey_static_loader_mua_cl(dataset,
                                     neuronal_data_files,
                                     image_cache_path,
                                     batch_size=64,
                                     seed=None,
                                     train_frac=0.8,
                                     subsample=1,
                                     crop=((96, 96), (96, 96)),
                                     scale=1.,
                                     time_bins_sum=tuple(range(12)),
                                     avg=False,
                                     image_file=None,
                                     return_data_info=False,
                                     store_data_info=True,
                                     image_frac=1.,
                                     image_selection_seed=None,
                                     randomize_image_selection=True,
                                     stimulus_location=None,
                                     img_mean=None,
                                     img_std=None,
                                     dataloader_keys=None):
    """
    Function that returns cached dataloaders for monkey ephys experiments.

     creates a nested dictionary of dataloaders in the format
            {'train' : dict_of_loaders,
             'validation'   : dict_of_loaders,
            'test'  : dict_of_loaders, }

        in each dict_of_loaders, there will be  one dataloader per data-key (refers to a unique session ID)
        with the format:
            {'data-key1': torch.utils.data.DataLoader,
             'data-key2': torch.utils.data.DataLoader, ... }

    requires the types of input files:
        - the neuronal data files. A list of pickle files, with one file per session
        - the image file. The pickle file that contains all images.
        - individual image files, stored as numpy array, in a subfolder

    Args:
        dataset: a string, identifying the Dataset:
            'PlosCB19_V1', 'CSRF19_V1', 'CSRF19_V4'
            This string will be parsed by a datajoint table

        neuronal_data_files: a list paths that point to neuronal data pickle files
        image_file: a path that points to the image file
        image_cache_path: The path to the cached images
        batch_size: int - batch size of the dataloaders
        seed: int - random seed, to calculate the random split
        train_frac: ratio of train/validation images
        subsample: int - downsampling factor
        crop: int or tuple - crops x pixels from each side. Example: Input image of 100x100, crop=10 => Resulting img = 80x80.
            if crop is tuple, the expected input is a list of tuples, the specify the exact cropping from all four sides
                i.e. [(crop_top, crop_bottom), (crop_left, crop_right)]
        scale: float or integer - up-scale or down-scale via interpolation hte input images (default= 1)
        time_bins_sum: sums the responses over x time bins.
        avg: Boolean - Sums oder Averages the responses across bins.

    Returns: nested dictionary of dataloaders
    """

    dataset_config = locals()

    # Set Dataset Specific Parameters
    train_test_split = 1
    image_id_offset = 0

    if not isinstance(time_bins_sum, Iterable):
        time_bins_sum = tuple(range(time_bins_sum))

    if isinstance(crop, int):
        crop = [(crop, crop), (crop, crop)]

    if not isinstance(image_frac, Iterable):
        image_frac = [image_frac for i in neuronal_data_files]

    if not isinstance(stimulus_location, Iterable):
        stimulus_location = [stimulus_location for i in neuronal_data_files]

    # clean up image path because of legacy folder structure
    image_cache_path = image_cache_path.split('individual')[0]

    # Load image statistics if present
    stats_filename = make_hash(dataset_config)

    stats_path = os.path.join(image_cache_path, 'statistics/', stats_filename)

    # Get mean and std
    if os.path.exists(stats_path):
        with open(stats_path, "rb") as pkl:
            data_info = pickle.load(pkl)
        if return_data_info:
            return data_info
        img_mean = list(data_info.values())[0]["img_mean"]
        img_std = list(data_info.values())[0]["img_std"]

        # Initialize cache
        cache = ImageCache(path=image_cache_path, subsample=subsample, crop=crop, scale=scale, img_mean=img_mean,
                           img_std=img_std, transform=True, normalize=True)
    else:
        if img_mean is not None:
            cache = ImageCache(path=image_cache_path, subsample=subsample, crop=crop, scale=scale, img_mean=img_mean,
                               img_std=img_std, transform=True, normalize=True)
        else:

            # Initialize cache with no normalization
            cache = ImageCache(path=image_cache_path, subsample=subsample, crop=crop, scale=scale, transform=True,
                               normalize=False)

            # Compute mean and std of transformed images and zscore data (the cache wil be filled so first epoch will be fast)
            cache.zscore_images(update_stats=True)
            img_mean = cache.img_mean
            img_std = cache.img_std

    n_images = len(cache)
    data_info = {}
    if stimulus_location is not None:
        TestImageCaches = {}

    # initialize dataloaders as empty dict
    dataloaders = {'test': {},
                   'natural_imgs': {},
                   'noise_mei_resnet': {},
                   'noise_mei_cnn': {},
                   'natural_mei_resnet': {},
                   'natural_mei_cnn': {},}


    # cycling through all datafiles to fill the dataloaders with an entry per session
    for i, datapath in enumerate(neuronal_data_files):

        with open(datapath, "rb") as pkl:
            raw_data = pickle.load(pkl)

        subject_ids = raw_data["subject_id"]
        data_key = str(raw_data["session_id"])

        test_image_responses = raw_data["test_image_responses"].astype(np.float32)
        natural_image_responses = raw_data["natural_image_responses"].astype(np.float32)
        noise_mei_resnet_responses = raw_data["noise_mei_resnet_responses"].astype(np.float32)
        noise_mei_cnn_responses = raw_data["noise_mei_cnn_responses"].astype(np.float32)
        natural_mei_resnet_responses = raw_data["natural_mei_resnet_responses"].astype(np.float32)
        natural_mei_cnn_responses = raw_data["natural_mei_cnn_responses"].astype(np.float32)

        test_image_ids = raw_data["test_image_ids"].astype(np.int64)

        natural_image_ids = raw_data["natural_image_ids"].astype(np.int64)
        noise_mei_resnet_ids = raw_data["noise_mei_resnet_ids"].astype(np.int64)
        noise_mei_cnn_ids = raw_data["noise_mei_cnn_ids"].astype(np.int64)
        natural_mei_resnet_ids = raw_data["natural_mei_resnet_ids"].astype(np.int64)
        natural_mei_cnn_ids = raw_data["natural_mei_cnn_ids"].astype(np.int64)

        test_image_responses = test_image_responses.transpose((2, 0, 1))
        natural_image_responses = natural_image_responses.transpose((2, 0, 1))
        noise_mei_resnet_responses = noise_mei_resnet_responses.transpose((2, 0, 1))
        noise_mei_cnn_responses = noise_mei_cnn_responses.transpose((2, 0, 1))
        natural_mei_resnet_responses = natural_mei_resnet_responses.transpose((2, 0, 1))
        natural_mei_cnn_responses = natural_mei_cnn_responses.transpose((2, 0, 1))

        if time_bins_sum is not None:  # then average over given time bins
            test_image_responses = (np.mean if avg else np.sum)(test_image_responses[:, :, time_bins_sum], axis=-1)
            natural_image_responses = (np.mean if avg else np.sum)(natural_image_responses[:, :, time_bins_sum], axis=-1)
            noise_mei_resnet_responses = (np.mean if avg else np.sum)(noise_mei_resnet_responses[:, :, time_bins_sum], axis=-1)
            noise_mei_cnn_responses = (np.mean if avg else np.sum)(noise_mei_cnn_responses[:, :, time_bins_sum], axis=-1)
            natural_mei_resnet_responses = (np.mean if avg else np.sum)(natural_mei_resnet_responses[:, :, time_bins_sum], axis=-1)
            natural_mei_cnn_responses = (np.mean if avg else np.sum)(natural_mei_cnn_responses[:, :, time_bins_sum], axis=-1)


        #training_crop = get_crop_from_stimulus_location(stimulus_location[i], crop, monitor_scaling_factor=4.57)
        #test_crop = crop - np.clip(training_crop, -999, 0)

        #TestImageCaches[data_key] = ImageCache(path=image_cache_path, subsample=subsample, crop=test_crop, scale=scale,
        #                                       img_mean=img_mean, img_std=img_std, transform=True, normalize=True)

        TestImageCaches[data_key] = cache

        test_loader = get_cached_loader(test_image_ids,
                                        test_image_responses,
                                        batch_size=None,
                                        shuffle=None,
                                        image_cache=TestImageCaches[data_key],
                                        repeat_condition=test_image_ids)

        natural_img_loader = get_cached_loader(natural_image_ids,
                                        natural_image_responses,
                                        batch_size=None,
                                        shuffle=None,
                                        image_cache=TestImageCaches[data_key],
                                        repeat_condition=natural_image_ids)

        noise_mei_resnet_loader = get_cached_loader(noise_mei_resnet_ids,
                                                 noise_mei_resnet_responses,
                                                 batch_size=None,
                                                 shuffle=None,
                                                 image_cache=TestImageCaches[data_key],
                                                 repeat_condition=noise_mei_resnet_ids)

        noise_mei_cnn_loader = get_cached_loader(noise_mei_cnn_ids,
                                                     noise_mei_cnn_responses,
                                                     batch_size=None,
                                                     shuffle=None,
                                                     image_cache=TestImageCaches[data_key],
                                                     repeat_condition=noise_mei_cnn_ids)

        natural_mei_resnet_loader = get_cached_loader(natural_mei_resnet_ids,
                                               natural_mei_resnet_responses,
                                               batch_size=None,
                                               shuffle=None,
                                               image_cache=TestImageCaches[data_key],
                                               repeat_condition=natural_mei_resnet_ids)

        natural_mei_cnn_loader = get_cached_loader(natural_mei_cnn_ids,
                                                   natural_mei_cnn_responses,
                                                   batch_size=None,
                                                   shuffle=None,
                                                   image_cache=TestImageCaches[data_key],
                                                   repeat_condition=natural_mei_cnn_ids)

        dataloaders["test"][data_key] = test_loader
        dataloaders["natural_imgs"][data_key] = natural_img_loader
        dataloaders["noise_mei_resnet"][data_key] = noise_mei_resnet_loader
        dataloaders["noise_mei_cnn"][data_key] = noise_mei_cnn_loader
        dataloaders["natural_mei_resnet"][data_key] = natural_mei_resnet_loader
        dataloaders["natural_mei_cnn"][data_key] = natural_mei_cnn_loader


    if store_data_info and not os.path.exists(stats_path):

        in_name, out_name = next(iter(list(dataloaders["test"].values())[0]))._fields

        session_shape_dict = get_dims_for_loader_dict(dataloaders["test"])
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = {k: v[in_name][1] for k, v in session_shape_dict.items()}

        for data_key in session_shape_dict:
            data_info[data_key] = dict(input_dimensions=in_shapes_dict[data_key],
                                       input_channels=input_channels[data_key],
                                       output_dimension=n_neurons_dict[data_key],
                                       img_mean=img_mean,
                                       img_std=img_std)

        stats_path_base = str(Path(stats_path).parent)
        if not os.path.exists(stats_path_base):
            os.mkdir(stats_path_base)
        with open(stats_path, "wb") as pkl:
            pickle.dump(data_info, pkl)

    return dataloaders if not return_data_info else data_info


def monkey_v4_cl(dataset,
                neuronal_data_files,
                image_cache_path,
                batch_size=64,
                seed=None,
                train_frac=0.8,
                subsample=1,
                crop=((96, 96), (96, 96)),
                scale=1.,
                time_bins_sum=tuple(range(12)),
                avg=False,
                image_file=None,
                return_data_info=False,
                store_data_info=True,
                image_frac=1.,
                image_selection_seed=None,
                randomize_image_selection=True,
                stimulus_location=None,
                img_mean=None,
                img_std=None,
                dataloader_keys=None,
                zscore_images=True,
                 test_image_path=None,):
    """
    Function that returns cached dataloaders for monkey ephys experiments.

     creates a nested dictionary of dataloaders in the format
            {'train' : dict_of_loaders,
             'validation'   : dict_of_loaders,
            'test'  : dict_of_loaders, }

        in each dict_of_loaders, there will be  one dataloader per data-key (refers to a unique session ID)
        with the format:
            {'data-key1': torch.utils.data.DataLoader,
             'data-key2': torch.utils.data.DataLoader, ... }

    requires the types of input files:
        - the neuronal data files. A list of pickle files, with one file per session
        - the image file. The pickle file that contains all images.
        - individual image files, stored as numpy array, in a subfolder

    Args:
        dataset: a string, identifying the Dataset:
            'PlosCB19_V1', 'CSRF19_V1', 'CSRF19_V4'
            This string will be parsed by a datajoint table

        neuronal_data_files: a list paths that point to neuronal data pickle files
        image_file: a path that points to the image file
        image_cache_path: The path to the cached images
        batch_size: int - batch size of the dataloaders
        seed: int - random seed, to calculate the random split
        train_frac: ratio of train/validation images
        subsample: int - downsampling factor
        crop: int or tuple - crops x pixels from each side. Example: Input image of 100x100, crop=10 => Resulting img = 80x80.
            if crop is tuple, the expected input is a list of tuples, the specify the exact cropping from all four sides
                i.e. [(crop_top, crop_bottom), (crop_left, crop_right)]
        scale: float or integer - up-scale or down-scale via interpolation hte input images (default= 1)
        time_bins_sum: sums the responses over x time bins.
        avg: Boolean - Sums oder Averages the responses across bins.

    Returns: nested dictionary of dataloaders
    """

    dataset_config = locals()

    # Set Dataset Specific Parameters
    train_test_split = 1
    image_id_offset = 0

    if not isinstance(time_bins_sum, Iterable):
        time_bins_sum = tuple(range(time_bins_sum))

    if isinstance(crop, int):
        crop = [(crop, crop), (crop, crop)]

    if not isinstance(image_frac, Iterable):
        image_frac = [image_frac for i in neuronal_data_files]

    if not isinstance(stimulus_location, Iterable):
        stimulus_location = [stimulus_location for i in neuronal_data_files]

    # clean up image path because of legacy folder structure
    image_cache_path = image_cache_path.split('individual')[0]

    # Load image statistics if present
    stats_filename = make_hash(dataset_config)

    stats_path = os.path.join(image_cache_path, 'statistics/', stats_filename)

    # Get mean and std
    if os.path.exists(stats_path):
        with open(stats_path, "rb") as pkl:
            data_info = pickle.load(pkl)
        if return_data_info:
            return data_info
        img_mean = list(data_info.values())[0]["img_mean"]
        img_std = list(data_info.values())[0]["img_std"]

        # Initialize cache
        cache = ImageCache(path=image_cache_path, subsample=subsample, crop=crop, scale=scale, img_mean=img_mean,
                           img_std=img_std, transform=True, normalize=zscore_images)
        if test_image_path is not None:
            oracle_cache = ImageCache(path=test_image_path, subsample=subsample, crop=crop, scale=scale, img_mean=img_mean,
                                          img_std=img_std, transform=True, normalize=zscore_images)
    else:
        if img_mean is not None:
            cache = ImageCache(path=image_cache_path, subsample=subsample, crop=crop, scale=scale, img_mean=img_mean,
                               img_std=img_std, transform=True, normalize=zscore_images)
            if test_image_path is not None:
                oracle_cache = ImageCache(path=test_image_path, subsample=subsample, crop=crop, scale=scale, img_mean=img_mean,
                                   img_std=img_std, transform=True, normalize=zscore_images)
        else:
            # Initialize cache with no normalization
            cache = ImageCache(path=image_cache_path, subsample=subsample, crop=crop, scale=scale, transform=True,
                               normalize=False)
            if test_image_path is not None:
                oracle_cache = ImageCache(path=test_image_path, subsample=subsample, crop=crop, scale=scale, img_mean=img_mean,
                                          img_std=img_std, transform=True, normalize=False)
            if zscore_images:
            # Compute mean and std of transformed images and zscore data (the cache wil be filled so first epoch will be fast)
                cache.zscore_images(update_stats=True)
                img_mean = cache.img_mean
                img_std = cache.img_std

    n_images = len(cache)
    data_info = {}
    if stimulus_location is not None:
        TestImageCaches, OracleImageCaches = {}, {}

    dataloaders = {k:{} for k in dataloader_keys}
    pickle_file_dict = {k:{} for k in ['responses', 'ids']}

    for i, datapath in enumerate(neuronal_data_files):
        with open(datapath, "rb") as pkl:
            raw_data = pickle.load(pkl)
        subject_ids = raw_data["subject_id"]
        data_key = str(raw_data["session_id"])
        TestImageCaches[data_key] = cache
        if test_image_path is not None:
            OracleImageCaches[data_key] = oracle_cache

        for stim_key in dataloader_keys:
            pickle_file_dict["responses"][stim_key] = raw_data[stim_key+"_responses"].astype(np.float32).transpose((2, 0, 1))
            pickle_file_dict["ids"][stim_key] = raw_data[stim_key+"_ids"].astype(np.int64)
            if time_bins_sum is not None:  # then average over given time bins
                pickle_file_dict["responses"][stim_key] = (np.mean if avg else np.sum)(pickle_file_dict["responses"][stim_key][:, :, time_bins_sum], axis=-1)

            if stim_key == "test_image" and test_image_path is not None:
                stim_key_cache = OracleImageCaches[data_key]
            else:
                stim_key_cache = TestImageCaches[data_key]

            loader = get_cached_loader(pickle_file_dict["ids"][stim_key],
                                       pickle_file_dict["responses"][stim_key],
                                       batch_size=None,
                                       shuffle=None,
                                       image_cache=stim_key_cache,
                                       repeat_condition=pickle_file_dict["ids"][stim_key])

            dataloaders[stim_key][data_key] = loader

    if store_data_info and not os.path.exists(stats_path):

        in_name, out_name = next(iter(list(dataloaders[dataloader_keys[0]].values())[0]))._fields

        session_shape_dict = get_dims_for_loader_dict(dataloaders[dataloader_keys[0]])
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = {k: v[in_name][1] for k, v in session_shape_dict.items()}

        for data_key in session_shape_dict:
            data_info[data_key] = dict(input_dimensions=in_shapes_dict[data_key],
                                       input_channels=input_channels[data_key],
                                       output_dimension=n_neurons_dict[data_key],
                                       img_mean=img_mean,
                                       img_std=img_std)

        stats_path_base = str(Path(stats_path).parent)
        if not os.path.exists(stats_path_base):
            os.mkdir(stats_path_base)
        with open(stats_path, "wb") as pkl:
            pickle.dump(data_info, pkl)

    return dataloaders if not return_data_info else data_info