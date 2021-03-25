import torch
import torch.utils.data as utils
import numpy as np
import pickle
#from retina.retina import warp_image
from collections import namedtuple, Iterable
import os
from mlutils.data.samplers import RepeatsBatchSampler
from .utility import get_validation_split, ImageCache, get_cached_loader, get_fraction_of_training_images
from nnvision.datasets.utility import Rescale, Crop
from nnfabrik.utility.nn_helpers import get_module_output, set_random_seed, get_dims_for_loader_dict
from nnfabrik.utility.dj_helpers import make_hash
from torchvision import transforms
from bias_transfer.trainer.main_loop_modules import NoiseAugmentation
from functools import partial
from torchvision import datasets
from imagecorruptions import corrupt
from imagecorruptions.corruptions import *
from nnvision.datasets.utility import convert_to_np

def monkey_static_loader(dataset,
                         neuronal_data_files,
                         image_cache_path,
                         original_image_cache_path,
                         batch_size=64,
                         train_transformation=False,
                         normalize=True,
                         seed=None,
                         train_frac=0.8,
                         subsample=1,
                         crop=((96,96), (96,96)),
                         scale=1.,
                         time_bins_sum=tuple(range(12)),
                         avg=False,
                         image_file=None,
                         return_data_info=False,
                         store_data_info=True,
                         image_frac=1.,
                         image_selection_seed=None, load_all_in_memory=True,
                         randomize_image_selection=True, num_workers=1, pin_memory=True,
                         target_types=["v1"], stats={}, apply_augmentation=None, input_size=64, apply_grayscale=True,
                         add_fly_corrupted_test={}, resize=0, individual_image_paths=False):
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
                i.e. [(crop_left, crop_right), (crop_top, crop_bottom)]
        scale: float or integer - up-scale or down-scale via interpolation hte input images (default= 1)
        time_bins_sum: sums the responses over x time bins.
        avg: Boolean - Sums oder Averages the responses across bins.

    Returns: nested dictionary of dataloaders
    """

    seed = 1000
    torch.manual_seed(seed)
    np.random.seed(seed)

    def apply_one_noise(x, std_value=None):
        noise_config = {
            "std": {std_value: 1.0}
        }

        return NoiseAugmentation.apply_noise(x, device="cpu", **noise_config)[0]


    dataset_config = locals()

    # initialize dataloaders as empty dict
    dataloaders = {'train': {}, 'validation': {}, 'test': {}}

    if not isinstance(time_bins_sum, Iterable):
        time_bins_sum = tuple(range(time_bins_sum))

    if isinstance(crop, int):
        crop = [(crop, crop), (crop, crop)]

    # clean up image path because of legacy folder structure
    if not individual_image_paths:
        image_cache_path = image_cache_path.split('individual')[0]
    else:
        with open(image_cache_path, "rb") as pkl:
            paths_dict = pickle.load(pkl)
        image_cache_path = original_image_cache_path = paths_dict

    # Load image statistics if present
    stats_filename = make_hash(dataset_config)
    if not individual_image_paths:
        stats_path = os.path.join(image_cache_path, 'statistics/', stats_filename)
    else:
        stats_path = ""
    
    # Get mean and std
    if stats:
        img_mean = np.float32(stats['mean'])
        img_std = np.float32(stats['std'])
        # Initialize cache
        cache = ImageCache(path=image_cache_path, load_all_in_memory=load_all_in_memory)
        original_cache = ImageCache(path=original_image_cache_path, load_all_in_memory=load_all_in_memory)
    elif os.path.exists(stats_path):
        with open(stats_path, "rb") as pkl:
            data_info = pickle.load(pkl)
        if return_data_info:
            return data_info
        img_mean = list(data_info.values())[0]["img_mean"]
        img_std = list(data_info.values())[0]["img_std"]

        # Initialize cache
        cache = ImageCache(path=image_cache_path, load_all_in_memory=load_all_in_memory)
        original_cache = ImageCache(path=original_image_cache_path, load_all_in_memory=load_all_in_memory)
    else:
        # Initialize cache with no normalization
        cache = ImageCache(path=image_cache_path, subsample=subsample, crop=crop, scale=scale, load_all_in_memory=load_all_in_memory)
        original_cache = ImageCache(path=original_image_cache_path, load_all_in_memory=load_all_in_memory)
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

    names = tuple(['inputs'])
    if "img_classification" in target_types:
        names += ('labels',)
    if "v1" in target_types or "v4" in target_types:
        names += ('responses',)

    # cycling through all datafiles to fill the dataloaders with an entry per session
    for i, datapath in enumerate(neuronal_data_files):

        with open(datapath, "rb") as pkl:
            raw_data = pickle.load(pkl)

        subject_ids = raw_data["subject_id"]
        data_key = str(raw_data["session_id"])
        responses_train = raw_data["training_responses"].astype(np.float32)
        responses_test = raw_data["testing_responses"].astype(np.float32)
        if "img_classification" in target_types:
            labels_train = raw_data["training_labels"].astype(np.float32)
            labels_test = raw_data["testing_labels"].astype(np.float32)
        training_image_ids = raw_data["training_image_ids"] - image_id_offset
        testing_image_ids = raw_data["testing_image_ids"] - image_id_offset

        if dataset != 'PlosCB19_V1':
            responses_test = responses_test.transpose((2, 0, 1))
            responses_train = responses_train.transpose((2, 0, 1))

            if time_bins_sum is not None:  # then average over given time bins
                responses_train = (np.mean if avg else np.sum)(responses_train[:, :, time_bins_sum], axis=-1)
                responses_test = (np.mean if avg else np.sum)(responses_test[:, :, time_bins_sum], axis=-1)

        if image_frac < 1:
            if randomize_image_selection:
                image_selection_seed = int(image_selection_seed*image_frac)
            idx_out = get_fraction_of_training_images(image_ids=training_image_ids, fraction=image_frac, seed=image_selection_seed)
            training_image_ids = training_image_ids[idx_out]
            responses_train = responses_train[idx_out]

        train_idx = np.isin(training_image_ids, all_train_ids)
        val_idx = np.isin(training_image_ids, all_validation_ids)

        train_data = dict()
        val_data = dict()
        test_data = dict()
        validation_image_ids = training_image_ids[val_idx]
        training_image_ids = training_image_ids[train_idx]
        train_data['inputs'] = training_image_ids
        val_data['inputs'] = validation_image_ids
        test_data['inputs'] = testing_image_ids


        if "img_classification" in target_types:
            labels_val = labels_train[val_idx]
            labels_train = labels_train[train_idx]
            train_data['labels'] = labels_train
            val_data['labels'] = labels_val
            test_data['labels'] = labels_test
        if "v1" in target_types or "v4" in target_types:
            responses_val = responses_train[val_idx]
            responses_train = responses_train[train_idx]
            train_data['responses'] = responses_train
            val_data['responses'] = responses_val
            test_data['responses'] = responses_test

        transform_train = [
            transforms.ToPILImage() if (resize and train_transformation) else None,
            transforms.Resize((resize, resize)) if (resize and train_transformation) else None,
            transforms.Lambda(convert_to_np) if (resize and train_transformation) else None,
            transforms.Lambda(Crop(crop, subsample)),
            transforms.Lambda(Rescale(scale)),
            transforms.ToPILImage() if apply_augmentation else None,
            transforms.RandomCrop(input_size, padding=4) if apply_augmentation else None,
            transforms.RandomHorizontalFlip() if apply_augmentation else None,
            transforms.RandomRotation(15) if apply_augmentation else None,
            transforms.ToTensor(),
            transforms.Grayscale() if (apply_grayscale and train_transformation) else None,
            transforms.Normalize(img_mean, img_std)
            if normalize
            else None,
        ]

        transform_val = [
            transforms.ToPILImage() if (resize and train_transformation) else None,
            transforms.Resize((resize, resize)) if (resize and train_transformation) else None,
            transforms.Lambda(convert_to_np) if (resize and train_transformation) else None,
            transforms.Lambda(Crop(crop, subsample)),
            transforms.Lambda(Rescale(scale)),
            transforms.ToTensor(),
            transforms.Grayscale() if (apply_grayscale and train_transformation) else None,
            transforms.Normalize(img_mean, img_std)
            if normalize
            else None,
        ]

        transform_train = transforms.Compose(
            list(filter(lambda x: x is not None, transform_train))
        )
        transform_val = transforms.Compose(
            list(filter(lambda x: x is not None, transform_val))
        )

        train_loader = get_cached_loader(train_data, names=names, batch_size=batch_size,image_cache=cache, transform=transform_train,
                                         num_workers=num_workers, pin_memory=pin_memory,)
        val_loader = get_cached_loader(val_data, names=names, batch_size=batch_size, image_cache=cache, transform=transform_val,
                                       num_workers=num_workers, pin_memory=pin_memory,)

        if dataset in ["CSRF19_V1", "CSRF19_V4"]:
            test_loader = get_cached_loader(test_data,
                                            names=names,num_workers=num_workers, pin_memory=pin_memory,
                                            batch_size=None,
                                            shuffle=None,
                                            image_cache=cache,
                                            repeat_condition=testing_image_ids, transform=transform_val)
        else:
            test_loader = get_cached_loader(test_data, names=names, batch_size=batch_size, image_cache=cache, transform=transform_val,
                                       num_workers=num_workers, pin_memory=pin_memory,)

        dataloaders["train"][data_key] = train_loader
        dataloaders["validation"][data_key] = val_loader
        dataloaders["test"][data_key] = test_loader


        if 'labels' in names:
            transform_val_gauss_levels = {}
            for level in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
                transform_val_gauss_levels[level] = [
                    transforms.ToPILImage() if resize else None,
                    transforms.Resize((resize, resize)) if resize else None,
                    transforms.Lambda(convert_to_np) if resize else None,
                    transforms.Lambda(Crop(crop, subsample)),
                    transforms.Lambda(Rescale(scale)),
                    transforms.ToTensor(),
                    transforms.Lambda(partial(apply_one_noise, std_value=level)),
                    transforms.Grayscale() if apply_grayscale else None,
                    transforms.Normalize(img_mean, img_std)
                    if normalize
                    else None,
                ]
            for level in list(transform_val_gauss_levels.keys()):
                transform_val_gauss_levels[level] = transforms.Compose(
                    list(filter(lambda x: x is not None, transform_val_gauss_levels[level]))
                )
            val_gauss_loaders = {}
            for level in list(transform_val_gauss_levels.keys()):
                val_gauss_loaders[level] = get_cached_loader(val_data, names=names, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                                                             image_cache=original_cache, transform=transform_val_gauss_levels[level])

            dataloaders["validation_gauss"] = val_gauss_loaders

            if add_fly_corrupted_test:
                fly_test_loaders = {}
                for fly_noise_type, levels in add_fly_corrupted_test.items():
                    fly_test_loaders[fly_noise_type] = {}
                    for level in levels:

                        class Noise(object):
                            def __init__(self, noise_type, severity):
                                self.noise_type = noise_type
                                self.severity = severity

                            def __call__(self, pic):
                                pic = np.asarray(pic)
                                img = corrupt(pic, corruption_name=self.noise_type, severity=self.severity)
                                return img

                        transform_fly_test = [
                            transforms.ToPILImage() if resize else None,
                            transforms.Resize((resize, resize))if resize else None,
                            transforms.Lambda(convert_to_np) if resize else None,
                            transforms.Lambda(Crop(crop, subsample)),
                            transforms.Lambda(Rescale(scale)),
                            Noise(fly_noise_type, level),
                            transforms.ToPILImage() if apply_grayscale else None,
                            transforms.Grayscale() if apply_grayscale else None,
                            transforms.ToTensor(),
                            transforms.Normalize(img_mean, img_std)
                            if normalize
                            else None,
                        ]
                        transform_fly_test = transforms.Compose(
                            list(filter(lambda x: x is not None, transform_fly_test))
                        )
                        fly_test_loaders[fly_noise_type][level] = get_cached_loader(test_data if len(set(testing_image_ids)) > 1000 else val_data,
                                                                                    names=names, batch_size=batch_size, num_workers=num_workers,
                                                                                    pin_memory=pin_memory,
                                                                                    image_cache=original_cache, transform=transform_fly_test)

                dataloaders["fly_c_test"] = fly_test_loaders


    if (not stats) and store_data_info and (not os.path.exists(stats_path)) and ("v1" in target_types or "v4" in target_types):
        if 'labels' in names:
            in_name, _, out_name = next(iter(list(dataloaders["train"].values())[0]))._fields
        else:
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
                         mua_selector=None):
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
                i.e. [(crop_left, crop_right), (crop_top, crop_bottom)]
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

        validation_image_ids = training_image_ids[val_idx]
        training_image_ids = training_image_ids[train_idx]

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

        with open(stats_path, "wb") as pkl:
            pickle.dump(data_info, pkl)

    return dataloaders if not return_data_info else data_info