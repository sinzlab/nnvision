from torch.utils.data import DataLoader
import torch
import torch.utils.data as utils
import numpy as np
#from retina.retina import warp_image
from skimage.transform import rescale
from collections import namedtuple, Iterable
import os
from neuralpredictors.data.samplers import RepeatsBatchSampler


def get_oracle_dataloader(dat,
                          toy_data=False,
                          oracle_condition=None,
                          verbose=False,
                          file_tree=False):

    if toy_data:
        condition_hashes = dat.info.condition_hash
    else:
        dat_info = dat.info if not file_tree else dat.trial_info
        if 'image_id' in dir(dat_info):
            condition_hashes = dat_info.image_id
            image_class = dat_info.image_class

        elif 'colorframeprojector_image_id' in dir(dat_info):
            condition_hashes = dat_info.colorframeprojector_image_id
            image_class = dat_info.colorframeprojector_image_class
        elif 'frame_image_id' in dir(dat_info):
            condition_hashes = dat_info.frame_image_id
            image_class = dat_info.frame_image_class
        else:
            raise ValueError("'image_id' 'colorframeprojector_image_id', or 'frame_image_id' have to present in the dataset under dat.info "
                             "in order to load get the oracle repeats.")

    max_idx = condition_hashes.max() + 1
    classes, class_idx = np.unique(image_class, return_inverse=True)
    identifiers = condition_hashes + class_idx * max_idx

    dat_tiers = dat.tiers if not file_tree else dat.trial_info.tiers
    sampling_condition = np.where(dat_tiers == 'test')[0] if oracle_condition is None else \
        np.where((dat_tiers == 'test') & (class_idx == oracle_condition))[0]
    if (oracle_condition is not None) and verbose:
        print("Created Testloader for image class {}".format(classes[oracle_condition]))

    sampler = RepeatsBatchSampler(identifiers, sampling_condition)
    return DataLoader(dat, batch_sampler=sampler)


def get_validation_split(n_images, train_frac, seed):
    """
    Splits the total number of images into train and test set.
    This ensures that in every session, the same train and validation images are being used.

    Args:
        n_images: Total number of images. These will be plit into train and validation set
        train_frac: fraction of images used for the training set
        seed: random seed

    Returns: Two arrays, containing image IDs of the whole imageset, split into train and validation

    """
    if seed: np.random.seed(seed)
    train_idx, val_idx = np.split(np.random.permutation(int(n_images)), [int(n_images*train_frac)])
    assert not np.any(np.isin(train_idx, val_idx)), "train_set and val_set are overlapping sets"

    return train_idx, val_idx


def get_fraction_of_training_images(image_ids, fraction, seed=None, replace=False):

    if len(image_ids.shape) > 1:
        raise ValueError("image_ids must be a one dimensional vector ")

    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()

    n_images = len(image_ids)
    image_indices = np.arange(n_images)
    idx_out = np.random.choice(image_indices, int(n_images * fraction), replace=replace)
    return idx_out


class ImageCache:
    """
    A simple cache which loads images into memory given a path to the directory where the images are stored.
    Images need to be present as 2D .npy arrays
    """

    def __init__(self, path=None,
                 subsample=1,
                 crop=0,
                 scale=1,
                 img_mean=None,
                 img_std=None,
                 transform=True,
                 normalize=True,
                 filename_precision=6,
                 ):
        """

        path: str - pointing to the directory, where the individual .npy files are present
        subsample: int - amount of downsampling
        crop:  the expected input is a list of tuples, the specify the exact cropping from all four sides
                i.e. [(crop_left, crop_right), (crop_top, crop_down)]
        scale: - the scale factor to upsample or downsample images via interpolation
        img_mean: - mean luminance across all images
        img_std: - std of the luminance across all images
        transform: - whether to apply a transformation to an image
        normalize: - whether to standarized inputs by the mean and variance
        filename_precision: - amount leading zeros of the files in the specified folder
        """
        
        self.cache = {}
        self.path = path
        self.subsample = subsample
        self.crop = crop
        self.scale = scale
        self.img_mean = img_mean
        self.img_std = img_std
        self.transform = transform
        self.normalize = normalize
        self.leading_zeros = filename_precision

    def __len__(self):
        return len([file for file in os.listdir(self.path) if file.endswith('.npy')])

    def __contains__(self, key):
        return key in self.cache

    def __getitem__(self, item):
        item = item.tolist() if isinstance(item, Iterable) else item
        return [self[i] for i in item] if isinstance(item, Iterable) else self.update(item)

    def update(self, key):
        if key in self.cache:
            return self.cache[key]
        else:
            filename = os.path.join(self.path, str(key).zfill(self.leading_zeros) + '.npy')
            image = np.load(filename, allow_pickle=True)
            image = self.transform_image(image) if self.transform else image
            image = self.normalize_image(image) if self.normalize else image
            image = torch.tensor(image).to(torch.float)
            self.cache[key] = image
            return image

    def transform_image(self, image):
        """
        applies transformations to the image: downsampling, cropping, rescaling, and dimension expansion.
        """
        if len(image.shape) == 2:
            h, w = image.shape
            rescale_fn = lambda x, s: rescale(x,
                                              s,
                                              mode='reflect',
                                              multichannel=False,
                                              anti_aliasing=False,
                                              preserve_range=True).astype(x.dtype)
            image = image[self.crop[0][0]:h - self.crop[0][1]:self.subsample,
                    self.crop[1][0]:w - self.crop[1][1]:self.subsample]
            image = image if self.scale == 1 else rescale_fn(image, self.scale)
            image = image[None,]
            return image

        elif len(image.shape) == 3:
            h, w = image.shape[:2]
            rescale_fn = lambda x, s: rescale(x, s, mode='reflect', multichannel=True, anti_aliasing=False,
                                              preserve_range=True).astype(x.dtype)
            image = image[self.crop[0][0]:h - self.crop[0][1]:self.subsample,
                    self.crop[1][0]:w - self.crop[1][1]:self.subsample, ...]
            image = image if self.scale == 1 else rescale_fn(image, self.scale)
            image = image[None,].permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Image shape has to be two dimensional (grayscale) or three dimensional "
                             f"(color, with w x h x c). got image shape {image.shape}")
        return image
    
    def normalize_image(self, image):
        """
        standarizes image
        """
        image = (image - self.img_mean) / self.img_std
        return image

    @property
    def cache_size(self):
        return len(self.cache)
    
    @property
    def loaded_images(self):
        print('Loading images ...')
        items = [int(file.split('.')[0]) for file in os.listdir(self.path) if file.endswith('.npy')]
        images = torch.stack([self.update(item) for item in items])
        return images
    
    
    def zscore_images(self, update_stats=True):
        """
        zscore images in cache
        """
        images   = self.loaded_images
        img_mean = images.mean()
        img_std  = images.std()
        
        for key in self.cache:
            self.cache[key] = (self.cache[key] - img_mean) / img_std
        
        if update_stats:
            self.img_mean = np.float32(img_mean.item())
            self.img_std  = np.float32(img_std.item())
        
    @property
    def image_shape(self):
        if self.cache_size > 0:
            image = next(iter(self.cache.values()))
            return image.shape
        else:
            self.update(1)
            return self.image_shape


class CachedTensorDataset(utils.Dataset):
    """
    Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors, names=('inputs', 'targets'), image_cache=None, prev_img=None, trial_id=None):
        if not all(tensors[0].size(0) == tensor.size(0) for tensor in tensors):
            raise ValueError('The tensors of the dataset have unequal lenghts. The first dim of all tensors has to match exactly.')

        self.tensors = tensors
        self.input_position = names.index("inputs")
        self.DataPoint = namedtuple('DataPoint', names)
        self.image_cache = image_cache
        self.prev_img = prev_img
        self.trial_id = trial_id

    def __getitem__(self, index):
        """
        retrieves the inputs (= tensors[0]) from the image cache. If the image ID is not present in the cache,
            the cache is updated to load the corresponding image into memory.

        Also has the functionality to include the previous image. In that case, len(tensors) will be 3
        """

        if len(self.tensors) == 2:
            if type(index) == int:
                key = self.tensors[0][index].item()
            else:
                key = self.tensors[0][index].numpy().astype(np.int64)
            tensors_expanded = [tensor[index] if pos != self.input_position else torch.stack(list(self.image_cache[key]))
                            for pos, tensor in enumerate(self.tensors)]

        elif len(self.tensors) > 2:
            if type(index) == int:
                key_img = self.tensors[0][index].item()
                if self.prev_img:
                    key_prev_img = self.tensors[1][index].item()
            else:
                key_img = self.tensors[0][index].numpy().astype(np.int64)
                if self.prev_img:
                    key_prev_img = self.tensors[1][index].numpy().astype(np.int64)

            img = torch.stack(list(self.image_cache[key_img]))
            if self.prev_img:
                prev_img = torch.stack(list(self.image_cache[key_prev_img]))

            if len(self.tensors) == 3:
                if self.trial_id:
                    trial_id = self.tensors[1][index]
                    trial_id_img = torch.ones_like(img).to(img.dtype)*trial_id

                img_channel_2 = prev_img if self.prev_img else trial_id_img
                targets = self.tensors[2][index]
                full_img = torch.cat([img, img_channel_2], dim=1) if len(img.shape) > 3 else torch.cat([img, img_channel_2], dim=0)
            else:
                trial_id = self.tensors[2][index]
                targets = self.tensors[3][index]
                trial_id_img= torch.ones_like(img).to(img.dtype)*trial_id
                full_img = torch.cat([img, prev_img, trial_id_img], dim=1) if len(img.shape) > 3 else torch.cat([img, prev_img, trial_id_img], dim=0)

            tensors_expanded = [full_img, targets]

        return self.DataPoint(*tensors_expanded)

    def __len__(self):
        return self.tensors[0].size(0)


def get_cached_loader(*args,
                      batch_size=None,
                      shuffle=True,
                      image_cache=None,
                      repeat_condition=None,
                      names=('inputs', 'targets'),
                      include_prev_image=False,
                      include_trial_id=False):
    """

    Args:
        image_ids: an array of image IDs
        responses: Numpy Array, Dimensions: N_images x Neurons
        batch_size: int - batch size for the dataloader
        shuffle: Boolean, shuffles image in the dataloader if True
        image_cache: a cache object which stores the images

    Returns: a PyTorch DataLoader object
    """

    image_ids = torch.from_numpy(args[0].astype(np.int64))
    #breakhere
    tensors = [image_ids]

    if len(args) > 2 and "eye_position" not in names:
        if include_prev_image:
            prev_image_ids = torch.from_numpy(args[1].astype(np.int64))
            tensors.append(prev_image_ids)
        if include_trial_id and include_prev_image:
            tensors.append(torch.from_numpy(args[2]).to(torch.float))
            responses = torch.tensor(args[3]).to(torch.float)
        if include_trial_id and not include_prev_image:
            tensors.append(torch.from_numpy(args[1]).to(torch.float))
            responses = torch.tensor(args[2]).to(torch.float)
    else:
        responses = torch.tensor(args[1]).to(torch.float)

    tensors.append(responses)
    if len(args) > 2 and "eye_position" in names:
        eye_position = torch.tensor(args[2]).to(torch.float)
        tensors.append(eye_position)
    dataset = CachedTensorDataset(*tensors, image_cache=image_cache, names=names,prev_img=include_prev_image, trial_id=include_trial_id, )
    sampler = RepeatsBatchSampler(repeat_condition) if repeat_condition is not None else None

    dataloader = utils.DataLoader(dataset, batch_sampler=sampler) if batch_size is None else utils.DataLoader(dataset,
                                                                                                            batch_size=batch_size,
                                                                                                            shuffle=shuffle,
                                                                                                            )
    return dataloader


def get_crop_from_stimulus_location(stimulus_location, crop, monitor_scaling_factor=4.57):
    """

    Args:
        stimulus_location: in pixels on the presentation monitor (1920 x 1080 resolution).
        crop (tuple): [(crop_top, crop_bottom), (crop_left, crop_right)]
        monitor_scaling_factor: upscaling factor from imagenet images (default 420x420 px) to monitor presentation
            (1920 x 1080 px)

    Returns:
        crop_shifted (tuple):
    """

    # rounding down the pixels after dividing by the scaling factor
    w_shift = int(stimulus_location[0] / monitor_scaling_factor)
    h_shift = int(stimulus_location[1] / monitor_scaling_factor)

    crop_shifted = ((crop[0][0] + h_shift, crop[0][1] - h_shift),
                    (crop[1][0] + w_shift, crop[1][1] - w_shift))

    return crop_shifted

