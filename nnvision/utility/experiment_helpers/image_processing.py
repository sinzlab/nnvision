import torch
import numpy as np
import random
import math
import os
import scipy

from copy import copy
from torch import load
from scipy import ndimage
from skimage import morphology
from skimage.transform import rescale
from scipy.io import savemat
import scipy.io as sio


def rescale_image(image, scale):
    """
    rescales the image by the specified scale factor; adapted from the transformations defined within the
    ImageCache class; makes use of skimage.transform.rescale to interpolate the image

    image: ndarray
    scale: float or tuple of floats


    mode: what to do to values outside the boundaries of the input
    multichannel: whether the last axis of the image should be interpreted as channels or spatial dimension
    anti_aliasing: whether or not to Gaussian blur the image before interpolation to avoid aliasing artifacts
    preserve_range: whether to keep original range of values
    """

    rescale_fn = lambda x, s: rescale(x, s, mode='reflect', multichannel=False,
                                      anti_aliasing=False, preserve_range=True).astype(x.dtype)

    image = image if scale == 1 else rescale_fn(image, scale)

    return image


from scipy.ndimage import center_of_mass, shift


def shift_image_based_on_masks(img, masks):
    """
    takes a single image as an input, as well as n masks, and returns a list of images, each shifted to a particular location

    img (np.array): the image to be shifted
    masks (np.arrray or list): the masks as basis for the shift
    """
    h, w = img.shape

    shifted_images = []
    for mask in masks:
        com = center_of_mass(mask, labels=None, index=None)
        shifted_image = shift(img, (h - int(com[0]), w - int(com[1])))
        shifted_images.append(shifted_image)
    return shifted_images


def normalize_image(image, img_mean, img_std):
    """
    standarizes image
    """
    image = (image - img_mean) / img_std
    return image

def transform_image(image, scale, subsample, crop):
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
        image = image[crop[0][0]:h - crop[0][1]:subsample,
                crop[1][0]:w - crop[1][1]:subsample]
        image = image if scale == 1 else rescale_fn(image, scale)
        image = image[None,]
        return image

def get_norm(img):
    norm = torch.norm(torch.from_numpy(img))
    return norm

def re_norm(img, norm):
    desiredNorm = ChangeNorm(norm)
    height, width = img.shape
    imgTensor = torch.from_numpy(img)
    renormedImage = desiredNorm.__call__(x=imgTensor.view(1, height, width))
    return renormedImage


