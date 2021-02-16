import warnings

import torch
import torch.nn.functional as F
from scipy import signal

from mei.legacy.utils import varargin
from ..utility.measure_helpers import get_cosine_mask


class BlurAndCut:
    """ Blur an image with a Gaussian window.

    Arguments:
        sigma (float or tuple): Standard deviation in y, x used for the gaussian blurring.
        decay_factor (float): Compute sigma every iteration as `sigma + decay_factor *
            (iteration - 1)`. Ignored if None.
        truncate (float): Gaussian window is truncated after this number of standard
            deviations to each side. Size of kernel = 8 * sigma + 1
        pad_mode (string): Mode for the padding used for the blurring. Valid values are:
            'constant', 'reflect' and 'replicate'
    """

    def __init__(self, sigma, decay_factor=None, truncate=4, pad_mode="reflect", cut_channel=None):
        self.sigma = sigma if isinstance(sigma, tuple) else (sigma,) * 2
        self.decay_factor = decay_factor
        self.truncate = truncate
        self.pad_mode = pad_mode
        self.cut_channel = cut_channel
        if cut_channel is not None:
            print(f"cutting channel: {cut_channel}")

    @varargin
    def __call__(self, x, iteration=None):
        num_channels = x.shape[1]

        # Update sigma if needed
        if self.decay_factor is None:
            sigma = self.sigma
        else:
            sigma = tuple(s + self.decay_factor * (iteration - 1) for s in self.sigma)

        # Define 1-d kernels to use for blurring
        y_halfsize = max(int(round(sigma[0] * self.truncate)), 1)
        y_gaussian = signal.gaussian(2 * y_halfsize + 1, std=sigma[0])
        x_halfsize = max(int(round(sigma[1] * self.truncate)), 1)
        x_gaussian = signal.gaussian(2 * x_halfsize + 1, std=sigma[1])
        y_gaussian = torch.as_tensor(y_gaussian, device=x.device, dtype=x.dtype)
        x_gaussian = torch.as_tensor(x_gaussian, device=x.device, dtype=x.dtype)

        # Blur
        padded_x = F.pad(x, pad=(x_halfsize, x_halfsize, y_halfsize, y_halfsize), mode=self.pad_mode)
        blurred_x = F.conv2d(padded_x, y_gaussian.repeat(num_channels, 1, 1)[..., None], groups=num_channels)
        blurred_x = F.conv2d(blurred_x, x_gaussian.repeat(num_channels, 1, 1, 1), groups=num_channels)
        final_x = blurred_x / (y_gaussian.sum() * x_gaussian.sum())  # normalize

        if self.cut_channel is not None:
            final_x[:, self.cut_channel, ...] *= 0

        return final_x


class ChangeNormAndClip:
    """ Change the norm of the input.

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(self, norm, x_min, x_max):
        self.norm = norm
        self.x_min = x_min
        self.x_max = x_max

    @varargin
    def __call__(self, x, iteration=None):
        x_norm = torch.norm(x.view(len(x), -1), dim=-1)
        renorm = x * (self.norm / x_norm).view(len(x), *[1] * (x.dim() - 1))
        return torch.clamp(renorm, self.x_min, self.x_max)


class ChangeNormClipSetBackground:
    """ Change the norm of the input.

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(self, norm, x_min, x_max, background):
        self.norm = norm
        self.x_min = x_min
        self.x_max = x_max
        self.background = background

    @varargin
    def __call__(self, x, iteration=None):
        x_norm = torch.norm((x + self.background).view(len(x), -1), dim=-1)
        renorm = x * (self.norm / x_norm).view(len(x), *[1] * (x.dim() - 1)) + self.background
        return torch.clamp(renorm, self.x_min, self.x_max)


class MaskChangeNormClip:
    """ Change the norm of the input.

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(self, norm, x_min, x_max, mask_width, mask_height, ppd, fade_start_degrees):

        self.mask = get_cosine_mask(mask_width, mask_height, ppd, fade_start_degrees)
        self.norm = norm
        self.x_min = x_min
        self.x_max = x_max



    @varargin
    def __call__(self, x, iteration=None):

        mask = torch.as_tensor(self.mask, device=x.device, dtype=x.dtype)
        x = x * mask
        x_norm = torch.norm(x.view(len(x), -1), dim=-1)
        renorm = x * (self.norm / x_norm).view(len(x), *[1] * (x.dim() - 1))
        return torch.clamp(renorm, self.x_min, self.x_max)


class BatchedCropsPadded:
    """ Create a batch of crops of the original image.

    Arguments:
        height (int): Height of the crop
        width (int): Width of the crop
        step_size (int or tuple): Number of pixels in y, x to step for each crop.
        sigma (float or tuple): Sigma in y, x for the gaussian mask applied to each batch.
            None to avoid masking

    Note:
        Increasing the stride of every convolution to stride * step_size produces the same
        effect in a much more memory efficient way but it will be architecture dependent
        and may not play nice with the rest of transforms.
    """

    def __init__(self, height, width, step_size, sigma=None, padding=0):
        self.height = height
        self.width = width
        self.padding = padding
        self.step_size = step_size if isinstance(step_size, tuple) else (step_size,) * 2
        self.sigma = sigma if sigma is None or isinstance(sigma, tuple) else (sigma,) * 2

        # If needed, create gaussian mask
        if sigma is not None:
            y_gaussian = signal.gaussian(height, std=self.sigma[0])
            x_gaussian = signal.gaussian(width, std=self.sigma[1])
            self.mask = y_gaussian[:, None] * x_gaussian

    @varargin
    def __call__(self, x, iteration=None):
        if len(x) > 1:
            raise ValueError("x can only have one example.")
        if x.shape[-2] < self.height or x.shape[-1] < self.width:
            raise ValueError("x should be larger than the expected crop")

        # Take crops
        crops = []
        for i in range(0 + self.padding, x.shape[-2] - self.height + 1 - self.padding, self.step_size[0]):
            for j in range(0 + self.padding, x.shape[-1] - self.width + 1 - self.padding, self.step_size[1]):
                crops.append(x[..., i : i + self.height, j : j + self.width])
        crops = torch.cat(crops, dim=0)

        # Multiply by a gaussian mask if needed
        if self.sigma is not None:
            mask = torch.as_tensor(self.mask, device=crops.device, dtype=crops.dtype)
            crops = crops * mask

        return crops