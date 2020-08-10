import warnings

import torch
import torch.nn.functional as F
from scipy import signal

from mei.legacy.utils import varargin


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

