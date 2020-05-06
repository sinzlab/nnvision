from featurevis.ops import ChangeStd, GaussianBlur, Jitter, TotalVariation, ChangeNorm, ClipRange
from featurevis.utils import Compose

from .utility import cumstom_initial_guess
from functools import partial


# Contrast
postup_contrast_01 = ChangeStd(0.1)
postup_contrast_1 = ChangeStd(1)
postup_contrast_125 = ChangeStd(12.5)
postup_contrast_100 = ChangeStd(10)
postup_contrast_5 = ChangeStd(5)

# Blurring
Blur_sigma_1 = GaussianBlur(1)


# Regularizers of DiCarlo 2019
jitter_DiCarlo = Jitter(max_jitter=(2, 4))
total_variation_DiCarlo = TotalVariation(weight=0.001)
gradient_DiCarlo = Compose([ChangeNorm(1), ClipRange(-1, 1)])

# Initial Guess
rgb_initial_guess = partial(cumstom_initial_guess, mean=111, std=60)
