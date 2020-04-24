from featurevis.ops import ChangeStd, GaussianBlur

postup_contrast_01 = ChangeStd(0.1)
postup_contrast_1 = ChangeStd(1)
postup_contrast_125 = ChangeStd(12.5)
postup_contrast_100 = ChangeStd(10)
postup_contrast_5 = ChangeStd(5)

Blur_sigma_1 = GaussianBlur(1)