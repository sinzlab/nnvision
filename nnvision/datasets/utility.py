import numpy as np
from torch.utils.data import DataLoader
from mlutils.data.datasets import StaticImageSet, FileTreeDataset
from mlutils.data.samplers import RepeatsBatchSampler


def get_oracle_dataloader(dat,
                          toy_data=False,
                          oracle_condition=None,
                          verbose=False):

    if toy_data:
        condition_hashes = dat.info.condition_hash
    else:

        if 'image_id' in dir(dat.info):
            condition_hashes = dat.info.image_id
            image_class =  dat.info.image_class
        elif 'colorframeprojector_image_id' in dir(dat.info):
            condition_hashes = dat.info.colorframeprojector_image_id
            image_class = dat.info.colorframeprojector_image_class
        elif 'frame_image_id' in dir(dat.info):
            condition_hashes = dat.info.frame_image_id
            image_class = dat.info.frame_image_class
        else:
            raise ValueError("'image_id' 'colorframeprojector_image_id', or 'frame_image_id' have to present in the dataset under dat.info "
                             "in order to load get the oracle repeats.")

    max_idx = condition_hashes.max() + 1
    classes, class_idx = np.unique(image_class, return_inverse=True)
    identifiers = condition_hashes + class_idx * max_idx


    sampling_condition = np.where(dat.tiers == 'test')[0] if oracle_condition is None else \
        np.where((dat.tiers == 'test') & (class_idx == oracle_condition))[0]
    if (oracle_condition is not None) and verbose:
        print("Created Testloader for image class {}".format(classes[oracle_condition]))

    sampler = RepeatsBatchSampler(identifiers, sampling_condition)
    return DataLoader(dat, sampler=sampler)
