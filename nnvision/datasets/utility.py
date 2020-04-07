import numpy as np
from torch.utils.data import DataLoader
from mlutils.data.datasets import StaticImageSet, FileTreeDataset
from mlutils.data.samplers import RepeatsBatchSampler


def get_oracle_dataloader(dat,
                          toy_data=False,
                          file_tree_dataset=False,
                          oracle_condition=None,
                          ):

    if toy_data:
        condition_hashes = dat.info.condition_hash
    else:

        if 'image_id' in dir(dat.info):
            condition_hashes = dat.info.image_id
        elif 'colorframeprojector_image_id' in dir(dat.info):
            condition_hashes = dat.info.colorframeprojector_image_id
        elif 'frame_image_id' in dir(dat.info):
            condition_hashes = dat.info.frame_image_id
        else:
            raise ValueError("'image_id' 'colorframeprojector_image_id', or 'frame_image_id' have to present in the dataset under dat.info "
                             "in order to load get the oracle repeats.")

    image_class = dat.info.image_class if "image_class" in dir(dat.info) else dat.info.frame_image_class
    max_idx = condition_hashes.max() + 1
    _, class_idx = np.unique(image_class, return_inverse=True)
    identifiers = condition_hashes + class_idx * max_idx

    sampling_condition = np.where(dat.tiers == 'test')[0] if oracle_condition is None else \
        np.where((dat.tiers == 'test') & (class_idx == oracle_condition))[0]

    sampler = RepeatsBatchSampler(identifiers, sampling_condition)
    return DataLoader(dat, sampler=sampler)
