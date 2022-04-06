import warnings

import datajoint as dj
from scipy import ndimage

from nnfabrik.utility.dj_helpers import CustomSchema
schema = CustomSchema(dj.config.get('nnfabrik.schema_name', 'nnfabrik_core'))


@schema
class StaticImageClass(dj.Lookup):
    definition = """

    image_class : varchar(64)
    """
    contents = [["imagenet"], ["imagenet_v2"]]

    def pull(self, restrictions=None):
        from .external import stim
        restrictions = dict() if restrictions is None else restrictions
        self.insert(
            (stim.StaticImageClass & restrictions).fetch(), skip_duplicates=True
        )


@schema
class StaticImage(dj.Manual):
    definition = """ # static images for stimulus presentation

    -> StaticImageClass
    ---
    frame_width=256      : int                          # pixels
    frame_height=144     : int                          # pixels
    num_channels=1       : tinyint                      # whether in color or not
    """

    class Image(dj.Part):
        definition = """
        -> master
        image_id        : int            # image id
        ---
        image           : longblob       # actual image
        """

    class ImageNet(dj.Part):
        definition = """
        -> master
        image_id        : int            # image id
        imagenet_id     : varchar(25)    # id used in the actual imagenet dataset
        ---
        description     : varchar(255)   # image content
        """

    def pull(self, key):
        from .external import stim
        parent_keys = (stim.StaticImage & key).fetch()
        self.insert(parent_keys, skip_duplicates=True)

        for parent_key in parent_keys:
            child_keys = (stim.StaticImage.Image & parent_key & key).fetch(as_dict=True)
            self.Image().insert(child_keys, skip_duplicates=True)

            child_keys = (stim.StaticImage.ImageNet & parent_key & key).fetch(as_dict=True)
            self.ImageNet().insert(child_keys, skip_duplicates=True)


@schema
class GaussianCLMaskedControl(dj.Manual):
    definition = """
    -> StaticImage.Image
    cl_image_id          : int
    ---
    image_mask           : longblob
    image_mask_key       : longblob
    image                : longblob
    """

@schema
class GaussianCLMaskedMEI(dj.Manual):
    definition = """
    cl_image_id          : int
    method_fn            : varchar(64)                  # name of the method function
    method_hash          : varchar(32)                  # hash of the method config
    dataset_fn           : varchar(64)                  # name of the dataset loader function
    dataset_hash         : varchar(64)                  # hash of the configuration object
    ensemble_hash        : char(32)                     # the hash of the ensemble
    data_key             : varchar(64)                  # 
    unit_id              : int                          # 
    unit_type            : int                          # 
    mei_seed             : tinyint unsigned             # MEI seed
    ---
    image_mask           : longblob
    image                : longblob
    """

@schema
class ShiftCLMaskedControl(dj.Manual):
    definition = """
    -> StaticImage.Image
    cl_image_id          : int
    ---
    image_mask           : longblob
    shift_mask           : longblob
    image_mask_key       : longblob
    shift_mask_key       : longblob
    image                : longblob
    """

@schema
class ShiftCLMaskedMEI(dj.Manual):
    definition = """
    cl_image_id          : int
    method_fn            : varchar(64)                  # name of the method function
    method_hash          : varchar(32)                  # hash of the method config
    dataset_fn           : varchar(64)                  # name of the dataset loader function
    dataset_hash         : varchar(64)                  # hash of the configuration object
    ensemble_hash        : char(32)                     # the hash of the ensemble
    data_key             : varchar(64)                  # 
    unit_id              : int                          # 
    unit_type            : int                          # 
    mei_seed             : tinyint unsigned             # MEI seed
    ---
    image_mask           : longblob
    shift_mask           : longblob
    shift_mask_key       : longblob
    image                : longblob
    """