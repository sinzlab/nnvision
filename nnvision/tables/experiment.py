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


from nnvision.utility.experiment_helpers.image_processing import shift_image_based_on_masks
from scipy.ndimage import center_of_mass

@schema
class GaussianCLMaskedControlExpanded(dj.Computed):
    definition = """
    -> GaussianCLMaskedControl
    ---
    method_fn         : varchar(64)
    method_hash       : varchar(64)
    dataset_fn        : varchar(64)
    dataset_hash      : varchar(64)
    ensemble_hash     : varchar(64)
    data_key          : varchar(64)
    unit_id           : int
    unit_type         : int
    mei_seed          : int
    com_h             : float
    com_w             : float
    """

    def make(self, key):
        primary_keys, secondary_keys, mask = (GaussianCLMaskedControl & key).fetch1("KEY", "image_mask_key", "image_mask")
        com_h, com_w = center_of_mass(mask)
        insert_key = {**primary_keys, **secondary_keys, 'com_h': com_h, 'com_w': com_w}
        self.insert1(insert_key)


@schema
class ShiftCLMaskedControlExpanded(dj.Computed):
    definition = """
    -> ShiftCLMaskedControl
    ---
    method_fn         : varchar(64)
    method_hash       : varchar(64)
    dataset_fn        : varchar(64)
    dataset_hash      : varchar(64)
    ensemble_hash     : varchar(64)
    data_key          : varchar(64)
    unit_id           : int
    unit_type         : int
    image_mei_seed    : int
    shift_mei_seed    : int
    com_h             : float
    com_w             : float
    shifted_mask      : longblob
    """

    def make(self, key):
        keys = (ShiftCLMaskedControl & key).fetch1()

        com_h, com_w = center_of_mass(keys["shift_mask"])
        shifted_mask = shift_image_based_on_masks(keys["image_mask"], keys["image_mask"], [keys["shift_mask"]])[0]
        image_mei_seed = keys["image_mask_key"]["mei_seed"]
        shift_mei_seed = keys["shift_mask_key"]["mei_seed"]


        insert_key = {**keys,
                      **keys["image_mask_key"],
                      'com_h': com_h,
                      'com_w': com_w,
                      'image_mei_seed': image_mei_seed,
                      'shift_mei_seed': shift_mei_seed,
                      'shifted_mask': shifted_mask}
        self.insert1(insert_key, ignore_extra_fields=True)


@schema
class GaussianCLMaskedMEIExpanded(dj.Computed):
    definition = """
    -> GaussianCLMaskedMEI
    ---
    com_h           : longblob
    com_w                : longblob
    """

    def make(self, key):
        primary_keys, mask = (GaussianCLMaskedMEI & key).fetch1("KEY", "image_mask")
        com_h, com_w = center_of_mass(mask)
        insert_key = {**primary_keys, 'com_h': com_h, 'com_w': com_w}
        self.insert1(insert_key)


@schema
class ShiftCLMaskedMEIExpanded(dj.Computed):
    definition = """
    -> ShiftCLMaskedMEI
    ---
    shift_mei_seed    : int
    com_h             : float
    com_w             : float
    shifted_mask      : longblob
    """

    def make(self, key):
        primary_keys, shift_key, shift_mask, image_mask = (ShiftCLMaskedMEI & key).fetch1("KEY", "shift_mask_key", "shift_mask", "image_mask")
        com_h, com_w = center_of_mass(shift_mask)

        shifted_mask = shift_image_based_on_masks(image_mask, image_mask, [shift_mask])[0]
        shift_mei_seed = shift_key["mei_seed"]
        insert_key = {**primary_keys,
                      'com_h': com_h,
                      'com_w': com_w,
                      'shifted_mask': shifted_mask,
                      'shift_mei_seed':shift_mei_seed }
        self.insert1(insert_key)
