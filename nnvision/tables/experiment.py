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