import datajoint as dj
from nnfabrik.template import TrainedModelBase
from nnfabrik.main import Model, Dataset, Trainer, Seed, Fabrikant
from nnfabrik.utility.dj_helpers import gitlog, make_hash
from nnfabrik.template import DataInfoBase
from nnfabrik.builder import resolve_data
import os
import pickle
from pathlib import Path
from ..utility.dj_helpers import get_default_args

schema = dj.schema(dj.config.get('schema_name', 'nnfabrik_core'))


@schema
class DataInfo(DataInfoBase):

    def create_stats_files(self, key=None, path=None):

        if key == None:
            key = self.fetch('KEY')

            for restr in key:
                dataset_config = (self.dataset_table & restr).fetch1("dataset_config")
                image_cache_path = dataset_config.get("image_cache_path", None)
                if image_cache_path is None:
                    raise ValueError("The argument image_cache_path has to be specified in the dataset_config in order "
                                     "to create the DataInfo")

                image_cache_path = image_cache_path.split('individual')[0]
                default_args = get_default_args(resolve_data((self.dataset_table & restr).fetch1("dataset_fn")))

                stats_filename = make_hash(default_args.update(dataset_config))
                stats_path = os.path.join(path if path is not None else image_cache_path, 'statistics/', stats_filename)

                if not os.path.exists(stats_path):
                    data_info = (self & restr).fetch1("data_info")

                    with open(stats_path, "wb") as pkl:
                        pickle.dump(data_info, pkl)


@schema
class TrainedModel(TrainedModelBase):
    table_comment = "Trained models"
    data_info_table = DataInfo
