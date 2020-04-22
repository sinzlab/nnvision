import datajoint as dj
from nnfabrik.template import TrainedModelBase
from nnfabrik.main import Model, Dataset, Trainer, Seed, Fabrikant
from nnfabrik.utility.dj_helpers import gitlog, make_hash
from nnfabrik.template import DataInfoBase
from nnfabrik.builder import resolve_data
import os
import pickle

schema = dj.schema(dj.config.get('schema_name', 'nnfabrik_core'))


@schema
class TrainedModel(TrainedModelBase):
    table_comment = "Trained models"

@schema
class DataInfo(DataInfoBase):

    def create_stats_files(self, key=None, path=None):

        if key == None:
            key = self.fetch('KEY')

            for restr in key:
                dataset_config = (self.dataset_table & restr).fetch1("dataset_config")
                image_cache_path = dataset_config.get("image_cache_path", None)
                image_cache_path = os.path.dirname(image_cache_path) if 'individual' in image_cache_path else image_cache_path
                stats_filename = make_hash(dataset_config)
                stats_path = os.path.join(path if path is not None else image_cache_path, 'statistics/', stats_filename)

                if not os.path.exists(stats_path):
                    data_info = (self & restr).fetch1("data_info")

                    with open(stats_path, "wb") as pkl:
                        pickle.dump(data_info, pkl)
