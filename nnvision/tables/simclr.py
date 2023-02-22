import tempfile
import torch
import pickle
from nnfabrik.utility.dj_helpers import make_hash



import os
import datajoint as dj
dj.config['nnfabrik.schema_name'] = "nnfabrik_v4_contrastive_clustering"

from nnfabrik.templates import TrainedModelBase, DataInfoBase
from nnfabrik.main import Model, Dataset, Trainer, Seed, Fabrikant
from nnfabrik.utility.dj_helpers import CustomSchema


schema = CustomSchema(dj.config.get("nnfabrik.schema_name", "nnfabrik_core"))

if not "stores" in dj.config:
    dj.config["stores"] = {}
dj.config["stores"]["minio"] = {  # store in s3
    "protocol": "s3",
    "endpoint": os.environ.get("MINIO_ENDPOINT", "DUMMY_ENDPOINT"),
    "bucket": "nnfabrik",
    "location": "dj-store",
    "access_key": os.environ.get("MINIO_ACCESS_KEY", "FAKEKEY"),
    "secret_key": os.environ.get("MINIO_SECRET_KEY", "FAKEKEY"),
    "secure": True,
    'subfolding': (2, 2)
}


class CustomTrainedModelBase(TrainedModelBase):
    disk_storage_path = "/data/simclr/"

    def make(self, key):
        """
        Given key specifying configuration for dataloaders, model and trainer,
        trains the model and saves the trained model.
        """
        # lookup the fabrikant corresponding to the current DJ user
        fabrikant_name = self.user_table.get_current_user()
        seed = (self.seed_table & key).fetch1("seed")

        # load everything
        dataloaders, model, trainer = self.load_model(key, include_trainer=True, include_state_dict=False, seed=seed)

        # define callback with pinging
        def call_back(**kwargs):
            self.connection.ping()
            self.call_back(**kwargs)

        print("pinging in training")
        # model training
        score, output, model_state = trainer(model=model, dataloaders=dataloaders, seed=seed, uid=key, cb=call_back)

        # save resulting model_state into a temporary file to be attached
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = make_hash(key) + ".pth.tar"
            filepath = os.path.join(temp_dir, filename)
            torch.save(model_state, filepath)

            # also save key, output, and model weight on disk
            base_name = make_hash(key)

            # DJ key
            filename = base_name + "_key.pickle"
            f = os.path.join(self.disk_storage_path, filename)
            with open(f, 'wb') as pkl:
                pickle.dump(key, pkl)

            # output dict
            filename = base_name + "_output.pickle"
            f = os.path.join(self.disk_storage_path, filename)
            with open(f, 'wb') as pkl:
                pickle.dump(output, pkl)

            # model state dict
            filename = base_name + "_model_state.pth.tar"
            f = os.path.join(self.disk_storage_path, filename)
            torch.save(model_state, f)




            key["score"] = score
            key["output"] = output
            key["fabrikant_name"] = fabrikant_name
            comments = []
            comments.append((self.trainer_table & key).fetch1("trainer_comment"))
            comments.append((self.model_table & key).fetch1("model_comment"))
            comments.append((self.dataset_table & key).fetch1("dataset_comment"))
            key["comment"] = self.comment_delimitter.join(comments)
            self.insert1(key)

            key["model_state"] = filepath

            self.ModelStorage.insert1(key, ignore_extra_fields=True)

@schema
class TrainedModel(CustomTrainedModelBase):
    table_comment = "Trained models"
    data_info_table = None
    storage = "minio"

    model_table = Model
    dataset_table = Dataset
    trainer_table = Trainer
    seed_table = Seed
    user_table = Fabrikant