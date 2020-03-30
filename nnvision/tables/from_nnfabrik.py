import datajoint as dj
from nnfabrik.template import TrainedModelBase
import tempfile
import torch
import os
from nnfabrik.main import Model, Dataset, Trainer, Seed, Fabrikant
from nnfabrik.utility.dj_helpers import gitlog, make_hash
import numpy as np
from .main import MonkeyExperiment
schema = dj.schema(dj.config.get('schema_name', 'nnfabrik_core'))


@schema
class TrainedModel(TrainedModelBase):
    table_comment = "Trained models"


class TrainedModel_Scores(TrainedModelBase):
    table_comment = "Trained models"

    class Scores(dj.Part):
        definition = """
        -> master
        ---
        validation_corr:          float
        test_corr:               float
        fraction_oracle:        float
        feve:                   float
        """

    class UnitScores(dj.Part):
        definition = """
        -> master
        -> MonkeyExperiment.Units
        ---
        unit_val_corr: float
        unit_test_corr: float
        unit_feve:  float

        """

    def make(self, key):
        """
        Given key specifying configuration for dataloaders, model and trainer,
        trains the model and saves the trained model.

        Part Tables will re refactored into computed tables. Additional metrics to be added:
        avg_correlation:        float
        connor_ev               float
        poisson_loss:           float
        information_gain:       float
        """

        # lookup the fabrikant corresponding to the current DJ user
        fabrikant_name = Fabrikant.get_current_user()
        seed = (Seed & key).fetch1('seed')

        # load everything
        dataloaders, model, trainer = self.load_model(key, include_trainer=True, include_state_dict=False, seed=seed)

        # define callback with pinging
        def call_back(**kwargs):
            self.connection.ping()
            self.call_back(**kwargs)

        # model training
        score, output, model_state = trainer(model=model, dataloaders=dataloaders, seed=seed, uid=key, cb=call_back)

        with tempfile.TemporaryDirectory() as temp_dir:
            filename = make_hash(key) + '.pth.tar'
            filepath = os.path.join(temp_dir, filename)
            torch.save(model_state, filepath)

            key['score'] = score
            key['output'] = output
            key['fabrikant_name'] = fabrikant_name
            comments = []
            comments.append((self.trainer_table & key).fetch1("trainer_comment"))
            comments.append((self.model_table & key).fetch1("model_comment"))
            comments.append((self.dataset_table & key).fetch1("dataset_comment"))
            key['comment'] = self.comment_delimitter.join(comments)
            self.insert1(key)

            key['model_state'] = filepath

            self.ModelStorage.insert1(key, ignore_extra_fields=True)

            key['validation_corr'] = output["validation_corr"]
            key['test_corr'] = output["test_corr"]
            key['feve'] = output["feve"]
            key['fraction_oracle'] = 0.1

            self.Scores().insert1(key, ignore_extra_fields=True)

            validation_correlation = output["unit_val_corr"]
            test_correlation = output["unit_test_corr"]
            for data_key, val_corrs in validation_correlation.items():
                for i, unit_val_corr in enumerate(val_corrs):
                    key.pop("unit_id") if "unit_id" in key else None
                    key.pop("session_id") if "session_id" in key else None
                    unit_id = (MonkeyExperiment.Units() & key & "session_id = '{}'".format(
                        data_key) & "unit_position = {}".format(i)).fetch1("unit_id")

                    key["unit_id"] = unit_id
                    key["session_id"] = data_key
                    key["unit_val_corr"] = unit_val_corr
                    key["unit_test_corr"] = test_correlation[data_key][i]
                    key["unit_feve"] = np.sqrt(test_correlation[data_key][i])
                    self.UnitScores.insert1(key, ignore_extra_fields=True)