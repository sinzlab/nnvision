import pickle
import numpy as np
import datajoint as dj
from nnfabrik.main import Dataset
from nnfabrik.utility.dj_helpers import CustomSchema
from nnvision.legacy.featurevis import integration

schema = CustomSchema(dj.config.get("nnfabrik.schema_name", "nnfabrik_core"))


@schema
class MonkeyExperiment(dj.Computed):
    definition = """
    # This table stores something which is awesome by design

    -> Dataset
    ---
    brain_area:            varchar(64)   # some string
    experiment_name:       varchar(64)   # another string
    n_sessions:            int
    total_n_neurons:       int
    """

    class Sessions(dj.Part):
        definition = """
        # This table stores something.

        -> master
        data_key:              varchar(64)
        ---
        animal_id:             varchar(64)
        n_neurons:             int
        x_grid:                float
        y_grid:                float
        """

    class Units(dj.Part):
        definition = """
        # All Units

        -> MonkeyExperiment.Sessions
        unit_id:     int
        ---
        unit_index:  int
        electrode:   int
        relative_depth:  float
        """

        def get_output_selected_model(self, model, key):
            unit_index, data_key = (self & key).fetch1("unit_index", "data_key")
            return integration.get_output_selected_model(unit_index, data_key, model)

    def make(self, key):

        dataset_fn, dataset_config = (Dataset & key).fn_config
        filenames = dataset_config["neuronal_data_files"]
        experiment_name, brain_area = dataset_config["dataset"].split("_")

        session_dict = {}

        for file in filenames:
            with open(file, "rb") as pkl:
                raw_data = pickle.load(pkl)

            data_key = str(raw_data["session_id"])

            unit_ids = (
                raw_data["unit_ids"]
                if "unit_ids" in raw_data
                else np.arange(raw_data["testing_responses"].shape[1])
            )

            electrode = (
                raw_data["electrode_nums"]
                if "electrode_nums" in raw_data
                else np.zeros_like(unit_ids, dtype=np.double)
            )
            x_grid = raw_data["x_grid_location"] if "x_grid_location" in raw_data else 0
            y_grid = raw_data["y_grid_location"] if "y_grid_location" in raw_data else 0
            relative_depth = (
                raw_data["relative_micron_depth"]
                if "relative_micron_depth" in raw_data
                else np.zeros_like(unit_ids, dtype=np.double)
            )
            session_dict[data_key] = dict(
                animal_id=raw_data["subject_id"],
                n_neurons=int(len(unit_ids)),
                x_grid=x_grid,
                y_grid=y_grid,
            )

            session_dict[data_key]["unit_id"] = unit_ids
            session_dict[data_key]["electrode"] = electrode
            session_dict[data_key]["x_grid"] = x_grid
            session_dict[data_key]["y_grid"] = y_grid
            session_dict[data_key]["relative_depth"] = relative_depth

        key["brain_area"] = brain_area
        key["experiment_name"] = experiment_name
        key["n_sessions"] = len(session_dict)
        key["total_n_neurons"] = int(
            np.sum([v["n_neurons"] for v in session_dict.values()])
        )

        self.insert1(key, ignore_extra_fields=True)

        for k, v in session_dict.items():
            key["data_key"] = k
            key["animal_id"] = str(v["animal_id"])
            key["n_neurons"] = v["n_neurons"]
            key["x_grid"] = v["x_grid"]
            key["y_grid"] = v["y_grid"]

            self.Sessions().insert1(key, ignore_extra_fields=True)

            for i, neuron_id in enumerate(session_dict[k]["unit_id"]):
                key["unit_id"] = int(neuron_id)
                key["unit_index"] = i
                key["electrode"] = session_dict[k]["electrode"][i]
                key["relative_depth"] = session_dict[k]["relative_depth"][i]
                self.Units().insert1(key, ignore_extra_fields=True)


# legacy
class UnitMeasures(dj.Manual):
    definition = """
    # again, here some witty comment. 

    -> MonkeyExperiment.Units
    ---
    unit_oracle:            float
    unit_explainable_var:   float
    """
    pass


class UnitStatistics(dj.Manual):
    definition = """
    # some string ...

    -> MonkeyExperiment.Units
    ---
    unit_avg_firing:      float
    unit_fano_factor:     float
    """

    def make(self):
        dataset_fn, dataset_config = (Dataset & key).fn_config
        dataloaders = (Dataset & key).get_dataloader()

        responses = dataloaders["train"][data_key].dataset[:].targets

        avg_firing = responses.mean(dim=0)
        fano_factor = responses.var(dim=0) / (responses.mean(dim=0) + 1e-9)
        session_dict[data_key]["avg_firing"] = avg_firing.numpy()
        session_dict[data_key]["fano_factor"] = fano_factor.numpy()

        session_dict[data_key]["unit_oracles"] = oracles[data_key]
        session_dict[data_key]["unit_explainable_variance"] = explainable_var[data_key]
