import pickle
import numpy as np
import datajoint as dj
from ..utility.measures import get_oracles, get_explainable_var
from nnfabrik.main import Dataset

schema = dj.schema(dj.config.get('schema_name', 'nnfabrik_core'))
print("Schema name: {}".format(dj.config["schema_name"]))


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
    avg_oracle:            float
    avg_explainable_var:   float
    """

    class Sessions(dj.Part):
        definition = """
        # This table stores something.

        -> master
        session_id:            varchar(64)
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
        unit_position:  int
        """

    class UnitMeasures(dj.Part):
        definition = """
        # again, here some witty comment. 

        -> MonkeyExperiment.Units
        ---
        unit_oracle:            float
        unit_explainable_var:   float
        """

    class UnitStatistics(dj.Part):
        definition = """
        # some string ...

        -> MonkeyExperiment.Units
        ---
        unit_avg_firing:      float
        unit_fano_factor:     float
        """

    class UnitPhysiology(dj.Part):
        definition = """
            # Information about the recording depth as well as spike sorting details

            -> MonkeyExperiment.Units
            ---
            electrode:   int
            relative_depth:  float
            """

    def make(self, key):
        print(key)
        dataset_fn, dataset_config = (Dataset & key).fn_config
        dataloaders = (Dataset & key).get_dataloader()

        filenames = dataset_config["neuronal_data_files"]
        experiment_name, brain_area = dataset_config["dataset"].split('_')

        oracles = get_oracles(dataloaders["test"], as_dict=True)
        explainable_var = get_explainable_var(dataloaders["test"], as_dict=True)

        session_dict = {}

        for file in filenames:
            with open(file, "rb") as pkl:
                raw_data = pickle.load(pkl)

            data_key = str(raw_data["session_id"])


            unit_ids = raw_data["unit_ids"] if "unit_ids" in raw_data else np.arange(raw_data["testing_responses"].shape[1])

            electrode = raw_data["electrode_nums"] if "electrode_nums" in raw_data else np.zeros_like(unit_ids,
                                                                                                     dtype=np.double)
            x_grid = raw_data["x_grid_location"] if "x_grid_location" in raw_data else 0
            y_grid = raw_data["y_grid_location"] if "y_grid_location" in raw_data else 0
            relative_depth = raw_data["relative_micron_depth"] if "relative_micron_depth" in raw_data else np.zeros_like(unit_ids,
                                                                                                                        dtype=np.double)
            session_dict[data_key] = dict(animal_id=raw_data["subject_id"],
                                          n_neurons=int(len(unit_ids)),
                                          x_grid=x_grid,
                                          y_grid=y_grid)

            session_dict[data_key]['unit_id'] = unit_ids
            session_dict[data_key]['electrode'] = electrode
            session_dict[data_key]['x_grid'] = x_grid
            session_dict[data_key]['y_grid'] = y_grid
            session_dict[data_key]['relative_depth'] = relative_depth

            responses   = dataloaders["train"][data_key].dataset[:].targets

            avg_firing  = responses.mean(dim=0)
            fano_factor = responses.var(dim=0) / responses.mean(dim=0)
            session_dict[data_key]['avg_firing'] = avg_firing.numpy()
            session_dict[data_key]['fano_factor'] = fano_factor.numpy()

            session_dict[data_key]['unit_oracles'] = oracles[data_key]
            session_dict[data_key]['unit_explainable_variance'] = explainable_var[data_key]

        key["brain_area"] = brain_area
        key["experiment_name"] = experiment_name
        key["n_sessions"] = len(session_dict)
        key["total_n_neurons"] = int(np.sum([v["n_neurons"] for v in session_dict.values()]))
        key["avg_oracle"] = np.nanmean(np.hstack([v["unit_oracles"] for v in session_dict.values()]))
        key["avg_explainable_var"] = np.nanmean(np.hstack([v["unit_explainable_variance"] for v in session_dict.values()]))

        self.insert1(key, ignore_extra_fields=True)

        for k, v in session_dict.items():
            key['session_id'] = k
            key['animal_id'] = str(v["animal_id"])
            key['n_neurons'] = v["n_neurons"]

            self.Sessions().insert1(key, ignore_extra_fields=True)

            for i, neuron_id in enumerate(session_dict[k]["unit_id"]):
                key["unit_id"] = int(neuron_id)
                key["unit_position"] = i
                self.Units().insert1(key, ignore_extra_fields=True)

                key['unit_avg_firing'] = session_dict[k]['avg_firing'][i]
                key['unit_fano_factor'] = session_dict[k]['fano_factor'][i] if not np.isnan(session_dict[k]['fano_factor'][i]) else 0
                self.UnitStatistics().insert1(key, ignore_extra_fields=True)

                key['unit_oracle'] = session_dict[k]['unit_oracles'][i] if not np.isnan(session_dict[k]['unit_oracles'][i]) else 0
                key['unit_explainable_var'] = session_dict[k]['unit_explainable_variance'][i] if not np.isnan(session_dict[k]['unit_explainable_variance'][i]) else 0
                self.UnitMeasures().insert1(key, ignore_extra_fields=True)

                key['electrode'] = session_dict[k]['electrode'][i]
                key['relative_depth'] = session_dict[k]['relative_depth'][i]
                self.UnitPhysiology().insert1(key, ignore_extra_fields=True)
