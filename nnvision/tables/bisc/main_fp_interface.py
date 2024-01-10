from __future__ import annotations

from typing import Callable, Iterable, Mapping, Optional, Tuple, Dict, Any

from torch.nn import Module
from torch.utils.data import DataLoader

from mei.modules import ConstrainedOutputModel


Key = Dict[str, Any]
Dataloaders = Dict[str, DataLoader]

import pickle
import numpy as np
import datajoint as dj
from nnfabrik.main import Dataset
from nnfabrik.utility.dj_helpers import CustomSchema
from nnvision.datasets.conventions import unit_type_conventions

schema = CustomSchema(dj.config.get("nnfabrik.schema_name", "nnfabrik_core"))


@schema
class RecordingInterfaceFP(dj.Computed):
    definition = """	
    # Overview table that summarizes the experiment, session, and unit data of table entry in nnfabrik's Dataset.	
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
        data_key_original:     varchar(64)	
        data_key:              varchar(64)	
        ---	
        animal_id:             varchar(64)	
        n_neurons:             int	
        x_grid:                float	
        y_grid:                float	
        fp_proc_method:        int
        """

    class Units(dj.Part):
        definition = """	
        # All Units	
        -> master.Sessions	
        unit_id_original:       int     	
        unit_id:                int	
        unit_type:              int	
        ---	
        unit_index_original:             int	
        unit_index:    int	
        electrode:              int	
        relative_depth:         float	
        fp_proc_method:         int
        """
        constrained_output_model = ConstrainedOutputModel

        def get_output_selected_model(
            self, model: Module, key: Key
        ) -> constrained_output_model:
            unit_index, data_key = (self & key).fetch1("unit_index", "data_key")
            return self.constrained_output_model(
                model, unit_index, forward_kwargs=dict(data_key=data_key)
            )

    def make(self, key):

        dataset_fn, dataset_config = (Dataset & key).fn_config

        # take care of the case of the combined dataloader:
        if "monkey_static_loader_combined" in dataset_fn:
            combined_data_key = "all_sessions"
        else:
            combined_data_key = None

        dataset = dataset_config.get("dataset", None)
        filenames = dataset_config.get("neuronal_data_files", None)
        if filenames is None:
            filenames = dataset_config.get("sua_data_files", None)
        filenames_mua = dataset_config.get("mua_data_files", None)
        if filenames_mua is not None and filenames is None:
            print("found MUA only ...")
            filenames = filenames_mua
            filenames_mua = None
        mua_selector = dataset_config.get("mua_selector", None)
        experiment_name, brain_area = dataset_config["dataset"].split("_")

        session_dict = {}
        for file_num, file in enumerate(filenames):
            with open(file, "rb") as pkl:
                raw_data = pickle.load(pkl)
            data_key = str(raw_data["session_id"])

            if filenames_mua is None:
                unit_ids_mua, electrode_mua, relative_depth_mua, unit_types_mua = (
                    [],
                    [],
                    [],
                    [],
                )
            else:
                for mua_data_path in filenames_mua:
                    with open(mua_data_path, "rb") as mua_pkl:
                        mua_data = pickle.load(mua_pkl)

                    if str(mua_data["session_id"]) == data_key:
                        if mua_selector is not None:
                            selected_mua = mua_selector[data_key]
                        else:
                            selected_mua = np.ones(len(mua_data["unit_ids"])).astype(
                                bool
                            )
                        unit_ids_mua = mua_data["unit_ids"][selected_mua]
                        electrode_mua = mua_data["electrode_nums"][selected_mua]
                        relative_depth_mua = mua_data["relative_micron_depth"][
                            selected_mua
                        ]
                        unit_types_mua = mua_data["unit_type"][selected_mua]
                        break

                if not str(mua_data["session_id"]) == data_key:
                    print(
                        "session {} does not exist in MUA. Skipping MUA for that session".format(
                            data_key
                        )
                    )
                    unit_ids_mua, electrode_mua, relative_depth_mua, unit_types_mua = (
                        [],
                        [],
                        [],
                        [],
                    )

            if dataset == "PlosCB19_V1":
                n_neurons = raw_data["testing_responses"].shape[1]
            else:
                responses_test = raw_data["testing_responses"]

                if (len(responses_test.shape) < 3) and responses_test.shape[0] == 12:
                    responses_test = responses_test[None, ...]
                n_neurons = responses_test.shape[0]

            unit_ids = raw_data.get("unit_ids", np.arange(n_neurons))
            unit_type = raw_data.get("unit_type", np.ones(n_neurons))
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
            if not isinstance(unit_ids, Iterable):
                unit_ids = [unit_ids]
                electrode = [electrode]
                relative_depth = [relative_depth]

            unit_ids = np.concatenate([unit_ids, unit_ids_mua])
            unit_type = np.concatenate([unit_type, unit_types_mua])
            unit_type_int = []
            for unit in unit_type:
                unit_type_int.append(unit_type_conventions.get(unit, unit))
            unit_type_int = np.array(unit_type_int).astype(np.float)

            electrode = np.concatenate([electrode, electrode_mua])
            relative_depth = np.concatenate([relative_depth, relative_depth_mua])

            session_dict[file_num] = dict(
                animal_id=raw_data["subject_id"],
                n_neurons=int(len(unit_ids)),
                x_grid=x_grid,
                y_grid=y_grid,
            )

            session_dict[file_num]["unit_id"] = unit_ids
            session_dict[file_num]["unit_type"] = unit_type_int
            session_dict[file_num]["electrode"] = electrode
            session_dict[file_num]["x_grid"] = x_grid
            session_dict[file_num]["y_grid"] = y_grid
            session_dict[file_num]["data_key"] = data_key
            session_dict[file_num]["relative_depth"] = relative_depth
            session_dict[file_num]["fp_proc_method"] = raw_data["fp_proc_method"]

        key["brain_area"] = brain_area
        key["experiment_name"] = experiment_name
        key["n_sessions"] = len(session_dict)
        key["total_n_neurons"] = int(
            np.sum([v["n_neurons"] for v in session_dict.values()])
        )

        self.insert1(key, ignore_extra_fields=True)

        combined_neuron_counter = None if combined_data_key is None else 0
        for k, v in session_dict.items():
            key["data_key"] = (
                v["data_key"] if combined_data_key is None else combined_data_key
            )
            key["data_key_original"] = v["data_key"]
            key["animal_id"] = str(v["animal_id"])
            key["n_neurons"] = v["n_neurons"]
            key["x_grid"] = v["x_grid"]
            key["y_grid"] = v["y_grid"]
            key["fp_proc_method"] = v["fp_proc_method"]

            self.Sessions().insert1(key, ignore_extra_fields=True, skip_duplicates=True)

            for i, neuron_id in enumerate(session_dict[k]["unit_id"]):
                key["unit_id_original"] = int(neuron_id)
                key["unit_id"] = combined_neuron_counter
                key["unit_index_original"] = i
                key["unit_index"] = combined_neuron_counter
                key["unit_type"] = int(
                    (session_dict[k]["unit_type"][i]).astype(np.float)
                )
                key["electrode"] = session_dict[k]["electrode"][i]
                key["relative_depth"] = session_dict[k]["relative_depth"][i]
                key["fp_proc_method"] = session_dict[k]["fp_proc_method"]
                self.Units().insert1(key, ignore_extra_fields=True)

                if combined_neuron_counter is not None:
                    combined_neuron_counter += 1
