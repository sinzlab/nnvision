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
from ..datasets.conventions import unit_type_conventions


schema = CustomSchema(dj.config.get('schema_name', 'nnfabrik_core'))


@schema
class Recording(dj.Computed):
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

        -> Recording.Sessions
        unit_id:            int     
        unit_type:          int
        ---
        unit_index:         int
        electrode:          int
        relative_depth:     float
        """
        constrained_output_model = ConstrainedOutputModel
        def get_output_selected_model(self, model: Module, key: Key) -> constrained_output_model:
            unit_index, data_key = (self & key).fetch1("unit_index", "data_key")
            return self.constrained_output_model(model, unit_index, forward_kwargs=dict(data_key=data_key))

    def make(self, key):

        dataset_fn, dataset_config = (Dataset & key).fn_config
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
        experiment_name, brain_area = dataset_config["dataset"].split('_')

        session_dict = {}
        for file in filenames:
            with open(file, "rb") as pkl:
                raw_data = pickle.load(pkl)
            data_key = str(raw_data["session_id"])

            if filenames_mua is None:
                unit_ids_mua, electrode_mua, relative_depth_mua, unit_types_mua = [], [], [], []
            else:
                for mua_data_path in filenames_mua:
                    with open(mua_data_path, "rb") as mua_pkl:
                        mua_data = pickle.load(mua_pkl)

                    if str(mua_data["session_id"]) == data_key:
                        if mua_selector is not None:
                            selected_mua = mua_selector[data_key]
                        else:
                            selected_mua = np.ones(len(mua_data["unit_ids"])).astype(bool)
                        unit_ids_mua = mua_data["unit_ids"][selected_mua]
                        electrode_mua = mua_data["electrode_nums"][selected_mua]
                        relative_depth_mua = mua_data["relative_micron_depth"][selected_mua]
                        unit_types_mua = mua_data["unit_type"][selected_mua]
                        break

                if not str(mua_data["session_id"]) == data_key:
                    print("session {} does not exist in MUA. Skipping MUA for that session".format(data_key))
                    unit_ids_mua, electrode_mua, relative_depth_mua, unit_types_mua = [], [], [], []

            if dataset == 'PlosCB19_V1':
                n_neurons = raw_data["testing_responses"].shape[1]
            else:
                n_neurons = raw_data["testing_responses"].shape[0]

            unit_ids = raw_data.get("unit_ids", np.arange(n_neurons))
            unit_type = raw_data.get("unit_type", np.ones(n_neurons))

            electrode = raw_data["electrode_nums"] if "electrode_nums" in raw_data else np.zeros_like(unit_ids,
                                                                                                      dtype=np.double)
            x_grid = raw_data["x_grid_location"] if "x_grid_location" in raw_data else 0
            y_grid = raw_data["y_grid_location"] if "y_grid_location" in raw_data else 0
            relative_depth = raw_data[
                "relative_micron_depth"] if "relative_micron_depth" in raw_data else np.zeros_like(unit_ids,
                                                                                                   dtype=np.double)
            unit_ids = np.concatenate([unit_ids, unit_ids_mua])
            unit_type = np.concatenate([unit_type, unit_types_mua])
            unit_type_int = []
            for unit in unit_type:
                unit_type_int.append(unit_type_conventions.get(unit, unit))
            unit_type_int = np.array(unit_type_int).astype(np.float)

            electrode = np.concatenate([electrode, electrode_mua])
            relative_depth = np.concatenate([relative_depth, relative_depth_mua])

            session_dict[data_key] = dict(animal_id=raw_data["subject_id"],
                                          n_neurons=int(len(unit_ids)),
                                          x_grid=x_grid,
                                          y_grid=y_grid)

            session_dict[data_key]['unit_id'] = unit_ids
            session_dict[data_key]['unit_type'] = unit_type_int
            session_dict[data_key]['electrode'] = electrode
            session_dict[data_key]['x_grid'] = x_grid
            session_dict[data_key]['y_grid'] = y_grid
            session_dict[data_key]['relative_depth'] = relative_depth

        key["brain_area"] = brain_area
        key["experiment_name"] = experiment_name
        key["n_sessions"] = len(session_dict)
        key["total_n_neurons"] = int(np.sum([v["n_neurons"] for v in session_dict.values()]))

        self.insert1(key, ignore_extra_fields=True)

        for k, v in session_dict.items():
            key['data_key'] = k
            key['animal_id'] = str(v["animal_id"])
            key['n_neurons'] = v["n_neurons"]
            key['x_grid'] = v["x_grid"]
            key['y_grid'] = v["y_grid"]

            self.Sessions().insert1(key, ignore_extra_fields=True)

            for i, neuron_id in enumerate(session_dict[k]["unit_id"]):
                key["unit_id"] = int(neuron_id)
                key["unit_index"] = i
                key['unit_type'] = int((session_dict[k]['unit_type'][i]).astype(np.float))
                key['electrode'] = session_dict[k]['electrode'][i]
                key['relative_depth'] = session_dict[k]['relative_depth'][i]
                self.Units().insert1(key, ignore_extra_fields=True)
                
                
                
@schema       
class SharedReadoutRecording(dj.Computed):
    definition = """
    # Overview table that summarizes the experiment, session, and unit data of table entry in nnfabrik's Dataset.
    
    -> Dataset
    ---
    brain_area:            varchar(64)   # some string
    experiment_name:       varchar(64)   # another string
    n_sessions:            int
    total_n_neurons:       int
    """


    class Units(dj.Part):
        definition = """
        # All Units

        -> master
        data_key:           varchar(64)
        unit_index:         int     
        unit_type:          int
        ---
        old_data_key:       varchar(64)
        old_unit_index:     int
        old_unit_id:        int
        electrode:          int
        relative_depth:     float
        """
        constrained_output_model = ConstrainedOutputModel
        def get_output_selected_model(self, model: Module, key: Key) -> constrained_output_model:
            unit_index, data_key = (self & key).fetch1("unit_index", "data_key")
            return self.constrained_output_model(model, unit_index, forward_kwargs=dict(data_key=data_key))

    def make(self, key):
        print(key)
        dataloaders=0
        seed=1000
        model_hash='fff34fa29b200777a16469ab7305112f'
        model_fn='nnvision.models.se_core_full_gauss_readout'
        dataset_fn='nnvision.datasets.monkey_loaders.monkey_static_loader'       
        dataset_hash='293317b0600244345fe605961ef9f170' 
        trainer_fn='nnvision.training.trainers.nnvision_trainer',
        trainer_hash='ca71a6398d0131cac49323c0d0daa560'        
        
        session_list = se_core_shared_gaussian_readout(dataloaders=dataloaders, seed=seed, model_fn=model_fn, 
                                                       model_hash=model_hash, dataset_fn=dataset_fn, 
                                                       dataset_hash=dataset_hash, trainer_fn=trainer_fn,
                                                       trainer_hash=trainer_hash, session_list=True)
        
        dataset_fn, dataset_config = (Dataset & key).fn_config
       
        dataset = dataset_config.get("dataset", None)
        filenames = dataset_config.get("neuronal_data_files", None)
        
        print('Got files')
        
        if filenames is None:
            filenames = dataset_config.get("sua_data_files", None)
        filenames_mua = dataset_config.get("mua_data_files", None)
        if filenames_mua is not None and filenames is None:
            print("found MUA only ...")
            filenames = filenames_mua
            filenames_mua = None
        mua_selector = dataset_config.get("mua_selector", None)
        experiment_name, brain_area = dataset_config["dataset"].split('_')
            
        print(session_list)
        sorted_filenames = []
        for session in session_list:
            for file in filenames:
                if session in file:
                    sorted_filenames.append(file)
                    
        print(sorted_filenames)

        session_dict = {}
        for file in sorted_filenames:
            with open(file, "rb") as pkl:
                raw_data = pickle.load(pkl)
            data_key = str(raw_data["session_id"])

            if filenames_mua is None:
                unit_ids_mua, electrode_mua, relative_depth_mua, unit_types_mua = [], [], [], []
            else:
                for mua_data_path in filenames_mua:
                    with open(mua_data_path, "rb") as mua_pkl:
                        mua_data = pickle.load(mua_pkl)

                    if str(mua_data["session_id"]) == data_key:
                        if mua_selector is not None:
                            selected_mua = mua_selector[data_key]
                        else:
                            selected_mua = np.ones(len(mua_data["unit_ids"])).astype(bool)
                        unit_ids_mua = mua_data["unit_ids"][selected_mua]
                        electrode_mua = mua_data["electrode_nums"][selected_mua]
                        relative_depth_mua = mua_data["relative_micron_depth"][selected_mua]
                        unit_types_mua = mua_data["unit_type"][selected_mua]
                        break

                if not str(mua_data["session_id"]) == data_key:
                    print("session {} does not exist in MUA. Skipping MUA for that session".format(data_key))
                    unit_ids_mua, electrode_mua, relative_depth_mua, unit_types_mua = [], [], [], []

            if dataset == 'PlosCB19_V1':
                n_neurons = raw_data["testing_responses"].shape[1]
            else:
                n_neurons = raw_data["testing_responses"].shape[0]

            unit_ids = raw_data.get("unit_ids", np.arange(n_neurons))
            unit_type = raw_data.get("unit_type", np.ones(n_neurons))

            electrode = raw_data["electrode_nums"] if "electrode_nums" in raw_data else np.zeros_like(unit_ids,
                                                                                                      dtype=np.double)
            x_grid = raw_data["x_grid_location"] if "x_grid_location" in raw_data else 0
            y_grid = raw_data["y_grid_location"] if "y_grid_location" in raw_data else 0
            relative_depth = raw_data[
                "relative_micron_depth"] if "relative_micron_depth" in raw_data else np.zeros_like(unit_ids,
                                                                                                   dtype=np.double)
            unit_ids = np.concatenate([unit_ids, unit_ids_mua])
            unit_type = np.concatenate([unit_type, unit_types_mua])
            unit_type_int = []
            for unit in unit_type:
                unit_type_int.append(unit_type_conventions.get(unit, unit))
            unit_type_int = np.array(unit_type_int).astype(np.float)

            electrode = np.concatenate([electrode, electrode_mua])
            relative_depth = np.concatenate([relative_depth, relative_depth_mua])

            session_dict[data_key] = dict(animal_id=raw_data["subject_id"],
                                          n_neurons=int(len(unit_ids)),
                                          x_grid=x_grid,
                                          y_grid=y_grid)

            session_dict[data_key]['unit_id'] = unit_ids
            session_dict[data_key]['unit_type'] = unit_type_int
            session_dict[data_key]['electrode'] = electrode
            session_dict[data_key]['x_grid'] = x_grid
            session_dict[data_key]['y_grid'] = y_grid
            session_dict[data_key]['relative_depth'] = relative_depth
        
        total_num_neurons = []
        for value in session_dict.values():
            total_num_neurons.append(value['n_neurons'])
        key["brain_area"] = brain_area
        key["experiment_name"] = experiment_name
        
        key["n_sessions"] = len(session_list)
        key["total_n_neurons"] = int(np.sum([v["n_neurons"] for v in session_dict.values()]))

        self.insert1(key, ignore_extra_fields=True)
        
        print('Inserted into main table')
        
        index = 0
        for k, v in session_dict.items():
            key['data_key'] = 'augmentation'
            key['old_data_key'] = k
            key['animal_id'] = str(v["animal_id"])
            key['n_neurons'] = v["n_neurons"]
            key['x_grid'] = v["x_grid"]
            key['y_grid'] = v["y_grid"]


            for i, neuron_id in enumerate(session_dict[k]["unit_id"]):
                key['unit_index'] = index
                #key['old_data_key'] = k
                key['old_unit_index'] = i
                key["old_unit_id"] = int(neuron_id)
                key['unit_type'] = int((session_dict[k]['unit_type'][i]).astype(np.float))
                key['electrode'] = session_dict[k]['electrode'][i]
                key['relative_depth'] = session_dict[k]['relative_depth'][i]
                print(key)
                self.Units().insert1(key, ignore_extra_fields=True)
                index += 1