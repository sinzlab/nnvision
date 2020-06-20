import torch
import copy


def unpack_data_info(data_info):

    in_shapes_dict = {k: v["input_dimensions"] for k, v in data_info.items()}
    input_channels = [v["input_channels"] for k, v in data_info.items()]
    n_neurons_dict = {k: v["output_dimension"] for k, v in data_info.items()}
    return n_neurons_dict, in_shapes_dict, input_channels


def purge_state_dict(state_dict, key):
    purged_state_dict = copy.deepcopy(state_dict)
    for dict_key in purged_state_dict.keys():
        if key in dict_key:
            purged_state_dict.pop(dict_key)
    return purged_state_dict