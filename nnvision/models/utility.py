import torch
import copy
import mlutils
from mlutils.layers.readouts import PointPooled2d, FullGaussian2d
from mlutils.layers.readouts.legacy import Gaussian2d
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


def get_readout_key_names(model):
    data_key = list(model.readout.keys())[0]
    readout = model.readout[data_key]

    if type(readout) == PointPooled2d:
        feature_name = "features"
        grid_name = "grid"
        bias_name = "bias"
    elif type(readout) == FullGaussian2d:
        feature_name = "_features"
        grid_name = "_mu"
        bias_name = "bias"
    elif type(readout) == FullGaussian2d:
        feature_name = "features"
        grid_name = "mu"
        bias_name = "bias"
    else:
        raise NotImplementedError("Readout Transfer is only possible for the PointPooled2d, FullGaussian2d, "
                                  "and Gaussian2d at the present time.")
    return feature_name, grid_name, bias_name