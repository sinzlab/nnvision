import numpy as np
import torch
import copy

from neuralpredictors.layers.cores import Stacked2dCore
from neuralpredictors.layers.legacy import Gaussian2d
from neuralpredictors.layers.readouts import PointPooled2d
from nnfabrik.utility.nn_helpers import set_random_seed, get_dims_for_loader_dict

from neuralpredictors.utils import get_module_output
from torch import nn
from torch.nn import functional as F

from .readouts import (
    MultipleFullGaussian2d,
    MultiReadout,
    MultipleSpatialXFeatureLinear,
    MultipleRemappedGaussian2d,
    MultipleSelfAttention2d,
    MultipleMultiHeadAttention2d,
    MultipleSharedMultiHeadAttention2d,
)

from .utility import unpack_data_info
from .encoders import EncoderShifter, Encoder
from .shifters import MLPShifter, StaticAffine2dShifter

try:
    from ptrnets.cores.cores import TaskDrivenCore, TaskDrivenCore2
except:
    pass
from .cores import TaskDrivenCore3


def task_core_gauss_readout(
    dataloaders,
    seed,
    input_channels=1,
    model_name="vgg19",  # begin of core args
    layer_name="features.10",
    pretrained=True,
    bias=False,
    final_batchnorm=True,
    final_nonlinearity=True,
    momentum=0.1,
    fine_tune=False,
    init_mu_range=0.4,
    init_sigma_range=0.6,  # readout args,
    readout_bias=True,
    gamma_readout=0.01,
    gauss_type="isotropic",
    elu_offset=-1,
    data_info=None,
    shifter=None,
    shifter_type="MLP",
    input_channels_shifter=2,
    hidden_channels_shifter=5,
    shift_layers=3,
    gamma_shifter=0,
    shifter_bias=True,
):
    """
    A Model class of a predefined core (using models from ptrnets). Can be initialized pretrained or random.
    Can also be set to be trainable or not, independent of initialization.

    Args:
        dataloaders: a dictionary of train-dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: ..
        pool_steps:
        pool_kern:
        readout_bias:
        init_range:
        gamma_readout:

    Returns:
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields[:2]

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = (
        list(input_channels.values())[0]
        if isinstance(input_channels, dict)
        else input_channels[0]
    )

    set_random_seed(seed)

    core = TaskDrivenCore3(
        input_channels=core_input_channels,
        model_name=model_name,
        layer_name=layer_name,
        pretrained=pretrained,
        bias=bias,
        final_batchnorm=final_batchnorm,
        final_nonlinearity=final_nonlinearity,
        momentum=momentum,
        fine_tune=fine_tune,
    )

    set_random_seed(seed)

    core.initialize()

    readout = MultipleFullGaussian2d(
        core,
        in_shape_dict=in_shapes_dict,
        n_neurons_dict=n_neurons_dict,
        init_mu_range=init_mu_range,
        init_sigma=init_sigma_range,
        bias=readout_bias,
        gamma_readout=gamma_readout,
        gauss_type=gauss_type,
        grid_mean_predictor=None,  # not relevant for monkey data
        grid_mean_predictor_type=None,
        source_grids=None,
        share_features=None,
        share_grid=None,
        shared_match_ids=None,
    )

    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))[:2]
            readout[key].bias.data = targets.mean(0)

    if shifter is True:
        data_keys = [i for i in dataloaders.keys()]
        if shifter_type == "MLP":
            shifter = MLPShifter(
                data_keys=data_keys,
                input_channels=input_channels_shifter,
                hidden_channels_shifter=hidden_channels_shifter,
                shift_layers=shift_layers,
                gamma_shifter=gamma_shifter,
            )

        elif shifter_type == "StaticAffine":
            shifter = StaticAffine2dShifter(
                data_keys=data_keys,
                input_channels=input_channels_shifter,
                bias=shifter_bias,
                gamma_shifter=gamma_shifter,
            )

    model = EncoderShifter(core, readout, shifter=shifter, elu_offset=elu_offset)

    return model


def custom_task_core_gauss_readout(
    dataloaders,
    seed,
    input_channels=1,
    model_name="vgg19",  # begin of core args
    layer_name="features.10",
    pretrained=True,
    bias=False,
    final_batchnorm=True,
    final_nonlinearity=True,
    momentum=0.1,
    fine_tune=False,
    init_mu_range=0.4,
    init_sigma_range=0.6,  # readout args,
    readout_bias=True,
    gamma_readout=0.01,
    gauss_type="isotropic",
    elu_offset=-1,
    data_info=None,
    shifter=None,
    shifter_type="MLP",
    input_channels_shifter=2,
    hidden_channels_shifter=5,
    shift_layers=3,
    gamma_shifter=0,
    shifter_bias=True,
):
    """
    A Model class of a predefined core (using models from ptrnets). Can be initialized pretrained or random.
    Can also be set to be trainable or not, independent of initialization.

    Args:
        dataloaders: a dictionary of train-dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: ..
        pool_steps:
        pool_kern:
        readout_bias:
        init_range:
        gamma_readout:

    Returns:
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields[:2]

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = (
        list(input_channels.values())[0]
        if isinstance(input_channels, dict)
        else input_channels[0]
    )

    set_random_seed(seed)

    core = TaskDrivenCore3(
        input_channels=core_input_channels,
        model_name=model_name,
        layer_name=layer_name,
        pretrained=pretrained,
        bias=bias,
        final_batchnorm=final_batchnorm,
        final_nonlinearity=final_nonlinearity,
        momentum=momentum,
        fine_tune=fine_tune,
    )

    set_random_seed(seed)

    core.initialize()

    readout = MultipleFullGaussian2d(
        core,
        in_shape_dict=in_shapes_dict,
        n_neurons_dict=n_neurons_dict,
        init_mu_range=init_mu_range,
        init_sigma=init_sigma_range,
        bias=readout_bias,
        gamma_readout=gamma_readout,
        gauss_type=gauss_type,
        grid_mean_predictor=None,  # not relevant for monkey data
        grid_mean_predictor_type=None,
        source_grids=None,
        share_features=None,
        share_grid=None,
        shared_match_ids=None,
    )

    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))[:2]
            readout[key].bias.data = targets.mean(0)

    if shifter is True:
        data_keys = [i for i in dataloaders.keys()]
        if shifter_type == "MLP":
            shifter = MLPShifter(
                data_keys=data_keys,
                input_channels=input_channels_shifter,
                hidden_channels_shifter=hidden_channels_shifter,
                shift_layers=shift_layers,
                gamma_shifter=gamma_shifter,
            )

        elif shifter_type == "StaticAffine":
            shifter = StaticAffine2dShifter(
                data_keys=data_keys,
                input_channels=input_channels_shifter,
                bias=shifter_bias,
                gamma_shifter=gamma_shifter,
            )

    model = EncoderShifter(core, readout, shifter=shifter, elu_offset=elu_offset)

    return model


def custom_task_core_selfattention_readout(
    dataloaders,
    seed,
    input_channels=1,
    model_name="vgg19",  # begin of core args
    layer_name="features.10",
    pretrained=True,
    bias=False,
    final_batchnorm=True,
    final_nonlinearity=True,
    momentum=0.1,
    fine_tune=False,
    data_info=None,
    readout_bias=True,
    gamma_features=3.0,
    gamma_query=1,
    elu_offset=-1,
    shifter=None,
    shifter_type="MLP",
    input_channels_shifter=2,
    hidden_channels_shifter=5,
    shift_layers=3,
    gamma_shifter=0,
    shifter_bias=True,
    position_encoding=True,
    learned_pos=False,
    dropout_pos=0.1,
    stack_pos_encoding=False,
    n_pos_channels=None,
    temperature=1.0,
):
    """
    A Model class of a predefined core (using models from ptrnets). Can be initialized pretrained or random.
    Can also be set to be trainable or not, independent of initialization.

    Args:
        dataloaders: a dictionary of train-dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: ..
        pool_steps:
        pool_kern:
        readout_bias:
        init_range:
        gamma_readout:

    Returns:
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields[:2]

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = (
        list(input_channels.values())[0]
        if isinstance(input_channels, dict)
        else input_channels[0]
    )

    set_random_seed(seed)

    core = TaskDrivenCore3(
        input_channels=core_input_channels,
        model_name=model_name,
        layer_name=layer_name,
        pretrained=pretrained,
        bias=bias,
        final_batchnorm=final_batchnorm,
        final_nonlinearity=final_nonlinearity,
        momentum=momentum,
        fine_tune=fine_tune,
    )

    set_random_seed(seed)

    core.initialize()

    readout = MultipleSelfAttention2d(
        core,
        in_shape_dict=in_shapes_dict,
        n_neurons_dict=n_neurons_dict,
        bias=readout_bias,
        gamma_query=gamma_query,
        gamma_features=gamma_features,
        use_pos_enc=position_encoding,
        learned_pos=learned_pos,
        dropout_pos=dropout_pos,
        stack_pos_encoding=stack_pos_encoding,
        n_pos_channels=n_pos_channels,
        temperature=temperature,
    )

    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))[:2]
            readout[key].bias.data = targets.mean(0)

    if shifter is True:
        data_keys = [i for i in dataloaders.keys()]
        if shifter_type == "MLP":
            shifter = MLPShifter(
                data_keys=data_keys,
                input_channels=input_channels_shifter,
                hidden_channels_shifter=hidden_channels_shifter,
                shift_layers=shift_layers,
                gamma_shifter=gamma_shifter,
            )

        elif shifter_type == "StaticAffine":
            shifter = StaticAffine2dShifter(
                data_keys=data_keys,
                input_channels=input_channels_shifter,
                bias=shifter_bias,
                gamma_shifter=gamma_shifter,
            )

    model = EncoderShifter(core, readout, shifter=shifter, elu_offset=elu_offset)

    return model


def custom_task_core_multihead_attention(
    dataloaders,
    seed,
    input_channels=1,
    model_name="vgg19",  # begin of core args
    layer_name="features.10",
    pretrained=True,
    bias=False,
    final_batchnorm=True,
    final_nonlinearity=True,
    momentum=0.1,
    fine_tune=False,
    readout_bias=True,
    gamma_features=3,  # start of readout kwargs
    gamma_query=1,
    use_pos_enc=True,
    learned_pos=False,
    heads=1,
    scale=False,
    key_embedding=False,
    value_embedding=False,
    temperature=(False, 1.0),  # (learnable-per-neuron, value)
    dropout_pos=0.1,
    layer_norm=False,
    data_info=None,
    shifter=None,
    shifter_type="MLP",
    input_channels_shifter=2,
    hidden_channels_shifter=5,
    shift_layers=3,
    gamma_shifter=0,
    shifter_bias=True,
    elu_offset=-1,
    stack_pos_encoding=False,
    n_pos_channels=0,
    replace_downsampling=False,
):
    """
    A Model class of a predefined core (using models from ptrnets). Can be initialized pretrained or random.
    Can also be set to be trainable or not, independent of initialization.

    Args:
        dataloaders: a dictionary of train-dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: ..
        pool_steps:
        pool_kern:
        readout_bias:
        init_range:
        gamma_readout:

    Returns:
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields[:2]

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = (
        list(input_channels.values())[0]
        if isinstance(input_channels, dict)
        else input_channels[0]
    )

    set_random_seed(seed)

    core = TaskDrivenCore3(
        input_channels=core_input_channels,
        model_name=model_name,
        layer_name=layer_name,
        pretrained=pretrained,
        bias=bias,
        final_batchnorm=final_batchnorm,
        final_nonlinearity=final_nonlinearity,
        momentum=momentum,
        fine_tune=fine_tune,
        replace_downsampling=replace_downsampling,
    )

    set_random_seed(seed)

    core.initialize()

    readout = MultipleMultiHeadAttention2d(
        core,
        in_shape_dict=in_shapes_dict,
        n_neurons_dict=n_neurons_dict,
        bias=readout_bias,
        gamma_features=gamma_features,
        gamma_query=gamma_query,
        use_pos_enc=use_pos_enc,
        learned_pos=learned_pos,
        heads=heads,
        scale=scale,
        key_embedding=key_embedding,
        value_embedding=value_embedding,
        temperature=temperature,  # (learnable-per-neuron, value)
        dropout_pos=dropout_pos,
        layer_norm=layer_norm,
        stack_pos_encoding=stack_pos_encoding,
        n_pos_channels=n_pos_channels,
    )

    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))[:2]
            readout[key].bias.data = targets.mean(0)

    if shifter is True:
        data_keys = [i for i in dataloaders.keys()]
        if shifter_type == "MLP":
            shifter = MLPShifter(
                data_keys=data_keys,
                input_channels=input_channels_shifter,
                hidden_channels_shifter=hidden_channels_shifter,
                shift_layers=shift_layers,
                gamma_shifter=gamma_shifter,
            )

        elif shifter_type == "StaticAffine":
            shifter = StaticAffine2dShifter(
                data_keys=data_keys,
                input_channels=input_channels_shifter,
                bias=shifter_bias,
                gamma_shifter=gamma_shifter,
            )

    model = EncoderShifter(
        core,
        readout,
        shifter=shifter,
        elu_offset=elu_offset,
    )

    return model


def custom_task_core_shared_multihead_attention(
    dataloaders,
    seed,
    input_channels=1,
    model_name="vgg19",  # begin of core args
    layer_name="features.10",
    pretrained=True,
    bias=False,
    final_batchnorm=True,
    final_nonlinearity=True,
    momentum=0.1,
    fine_tune=False,
    readout_bias=True,
    gamma_features=3,  # start of readout kwargs
    gamma_query=1,
    use_pos_enc=True,
    learned_pos=False,
    heads=1,
    scale=False,
    key_embedding=False,
    value_embedding=False,
    temperature=(False, 1.0),  # (learnable-per-neuron, value)
    dropout_pos=0.1,
    layer_norm=False,
    data_info=None,
    shifter=None,
    shifter_type="MLP",
    input_channels_shifter=2,
    hidden_channels_shifter=5,
    shift_layers=3,
    gamma_shifter=0,
    shifter_bias=True,
    elu_offset=-1,
    stack_pos_encoding=False,
    n_pos_channels=0,
    replace_downsampling=False,
    gamma_embedding=0,
    embed_out_dim=None,
):
    """
    A Model class of a predefined core (using models from ptrnets). Can be initialized pretrained or random.
    Can also be set to be trainable or not, independent of initialization.

    Args:
        dataloaders: a dictionary of train-dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: ..
        pool_steps:
        pool_kern:
        readout_bias:
        init_range:
        gamma_readout:

    Returns:
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields[:2]

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = (
        list(input_channels.values())[0]
        if isinstance(input_channels, dict)
        else input_channels[0]
    )

    set_random_seed(seed)

    core = TaskDrivenCore3(
        input_channels=core_input_channels,
        model_name=model_name,
        layer_name=layer_name,
        pretrained=pretrained,
        bias=bias,
        final_batchnorm=final_batchnorm,
        final_nonlinearity=final_nonlinearity,
        momentum=momentum,
        fine_tune=fine_tune,
        replace_downsampling=replace_downsampling,
    )

    set_random_seed(seed)
    core.initialize()

    readout = MultipleSharedMultiHeadAttention2d(
        core,
        in_shape_dict=in_shapes_dict,
        n_neurons_dict=n_neurons_dict,
        bias=readout_bias,
        gamma_features=gamma_features,
        gamma_query=gamma_query,
        use_pos_enc=use_pos_enc,
        learned_pos=learned_pos,
        heads=heads,
        scale=scale,
        key_embedding=key_embedding,
        value_embedding=value_embedding,
        temperature=temperature,  # (learnable-per-neuron, value)
        dropout_pos=dropout_pos,
        layer_norm=layer_norm,
        stack_pos_encoding=stack_pos_encoding,
        n_pos_channels=n_pos_channels,
        embed_out_dim=embed_out_dim,
        gamma_embedding=gamma_embedding,
    )

    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))[:2]
            readout[key].bias.data = targets.mean(0)

    if shifter is True:
        data_keys = [i for i in dataloaders.keys()]
        if shifter_type == "MLP":
            shifter = MLPShifter(
                data_keys=data_keys,
                input_channels=input_channels_shifter,
                hidden_channels_shifter=hidden_channels_shifter,
                shift_layers=shift_layers,
                gamma_shifter=gamma_shifter,
            )

        elif shifter_type == "StaticAffine":
            shifter = StaticAffine2dShifter(
                data_keys=data_keys,
                input_channels=input_channels_shifter,
                bias=shifter_bias,
                gamma_shifter=gamma_shifter,
            )

    model = EncoderShifter(
        core,
        readout,
        shifter=shifter,
        elu_offset=elu_offset,
    )

    return model


def task_core_remapped_gauss_readout(
    dataloaders,
    seed,
    input_channels=1,
    model_name="vgg19",  # begin of core args
    layer_name="features.10",
    pretrained=True,
    bias=False,
    final_batchnorm=True,
    final_nonlinearity=True,
    momentum=0.1,
    fine_tune=False,
    init_mu_range=0.4,
    init_sigma_range=0.6,  # readout args,
    readout_bias=True,
    gamma_readout=0.01,
    gauss_type="isotropic",
    elu_offset=-1,
    data_info=None,
    remap_layers=2,
    remap_kernel=3,
    max_remap_amplitude=0.2,  # output and data_info
):
    """
    A Model class of a predefined core (using models from ptrnets). Can be initialized pretrained or random.
    Can also be set to be trainable or not, independent of initialization.

    Args:
        dataloaders: a dictionary of train-dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: ..
        pool_steps:
        pool_kern:
        readout_bias:
        init_range:
        gamma_readout:

    Returns:
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields[:2]

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = (
        list(input_channels.values())[0]
        if isinstance(input_channels, dict)
        else input_channels[0]
    )

    set_random_seed(seed)

    core = TaskDrivenCore2(
        input_channels=core_input_channels,
        model_name=model_name,
        layer_name=layer_name,
        pretrained=pretrained,
        bias=bias,
        final_batchnorm=final_batchnorm,
        final_nonlinearity=final_nonlinearity,
        momentum=momentum,
        fine_tune=fine_tune,
    )

    set_random_seed(seed)

    core.initialize()

    readout = MultipleRemappedGaussian2d(
        core,
        in_shape_dict=in_shapes_dict,
        n_neurons_dict=n_neurons_dict,
        init_mu_range=init_mu_range,
        bias=readout_bias,
        init_sigma=init_sigma_range,
        gamma_readout=gamma_readout,
        gauss_type=gauss_type,
        grid_mean_predictor=None,
        grid_mean_predictor_type=None,
        remap_layers=remap_layers,
        remap_kernel=remap_kernel,
        max_remap_amplitude=max_remap_amplitude,
        source_grids=None,
        share_features=None,
        share_grid=None,
        shared_match_ids=None,
    )

    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))[:2]
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


class MultiplePointPooled2d(MultiReadout, torch.nn.ModuleDict):
    def __init__(
        self,
        core,
        in_shape_dict,
        n_neurons_dict,
        pool_steps,
        pool_kern,
        bias,
        init_range,
        gamma_readout,
    ):
        # super init to get the _module attribute
        super(MultiplePointPooled2d, self).__init__()
        for k in n_neurons_dict:
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]
            self.add_module(
                k,
                PointPooled2d(
                    in_shape,
                    n_neurons,
                    pool_steps=pool_steps,
                    pool_kern=pool_kern,
                    bias=bias,
                    init_range=init_range,
                ),
            )
        self.gamma_readout = gamma_readout


def task_core_point_readout(
    dataloaders,
    seed,
    input_channels=1,
    model_name="vgg19_original",  # begin of core args
    layer_name="features.10",
    pretrained=True,
    bias=False,
    final_batchnorm=True,
    final_nonlinearity=True,
    momentum=0.1,
    fine_tune=False,
    pool_steps=2,
    pool_kern=3,
    init_range=0.2,  # readout args
    readout_bias=True,
    gamma_readout=5.8,
    elu_offset=-1,
    data_info=None,  # output and data_info
):
    """
    A Model class of a predefined core (using models from ptrnets). Can be initialized pretrained or random.
    Can also be set to be trainable or not, independent of initialization.

    Args:
        dataloaders: a dictionary of train-dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: ..
        pool_steps:
        pool_kern:
        readout_bias:
        init_range:
        gamma_readout:

    Returns:
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields[:2]

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = (
        list(input_channels.values())[0]
        if isinstance(input_channels, dict)
        else input_channels[0]
    )

    set_random_seed(seed)

    core = TaskDrivenCore2(
        input_channels=core_input_channels,
        model_name=model_name,
        layer_name=layer_name,
        pretrained=pretrained,
        bias=bias,
        final_batchnorm=final_batchnorm,
        final_nonlinearity=final_nonlinearity,
        momentum=momentum,
        fine_tune=fine_tune,
    )

    set_random_seed(seed)

    core.initialize()

    print(core)

    readout = MultiplePointPooled2d(
        core,
        in_shape_dict=in_shapes_dict,
        n_neurons_dict=n_neurons_dict,
        pool_steps=pool_steps,
        pool_kern=pool_kern,
        bias=readout_bias,
        gamma_readout=gamma_readout,
        init_range=init_range,
    )

    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))[:2]
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


def task_core_factorized_readout(
    dataloaders,
    seed,
    input_channels=1,
    model_name="vgg19_original",  # begin of core args
    layer_name="features.10",
    pretrained=True,
    bias=False,
    final_batchnorm=True,
    final_nonlinearity=True,
    momentum=0.1,
    fine_tune=False,
    init_noise=1e-3,
    normalize=True,
    constrain_pos=False,
    readout_bias=True,
    gamma_readout=5.8,
    elu_offset=-1,
    data_info=None,  # output and data_info
):
    """
    A Model class of a predefined core (using models from ptrnets). Can be initialized pretrained or random.
    Can also be set to be trainable or not, independent of initialization.

    Args:
        dataloaders: a dictionary of train-dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: ..
        pool_steps:
        pool_kern:
        readout_bias:
        init_range:
        gamma_readout:

    Returns:
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields[:2]

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = (
        list(input_channels.values())[0]
        if isinstance(input_channels, dict)
        else input_channels[0]
    )

    set_random_seed(seed)

    core = TaskDrivenCore2(
        input_channels=core_input_channels,
        model_name=model_name,
        layer_name=layer_name,
        pretrained=pretrained,
        bias=bias,
        final_batchnorm=final_batchnorm,
        final_nonlinearity=final_nonlinearity,
        momentum=momentum,
        fine_tune=fine_tune,
    )

    set_random_seed(seed)

    core.initialize()

    readout = MultipleSpatialXFeatureLinear(
        core,
        in_shape_dict=in_shapes_dict,
        n_neurons_dict=n_neurons_dict,
        bias=readout_bias,
        gamma_readout=gamma_readout,
        init_noise=init_noise,
        normalize=normalize,
        constrain_pos=constrain_pos,
    )

    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))[:2]
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model
