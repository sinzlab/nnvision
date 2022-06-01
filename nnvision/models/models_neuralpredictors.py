from neuralpredictors.layers.readouts import MultipleFullGaussian2d
from neuralpredictors.layers.cores import Stacked2dCore, SE2dCore, RotationEquivariant2dCore
from nnfabrik.utility.nn_helpers import set_random_seed, get_dims_for_loader_dict
import torch
from torch import nn
from torch.nn import functional as F
from neuralpredictors.layers.encoders import FiringRateEncoder


def se_core_full_gauss_readout(dataloaders,
                               seed,
                               hidden_channels=32,
                               input_kern=13,
                               hidden_kern=3,
                               layers=3,
                               gamma_input=15.5,
                               skip=0,
                               final_nonlinearity=True,
                               momentum=0.9,
                               pad_input=False,
                               batch_norm=True,
                               hidden_dilation=1,
                               laplace_padding=None,
                               input_regularizer='LaplaceL2norm',
                               init_mu_range=0.2,
                               init_sigma=1.,
                               readout_bias=True,
                               gamma_readout=4,
                               elu_offset=0,
                               stack=None,
                               se_reduction=32,
                               n_se_blocks=1,
                               depth_separable=False,
                               linear=False,
                               gauss_type='full',
                               grid_mean_predictor=None,
                               share_features=False,
                               share_grid=False,
                               data_info=None,
                               gamma_grid_dispersion=0,
                               attention_conv=False,
                               ):
    """
    Model class of a stacked2dCore (from neuralpredictors) and a pointpooled (spatial transformer) readout

    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: random seed
        elu_offset: Offset for the output non-linearity [F.elu(x + self.offset)]
        isotropic: whether the Gaussian readout should use isotropic Gaussians or not
        grid_mean_predictor: if not None, needs to be a dictionary of the form
            {
            'type': 'cortex',
            'input_dimensions': 2,
            'hidden_layers':0,
            'hidden_features':20,
            'final_tanh': False,
            }
            In that case the datasets need to have the property `neurons.cell_motor_coordinates`
        share_features: whether to share features between readouts. This requires that the datasets
            have the properties `neurons.multi_match_id` which are used for matching. Every dataset
            has to have all these ids and cannot have any more.
        all other args: See Documentation of Stacked2dCore in neuralpredictors.layers.cores and
            PointPooled2D in neuralpredictors.layers.readouts

    Returns: An initialized model which consists of model.core and model.readout
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
        print("fail")
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name][1:] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = list(input_channels.values())[0] if isinstance(input_channels, dict) else input_channels[0]

    source_grids = None
    grid_mean_predictor_type = None
    if grid_mean_predictor is not None:
        grid_mean_predictor = copy.deepcopy(grid_mean_predictor)
        grid_mean_predictor_type = grid_mean_predictor.pop('type')
        if grid_mean_predictor_type == 'cortex':
            input_dim = grid_mean_predictor.pop('input_dimensions', 2)
            source_grids = {k: v.dataset.neurons.cell_motor_coordinates[:, :input_dim] for k, v in dataloaders.items()}
        elif grid_mean_predictor_type == 'shared':
            pass

    shared_match_ids = None
    if share_features or share_grid:
        shared_match_ids = {k: v.dataset.neurons.multi_match_id for k, v in dataloaders.items()}
        all_multi_unit_ids = set(np.hstack(shared_match_ids.values()))

        for match_id in shared_match_ids.values():
            assert len(set(match_id) & all_multi_unit_ids) == len(all_multi_unit_ids), \
                'All multi unit IDs must be present in all datasets'

    set_random_seed(seed)

    core = SE2dCore(input_channels=core_input_channels,
                    hidden_channels=hidden_channels,
                    input_kern=input_kern,
                    hidden_kern=hidden_kern,
                    layers=layers,
                    gamma_input=gamma_input,
                    skip=skip,
                    final_nonlinearity=final_nonlinearity,
                    bias=False,
                    momentum=momentum,
                    pad_input=pad_input,
                    batch_norm=batch_norm,
                    hidden_dilation=hidden_dilation,
                    laplace_padding=laplace_padding,
                    input_regularizer=input_regularizer,
                    stack=stack,
                    se_reduction=se_reduction,
                    n_se_blocks=n_se_blocks,
                    depth_separable=depth_separable,
                    linear=linear,
                    attention_conv=attention_conv)

    readout = MultipleFullGaussian2d(in_shape_dict=in_shapes_dict,
                                     loader=dataloaders,

                                     n_neurons_dict=n_neurons_dict,
                                     init_mu_range=init_mu_range,
                                     bias=readout_bias,
                                     init_sigma=init_sigma,
                                     gamma_readout=gamma_readout,
                                     gauss_type=gauss_type,
                                     grid_mean_predictor=grid_mean_predictor,
                                     grid_mean_predictor_type=grid_mean_predictor_type,
                                     source_grids=source_grids,
                                     share_features=share_features,
                                     share_grid=share_grid,
                                     shared_match_ids=shared_match_ids,
                                     gamma_grid_dispersion=gamma_grid_dispersion,
                                     )

    # initializing readout bias to mean response
    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))
            readout[key].bias.data = targets.mean(0)

    model = FiringRateEncoder(core, readout, elu_offset)

    return model