import torch

from torch import nn
from neuralpredictors.utils import get_module_output
from torch.nn import Parameter
from neuralpredictors.layers.readouts import PointPooled2d, FullGaussian2d, SpatialXFeatureLinear, RemappedGaussian2d, AttentionReadout


from neuralpredictors.layers.legacy import Gaussian2d


class MultiplePointPooled2d(torch.nn.ModuleDict):
    def __init__(self, core, in_shape_dict, n_neurons_dict, pool_steps, pool_kern, bias, init_range, gamma_readout):
        # super init to get the _module attribute
        super().__init__()
        for k in n_neurons_dict:
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]
            self.add_module(k, PointPooled2d(
                in_shape,
                n_neurons,
                pool_steps=pool_steps,
                pool_kern=pool_kern,
                bias=bias,
                init_range=init_range)
                            )
        self.gamma_readout = gamma_readout

    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)

    def regularizer(self, data_key):
        return self[data_key].feature_l1(average=False) * self.gamma_readout


class MultipleGaussian2d(torch.nn.ModuleDict):
    def __init__(self, core, in_shape_dict, n_neurons_dict, init_mu_range, init_sigma_range, bias, gamma_readout):
        # super init to get the _module attribute
        super().__init__()
        for k in n_neurons_dict:
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]
            self.add_module(k, Gaussian2d(
                in_shape=in_shape,
                outdims=n_neurons,
                init_mu_range=init_mu_range,
                init_sigma_range=init_sigma_range,
                bias=bias)
                            )
        self.gamma_readout = gamma_readout

    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)

    def regularizer(self, data_key):
        return self[data_key].feature_l1(average=False) * self.gamma_readout


class MultiReadout:

    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)

    def regularizer(self, data_key):
        return self[data_key].feature_l1(average=False) * self.gamma_readout


class MultipleSpatialXFeatureLinear(MultiReadout, torch.nn.ModuleDict):
    def __init__(self, core, in_shape_dict, n_neurons_dict, init_noise, bias, normalize, gamma_readout, constrain_pos=False):
        # super init to get the _module attribute
        super().__init__()
        for k in n_neurons_dict:
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]
            self.add_module(k, SpatialXFeatureLinear(
                in_shape=in_shape,
                outdims=n_neurons,
                init_noise=init_noise,
                bias=bias,
                normalize=normalize,
                constrain_pos=constrain_pos
            )
                            )
        self.gamma_readout = gamma_readout

    def regularizer(self, data_key):
        return self[data_key].l1(average=False) * self.gamma_readout


class MultipleFullGaussian2d(MultiReadout, torch.nn.ModuleDict):
    def __init__(self, core, in_shape_dict, n_neurons_dict, init_mu_range, init_sigma, bias, gamma_readout,
                 gauss_type, grid_mean_predictor, grid_mean_predictor_type, source_grids,
                 share_features, share_grid, shared_match_ids, gamma_grid_dispersion=0):
        # super init to get the _module attribute
        super().__init__()
        k0 = None
        for i, k in enumerate(n_neurons_dict):
            k0 = k0 or k
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]

            source_grid = None
            shared_grid = None
            if grid_mean_predictor is not None:
                if grid_mean_predictor_type == 'cortex':
                    source_grid = source_grids[k]
                else:
                    raise KeyError('grid mean predictor {} does not exist'.format(grid_mean_predictor_type))
            elif share_grid:
                shared_grid = {
                    'match_ids': shared_match_ids[k],
                    'shared_grid': None if i == 0 else self[k0].shared_grid
                }

            if share_features:
                shared_features = {
                    'match_ids': shared_match_ids[k],
                    'shared_features': None if i == 0 else self[k0].shared_features
                }
            else:
                shared_features = None

            self.add_module(k, FullGaussian2d(
                in_shape=in_shape,
                outdims=n_neurons,
                init_mu_range=init_mu_range,
                init_sigma=init_sigma,
                bias=bias,
                gauss_type=gauss_type,
                grid_mean_predictor=grid_mean_predictor,
                shared_features=shared_features,
                shared_grid=shared_grid,
                source_grid=source_grid
            )
                            )
        self.gamma_readout = gamma_readout
        self.gamma_grid_dispersion = gamma_grid_dispersion

    def regularizer(self, data_key):
        if hasattr(FullGaussian2d, 'mu_dispersion'):
            return self[data_key].feature_l1(average=False) * self.gamma_readout \
                   + self[data_key].mu_dispersion() * self.gamma_grid_dispersion
        else:
            return self[data_key].feature_l1(average=False) * self.gamma_readout


class MultipleRemappedGaussian2d(MultiReadout, torch.nn.ModuleDict):
    def __init__(self, core, in_shape_dict, n_neurons_dict, remap_layers, remap_kernel, max_remap_amplitude,
                 init_mu_range, init_sigma, bias, gamma_readout,
                 gauss_type, grid_mean_predictor, grid_mean_predictor_type, source_grids,
                 share_features, share_grid, shared_match_ids, ):
        # super init to get the _module attribute
        super().__init__()
        k0 = None
        for i, k in enumerate(n_neurons_dict):
            k0 = k0 or k
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]

            source_grid = None
            shared_grid = None
            shared_transform = None
            if grid_mean_predictor is not None:
                if grid_mean_predictor_type == 'cortex':
                    source_grid = source_grids[k]
                else:
                    raise KeyError('grid mean predictor {} does not exist'.format(grid_mean_predictor_type))

            elif share_grid:
                shared_grid = {
                    'match_ids': shared_match_ids[k],
                    'shared_grid': None if i == 0 else self[k0].shared_grid
                }

            if share_features:
                shared_features = {
                    'match_ids': shared_match_ids[k],
                    'shared_features': None if i == 0 else self[k0].shared_features
                }
            else:
                shared_features = None

            self.add_module(k, RemappedGaussian2d(
                in_shape=in_shape,
                outdims=n_neurons,
                remap_layers=remap_layers,
                remap_kernel=remap_kernel,
                max_remap_amplitude=max_remap_amplitude,
                init_mu_range=init_mu_range,
                init_sigma=init_sigma,
                bias=bias,
                gauss_type=gauss_type,
                grid_mean_predictor=grid_mean_predictor,
                shared_features=shared_features,
                shared_grid=shared_grid,
                source_grid=source_grid,
            )
                            )
        self.gamma_readout = gamma_readout


class MultipleAttention2d(MultiReadout, torch.nn.ModuleDict):
    def __init__(
        self,
        core,
        in_shape_dict,
        n_neurons_dict,
        attention_layers,
        attention_kernel,
        bias,
        gamma_readout,
    ):
        # super init to get the _module attribute
        super().__init__()
        k0 = None
        for i, k in enumerate(n_neurons_dict):
            k0 = k0 or k
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]

            self.add_module(
                k,
                AttentionReadout(
                    in_shape=in_shape,
                    outdims=n_neurons,
                    attention_layers=attention_layers,
                    attention_kernel=attention_kernel,
                    bias=bias
                ),
            )
        self.gamma_readout = gamma_readout


class DenseReadout(nn.Module):
    """
    Fully connected readout layer.
    """

    def __init__(self, in_shape, outdims, bias, init_noise=1e-3):
        super().__init__()
        self.in_shape = in_shape
        self.outdims = outdims
        self.init_noise = init_noise
        c, w, h = in_shape

        self.linear = torch.nn.Linear(in_features=c*w*h,
                                      out_features=outdims,
                                      bias=False)
        if bias:
            bias = Parameter(torch.Tensor(outdims))
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        self.initialize()

    @property
    def features(self):
        return next(iter(self.linear.parameters()))

    def feature_l1(self, average=False):
        if average:
            return self.features.abs().mean()
        else:
            return self.features.abs().sum()

    def initialize(self):
        self.features.data.normal_(0, self.init_noise)

    def forward(self, x):

        b, c, w, h = x.shape

        x = x.view(b, c * w * h)
        y = self.linear(x)
        if self.bias is not None:
            y = y + self.bias
        return y

    def __repr__(self):
        return self.__class__.__name__ + \
               ' (' + '{} x {} x {}'.format(*self.in_shape) + ' -> ' + str(
            self.outdims) + ')'


class MultipleDense(MultiReadout, torch.nn.ModuleDict):
    def __init__(
        self,
        core,
        in_shape_dict,
        n_neurons_dict,
        bias,
        gamma_readout,
        init_noise,
    ):
        # super init to get the _module attribute
        super().__init__()
        k0 = None
        for i, k in enumerate(n_neurons_dict):
            k0 = k0 or k
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]

            self.add_module(
                k,
                DenseReadout(
                    in_shape=in_shape,
                    outdims=n_neurons,
                    bias=bias,
                    init_noise=init_noise,
                ),
            )
        self.gamma_readout = gamma_readout