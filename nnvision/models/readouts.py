import torch
from torch import nn
from einops import rearrange

from torch import nn
from neuralpredictors.utils import get_module_output
from torch.nn import Parameter
from neuralpredictors.layers.readouts import (
    PointPooled2d,
    FullGaussian2d,
    FullGaussian2dModulators,
    SpatialXFeatureLinear,
    RemappedGaussian2d,
    AttentionReadout,
)
from neuralpredictors.layers.legacy import Gaussian2d
from neuralpredictors.layers.attention_readout import (
    Attention2d,
    MultiHeadAttention2d,
    SharedMultiHeadAttention2d,
)
from neuralpredictors.utils import PositionalEncoding2D


class MultiReadout:
    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)

    def regularizer(self, data_key):
        return self[data_key].feature_l1(average=False) * self.gamma_readout


class MultiplePointPooled2d(MultiReadout, torch.nn.ModuleDict):
    base_r = PointPooled2d


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
        super().__init__()
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


class MultipleGaussian2d(torch.nn.ModuleDict):
    def __init__(
        self,
        core,
        in_shape_dict,
        n_neurons_dict,
        init_mu_range,
        init_sigma_range,
        bias,
        gamma_readout,
    ):
        # super init to get the _module attribute
        super().__init__()
        for k in n_neurons_dict:
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]
            self.add_module(
                k,
                Gaussian2d(
                    in_shape=in_shape,
                    outdims=n_neurons,
                    init_mu_range=init_mu_range,
                    init_sigma_range=init_sigma_range,
                    bias=bias,
                ),
            )
        self.gamma_readout = gamma_readout

    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)

    def regularizer(self, data_key):
        return self[data_key].feature_l1(average=False) * self.gamma_readout


class MultipleSelfAttention2d(torch.nn.ModuleDict):
    def __init__(
        self, core, in_shape_dict, n_neurons_dict, bias, gamma_features=0, gamma_query=0
    ):
        # super init to get the _module attribute
        super().__init__()
        for k in n_neurons_dict:
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]
            self.add_module(
                k, Attention2d(in_shape=in_shape, outdims=n_neurons, bias=bias)
            )
        self.gamma_features = gamma_features
        self.gamma_query = gamma_query

    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)

    def regularizer(self, data_key):
        return (
            self[data_key].feature_l1(average=False) * self.gamma_features
            + self[data_key].query_l1(average=False) * self.gamma_query
        )


class MultipleMultiHeadAttention2d(torch.nn.ModuleDict):
    def __init__(
        self,
        core,
        in_shape_dict,
        n_neurons_dict,
        bias,
        gamma_features=0,
        gamma_query=0,
        use_pos_enc=True,
        learned_pos=False,
        heads=1,
        scale=False,
        key_embedding=False,
        value_embedding=False,
        temperature=(False, 1.0),  # (learnable-per-neuron, value)
        dropout_pos=0.1,
        layer_norm=False,
        stack_pos_encoding=False,
        n_pos_channels=None,
    ):

        # super init to get the _module attribute
        super().__init__()
        for k in n_neurons_dict:
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]
            self.add_module(
                k,
                MultiHeadAttention2d(
                    in_shape=in_shape,
                    outdims=n_neurons,
                    bias=bias,
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
                ),
            )
        self.gamma_features = gamma_features
        self.gamma_query = gamma_query

    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)

    def regularizer(self, data_key):
        return (
            self[data_key].feature_l1(average=False) * self.gamma_features
            + self[data_key].query_l1(average=False) * self.gamma_query
        )


class MultipleSharedMultiHeadAttention2d(torch.nn.ModuleDict):
    def __init__(
        self,
        core,
        in_shape_dict,
        n_neurons_dict,
        bias,
        gamma_features=0,
        gamma_query=0,
        gamma_embedding=0,
        use_pos_enc=True,
        learned_pos=False,
        heads=1,
        scale=False,
        key_embedding=False,
        value_embedding=False,
        temperature=(False, 1.0),  # (learnable-per-neuron, value)
        dropout_pos=0.1,
        layer_norm=False,
        stack_pos_encoding=False,
        n_pos_channels=None,
        embed_out_dim=None,
    ):

        if bool(n_pos_channels) ^ bool(stack_pos_encoding):
            raise ValueError(
                "when stacking the position embedding, the number of channels must be specified."
                "Similarly, when not stacking the position embedding, n_pos_channels must be None"
            )

        super().__init__()
        self.n_data_keys = len(n_neurons_dict.keys())
        self.heads = heads
        self.key_embedding = key_embedding
        self.value_embedding = value_embedding
        self.use_pos_enc = use_pos_enc

        # get output of first dim
        k = list(in_shape_dict.keys())[0]
        in_shape = get_module_output(core, in_shape_dict[k])[1:]

        c, w, h = in_shape
        if n_pos_channels and stack_pos_encoding:
            c = c + n_pos_channels
        c_out = c if not embed_out_dim else embed_out_dim

        d_model = n_pos_channels if n_pos_channels else c
        if self.use_pos_enc:
            self.position_embedding = PositionalEncoding2D(
                d_model=d_model,
                width=w,
                height=h,
                learned=learned_pos,
                dropout=dropout_pos,
                stack_channels=stack_pos_encoding,
            )

        if layer_norm:
            self.norm = nn.LayerNorm((c, w * h))
        else:
            self.norm = None

        if self.key_embedding and self.value_embedding:
            self.to_kv = nn.Linear(c, c_out * 2, bias=False)
        elif self.key_embedding:
            self.to_key = nn.Linear(c, c_out, bias=False)

        # super init to get the _module attribute
        for k in n_neurons_dict:
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]
            self.add_module(
                k,
                SharedMultiHeadAttention2d(
                    in_shape=in_shape,
                    outdims=n_neurons,
                    bias=bias,
                    use_pos_enc=False,
                    key_embedding=key_embedding,
                    value_embedding=value_embedding,
                    heads=heads,
                    scale=scale,
                    temperature=temperature,  # (learnable-per-neuron, value)
                    layer_norm=layer_norm,
                    stack_pos_encoding=stack_pos_encoding,
                    n_pos_channels=n_pos_channels,
                    embed_out_dim=embed_out_dim,
                ),
            )
        self.gamma_features = gamma_features
        self.gamma_query = gamma_query
        self.gamma_embedding = gamma_embedding

    def forward(self, x, data_key=None, **kwargs):

        i, c, w, h = x.size()
        x = x.flatten(2, 3)  # [Images, Channels, w*h]
        if self.use_pos_enc:
            x_embed = self.position_embedding(x)  # -> [Images, Channels, w*h]
        else:
            x_embed = x

        if self.norm is not None:
            x_embed = self.norm(x_embed)

        if self.key_embedding and self.value_embedding:
            key, value = self.to_kv(rearrange(x_embed, "i c s -> (i s) c")).chunk(
                2, dim=-1
            )
            key = rearrange(key, "(i s) (h d) -> i h d s", h=self.heads, i=i)
            value = rearrange(value, "(i s) (h d) -> i h d s", h=self.heads, i=i)
        elif self.key_embedding:
            key = self.to_key(rearrange(x_embed, "i c s -> (i s) c"))
            key = rearrange(key, "(i s) (h d) -> i h d s", h=self.heads, i=i)
            value = rearrange(x, "i (h d) s -> i h d s", h=self.heads)
        else:
            key = rearrange(x_embed, "i (h d) s -> i h d s", h=self.heads)
            value = rearrange(x, "i (h d) s -> i h d s", h=self.heads)

        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](key, value, **kwargs)

    def embedding_l1(
        self,
    ):
        if self.key_embedding and self.value_embedding:
            return self.to_kv.weight.abs().mean()
        elif self.key_embedding:
            return self.to_key.weight.abs().mean()
        else:
            return 0

    def regularizer(self, data_key):
        return (
            self[data_key].feature_l1(average=False) * self.gamma_features
            + self[data_key].query_l1(average=False) * self.gamma_query
            + self.embedding_l1() * self.gamma_embedding
        )


class MultipleSpatialXFeatureLinear(MultiReadout, torch.nn.ModuleDict):
    def __init__(
        self,
        core,
        in_shape_dict,
        n_neurons_dict,
        init_noise,
        bias,
        normalize,
        gamma_readout,
        constrain_pos=False,
    ):
        # super init to get the _module attribute
        super().__init__()
        for k in n_neurons_dict:
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]
            self.add_module(
                k,
                SpatialXFeatureLinear(
                    in_shape=in_shape,
                    outdims=n_neurons,
                    init_noise=init_noise,
                    bias=bias,
                    normalize=normalize,
                    constrain_pos=constrain_pos,
                ),
            )
        self.gamma_readout = gamma_readout

    def regularizer(self, data_key):
        return self[data_key].l1(average=False) * self.gamma_readout


class MultipleFullGaussian2d(MultiReadout, torch.nn.ModuleDict):
    def __init__(
        self,
        core,
        in_shape_dict,
        n_neurons_dict,
        init_mu_range,
        init_sigma,
        bias,
        gamma_readout,
        gauss_type,
        grid_mean_predictor,
        grid_mean_predictor_type,
        source_grids,
        share_features,
        share_grid,
        shared_match_ids,
        gamma_grid_dispersion=0,
        **kwargs
    ):
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
                if grid_mean_predictor_type == "cortex":
                    source_grid = source_grids[k]
                else:
                    raise KeyError(
                        "grid mean predictor {} does not exist".format(
                            grid_mean_predictor_type
                        )
                    )
            elif share_grid:
                shared_grid = {
                    "match_ids": shared_match_ids[k],
                    "shared_grid": None if i == 0 else self[k0].shared_grid,
                }

            if share_features:
                shared_features = {
                    "match_ids": shared_match_ids[k],
                    "shared_features": None if i == 0 else self[k0].shared_features,
                }
            else:
                shared_features = None

            self.add_module(
                k,
                FullGaussian2d(
                    in_shape=in_shape,
                    outdims=n_neurons,
                    init_mu_range=init_mu_range,
                    init_sigma=init_sigma,
                    bias=bias,
                    gauss_type=gauss_type,
                    grid_mean_predictor=grid_mean_predictor,
                    shared_features=shared_features,
                    shared_grid=shared_grid,
                    source_grid=source_grid,
                    **kwargs,
                ),
            )

        self.context_regularizer = kwargs.get("context_regularizer", False)
        self.gamma_context_regularizer = kwargs.get("gamma_context_regularizer", 3)
        self.prev_regularizer = kwargs.get("prev_regularizer", False)
        self.gamma_prev_regularizer = kwargs.get("gamma_prev_regularizer", 3)
        if self.context_regularizer:
            print("context regularizer, gamma:")
            print(self.gamma_context_regularizer)
        if self.prev_regularizer:
            print("prev regularizer, gamma:")
            print(self.gamma_prev_regularizer)
        self.gamma_readout = gamma_readout
        self.gamma_grid_dispersion = gamma_grid_dispersion

    def regularizer(self, data_key):
        if hasattr(FullGaussian2d, "mu_dispersion"):
            if self.context_regularizer and self.prev_regularizer:
                return (
                    self[data_key].feature_l1(average=False) * self.gamma_readout
                    + self[data_key].mu_dispersion * self.gamma_grid_dispersion
                    + self[data_key].context_modulator_l1()
                    * self.gamma_context_regularizer
                    + self[data_key].prev_modulator_l1() * self.gamma_prev_regularizer
                )
            elif self.context_regularizer:
                return (
                    self[data_key].feature_l1(average=False) * self.gamma_readout
                    + self[data_key].mu_dispersion * self.gamma_grid_dispersion
                    + self[data_key].context_modulator_l1()
                    * self.gamma_context_regularizer
                )
            elif self.prev_regularizer:
                return (
                    self[data_key].feature_l1(average=False) * self.gamma_readout
                    + self[data_key].mu_dispersion * self.gamma_grid_dispersion
                    + self[data_key].prev_modulator_l1() * self.gamma_prev_regularizer
                )
            else:
                return (
                    self[data_key].feature_l1(average=False) * self.gamma_readout
                    + self[data_key].mu_dispersion * self.gamma_grid_dispersion
                )
        else:
            return self[data_key].feature_l1(average=False) * self.gamma_readout


class MultipleFullGaussian2dModulators(MultiReadout, torch.nn.ModuleDict):
    def __init__(
        self,
        core,
        in_shape_dict,
        n_neurons_dict,
        init_mu_range,
        init_sigma,
        bias,
        gamma_readout,
        gauss_type,
        grid_mean_predictor,
        grid_mean_predictor_type,
        source_grids,
        share_features,
        share_grid,
        shared_match_ids,
        prev_resps,
        prev_hidden_layers,
        prev_hidden_features,
        prev_combine_addition,
        prev_self,
        prev_minus_self,
        prev_final_nonlin,
        prev_hidden_bias,
        prev_output_bias,
        other_resps,
        other_hidden_layers,
        other_hidden_features,
        other_combine_addition,
        context_resps,
        context_hidden_layers,
        context_hidden_features,
        context_combine_addition,
        context_session,
        context_final_nonlin,
        context_hidden_bias,
        context_output_bias,
        n_neurons,
        gamma_grid_dispersion=0,
        **kwargs
    ):
        # super init to get the _module attribute
        super().__init__()
        k0 = None
        for i, k in enumerate(n_neurons_dict):
            k0 = k0 or k
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            outdims = n_neurons_dict[k]

            source_grid = None
            shared_grid = None
            if grid_mean_predictor is not None:
                if grid_mean_predictor_type == "cortex":
                    source_grid = source_grids[k]
                else:
                    raise KeyError(
                        "grid mean predictor {} does not exist".format(
                            grid_mean_predictor_type
                        )
                    )
            elif share_grid:
                shared_grid = {
                    "match_ids": shared_match_ids[k],
                    "shared_grid": None if i == 0 else self[k0].shared_grid,
                }

            if share_features:
                shared_features = {
                    "match_ids": shared_match_ids[k],
                    "shared_features": None if i == 0 else self[k0].shared_features,
                }
            else:
                shared_features = None
            self.add_module(
                k,
                FullGaussian2dModulators(
                    in_shape=in_shape,
                    outdims=outdims,
                    init_mu_range=init_mu_range,
                    init_sigma=init_sigma,
                    bias=bias,
                    gauss_type=gauss_type,
                    grid_mean_predictor=grid_mean_predictor,
                    shared_features=shared_features,
                    shared_grid=shared_grid,
                    source_grid=source_grid,
                    prev_resps=prev_resps,
                    prev_hidden_layers=prev_hidden_layers,
                    prev_hidden_features=prev_hidden_features,
                    prev_combine_addition=prev_combine_addition,
                    prev_self=prev_self,
                    prev_minus_self=prev_minus_self,
                    prev_final_nonlin=prev_final_nonlin,
                    prev_hidden_bias=prev_hidden_bias,
                    prev_output_bias=prev_output_bias,
                    other_resps=other_resps,
                    other_hidden_layers=other_hidden_layers,
                    other_hidden_features=other_hidden_features,
                    other_combine_addition=other_combine_addition,
                    context_resps=context_resps,
                    context_hidden_layers=context_hidden_layers,
                    context_hidden_features=context_hidden_features,
                    context_combine_addition=context_combine_addition,
                    context_session=context_session,
                    context_final_nonlin=context_final_nonlin,
                    context_hidden_bias=context_hidden_bias,
                    context_output_bias=context_output_bias,
                    n_neurons=n_neurons,
                    **kwargs,
                ),
            )

        self.context_regularizer = kwargs.get("context_regularizer", False)
        self.gamma_context_regularizer = kwargs.get("gamma_context_regularizer", 3)
        self.prev_regularizer = kwargs.get("prev_regularizer", False)
        self.gamma_prev_regularizer = kwargs.get("gamma_prev_regularizer", 3)
        if self.context_regularizer:
            print("context regularizer, gamma:")
            print(self.gamma_context_regularizer)
        if self.prev_regularizer:
            print("prev regularizer, gamma:")
            print(self.gamma_prev_regularizer)
        self.gamma_readout = gamma_readout
        self.gamma_grid_dispersion = gamma_grid_dispersion

    def regularizer(self, data_key):
        if hasattr(FullGaussian2d, "mu_dispersion"):
            if self.context_regularizer and self.prev_regularizer:
                return (
                    self[data_key].feature_l1(average=False) * self.gamma_readout
                    + self[data_key].mu_dispersion * self.gamma_grid_dispersion
                    + self[data_key].context_modulator_l1()
                    * self.gamma_context_regularizer
                    + self[data_key].prev_modulator_l1() * self.gamma_prev_regularizer
                )
            elif self.context_regularizer:
                return (
                    self[data_key].feature_l1(average=False) * self.gamma_readout
                    + self[data_key].mu_dispersion * self.gamma_grid_dispersion
                    + self[data_key].context_modulator_l1()
                    * self.gamma_context_regularizer
                )
            elif self.prev_regularizer:
                return (
                    self[data_key].feature_l1(average=False) * self.gamma_readout
                    + self[data_key].mu_dispersion * self.gamma_grid_dispersion
                    + self[data_key].prev_modulator_l1() * self.gamma_prev_regularizer
                )
            else:
                return (
                    self[data_key].feature_l1(average=False) * self.gamma_readout
                    + self[data_key].mu_dispersion * self.gamma_grid_dispersion
                )
        else:
            return self[data_key].feature_l1(average=False) * self.gamma_readout


class MultipleRemappedGaussian2d(MultiReadout, torch.nn.ModuleDict):
    def __init__(
        self,
        core,
        in_shape_dict,
        n_neurons_dict,
        remap_layers,
        remap_kernel,
        max_remap_amplitude,
        init_mu_range,
        init_sigma,
        bias,
        gamma_readout,
        gauss_type,
        grid_mean_predictor,
        grid_mean_predictor_type,
        source_grids,
        share_features,
        share_grid,
        shared_match_ids,
    ):
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
                if grid_mean_predictor_type == "cortex":
                    source_grid = source_grids[k]
                else:
                    raise KeyError(
                        "grid mean predictor {} does not exist".format(
                            grid_mean_predictor_type
                        )
                    )

            elif share_grid:
                shared_grid = {
                    "match_ids": shared_match_ids[k],
                    "shared_grid": None if i == 0 else self[k0].shared_grid,
                }

            if share_features:
                shared_features = {
                    "match_ids": shared_match_ids[k],
                    "shared_features": None if i == 0 else self[k0].shared_features,
                }
            else:
                shared_features = None

            self.add_module(
                k,
                RemappedGaussian2d(
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
                ),
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
                    bias=bias,
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

        self.linear = torch.nn.Linear(
            in_features=c * w * h, out_features=outdims, bias=False
        )
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
        return (
            self.__class__.__name__
            + " ("
            + "{} x {} x {}".format(*self.in_shape)
            + " -> "
            + str(self.outdims)
            + ")"
        )


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
