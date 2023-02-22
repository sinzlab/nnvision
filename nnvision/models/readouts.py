import torch
from torch import nn
from einops import rearrange

from torch import nn
from neuralpredictors.utils import get_module_output
from torch.nn import Parameter
from neuralpredictors.layers.readouts import (
    PointPooled2d,
    FullGaussian2d,
    SpatialXFeatureLinear,
    RemappedGaussian2d,
    AttentionReadout,
)

try:
    from neuralpredictors.layers.slot_attention import SlotAttention
except:
    pass
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
        self,
        core,
        in_shape_dict,
        n_neurons_dict,
        bias,
        gamma_features=0,
        gamma_query=0,
        use_pos_enc=True,
        learned_pos=False,
        dropout_pos=0.1,
        stack_pos_encoding=False,
        n_pos_channels=None,
        temperature=1.0,
    ):
        # super init to get the _module attribute
        super().__init__()
        for k in n_neurons_dict:
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]
            self.add_module(
                k,
                Attention2d(
                    in_shape=in_shape,
                    outdims=n_neurons,
                    bias=bias,
                    use_pos_enc=use_pos_enc,
                    learned_pos=learned_pos,
                    dropout_pos=dropout_pos,
                    stack_pos_encoding=stack_pos_encoding,
                    n_pos_channels=n_pos_channels,
                    temperature=temperature,
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


class SharedSlotAttention2d(torch.nn.ModuleDict):
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
        slot_num_iterations=1,  # begin slot_attention arguments
        num_slots=10,
        slot_size=None,
        slot_input_size=None,
        slot_mlp_hidden_size_factor=2,
        slot_epsilon=1e-8,
        draw_slots=True,
        use_slot_gru=True,
        use_weighted_mean=True,
        full_skip=False,
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

        self.slot_size = slot_size if slot_size is not None else c

        if self.key_embedding and self.value_embedding:
            self.to_kv = nn.Linear(self.slot_size, c_out * 2, bias=False)
        elif self.key_embedding:
            self.to_key = nn.Linear(self.slot_size, c_out, bias=False)

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

        self.slot_attention = SlotAttention(
            num_iterations=slot_num_iterations,
            num_slots=num_slots,
            slot_size=self.slot_size,
            input_size=c
            if slot_input_size is None
            else slot_input_size,  # number of core output channels
            mlp_hidden_size=slot_mlp_hidden_size_factor * c,
            epsilon=slot_epsilon,
            draw_slots=draw_slots,
            use_slot_gru=use_slot_gru,
            use_weighted_mean=use_weighted_mean,
            full_skip=full_skip,
        )

    def forward(self, x, data_key=None, **kwargs):
        output_slot_attention = kwargs.get("output_attn_weights", None)
        i, c, w, h = x.size()
        x = x.flatten(2, 3)  # [Images, Channels, w*h]
        if self.use_pos_enc:
            x_embed = self.position_embedding(x)  # -> [Images, Channels, w*h]
        else:
            x_embed = x

        #TODO insert MLP


        x_embed = x_embed.permute(0, 2, 1)  # batch_sizes, w*h, c

        slots, slot_attention_maps = self.slot_attention(x_embed, )
        slots = slots.permute(0, 2, 1)  # batch_size, channels, w*h


        if self.key_embedding and self.value_embedding:
            key, value = self.to_kv(rearrange(slots, "i c s -> (i s) c")).chunk(
                2, dim=-1
            )
            key = rearrange(key, "(i s) (h d) -> i h d s", h=self.heads, i=i)
            value = rearrange(value, "(i s) (h d) -> i h d s", h=self.heads, i=i)
        elif self.key_embedding:
            key = self.to_key(rearrange(slots, "i c s -> (i s) c"))
            key = rearrange(key, "(i s) (h d) -> i h d s", h=self.heads, i=i)
            value = rearrange(x, "i (h d) s -> i h d s", h=self.heads)
        else:
            key = rearrange(slots, "i (h d) s -> i h d s", h=self.heads)
            value = rearrange(x, "i (h d) s -> i h d s", h=self.heads)

        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](key, value, slot_attention_maps=slot_attention_maps, **kwargs,)

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


class FullSlotAttention2d(torch.nn.ModuleDict):
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
            slot_num_iterations=3,  # begin slot_attention arguments
            num_slots=10,
            slot_size=None,
            slot_input_size=None,
            slot_mlp_hidden_size_factor=1,
            slot_epsilon=1e-8,
            draw_slots=True,
            use_slot_gru=True,
            use_weighted_mean=True,
            full_skip=False,
            slot_temperature=(False, 1.0),
            use_post_embed_mlp=True,
            learn_slot_weights=False,
            post_slot_heads=1,
            position_invariant=True,
            slot_pool_size=None,
            chunk_size=None,
    ):

        if bool(n_pos_channels) ^ bool(stack_pos_encoding):
            raise ValueError(
                "when stacking the position embedding, the number of channels must be specified."
                "Similarly, when not stacking the position embedding, n_pos_channels must be None"
            )

        super().__init__()

        # get output of first dim
        k = list(in_shape_dict.keys())[0]
        in_shape = get_module_output(core, in_shape_dict[k])[1:]
        n_neurons = n_neurons_dict[k]

        self.in_shape = in_shape
        self.n_neurons = n_neurons
        self.outdims = n_neurons
        self.n_data_keys = len(n_neurons_dict.keys())
        if self.n_data_keys > 1:
            raise NotImplementedError("Multiple data keys not yet supported")

        self.heads = heads
        self.key_embedding = key_embedding
        self.value_embedding = value_embedding
        self.use_pos_enc = use_pos_enc
        self.gamma_features = gamma_features
        self.gamma_query = gamma_query
        self.gamma_embedding = gamma_embedding


        # classical readout args
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
        self._features = Parameter(torch.Tensor(1, c_out, self.n_neurons))
        self._features.data.fill_(1 / self.in_shape[0])

        self.neuron_query = Parameter(torch.Tensor(1, c_out, self.outdims))
        self.neuron_query.data.fill_(1 / self.in_shape[0])
        if bias:
            self.bias = Parameter(torch.Tensor(self.n_neurons))
            self.bias.data.fill_(0)
            self.register_parameter("bias", self.bias)
        else:
            self.register_parameter("bias", None)

        if value_embedding and embed_out_dim and not learn_slot_weights:
            self._features = Parameter(torch.Tensor(1, embed_out_dim, self.outdims))
            self._features.data.fill_(1 / self.in_shape[0])
        if key_embedding and embed_out_dim:
            self.neuron_query = Parameter(torch.Tensor(1, embed_out_dim, self.outdims))
            self.neuron_query.data.fill_(1 / self.in_shape[0])

        if scale:
            dim_head = c // self.heads
            self.scale = dim_head ** -0.5  # prevent softmax gradients from vanishing (for large dim_head)
        else:
            self.scale = 1.0
        if temperature[0]:
            self.T = temperature[1]
        else:
            self.T = Parameter(torch.ones(self.n_neurons) * temperature[1])
        if layer_norm:
            self.norm = nn.LayerNorm((c, w * h))
        else:
            self.norm = None

        self.slot_size = slot_size if slot_size is not None else c
        self.num_slots = num_slots
        if learn_slot_weights:
            self._features = Parameter(torch.Tensor(1, c_out, self.outdims))
            self._features.data.fill_(1 / self.in_shape[0])

        if self.key_embedding and self.value_embedding:
            self.to_kv = nn.Linear(self.slot_size, c_out * 2, bias=False)
        elif self.key_embedding:
            self.to_key = nn.Linear(self.slot_size, c_out, bias=False)

        self.use_post_embed_mlp = use_post_embed_mlp
        self.slot_attention = SlotAttention(
            num_iterations=slot_num_iterations,
            num_slots=num_slots,
            slot_size=self.slot_size,
            input_size=c
            if slot_input_size is None
            else slot_input_size,  # number of core output channels
            mlp_hidden_size=slot_mlp_hidden_size_factor * c,
            epsilon=slot_epsilon,
            draw_slots=draw_slots,
            use_slot_gru=use_slot_gru,
            use_weighted_mean=use_weighted_mean,
            full_skip=full_skip,
            slot_temperature=slot_temperature,
            position_invariant=position_invariant,
            pool_size=slot_pool_size,
        )

        if self.use_post_embed_mlp:
            self.post_embedding_mlp = nn.Sequential(
                nn.Linear(c, c),
                nn.ReLU(True),
                nn.Linear(c, c if slot_input_size is None else slot_input_size)
            )

        self.learn_slot_weights = learn_slot_weights
        self.neuron_slot_weight = torch.nn.Parameter(torch.randn(1, 1, self.num_slots, self.n_neurons))
        self.position_invariant = position_invariant
        self.post_slot_heads = post_slot_heads
        self.post_slot_kv = nn.Linear(self.slot_size, c_out * 2, bias=False)
        self.post_slot_neuron_query = Parameter(torch.Tensor(1, embed_out_dim, self.n_neurons))
        self.post_slot_neuron_query.data.fill_(1 / embed_out_dim)
        self.chunk_size = chunk_size

    def forward(self, x, **kwargs):
        output_attn_weights = kwargs.get("output_attn_weights", None)
        i, c, w, h = x.size()
        x = x.flatten(2, 3)  # [Images, Channels, w*h]
        if self.use_pos_enc:
            x_embed = self.position_embedding(x)  # -> [Images, Channels, w*h]
        else:
            x_embed = x

        # Start Slot Attention
        x_embed = x_embed.permute(0, 2, 1)  # batch_sizes, w*h, c
        if self.use_post_embed_mlp:
            x_embed = self.post_embedding_mlp(x_embed)

        slots, slot_attention_maps = self.slot_attention(x_embed, width=w, height=h)  # batch_sizes, slots, channels
        if self.position_invariant:
            slots = slots.permute(0, 2, 1)  # batch_size, slot_size, n_slots
        else:
            # slots are of shape [batch_size, w_ * h_, num_slots, slot_size]
            slots = slots.permute(0, 3, 1, 2)  # batch_size, slot_size, w_ * h_, num_slots

        if not self.learn_slot_weights:
            # Self Attention based on slots

            assert self.position_invariant, "Position invariant must be true for dynamic slot weights"

            if self.key_embedding and self.value_embedding:
                key, value = self.to_kv(rearrange(slots, "i c s -> (i s) c")).chunk(
                    2, dim=-1
                )
                key = rearrange(key, "(i s) (h d) -> i h d s", h=self.heads, i=i)
                value = rearrange(value, "(i s) (h d) -> i h d s", h=self.heads, i=i)
            elif self.key_embedding:
                key = self.to_key(rearrange(slots, "i c s -> (i s) c"))
                key = rearrange(key, "(i s) (h d) -> i h d s", h=self.heads, i=i)
                value = rearrange(x, "i (h d) s -> i h d s", h=self.heads)
            else:
                key = rearrange(slots, "i (h d) s -> i h d s", h=self.heads)
                value = rearrange(x, "i (h d) s -> i h d s", h=self.heads)
            query = rearrange(self.neuron_query, "o (h d) n -> o h d n", h=self.heads)

            # compare neuron query with each spatial position (dot-product)
            dot = torch.einsum("ihds,ohdn->ihsn", key, query)  # -> [Images, Heads, w*h, Neurons]
            weights = dot * self.scale / self.T
            attention_weights = torch.nn.functional.softmax(weights, dim=2)  # -> [Images, Heads, n_slots, Neurons]
        else:
            slot_weights = torch.nn.functional.softmax(self.neuron_slot_weight, dim=2)

        if self.position_invariant:
            if not self.learn_slot_weights:
                y = torch.einsum("ihds,ihsn->ihdn", value, attention_weights)  # -> [Images, Heads, Head_Dim, Neurons]
                y = rearrange(y, "i h d n -> i (h d) n")  # -> [Images, Channels, Neurons]
            else:
                y = (slots.unsqueeze(3) * slot_weights).sum(2)  # [batch_size, slot_size, n_slots, 1] * [1, 1, n_slot, n_neuron] -> [Images, Channels, Neurons]

        else:

            # weighted sum over slots per neuron
            #weighted_slots = torch.einsum("icds,absn->icdn", slots, slot_weights)

            # loop over chunks of neurons (slot_weights is too large to be multiplied with slots)
            y_list = []
            for ii in range(0, self.n_neurons, self.chunk_size):
                chunk_size = min(self.chunk_size, self.n_neurons - ii)
                chunked_slot_weights = slot_weights[:, :, :, ii:ii+chunk_size]

                weighted_slots = slots.unsqueeze(-1) * chunked_slot_weights.unsqueeze(2) # [batch_size, slot_size, w_ * h_, num_slots, 1] * [1, 1, 1, num_slots, n_neurons]
                weighted_slots = weighted_slots.sum(3) # [batch_size, slot_size, w_ * h_, n_neurons]
                # -> [Images, Channels, w_*h_, Neurons]

                key, value = self.post_slot_kv(rearrange(weighted_slots, "i c s n -> (i s n) c")).chunk(
                    2, dim=-1
                )
                key = rearrange(key, "(i s n) (h c) -> i h c s n", h=self.post_slot_heads, i=i, n=chunk_size) # [Images, heads, Channels, w*h, Neurons]
                value = rearrange(value, "(i s n ) (h c) -> i h c s n", h=self.post_slot_heads, i=i, n=chunk_size)

                query = self.post_slot_neuron_query[..., ii:ii + chunk_size]
                query = rearrange(query, "o (h c) n -> o h c n", h=self.post_slot_heads) # [1, heads, channels, neurons]
                dot = torch.einsum("ihcsn,ohcn->ihsn", key, query)  # -> [Images, Heads, w*h, Neurons]

                weights = dot * self.scale / self.T
                attention_weights = torch.nn.functional.softmax(weights, dim=2)
                y = torch.einsum("ihcsn,ihsn->ihcn", value, attention_weights)
                y = rearrange(y, "i h d n -> i (h d) n")  # -> [Images, Channels, Neurons]
                y_list.append(y)
            y = torch.cat(y_list, dim=2)

        # new version:
        # get slots out like normal
        # then: just do a query + feature readout. No need for key/value


        # fix bug and initialize features with number of channels, c_out that come out of kv_embedding
        feat = self._features.view(1, -1, self.outdims)
        y = torch.einsum("icn,ocn->in", y, feat)  # -> [Images, Neurons]

        if self.bias is not None:
            y = y + self.bias

        # return neuronal responses, and attention maps if requested
        if output_attn_weights:
            return y, slot_attention_maps
        return y

    def feature_l1(self, average=True):
        """
        Returns the l1 regularization term either the mean or the sum of all weights
        Args:
            average(bool): if True, use mean of weights for regularization
        """
        if average:
            return self._features.abs().mean()
        else:
            return self._features.abs().sum()

    def query_l1(self, average=True):
        """
        Returns the l1 regularization term either the mean or the sum of all weights
        Args:
            average(bool): if True, use mean of weights for regularization
        """
        if average:
            return self.neuron_query.abs().mean()
        else:
            return self.neuron_query.abs().sum()

    def embedding_l1(
            self,
    ):
        if self.key_embedding and self.value_embedding:
            return self.to_kv.weight.abs().mean()
        elif self.key_embedding:
            return self.to_key.weight.abs().mean()
        else:
            return 0

    def regularizer(self, data_key=None):
        return (
                self.feature_l1(average=False) * self.gamma_features
                + self.query_l1(average=False) * self.gamma_query
                + self.embedding_l1() * self.gamma_embedding
        )


class GroupSlotAttention2d(torch.nn.ModuleDict):
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
            temperature=(False, 1.0),  # (learnable-per-neuron, value)
            dropout_pos=0.1,
            stack_pos_encoding=False,
            n_pos_channels=None,
            slot_num_iterations=3,  # begin slot_attention arguments
            num_slots=10,
            slot_size=None,
            slot_input_size=None,
            slot_mlp_hidden_size_factor=1,
            slot_epsilon=1e-8,
            draw_slots=True,
            use_slot_gru=True,
            use_weighted_mean=True,
            full_skip=False,
            slot_temperature=(False, 1.0),
            use_post_embed_mlp=True,
            broadcast_size=None,
            use_post_pos_enc=True,
            learned_post_pos=False,
            dropout_post_pos=0.1,
            stack_post_pos_encoding=None,
            n_post_pos_channels=None,
            self_attention=False,
            embed_out=None,
            neuron_slot_weight_temp=1.0,
    ):

        if bool(n_pos_channels) ^ bool(stack_pos_encoding):
            raise ValueError(
                "when stacking the position embedding, the number of channels must be specified."
                "Similarly, when not stacking the position embedding, n_pos_channels must be None"
            )

        super().__init__()

        # get output of first dim
        k = list(in_shape_dict.keys())[0]
        in_shape = get_module_output(core, in_shape_dict[k])[1:]
        n_neurons = n_neurons_dict[k]

        self.in_shape = in_shape
        self.n_neurons = n_neurons
        self.outdims = n_neurons
        self.n_data_keys = len(n_neurons_dict.keys())
        if self.n_data_keys > 1:
            raise NotImplementedError("Multiple data keys not yet supported")
        self.use_pos_enc = use_pos_enc
        self.use_post_pos_enc = use_post_pos_enc

        self.gamma_features = gamma_features
        self.gamma_query = gamma_query

        self.self_attention = self_attention

        if bias:
            self.bias = Parameter(torch.Tensor(self.n_neurons))
            self.bias.data.fill_(0)
            self.register_parameter("bias", self.bias)
        else:
            self.register_parameter("bias", None)

        c, w, h = in_shape
        if n_pos_channels and stack_pos_encoding:
            c = c + n_pos_channels
        self.slot_size = slot_size if slot_size else c

        if self.use_pos_enc:
            self.position_embedding = PositionalEncoding2D(
                d_model= n_pos_channels if n_pos_channels else c,
                width=w,
                height=h,
                learned=learned_pos,
                dropout=dropout_pos,
                stack_channels=stack_pos_encoding,
            )

        if self.use_post_pos_enc:
            self.post_position_embedding = PositionalEncoding2D(
                d_model=self.slot_size if n_post_pos_channels is None else n_post_pos_channels,
                width=broadcast_size,
                height=broadcast_size,
                learned=learned_post_pos,
                dropout=dropout_post_pos,
                stack_channels=stack_post_pos_encoding,
            )

        out_channels = self.slot_size if stack_post_pos_encoding is not True else self.slot_size + n_post_pos_channels

        if self.self_attention:
            self.k_proj = nn.Linear(out_channels, out_channels if embed_out is None else embed_out, bias=False)
            self.v_proj = nn.Linear(out_channels, out_channels if embed_out is None else embed_out, bias=False)
            if embed_out is not None:
                out_channels = embed_out

        self._features = Parameter(torch.Tensor(1, out_channels, self.n_neurons))
        self._features.data.fill_(1 / self.in_shape[0])

        self.neuron_query = Parameter(torch.Tensor(1, out_channels, self.outdims))
        self.neuron_query.data.fill_(1 / self.in_shape[0])

        if temperature[0]:
            self.T = temperature[1]
        else:
            self.T = Parameter(torch.ones(self.n_neurons) * temperature[1])

        self.slot_size = slot_size if slot_size is not None else c
        self.num_slots = num_slots

        self.use_post_embed_mlp = use_post_embed_mlp
        self.slot_attention = SlotAttention(
            num_iterations=slot_num_iterations,
            num_slots=num_slots,
            slot_size=self.slot_size,
            input_size=c
            if slot_input_size is None
            else slot_input_size,  # number of core output channels
            mlp_hidden_size=slot_mlp_hidden_size_factor * c,
            epsilon=slot_epsilon,
            draw_slots=draw_slots,
            use_slot_gru=use_slot_gru,
            use_weighted_mean=use_weighted_mean,
            full_skip=full_skip,
            slot_temperature=slot_temperature,
            position_invariant=True,
        )

        if self.use_post_embed_mlp:
            self.post_embedding_mlp = nn.Sequential(
                nn.Linear(c, c),
                nn.ReLU(True),
                nn.Linear(c, c if slot_input_size is None else slot_input_size)
            )

        self.neuron_slot_weight = torch.nn.Parameter(torch.randn(1, 1, self.num_slots, self.n_neurons))
        self.neuron_slot_weight_temp = neuron_slot_weight_temp
        self.broadcast_size = broadcast_size

    def broadcast(self, z):
        """Broadcasts the slot embeddings to the size of the input image.
        Args:
            z: [Images, Channels, Neurons]
        Returns:
            z: [Images, Channels, Width, Height, Neurons]

            """
        z = z[:, :, None, None, :].repeat((1, 1, self.broadcast_size, self.broadcast_size, 1))
        return z

    def forward(self, x, output_attn_weights=None, **kwargs):
        i, c, w, h = x.size()
        x = x.flatten(2, 3)  # [Images, Channels, w*h]
        if self.use_pos_enc:
            x_embed = self.position_embedding(x)  # -> [Images, Channels, w*h]
        else:
            x_embed = x

        # Start Slot Attention
        x_embed = x_embed.permute(0, 2, 1)  # batch_sizes, w*h, c
        if self.use_post_embed_mlp:
            x_embed = self.post_embedding_mlp(x_embed)

        slots, slot_attention_maps = self.slot_attention(x_embed, width=w, height=h)  # batch_sizes, slots, channels
        slots = slots.permute(0, 2, 1)  # batch_size, slot_size, n_slots

        slot_weights = torch.nn.functional.softmax(self.neuron_slot_weight / self.neuron_slot_weight_temp, dim=2)
        weighted_slots = slots.unsqueeze(-1) * slot_weights # [batch_size, slot_size, num_slots, 1] * [1, 1, num_slots, n_neurons] -> [bs, slot_size, num_slots, n_neurons]
        weighted_slots = weighted_slots.sum(2) # [bs, slot_size, n_neurons]

        broadcasted_slots = self.broadcast(weighted_slots)  # [Images, Channels, Width, Height, Neurons]
        broadcasted_slots = rearrange(broadcasted_slots, "b c w h n -> b c (w h) n")  # [Images, Channels, Width*Height, Neurons]
        if self.use_post_pos_enc:
            broadcasted_slots = self.post_position_embedding(broadcasted_slots)  # [Images, Channels, Width*Height, Neurons]

        keys, values = self.k_proj(broadcasted_slots.permute(0, 2, 3, 1)), self.v_proj(broadcasted_slots.permute(0, 2, 3, 1))  # Shape: (batch_size x w*h x slot_size)
        keys = keys.permute(0, 3, 1, 2)  # Shape: (batch_size x slot_size x w*h x neurons
        values = values.permute(0, 3, 1, 2)  # Shape: (batch_size x slot_size x w*h x neurons
        if self.self_attention:
            weights = torch.einsum("bdsn,adn->bsn", keys, self.neuron_query)  # [Images, Channels, Width, Height]
        else:
            weights = torch.einsum("bdsn,adn->bsn", broadcasted_slots, self.neuron_query)
        weights = weights / self.T
        weights = torch.nn.functional.softmax(weights, dim=2)

        if self.self_attention:
            y = torch.einsum("bcsn,bsn->bcsn", values, weights)
        else:
            y = torch.einsum("bcsn,bsn->bcsn", broadcasted_slots, weights)
        y = torch.einsum("bcsn,acn->bn", y, self._features)

        if self.bias is not None:
            y = y + self.bias

        # return neuronal responses, and attention maps if requested
        if output_attn_weights:
            return y, slot_attention_maps, weights
        return y

    def feature_l1(self, average=True):
        """
        Returns the l1 regularization term either the mean or the sum of all weights
        Args:
            average(bool): if True, use mean of weights for regularization
        """
        if average:
            return self._features.abs().mean()
        else:
            return self._features.abs().sum()

    def query_l1(self, average=True):
        """
        Returns the l1 regularization term either the mean or the sum of all weights
        Args:
            average(bool): if True, use mean of weights for regularization
        """
        if average:
            return self.neuron_query.abs().mean()
        else:
            return self.neuron_query.abs().sum()

    def regularizer(self, data_key=None):
        return (
                self.feature_l1(average=False) * self.gamma_features
                + self.query_l1(average=False) * self.gamma_query
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
                ),
            )
        self.gamma_readout = gamma_readout
        self.gamma_grid_dispersion = gamma_grid_dispersion

    def regularizer(self, data_key):
        if hasattr(FullGaussian2d, "mu_dispersion"):
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
