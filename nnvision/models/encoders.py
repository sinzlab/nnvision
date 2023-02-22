import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


class Encoder(nn.Module):
    def __init__(self, core, readout, elu_offset):
        super().__init__()
        self.core = core
        self.readout = readout
        self.offset = elu_offset

    def forward(self, x, data_key=None, repeat_channel_dim=None, **kwargs):
        if repeat_channel_dim is not None:
            x = x.repeat(1, repeat_channel_dim, 1, 1)
            x[:, 1:, ...] = 0

        x = self.core(x)
        x = self.readout(x, data_key=data_key, **kwargs)

        output_attn_weights = kwargs.get("output_attn_weights", False)
        if output_attn_weights:
            if len(x) == 2:
                x, attention_weights = x
            else:
                x, weights_1, weights_2 = x
                attention_weights = (weights_1, weights_2)
            return F.elu(x + self.offset) + 1, attention_weights
        return F.elu(x + self.offset) + 1

    def regularizer(self, data_key):
        return self.core.regularizer() + self.readout.regularizer(data_key=data_key)


class EncoderPNL(nn.Module):
    def __init__(self, core, readout, nonlinearity):
        super().__init__()
        self.core = core
        self.readout = readout
        self.nonlinearity = nonlinearity

    def forward(self, x, data_key=None, **kwargs):
        x = self.core(x)
        x = self.readout(x, data_key=data_key)
        x = self.nonlinearity(x, data_key=data_key)
        return x

    def regularizer(self, data_key):
        return self.core.regularizer() + self.readout.regularizer(data_key=data_key)


class EncoderShifter(nn.Module):
    def __init__(
        self,
        core,
        readout,
        shifter,
        elu_offset,
        stack_prev_image=False,
    ):
        super().__init__()
        self.core = core
        self.readout = readout
        self.offset = elu_offset
        self.shifter = shifter

    def forward(self, *args, data_key=None, eye_position=None, shift=None, **kwargs):
        x = self.core(args[0])
        if eye_position is not None and self.shifter is not None:
            eye_position = eye_position.to(x.device).to(dtype=x.dtype)
            shift = self.shifter[data_key](eye_position)

        sample = kwargs["sample"] if "sample" in kwargs else None
        x = self.readout(x, data_key=data_key, sample=sample, shift=shift, **kwargs)
        return F.elu(x + self.offset) + 1

    def regularizer(self, data_key):
        return self.core.regularizer() + self.readout.regularizer(data_key=data_key)
