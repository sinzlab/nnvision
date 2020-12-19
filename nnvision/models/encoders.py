import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):

    def __init__(self, core, readout, elu_offset):
        super().__init__()
        self.core = core
        self.readout = readout
        self.offset = elu_offset

    def forward(self, x, data_key=None, **kwargs):
        x = self.core(x)

        x = self.readout(x, data_key=data_key)
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

    def __init__(self, core, readout, shifter, elu_offset):
        super().__init__()
        self.core = core
        self.readout = readout
        self.offset = elu_offset
        self.shifter = shifter


    def forward(self, *args, data_key=None, eye_position=None, **kwargs):

        x = self.core(args[0])

        if len(args) == 3:
            if args[2].shape[1] == 2:
                eye_position = args[2]

        if len(args) == 4:
            if args[3].shape[1] == 2:
                eye_position = args[3]


        sample = kwargs["sample"] if 'sample' in kwargs else None
        x = self.readout(x, data_key=data_key, sample=sample)
        return F.elu(x + self.offset) + 1

    def regularizer(self, data_key):
        return self.core.regularizer() + self.readout.regularizer(data_key=data_key)