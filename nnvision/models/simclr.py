import torch
from torch import nn
from cnexp.models.simclr_like import make_model, make_projection_head
from cnexp.models.mutate_model import mutate_model

from .convnext import ConvNeXt
from ..tables.simclr import Model, TrainedModel


def tsimnce_resnets(
    dataloaders,
    seed=42,
    backbone="resnet18",
    in_channel: int = 1,
    out_dim=128,
) -> torch.nn.Module:
    """

    Args:
        dataloaders: Placeholder to fit with nnfabrik's trainer interface
        seed: random init seed
        backbone: can be resnet18, resnet34, resnet50, resnet101
        in_channel: n image channels
        out_dim: projection head dimensions, default of 128 in simclr

    Returns: torch.nn.module

    """
    model = make_model(
        seed=seed, backbone=backbone, in_channel=in_channel, out_dim=out_dim
    )
    return model


def get_mutated_model(
    key,
):
    model = (Model & key).build_model(dataloaders=0, seed=0)
    state_dict = torch.load((TrainedModel.ModelStorage & key).fetch1("model_state"))
    model = mutate_model(model, change="lastlin", freeze=True, out_dim=2)
    model.load_state_dict(state_dict)
    return model


class ConvNextFC(nn.Module):
    def __init__(
        self,
        dataloaders=None,
        seed=42,
        in_channels=1,
        channel_list=(64, 128, 256, 512),
        num_blocks_list=(2, 2, 2, 2),
        kernel_size=7,
        patch_size=1,
        res_p_drop=0.0,
        proj_head="mlp",
        out_dim=128,
        hidden_dim=1024,
    ):
        super().__init__()
        backbone = ConvNeXt(
            classes=1,
            in_channels=in_channels,
            channel_list=channel_list,
            num_blocks_list=num_blocks_list,
            kernel_size=kernel_size,
            patch_size=patch_size,
            res_p_drop=res_p_drop,
        )

        test_input = torch.ones(1, in_channels, 32, 32)
        with torch.no_grad():
            backbone_dim = backbone(test_input).shape[1]

        self.backbone_dim = backbone_dim
        self.out_dim = out_dim
        self.backbone = backbone

        self.projection_head = make_projection_head(
            proj_head,
            in_dim=backbone_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.projection_head(h)
        return z, h
