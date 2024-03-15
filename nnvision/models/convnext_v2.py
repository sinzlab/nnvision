try:
    from transformers import AutoImageProcessor, ConvNextV2ForImageClassification
except:
    pass
from torch import nn
from .utility import clip_convnext_layers


class ConvNextV2(nn.Module):
    def __init__(
        self,
        model_name,
        cutoff_layer=None,
        patch_embedding_stride=None,
        cut_classification_head=True,
        model_kwargs=None,
    ):
        super().__init__()
        self.cutoff_layer = cutoff_layer
        self.model = ConvNextV2ForImageClassification.from_pretrained(model_name)
        self.patch_embedding_stride = patch_embedding_stride

        if self.patch_embedding_stride is not None:
            self.replace_patch_embedding_stride()
        if self.cutoff_layer is not None:
            self.model = clip_convnext_layers(self.model, self.cutoff_layer)
        if cut_classification_head:
            self.model = self.model.convnextv2

    def replace_patch_embedding_stride(self):
        original_patch_embedding = self.model.convnextv2.embeddings.patch_embeddings
        conv_params = original_patch_embedding.weight.data
        bias_params = original_patch_embedding.bias.data

        new_patch_embedding = nn.Conv2d(
            in_channels=original_patch_embedding.in_channels,
            out_channels=original_patch_embedding.out_channels,
            kernel_size=original_patch_embedding.kernel_size,
            stride=self.patch_embedding_stride,
            dilation=original_patch_embedding.dilation,
            groups=original_patch_embedding.groups,
            padding=original_patch_embedding.padding,
            padding_mode=original_patch_embedding.padding_mode,
            bias=True,
        )
        new_patch_embedding.weight.data = conv_params
        new_patch_embedding.bias.data = bias_params
        self.model.convnextv2.embeddings.patch_embeddings = new_patch_embedding

    def forward(self, input_):
        out = self.model(input_)[0]
        return out
