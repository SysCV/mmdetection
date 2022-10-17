# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES
from .resnet import ResNet


@BACKBONES.register_module()
class ResNetFlow(ResNet):
    """ResNetFlow backbone."""

    def __init__(
        self,
        depth,
        extra_freeze=None,
        **kwargs
    ):
        self.extra_freeze = extra_freeze
        super(ResNetFlow, self).__init__(depth, **kwargs)

    def _freeze_stages(self):
        super()._freeze_stages()
        if self.extra_freeze is not None:
            for i in self.extra_freeze:
                m = getattr(self, f'layer{i}')
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            if i not in self.out_indices:
                break
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            outs.append(x)
        return tuple(outs)
