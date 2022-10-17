# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import BACKBONES
from .swin import SwinTransformer


@BACKBONES.register_module()
class SwinTransformerFlow(SwinTransformer):
    """SwinTransformerFlow backbone."""

    def __init__(self, extra_freeze=None, **kwargs):
        self.extra_freeze = extra_freeze
        super(SwinTransformerFlow, self).__init__(**kwargs)

    def _freeze_stages(self):
        super()._freeze_stages()
        if self.extra_freeze is not None:
            for i in self.extra_freeze:
                self.stages[i].eval()
                for param in self.stages[i].parameters():
                    param.requires_grad = False
                # for m in [self.stages[i], getattr(self, f'norm{i}')]:
                #     m.eval()
                #     for param in m.parameters():
                #         param.requires_grad = False

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            if i not in self.out_indices:
                break
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)

        return outs
