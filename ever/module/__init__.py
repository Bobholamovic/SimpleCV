from ever import registry
import torch.nn as nn

registry.OP.register('batchnorm', nn.BatchNorm2d)
registry.OP.register('groupnorm', nn.GroupNorm)
# basic component
from ever.module.aspp import AtrousSpatialPyramidPool
from ever.module.aspp import AtrousSpatialPyramidPoolv2
from ever.module.context_block import ContextBlock2d
from ever.module.sep_conv import SeparableConv2D
from ever.module.gap import GlobalAvgPool2D
from ever.module.se_block import SEBlock

# encoder
from ever.module.resnet import ResNetEncoder
from ever.module.fpn import FPN
from ever.module.fpn import LastLevelMaxPool
from ever.module.fpn import LastLevelP6P7

# loss
from ever.module import loss
