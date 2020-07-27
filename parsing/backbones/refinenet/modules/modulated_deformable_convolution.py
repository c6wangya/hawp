import math
import torch
from torch import nn
from torch.nn.modules.utils import _pair
from .functional import modulated_deformable_convolution

class ModulatedDeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 deformable_groups: int = 1,
                 im2col_step: int = 64,
                 bias=True,
                 mask_scale: float = 2.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.mask_scale = mask_scale
        self.im2col_step = im2col_step

        self.weight =  nn.Parameter(
                            torch.empty(out_channels, in_channels // groups,
                                        self.kernel_size[0], self.kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.conv_offset_mask = nn.Conv2d(self.in_channels,
                                          self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=False)
        self.init_offset
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, x):
        out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask) * self.mask_scale
        return modulated_deformable_convolution(x, offset, mask, self.weight, self.bias,
                                                self.stride, self.padding, self.dilation,
                                                self.im2col_step)
