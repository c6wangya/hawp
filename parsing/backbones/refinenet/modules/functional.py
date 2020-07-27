import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_fwd, custom_bwd
from .DCN.DCNv2 import modulated_deformable_conv2d_forward, modulated_deformable_conv2d_backward


class ModulatedDeformableConvolutionFunction(Function):

    @staticmethod
    def symbolic(g, input, offset, mask, weight, bias, stride, padding, dilation, im2col_step):
        args = [input, offset, mask, weight]
        if bias is not None:
            args.append(bias)
        kwargs = {"stride_i": list(stride), "padding_i": list(padding),
                  "dilation_i": list(dilation), "im2col_step_i": im2col_step}
        return g.op("ModulatedDeformableConv2d", *args, **kwargs)

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, offset, mask, weight, bias, stride, padding, dilation, im2col_step):
        if not input.is_cuda:
            raise NotImplementedError("ModulatedDeformableConvolution is only implemented on cuda now")

        stride_h, stride_w = stride
        padding_h, padding_w = padding
        dilation_h, dilation_w = dilation
        kernel_h, kernel_w = weight.shape[-2:]
        _, in_channels, in_h, in_w = input.shape
        out_channels = weight.shape[0]
        deformable_groups = offset.shape[1] // (2 * kernel_h * kernel_w)
        groups = in_channels // weight.shape[1]
        if deformable_groups == 0:
            raise RuntimeError(
                "the shape of the offset tensor at dimension 1 is not valid. It should "
                "be a multiple of 2 * weight.size[2] * weight.size[3].\n"
                "Got offset.shape[1]={}, while 2 * weight.size[2] * weight.size[3]={}".format(
                    offset.shape[1], 2 * kernel_h * kernel_w))

        if bias is None:
            bias = torch.zeros(out_channels, device=input.device, dtype=input.dtype)

        if input.requires_grad or offset.requires_grad or mask.requires_grad or bias.requires_grad:
            ctx.save_for_backward(input, offset, mask, weight, bias)
        ctx.stride_h = stride_h
        ctx.stride_w = stride_w
        ctx.padding_h = padding_h
        ctx.padding_w = padding_w
        ctx.dilation_h = dilation_h
        ctx.dilation_w = dilation_w
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step

        return modulated_deformable_conv2d_forward(input, weight, offset, mask, bias,
                                           stride_h, stride_w, padding_h, padding_w,
                                           dilation_h, dilation_w, groups,
                                           deformable_groups, im2col_step)

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input, grad_weight, grad_offset, grad_mask, grad_bias = \
            modulated_deformable_conv2d_backward(grad_output,
                                                 input,
                                                 weight,
                                                 offset,
                                                 mask,
                                                 bias,
                                                 ctx.stride_h, ctx.stride_w,
                                                 ctx.padding_h, ctx.padding_w,
                                                 ctx.dilation_h, ctx.dilation_w,
                                                 ctx.groups,
                                                 ctx.deformable_groups,
                                                 ctx.im2col_step)
        if not isinstance(bias, torch.nn.Parameter):
            grad_bias = None
        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias, \
            None, None, None, None, None, None, None, None, None


modulated_deformable_convolution = ModulatedDeformableConvolutionFunction.apply
