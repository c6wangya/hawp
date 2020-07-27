#include <torch/torch.h>
#include <torch/extension.h>

at::Tensor DeformConv2d_forward_cuda(
    const at::Tensor& input_param,
    const at::Tensor& weight_param,
    const at::Tensor& offset_param,
    const at::Tensor& mask_param,
    const at::Tensor& bias,
    std::pair<int, int> stride,
    std::pair<int, int> pad,
    std::pair<int, int> dilation,
    int n_weight_grps,
    int n_offset_grps,
    int im2col_step);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
DeformConv2d_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& bias,
    std::pair<int, int> stride,
    std::pair<int, int> pad,
    std::pair<int, int> dilation,
    int n_weight_grps,
    int n_offset_grps,
    int im2col_step);

at::Tensor DeformConv2d_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& bias,
    const int stride_h, const int stride_w,
    const int padding_h, const int padding_w,
    const int dilation_h, const int dilation_w,
    const int groups,
    const int offset_groups,
    const int im2col_step) {
  if (!input.is_cuda()) {
    AT_ERROR("CPU is not supported");
  }
  return DeformConv2d_forward_cuda(
      input.contiguous(),
      weight.contiguous(),
      offset.contiguous(),
      mask.contiguous(),
      bias.contiguous(),
      {stride_h, stride_w},
      {padding_h, padding_w},
      {dilation_h, dilation_w},
      groups,
      offset_groups,
      im2col_step);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> DeformConv2d_backward(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& bias,
    const int stride_h, const int stride_w,
    const int padding_h, const int padding_w,
    const int dilation_h, const int dilation_w,
    const int groups,
    const int offset_groups,
    const int im2col_step) {
  if (!grad.is_cuda()) {
    AT_ERROR("CPU is not supported");
  }
  return DeformConv2d_backward_cuda(
      grad.contiguous(),
      input.contiguous(),
      weight.contiguous(),
      offset.contiguous(),
      mask.contiguous(),
      bias.contiguous(),
      {stride_h, stride_w},
      {padding_h, padding_w},
      {dilation_h, dilation_w},
      groups,
      offset_groups,
      im2col_step);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("modulated_deformable_conv2d_forward", &DeformConv2d_forward, "modulated deformable conv2d forward");
  m.def("modulated_deformable_conv2d_backward", &DeformConv2d_backward, "modulated deformable conv2d backward");
}
