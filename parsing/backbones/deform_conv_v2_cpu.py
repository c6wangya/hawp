import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, stride=1,  padding=1, bias=False, modulation=True, attn=False, attn_only=False, attn_dim='', n_head=2):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.attn = attn
        self.attn_only = attn_only and attn
        self.modulation = modulation
        self.attn_dim = attn_dim
        self.n_head = n_head
        if not self.attn_only:
            self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        else:
            self.conv = nn.Conv2d(inc, outc, kernel_size=1, stride=stride, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        # local attn
        if self.attn:
            self.inc = inc
            self.outc = outc
            self.query_conv = nn.Conv2d(inc, inc, kernel_size=1, stride=stride, bias=bias)
            self.key_conv = nn.Conv2d(inc, inc, kernel_size=1, stride=stride, bias=bias)
        if self.attn_only:
            self.value_conv = nn.Conv2d(inc, inc, kernel_size=1, stride=stride, bias=bias)
            w_head = torch.empty((self.inc // self.n_head, self.n_head, self.inc),requires_grad=True)
            self.w_head = torch.nn.Parameter(w_head)
            torch.nn.init.normal_(w_head, std=0.1/np.sqrt(self.inc))
            self.register_parameter("head_weight",self.w_head)
        else:
            self.value_conv = nn.Identity()

        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def compute_attn(self, x, x_offset, offset, dim=''):
        """ compute local attention map indexed by offset """
        B, C, H, W = x.size()
        q_out = self.query_conv(x)  # [B, C, H, W]
        k_out = self.key_conv(x)  # [B, C, H, W]
        k_out = self.compute_x_off(k_out, offset)  # [B, C, H, W, k**2]
        if dim == '':
            # query
            q_out = q_out.view(B, self.inc, H, W, 1)
            # attention map
            attn_map = q_out * k_out
            attn_map = F.softmax(attn_map, dim=-1)  # [B, C, H, W, k**2]
            x_offset = x_offset * attn_map
        elif dim == 'c' and self.n_head == 1:
            q_out = q_out.permute(0, 2, 3, 1).unsqueeze(3)  # [B, H, W, 1, C]
            k_out = k_out.permute(0, 2, 3, 1, 4)  # [B, H, W, C, k**2]
            attn_map = torch.einsum('bhwmc,bhwck->bhwmk', q_out, k_out)  # [B, H, W, 1, k**2]
            attn_map = F.softmax(attn_map, dim=-1).permute(0, 3, 1, 2, 4)  # [B, 1, H, W, k**2]
            x_offset = x_offset * attn_map
        elif dim == 'c':
            qs = [q_out.permute(0, 2, 3, 1)[:, :, :, i*C//self.n_head:(i+1)*C//self.n_head].unsqueeze(3) for i in range(self.n_head)]
            ks = [k_out.permute(0, 2, 3, 1, 4)[:, :, :, i*C//self.n_head:(i+1)*C//self.n_head, :] for i in range(self.n_head)]
            attn_maps = [torch.einsum('bhwmc,bhwck->bhwmk', qs[i], ks[i]) for i in range(self.n_head)]
            attn_maps = [F.softmax(attn_maps[i], dim=-1).permute(0, 3, 1, 2, 4) for i in range(self.n_head)]
            x_offset = [x_offset[:, i*C//self.n_head:(i+1)*C//self.n_head, :, :, :] * attn_maps[i] for i in range(self.n_head)]
            x_offset = torch.stack(x_offset, dim=-1)
            x_offset = torch.einsum('behwkn,enm->bhwkm', x_offset, self.w_head).permute(0, 4, 1, 2, 3)  # [B, C, H, W, k**2]
        return x_offset

    def compute_x_off(self, x, offset):
        """ compute feature indexed by offset """
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt
        
        return x_offset

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))
        # compute feature map based on offset
        value = self.value_conv(x)
        x_offset = self.compute_x_off(value, offset)
        # compute attension map
        if self.attn:
            x_offset = self.compute_attn(x, x_offset, offset, dim=self.attn_dim)

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m
        
        if self.attn_only:
            x_offset = torch.sum(x_offset, dim=-1).squeeze(-1)
        else:
            x_offset = self._reshape_x_offset(x_offset, self.kernel_size)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset
