import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class DeformConv2d(nn.Module):
    def __init__(
            self, 
            inc, 
            outc, 
            kernel_size=3, 
            stride=1,  
            padding=1, 
            bias=False, 
            modulation=True, 
            attn=False, 
            attn_only=False, 
            attn_dim='', 
            n_head=2, 
            ctl_ks=9, 
            n_sample=9, 
            use_contrastive=False,
            share_weights=True, 
            attn_bottleneck=False
        ):
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
        self.ctl_ks = ctl_ks
        self.n_sample = n_sample
        self.use_contrastive = use_contrastive
        self.share_weights = share_weights
        self.attn_bottleneck = attn_bottleneck
        if not self.attn_only and not self.attn_bottleneck:
            self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        elif not self.attn_bottleneck:
            self.conv = nn.Conv2d(inc, outc, kernel_size=1, stride=stride, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        # local attn, keys and querys
        self.inc = inc
        self.outc = outc
        attn_outc = outc if self.attn_bottleneck else inc

        if self.attn and self.n_head > 1 and not self.share_weights:
            self.query_conv = nn.ModuleList([nn.Conv2d(inc, attn_outc, kernel_size=1, stride=stride, bias=bias) for i in range(self.n_head)])
            self.key_conv = nn.ModuleList([nn.Conv2d(inc, attn_outc, kernel_size=1, stride=stride, bias=bias) for i in range(self.n_head)])
        elif self.attn:
            self.query_conv = nn.Conv2d(inc, attn_outc, kernel_size=1, stride=stride, bias=bias)
            self.key_conv = nn.Conv2d(inc, attn_outc, kernel_size=1, stride=stride, bias=bias)
        
        # local attn, values
        # if self.attn_only and self.n_head > 1 and not self.share_weights:
        #     self.value_conv = nn.ModuleList([nn.Conv2d(inc, attn_outc, kernel_size=1, stride=stride, bias=bias) for i in range(self.n_head)])
        # elif self.attn_only:
        if self.attn_only:
            self.value_conv = nn.Conv2d(inc, attn_outc, kernel_size=1, stride=stride, bias=bias)
        else:
            self.value_conv = nn.Identity()
        
        # local attn, multihead merge operations
        if self.attn_only and self.n_head > 1 and self.share_weights:
            w_head = torch.empty((attn_outc // self.n_head, self.n_head, attn_outc),requires_grad=True)
            self.w_head = torch.nn.Parameter(w_head)
            torch.nn.init.normal_(w_head, std=0.1/np.sqrt(attn_outc))
            self.register_parameter("head_weight",self.w_head)

        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def compute_attn(self, x, x_offset, offset, dim='', share_weights=True):
        """ compute local attention map indexed by offset """
        B, C, H, W = x.size()
        if self.share_weights or dim == '' or self.n_head == 1:
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
        elif self.n_head == 1:
            q_out = q_out.permute(0, 2, 3, 1).unsqueeze(3)  # [B, H, W, 1, C]
            k_out = k_out.permute(0, 2, 3, 1, 4)  # [B, H, W, C, k**2]
            attn_map = torch.einsum('bhwmc,bhwck->bhwmk', q_out, k_out)  # [B, H, W, 1, k**2]
            attn_map = F.softmax(attn_map, dim=-1).permute(0, 3, 1, 2, 4)  # [B, 1, H, W, k**2]
            x_offset = x_offset * attn_map
        elif self.share_weights:
            qs = [q_out.permute(0, 2, 3, 1)[:, :, :, i*C//self.n_head:(i+1)*C//self.n_head].unsqueeze(3) for i in range(self.n_head)]
            ks = [k_out.permute(0, 2, 3, 1, 4)[:, :, :, i*C//self.n_head:(i+1)*C//self.n_head, :] for i in range(self.n_head)]
            attn_maps = [torch.einsum('bhwmc,bhwck->bhwmk', qs[i], ks[i]) for i in range(self.n_head)]
            attn_maps = [F.softmax(attn_maps[i], dim=-1).permute(0, 3, 1, 2, 4) for i in range(self.n_head)]
            x_offset = [x_offset[:, i*C//self.n_head:(i+1)*C//self.n_head, :, :, :] * attn_maps[i] for i in range(self.n_head)]
            x_offset = torch.stack(x_offset, dim=-1)
            x_offset = torch.einsum('behwkn,enm->bhwkm', x_offset, self.w_head).permute(0, 4, 1, 2, 3)  # [B, C, H, W, k**2]
        else:
            x_out = []
            for i in range(self.n_head):
                q = self.query_conv[i](x).permute(0, 2, 3, 1).unsqueeze(3)  # [B, H, W, 1, C]
                k = self.key_conv[i](x)
                k = self.compute_x_off(k, offset).permute(0, 2, 3, 1, 4)  # [B, H, W, C, k**2]
                attn_map = torch.einsum('bhwmc,bhwck->bhwmk', q, k)
                attn_map = F.softmax(attn_map, dim=-1).permute(0, 3, 1, 2, 4)
                x_out.append(x_offset * attn_map)
                # x_out.append(x_offset[i] * attn_map)
            x_offset = torch.sum(torch.stack(x_out, dim=-1), dim=-1)
        return x_offset

    def _quantize_offset(self, x, offset):
        dtype = offset.data.type()
        # ks = self.kernel_size
        N = offset.size(1) // 2

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype, padding=False)
        
        # (b, h, w, 2N)
        N = offset.size(1) // 2
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p 
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        q_x = torch.cat([q_lt[..., :N], q_rb[..., :N], q_lb[..., :N], q_rt[..., :N]], dim=-1)
        q_y = torch.cat([q_lt[..., N:], q_rb[..., N:], q_lb[..., N:], q_rt[..., N:]], dim=-1)

        return torch.cat([q_x, q_y], dim=-1)

    def compute_contrastive_offset(self, x, offset):
        ctl_ks = min(self.ctl_ks, x.size(3)-1)
        N = ctl_ks**2
        # assert not (offset < -1).any() and not (offset > 1).any() 
        qp = self._quantize_offset(x, offset)  # [B, H, W, 72]
        assert not (qp < 0).any() and not (qp > x.size(3)-1).any() and x.size(3) > 4

        p_n = torch.zeros(offset.size(0), N*2, offset.size(2), offset.size(3))
        p_n = self._get_p(p_n, torch.int64, padding=False).permute(0, 2, 3, 1)

        p_y_addition = torch.zeros([x.size(2), x.size(2)]).type(torch.int64)
        p_y_addition[:, :ctl_ks // 2] = \
            torch.tensor(list(range(ctl_ks//2))[::-1]).repeat(x.size(2), 1) + 1
        p_y_addition[:, x.size(2) - ctl_ks // 2:] = \
            torch.tensor(list(range(ctl_ks//2))).repeat(x.size(2), 1)*-1 - 1
        p_x_addition = torch.zeros([x.size(2), x.size(2)]).type(torch.int64)
        p_x_addition[:ctl_ks//2, :] = \
            torch.tensor(list(range(ctl_ks//2))[::-1]).repeat(x.size(2), 1).permute(1, 0) + 1
        p_x_addition[x.size(2) - ctl_ks // 2:, :] = \
            torch.tensor(list(range(ctl_ks//2))).repeat(x.size(2), 1).permute(1, 0)*-1 - 1
        p_n[:, :, :, :N] += p_x_addition.view(1, x.size(2), x.size(2), 1)
        p_n[:, :, :, N:] += p_y_addition.view(1, x.size(2), x.size(2), 1)

        qp_x = qp[:, :, :, :qp.size(-1)//2].permute(3, 0, 1, 2)
        qp_y = qp[:, :, :, qp.size(-1)//2:].permute(3, 0, 1, 2)
        p_n_x = p_n[:, :, :, :N].permute(3, 0, 1, 2)[:, None].to(qp.device)
        p_n_y = p_n[:, :, :, N:].permute(3, 0, 1, 2)[:, None].to(qp.device)

        x_match = (qp_x == p_n_x).int()
        y_match = (qp_y == p_n_y).int()

        iskeep = (x_match * y_match).bool()
        iskeep = ~iskeep.any(dim=1).permute(1, 2, 3, 0)
        # iskeep_bitmap = iskeep.int()
        idx = torch.where(iskeep.to('cpu'))

        def sample_k_pts(k, idx, b, h, w):
            pool = np.intersect1d(torch.where(idx[0] == b)[0], torch.where(idx[1] == h)[0])
            pool = np.intersect1d(torch.where(idx[2] == w)[0], pool)
            # randomly sample k idices
            selected = np.random.choice(pool, size=k)
            return torch.stack(idx)[:, selected]
        # selected = torch.zeros(tuple(iskeep.shape)[:-1] + (9, ))
        neg_off = torch.zeros(tuple(iskeep.shape)[:-1] + (2*self.n_sample, ))
        
        # # accelerated version
        # B, H, W, _ = iskeep.shape
        # neg_off = p_n.clone()
        # neg_off[..., :qp.size(-1)//2][~iskeep] = -1
        # neg_off[..., qp.size(-1)//2:][~iskeep] = -1
        
        # rand_idices = [[[list(torch.randperm(4) + i * 12 + j * 4) for w in range(W)] for h in range(H)] for b in range(B)]

        for b in range(iskeep.size(0)):
            for h in range(iskeep.size(1)):
                for w in range(iskeep.size(1)):
                    selected_idices = sample_k_pts(self.n_sample, idx, b, h, w)[-1, :].to('cuda')
                    neg_off[b, h, w, :self.n_sample] = p_n[([b]*self.n_sample, [h]*self.n_sample, [w]*self.n_sample, selected_idices)]
                    neg_off[b, h, w, self.n_sample:] = p_n[([b]*self.n_sample, [h]*self.n_sample, [w]*self.n_sample, selected_idices+N)]
        x_neg_off = self._get_x_q(x, neg_off.type(torch.int64), self.n_sample)
        # x_neg_off = self._get_x_q(x, neg_off.type(torch.int64).to(x.device), self.n_sample)
        return x_neg_off

    def compute_contrastive_loss(self, x, x_pos_offset, x_neg_offset, tao=0.1):
        """ x: [B, C, H, W]
            x_pos_offset: [B, C, H, W, 9]
            x_neg_offset: [B, C, H, W, 9]
        """
        epsilon = 1e-10
        pos_dot_prod = torch.einsum('bchw,bchwn->bhwn', x.detach(), x_pos_offset)  # [B, H, W, k**2]
        neg_dot_prod = torch.einsum('bchw,bchwn->bhwn', x.detach(), x_neg_offset)  # [B, H, W, k**2]
        x_norm = torch.norm(x.detach(), dim=1).unsqueeze(dim=-1)  # [B, H, W, 1]
        assert not torch.isnan(x_pos_offset).any() and not torch.isnan(x_pos_offset).any()
        # print("pos offset: {}".format(x_pos_offset[0, 0, 0, :]))
        # print("neg offset: {}".format(x_neg_offset[0, 0, 0, :]))
        pos_norm = torch.norm(x_pos_offset.detach(), dim=1)  # [B, H, W, k**2]
        neg_norm = torch.norm(x_neg_offset.detach(), dim=1)  # [B, H, W, k**2]
        pos_sim = pos_dot_prod / (x_norm + epsilon) / (pos_norm + epsilon) / tao  # [B, H, W, k**2]
        neg_sim = neg_dot_prod / (x_norm + epsilon) / (neg_norm + epsilon) / tao  # [B, H, W, k**2]
        # pos_sim = pos_dot_prod / (x_norm + (x_norm.detach() == 0).int()) / (pos_norm + (pos_norm.detach() == 0).int()) / tao  # [B, H, W, k**2]
        # neg_sim = neg_dot_prod / (x_norm + (x_norm.detach() == 0).int()) / (neg_norm + (neg_norm.detach() == 0).int()) / tao  # [B, H, W, k**2]
        pos_sim = torch.exp(pos_sim)  # [B, H, W, k**2]
        neg_sim = torch.exp(neg_sim)  # [B, H, W, k**2]
        total = pos_sim.sum(dim=-1, keepdim=True) + neg_sim.sum(dim=-1, keepdim=True) - pos_sim  # [B, H, W, k**2]
        l = torch.mean(-torch.log(pos_sim / total), dim=-1)  # [B, H, W, k**2]
        assert not torch.isnan(l).any()
        return l

    def compute_x_off(self, x, offset):
        """ compute feature indexed by offset """
        dtype = offset.data.type()
        # ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype, padding=True).to('cuda')

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
        # assert not (offset < -1).any() and not (offset > 1).any() 
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        # compute feature map based on offset
        value = self.value_conv(x)
        assert not torch.isnan(x).any()
        x_offset = self.compute_x_off(value, offset)
        # if self.share_weights:
        #     value = self.value_conv(x)
        #     x_offset = self.compute_x_off(value, offset)
        # else:
        #     # idt = nn.Identity()
        #     # value = idt(x)
        #     x_offset = [self.compute_x_off(self.value_conv[i](x), offset) for i in range(self.n_head)]

        # compute attension map
        if self.attn:
            x_offset = self.compute_attn(x, x_offset, offset, dim=self.attn_dim)

        # compute contrastive loss
        if self.use_contrastive and x.size(3) >= 8:
            x_neg_offset = self.compute_contrastive_offset(x, offset)
            # idt = nn.Identity()
            contrastive_loss = self.compute_contrastive_loss(x, x_offset, x_neg_offset)
        else: 
            contrastive_loss = 0

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
        
        if not self.attn_bottleneck:
            x_offset = self.conv(x_offset)

        return x_offset, contrastive_loss

    def _get_p_n(self, N, dtype, kernel_size, padding):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(kernel_size-1)//2, (kernel_size-1)//2+1),
            torch.arange(-(kernel_size-1)//2, (kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1)
        if padding:
            p_n = p_n.type(torch.float)
        return p_n

    def _get_p_0(self, h, w, N, dtype, padding):
        if padding:
            p_0_x, p_0_y = torch.meshgrid(
                torch.arange(1, h*self.stride+1, self.stride),
                torch.arange(1, w*self.stride+1, self.stride))
        else:
            p_0_x, p_0_y = torch.meshgrid(
                torch.arange(0, h*self.stride, self.stride),
                torch.arange(0, w*self.stride, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(torch.float)
        return p_0

    def _get_p(self, offset, dtype, padding=True):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)
        device = offset.device
        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype, int(N**0.5), padding)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype, padding)
        p = p_0 + p_n + offset.to('cpu')
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

        x_offset = x.gather(dim=-1, index=index.to('cuda')).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset
