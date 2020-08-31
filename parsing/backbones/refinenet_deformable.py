import torch.nn as nn
import torch.nn.functional as F
import torch

from .refinenet.modules.modulated_deformable_convolution import ModulatedDeformableConv2d as DCN
from .refinenet.modules.modulated_deformable_convolution import ModulatedDeformableConv2dResoff as DCNR
from .deform_conv_v2_cpu import DeformConv2d as DC

class RefineNetDeform(nn.Module):

    def __init__(
            self, 
            classes_num, 
            pretrained_weights=None, 
            cuda=True, 
            attn=False, 
            attn_only=False, 
            attn_dim='', 
            n_head=1, 
            use_contrastive=False, 
            share_weights=True, 
            attn_bottleneck=False,
            resoff=False,
            feature_dept=True
        ):
        self.use_cuda_version = cuda
        self.resoff = resoff
        if cuda and resoff:
            DC_module = DCNR
            extra_args = {"feature_dept": feature_dept}
        elif cuda:
            DC_module = DCN
            extra_args = {}
        else:
            DC_module = DC
            extra_args = {
                'attn': attn, 
                'attn_only': attn_only, 
                'attn_dim': attn_dim, 
                'n_head': n_head, 
                'use_contrastive': use_contrastive, 
                'share_weights': share_weights, 
                'attn_bottleneck': attn_bottleneck
            }

        super(RefineNetDeform, self).__init__()
        self.drop = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn_conv1 = nn.BatchNorm2d(64)
        self.conv1_relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # resnet block1
        self.res2a_branch1 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2a_branch1 = nn.BatchNorm2d(256)

        self.res2a_branch2a = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2a_branch2a = nn.BatchNorm2d(64)
        self.res2a_branch2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2a_branch2b = nn.BatchNorm2d(64)
        self.res2a_branch2c = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2a_branch2c = nn.BatchNorm2d(256)

        for i in "bc":
            self.__setattr__("res2{}_branch2a".format(i), nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False))
            self.__setattr__("bn2{}_branch2a".format(i), nn.BatchNorm2d(64))
            self.__setattr__("res2{}_branch2b".format(i), nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
            self.__setattr__("bn2{}_branch2b".format(i), nn.BatchNorm2d(64))
            self.__setattr__("res2{}_branch2c".format(i), nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False))
            self.__setattr__("bn2{}_branch2c".format(i), nn.BatchNorm2d(256))

        # resnet block2
        self.res3a_branch1 = nn.Conv2d(256, 512, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn3a_branch1 = nn.BatchNorm2d(512)

        self.res3a_branch2a = nn.Conv2d(256, 128, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn3a_branch2a = nn.BatchNorm2d(128)
        # self.res3a_branch2b = DC_module(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.res3a_branch2b = DC_module(*(128, 128), **{**{'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}, **extra_args})
        self.bn3a_branch2b = nn.BatchNorm2d(128)
        self.res3a_branch2c = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3a_branch2c = nn.BatchNorm2d(512)

        for i in range(1, 4):
            self.__setattr__("res3b{}_branch2a".format(i), nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False))
            self.__setattr__("bn3b{}_branch2a".format(i), nn.BatchNorm2d(128))
            # self.__setattr__("res3b{}_branch2b".format(i), DC_module(128, 128, kernel_size=3, stride=1, padding=1, bias=False))
            self.__setattr__("res3b{}_branch2b".format(i), DC_module(*(128, 128), **{**{'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}, **extra_args}))
            self.__setattr__("bn3b{}_branch2b".format(i), nn.BatchNorm2d(128))
            self.__setattr__("res3b{}_branch2c".format(i), nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False))
            self.__setattr__("bn3b{}_branch2c".format(i), nn.BatchNorm2d(512))

        # resnet block3
        self.res4a_branch1 = nn.Conv2d(512, 1024, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn4a_branch1 = nn.BatchNorm2d(1024)

        self.res4a_branch2a = nn.Conv2d(512, 256, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn4a_branch2a = nn.BatchNorm2d(256)
        # self.res4a_branch2b = DC_module(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.res4a_branch2b = DC_module(*(256, 256), **{**{'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}, **extra_args})
        self.bn4a_branch2b = nn.BatchNorm2d(256)
        self.res4a_branch2c = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4a_branch2c = nn.BatchNorm2d(1024)

        for i in range(1, 23):
            self.__setattr__("res4b{}_branch2a".format(i), nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False))
            self.__setattr__("bn4b{}_branch2a".format(i), nn.BatchNorm2d(256))
            # self.__setattr__("res4b{}_branch2b".format(i), DC_module(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
            self.__setattr__("res4b{}_branch2b".format(i), DC_module(*(256, 256), **{**{'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}, **extra_args}))
            self.__setattr__("bn4b{}_branch2b".format(i), nn.BatchNorm2d(256))
            self.__setattr__("res4b{}_branch2c".format(i), nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False))
            self.__setattr__("bn4b{}_branch2c".format(i), nn.BatchNorm2d(1024))

        # resnet block 4
        self.res5a_branch1 = nn.Conv2d(1024, 2048, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn5a_branch1 = nn.BatchNorm2d(2048)

        self.res5a_branch2a = nn.Conv2d(1024, 512, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn5a_branch2a = nn.BatchNorm2d(512)
        # self.res5a_branch2b = DC_module(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.res5a_branch2b = DC_module(*(512, 512), **{**{'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}, **extra_args})
        self.bn5a_branch2b = nn.BatchNorm2d(512)
        self.res5a_branch2c = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5a_branch2c = nn.BatchNorm2d(2048)

        for i in "bc":
            self.__setattr__("res5{}_branch2a".format(i), nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False))
            self.__setattr__("bn5{}_branch2a".format(i), nn.BatchNorm2d(512))
            # self.__setattr__("res5{}_branch2b".format(i), DC_module(512, 512, kernel_size=3, stride=1, padding=1, bias=False))
            self.__setattr__("res5{}_branch2b".format(i), DC_module(*(512, 512), **{**{'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}, **extra_args}))
            self.__setattr__("bn5{}_branch2b".format(i), nn.BatchNorm2d(512))
            self.__setattr__("res5{}_branch2c".format(i), nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False))
            self.__setattr__("bn5{}_branch2c".format(i), nn.BatchNorm2d(2048))

        # refinenet
        # self.p_ims1d2_outl1_dimred = DC_module(2048, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.p_ims1d2_outl1_dimred = DC_module(*(2048, 512), **{**{'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}, **extra_args})
        # self.p_ims1d2_outl2_dimred = DC_module(1024, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.p_ims1d2_outl2_dimred = DC_module(*(1024, 256), **{**{'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}, **extra_args})
        # self.p_ims1d2_outl3_dimred = DC_module(512, 256,  kernel_size=3, stride=1, padding=1, bias=False)
        self.p_ims1d2_outl3_dimred = DC_module(*(512, 256), **{**{'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}, **extra_args})
        self.p_ims1d2_outl4_dimred = nn.Conv2d(256, 256,  kernel_size=3, stride=1, padding=1, bias=False)
        self.p_ims1d2_outl5_dimred = nn.Conv2d(64, 64,    kernel_size=3, stride=1, padding=1, bias=False)
        self.p_ims1d2_outl6_dimred = nn.Conv2d(3, 64,     kernel_size=3, stride=1, padding=1, bias=False)

        channels = [512, 256, 256, 256, 64, 64]
        for i in range(1, 7):
            c = channels[i - 1]
            for j in range(1, 3):
                if i < 4:
                    # self.__setattr__("adapt_input_path{}_b{}_conv".format(i, j), DC_module(c, c, kernel_size=3, stride=1, padding=1, bias=True))
                    self.__setattr__("adapt_input_path{}_b{}_conv".format(i, j), DC_module(*(c, c), **{**{'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True}, **extra_args}))
                    # self.__setattr__("adapt_input_path{}_b{}_conv_relu_varout_dimred".format(i, j), DC_module(c, c, kernel_size=3, stride=1, padding=1, bias=False))
                    self.__setattr__("adapt_input_path{}_b{}_conv_relu_varout_dimred".format(i, j), DC_module(*(c, c), **{**{'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}, **extra_args}))
                else:
                    self.__setattr__("adapt_input_path{}_b{}_conv".format(i, j), nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=True))
                    self.__setattr__("adapt_input_path{}_b{}_conv_relu_varout_dimred".format(i, j), nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False))

        for i in range(2, 7):
            c = channels[i - 1]
            if i < 4:
                # self.__setattr__("adapt_input_path{}_b2_joint_varout_dimred".format(i), DC_module(c, c, kernel_size=3, stride=1, padding=1, bias=False))
                self.__setattr__("adapt_input_path{}_b2_joint_varout_dimred".format(i), DC_module(*(c, c), **{**{'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}, **extra_args}))
            else:
                self.__setattr__("adapt_input_path{}_b2_joint_varout_dimred".format(i), nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False))

        for i in range(1, 7):
            c = channels[i - 1]
            if i < 4:
                # self.__setattr__("mflow_conv_g{}_poolprev_relu_varout_pb1".format(i), DC_module(c, c, kernel_size=3, stride=1, padding=1, bias=False))
                self.__setattr__("mflow_conv_g{}_poolprev_relu_varout_pb1".format(i), DC_module(*(c, c), **{**{'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}, **extra_args}))
                # self.__setattr__("mflow_conv_g{}_pool1_outvar_pb2".format(i), DC_module(c, c, kernel_size=3, stride=1, padding=1, bias=False))
                self.__setattr__("mflow_conv_g{}_pool1_outvar_pb2".format(i), DC_module(*(c, c), **{**{'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}, **extra_args}))
            else:
                self.__setattr__("mflow_conv_g{}_poolprev_relu_varout_pb1".format(i), nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False))
                self.__setattr__("mflow_conv_g{}_pool1_outvar_pb2".format(i), nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False))
            self.__setattr__("mflow_conv_g{}_pool1".format(i), nn.MaxPool2d(kernel_size=5, stride=1, padding=2))
            self.__setattr__("mflow_conv_g{}_pool2".format(i), nn.MaxPool2d(kernel_size=5, stride=1, padding=2))
        for i in range(5, 7):
            c = channels[i - 1]
            self.__setattr__("mflow_conv_g{}_pool3_outvar_pb3".format(i), nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False))
            self.__setattr__("mflow_conv_g{}_pool3".format(i), nn.MaxPool2d(kernel_size=5, stride=1, padding=2))
            self.__setattr__("mflow_conv_g{}_pool4_outvar_pb4".format(i), nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False))
            self.__setattr__("mflow_conv_g{}_pool4".format(i), nn.MaxPool2d(kernel_size=5, stride=1, padding=2))

        for i in range(1, 7):
            c = channels[i - 1]
            if i < 4:
                for j in range(1, 4):
                    # self.__setattr__("mflow_conv_g{}_b{}_conv".format(i, j), DC_module(c, c, kernel_size=3, stride=1, padding=1, bias=True))
                    self.__setattr__("mflow_conv_g{}_b{}_conv".format(i, j), DC_module(*(c, c), **{**{'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True}, **extra_args}))
                    # self.__setattr__("mflow_conv_g{}_b{}_conv_relu_varout_dimred".format(i, j), DC_module(c, c, kernel_size=3, stride=1, padding=1, bias=False))
                    self.__setattr__("mflow_conv_g{}_b{}_conv_relu_varout_dimred".format(i, j), DC_module(*(c, c), **{**{'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}, **extra_args}))
            else:
                for j in range(1, 4):
                    self.__setattr__("mflow_conv_g{}_b{}_conv".format(i, j), nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=True))
                    self.__setattr__("mflow_conv_g{}_b{}_conv_relu_varout_dimred".format(i, j), nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False))

        for i in range(1, 6):
            c0 = channels[i - 1]
            c1 = channels[i]
            if i < 4:
                # self.__setattr__("mflow_conv_g{}_b3_joint_varout_dimred".format(i), DC_module(c0, c1, kernel_size=3, stride=1, padding=1, bias=False))
                self.__setattr__("mflow_conv_g{}_b3_joint_varout_dimred".format(i), DC_module(*(c0, c1), **{**{'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}, **extra_args}))
            else:
                self.__setattr__("mflow_conv_g{}_b3_joint_varout_dimred".format(i), nn.Conv2d(c0, c1, kernel_size=3, stride=1, padding=1, bias=False))

        self.mflow_conv_g6_b3_joint_drop_conv_new_2 = nn.Conv2d(64, classes_num, kernel_size=3, stride=1, padding=1, bias=True)

        if pretrained_weights:
            self.load_weights(pretrained_weights)

    def load_weights(self, weights):
        with torch.no_grad():
            for p in self.named_parameters():
                layer = p[0].rsplit(".", 1)[0]
                name = p[0].rsplit(".", 1)[1]
                assert layer in weights
                p[1].copy_(torch.from_numpy(weights[layer].pop(name)).float())

            for p in self.named_buffers():
                layer = p[0].rsplit(".", 1)[0]
                name = p[0].rsplit(".", 1)[1]
                if name == "num_batches_tracked":
                    continue
                assert layer in weights
                p[1].copy_(torch.from_numpy(weights[layer].pop(name)).float())

            for k in weights:
                assert not len(weights[k]), k


    def forward(self, x):
        data = x
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = self.conv1_relu(x)
        resconv1 = x
        contrastive_loss = []

        # resnet block1
        x = self.pool1(x)
        res2a_b1 = self.res2a_branch1(x)
        res2a_b1 = self.bn2a_branch1(res2a_b1)

        res2a_b2 = self.res2a_branch2a(x)
        res2a_b2 = self.bn2a_branch2a(res2a_b2)
        res2a_b2 = self.relu(res2a_b2)
        res2a_b2 = self.res2a_branch2b(res2a_b2)
        res2a_b2 = self.bn2a_branch2b(res2a_b2)
        res2a_b2 = self.relu(res2a_b2)
        res2a_b2 = self.res2a_branch2c(res2a_b2)
        res2a_b2 = self.bn2a_branch2c(res2a_b2)
        res2a = res2a_b1 + res2a_b2
        res2a = self.relu(res2a)

        x = res2a
        for i in "bc":
            y = self.__getattr__("res2{}_branch2a".format(i))(x)
            y = self.__getattr__("bn2{}_branch2a".format(i))(y)
            y = self.relu(y)
            y = self.__getattr__("res2{}_branch2b".format(i))(y)
            y = self.__getattr__("bn2{}_branch2b".format(i))(y)
            y = self.relu(y)
            y = self.__getattr__("res2{}_branch2c".format(i))(y)
            y = self.__getattr__("bn2{}_branch2c".format(i))(y)
            y += x
            y = self.relu(y)
            x = y
        res2c = y

        # resnet block2
        res3a_b1 = self.res3a_branch1(res2c)
        res3a_b1 = self.bn3a_branch1(res3a_b1)

        res3a_b2 = self.res3a_branch2a(res2c)
        res3a_b2 = self.bn3a_branch2a(res3a_b2)
        res3a_b2 = self.relu(res3a_b2)
        res3a_b2 = self.res3a_branch2b(res3a_b2)
        if self.resoff and self.use_cuda_version:
            prev_off = {}
            res3a_b2, prev_off[3] = res3a_b2
        if not self.use_cuda_version:
            contrastive_loss.append(res3a_b2[1])
            res3a_b2 = res3a_b2[0]
        res3a_b2 = self.bn3a_branch2b(res3a_b2)
        res3a_b2 = self.relu(res3a_b2)
        res3a_b2 = self.res3a_branch2c(res3a_b2)
        res3a_b2 = self.bn3a_branch2c(res3a_b2)
        res3a = res3a_b1 + res3a_b2
        res3a = self.relu(res3a)

        x = res3a
        for i in range(1, 4):
            y = self.__getattr__("res3b{}_branch2a".format(i))(x)
            y = self.__getattr__("bn3b{}_branch2a".format(i))(y)
            y = self.relu(y)
            if self.resoff and self.use_cuda_version:
                y, prev_off[3] = self.__getattr__("res3b{}_branch2b".format(i))(y, prev_off[3])
            else:
                y = self.__getattr__("res3b{}_branch2b".format(i))(y)
            if not self.use_cuda_version:
                contrastive_loss.append(y[1])
                y = y[0]
            y = self.__getattr__("bn3b{}_branch2b".format(i))(y)
            y = self.relu(y)
            y = self.__getattr__("res3b{}_branch2c".format(i))(y)
            y = self.__getattr__("bn3b{}_branch2c".format(i))(y)
            y += x
            y = self.relu(y)
            x = y
        res3b3 = y

        # resnet block3
        res4a_b1 = self.res4a_branch1(res3b3)
        res4a_b1 = self.bn4a_branch1(res4a_b1)

        res4a_b2 = self.res4a_branch2a(res3b3)
        res4a_b2 = self.bn4a_branch2a(res4a_b2)
        res4a_b2 = self.relu(res4a_b2)
        res4a_b2 = self.res4a_branch2b(res4a_b2)
        if self.resoff and self.use_cuda_version:
            res4a_b2, prev_off[4] = res4a_b2
        if not self.use_cuda_version:
            contrastive_loss.append(res4a_b2[1])
            res4a_b2 = res4a_b2[0]
        res4a_b2 = self.bn4a_branch2b(res4a_b2)
        res4a_b2 = self.relu(res4a_b2)
        res4a_b2 = self.res4a_branch2c(res4a_b2)
        res4a_b2 = self.bn4a_branch2c(res4a_b2)
        res4a = res4a_b1 + res4a_b2
        res4a = self.relu(res4a)

        x = res4a
        for i in range(1, 23):
            y = self.__getattr__("res4b{}_branch2a".format(i))(x)
            y = self.__getattr__("bn4b{}_branch2a".format(i))(y)
            y = self.relu(y)
            if self.resoff and self.use_cuda_version:
                y, prev_off[4] = self.__getattr__("res4b{}_branch2b".format(i))(y, prev_off[4])
            else:
                y = self.__getattr__("res4b{}_branch2b".format(i))(y)
            if not self.use_cuda_version:
                contrastive_loss.append(y[1])
                y = y[0]
            y = self.__getattr__("bn4b{}_branch2b".format(i))(y)
            y = self.relu(y)
            y = self.__getattr__("res4b{}_branch2c".format(i))(y)
            y = self.__getattr__("bn4b{}_branch2c".format(i))(y)
            y += x
            y = self.relu(y)
            x = y
        res4b22 = y

        # resnet block4
        res5a_b1 = self.res5a_branch1(res4b22)
        res5a_b1 = self.bn5a_branch1(res5a_b1)

        res5a_b2 = self.res5a_branch2a(res4b22)
        res5a_b2 = self.bn5a_branch2a(res5a_b2)
        res5a_b2 = self.relu(res5a_b2)
        res5a_b2 = self.res5a_branch2b(res5a_b2)
        if self.resoff and self.use_cuda_version:
            res5a_b2, prev_off[5] = res5a_b2
        if not self.use_cuda_version:
            contrastive_loss.append(res5a_b2[1])
            res5a_b2 = res5a_b2[0]
        res5a_b2 = self.bn5a_branch2b(res5a_b2)
        res5a_b2 = self.relu(res5a_b2)
        res5a_b2 = self.res5a_branch2c(res5a_b2)
        res5a_b2 = self.bn5a_branch2c(res5a_b2)
        res5a = res5a_b1 + res5a_b2
        res5a = self.relu(res5a)

        x = res5a
        for i in "bc":
            y = self.__getattr__("res5{}_branch2a".format(i))(x)
            y = self.__getattr__("bn5{}_branch2a".format(i))(y)
            y = self.relu(y)
            if self.resoff and self.use_cuda_version:
                y, prev_off[5] = self.__getattr__("res5{}_branch2b".format(i))(y, prev_off[5])
            else:
                y = self.__getattr__("res5{}_branch2b".format(i))(y)
            if not self.use_cuda_version:
                contrastive_loss.append(y[1])
                y = y[0]
            y = self.__getattr__("bn5{}_branch2b".format(i))(y)
            y = self.relu(y)
            y = self.__getattr__("res5{}_branch2c".format(i))(y)
            y = self.__getattr__("bn5{}_branch2c".format(i))(y)
            y += x
            y = self.relu(y)
            x = y
        res5c = y

        # refinenet
        res5c = self.drop(res5c)
        res4b22 = self.drop(res4b22)
        # outl1 = self.p_ims1d2_outl1_dimred(res5c)
        if self.resoff and self.use_cuda_version:
            outl1, prev_off[5] = self.p_ims1d2_outl1_dimred(res5c, prev_off[5])
        else:
            outl1 = self.p_ims1d2_outl1_dimred(res5c)
        if not self.use_cuda_version:
            contrastive_loss.append(outl1[1])
            outl1 = outl1[0]
        # outl2 = self.p_ims1d2_outl2_dimred(res4b22)
        if self.resoff and self.use_cuda_version:
            outl2, prev_off[4] = self.p_ims1d2_outl2_dimred(res4b22, prev_off[4])
        else:
            outl2 = self.p_ims1d2_outl2_dimred(res4b22)
        if not self.use_cuda_version:
            contrastive_loss.append(outl2[1])
            outl2 = outl2[0]
        # outl3 = self.p_ims1d2_outl3_dimred(res3b3)
        if self.resoff and self.use_cuda_version:
            outl3, prev_off[3] = self.p_ims1d2_outl3_dimred(res3b3, prev_off[3])
        else:
            outl3 = self.p_ims1d2_outl3_dimred(res3b3)
        if not self.use_cuda_version:
            contrastive_loss.append(outl3[1])
            outl3 = outl3[0]
        outl4 = self.p_ims1d2_outl4_dimred(res2c)
        outl5 = self.p_ims1d2_outl5_dimred(resconv1)
        outl6 = self.p_ims1d2_outl6_dimred(data)
        outl = [outl1, outl2, outl3, outl4, outl5, outl6]

        for i in range(1, 7):
            x = outl[i - 1]
            for j in range(1, 3):
                y = F.relu(x)
                # y = self.__getattr__("adapt_input_path{}_b{}_conv".format(i, j))(y)
                if self.resoff and self.use_cuda_version and i < 4:
                    y, prev_off[6-i] = self.__getattr__("adapt_input_path{}_b{}_conv".format(i, j))(y, prev_off[6-i])
                else:
                    y = self.__getattr__("adapt_input_path{}_b{}_conv".format(i, j))(y)
                if not self.use_cuda_version and i < 4:
                    contrastive_loss.append(y[1])
                    y = y[0]
                y = self.relu(y)
                # y = self.__getattr__("adapt_input_path{}_b{}_conv_relu_varout_dimred".format(i, j))(y)
                if self.resoff and self.use_cuda_version and i < 4:
                    y, prev_off[6-i] = self.__getattr__("adapt_input_path{}_b{}_conv_relu_varout_dimred".format(i, j))(y, prev_off[6-i])
                else:
                    y = self.__getattr__("adapt_input_path{}_b{}_conv_relu_varout_dimred".format(i, j))(y)
                if not self.use_cuda_version and i < 4:
                    contrastive_loss.append(y[1])
                    y = y[0]
                y += x
                x = y
            if i > 1:
                # y = self.__getattr__("adapt_input_path{}_b2_joint_varout_dimred".format(i))(y)
                if self.resoff and self.use_cuda_version and i < 4:
                    y, prev_off[6-i] = self.__getattr__("adapt_input_path{}_b2_joint_varout_dimred".format(i))(y, prev_off[6-i])
                else:
                    y = self.__getattr__("adapt_input_path{}_b2_joint_varout_dimred".format(i))(y)
                if not self.use_cuda_version and i < 4:
                    contrastive_loss.append(y[1])
                    y = y[0]
            outl[i - 1] = y
        outl1, outl2, outl3, outl4, outl5, outl6 = outl

        x0 = outl1
        x0 = self.relu(x0)
        # x1 = self.mflow_conv_g1_poolprev_relu_varout_pb1(x0)
        if self.resoff and self.use_cuda_version:
            x1, prev_off[5] = self.mflow_conv_g1_poolprev_relu_varout_pb1(x0, prev_off[5])
        else:
            x1 = self.mflow_conv_g1_poolprev_relu_varout_pb1(x0)
        if not self.use_cuda_version:
            contrastive_loss.append(x1[1])
            x1 = x1[0]
        x1 = self.mflow_conv_g1_pool1(x1)
        # x2 = self.mflow_conv_g1_pool1_outvar_pb2(x1)
        if self.resoff and self.use_cuda_version:
            x2, prev_off[5] = self.mflow_conv_g1_pool1_outvar_pb2(x1, prev_off[5])
        else:
            x2 = self.mflow_conv_g1_pool1_outvar_pb2(x1)
        if not self.use_cuda_version:
            contrastive_loss.append(x2[1])
            x2 = x2[0]
        x2 = self.mflow_conv_g1_pool2(x2)
        x = x0 + x1 + x2
        for i in range(1, 4):
            y = F.relu(x)
            # y = self.__getattr__("mflow_conv_g1_b{}_conv".format(i))(y)
            if self.resoff and self.use_cuda_version:
                y, prev_off[5] = self.__getattr__("mflow_conv_g1_b{}_conv".format(i))(y, prev_off[5])
            else:
                y = self.__getattr__("mflow_conv_g1_b{}_conv".format(i))(y)
            if not self.use_cuda_version:
                contrastive_loss.append(y[1])
                y = y[0]
            y = self.relu(y)
            # y = self.__getattr__("mflow_conv_g1_b{}_conv_relu_varout_dimred".format(i))(y)
            if self.resoff and self.use_cuda_version:
                y, prev_off[5] = self.__getattr__("mflow_conv_g1_b{}_conv_relu_varout_dimred".format(i))(y, prev_off[5])
            else:
                y = self.__getattr__("mflow_conv_g1_b{}_conv_relu_varout_dimred".format(i))(y)
            if not self.use_cuda_version:
                contrastive_loss.append(y[1])
                y = y[0]
            y += x
            x = y
        # y = self.mflow_conv_g1_b3_joint_varout_dimred(y)
        if self.resoff and self.use_cuda_version:
            y, prev_off[5] = self.mflow_conv_g1_b3_joint_varout_dimred(y, prev_off[5])
        else:
            y = self.mflow_conv_g1_b3_joint_varout_dimred(y)
        if not self.use_cuda_version:
            contrastive_loss.append(y[1])
            y = y[0]
        y = nn.Upsample(size=outl2.size()[2:], mode='bilinear', align_corners=True)(y)
        g2 = y + outl2

        x0 = g2
        x0 = self.relu(x0)
        # x1 = self.mflow_conv_g2_poolprev_relu_varout_pb1(x0)
        if self.resoff and self.use_cuda_version:
            x1, prev_off[4] = self.mflow_conv_g2_poolprev_relu_varout_pb1(x0, prev_off[4])
        else:
            x1 = self.mflow_conv_g2_poolprev_relu_varout_pb1(x0)
        if not self.use_cuda_version:
            contrastive_loss.append(x1[1])
            x1 = x1[0]
        x1 = self.mflow_conv_g2_pool1(x1)
        # x2 = self.mflow_conv_g2_pool1_outvar_pb2(x1)
        if self.resoff and self.use_cuda_version:
            x2, prev_off[4] = self.mflow_conv_g2_pool1_outvar_pb2(x1, prev_off[4])
        else:
            x2 = self.mflow_conv_g2_pool1_outvar_pb2(x1)
        if not self.use_cuda_version:
            contrastive_loss.append(x2[1])
            x2 = x2[0]
        x2 = self.mflow_conv_g2_pool2(x2)
        x = x0 + x1 + x2
        for i in range(1, 4):
            y = F.relu(x)
            # y = self.__getattr__("mflow_conv_g2_b{}_conv".format(i))(y)
            if self.resoff and self.use_cuda_version:
                y, prev_off[4] = self.__getattr__("mflow_conv_g2_b{}_conv".format(i))(y, prev_off[4])
            else:
                y = self.__getattr__("mflow_conv_g2_b{}_conv".format(i))(y)
            if not self.use_cuda_version:
                contrastive_loss.append(y[1])
                y = y[0]
            y = self.relu(y)
            # y = self.__getattr__("mflow_conv_g2_b{}_conv_relu_varout_dimred".format(i))(y)
            if self.resoff and self.use_cuda_version:
                y, prev_off[4] = self.__getattr__("mflow_conv_g2_b{}_conv_relu_varout_dimred".format(i))(y, prev_off[4])
            else:
                y = self.__getattr__("mflow_conv_g2_b{}_conv_relu_varout_dimred".format(i))(y)
            if not self.use_cuda_version:
                contrastive_loss.append(y[1])
                y = y[0]
            y += x
            x = y
        # y = self.mflow_conv_g2_b3_joint_varout_dimred(y)
        if self.resoff and self.use_cuda_version:
            y, prev_off[4] = self.mflow_conv_g2_b3_joint_varout_dimred(y, prev_off[4])
        else:
            y = self.mflow_conv_g2_b3_joint_varout_dimred(y)
        if not self.use_cuda_version:
            contrastive_loss.append(y[1])
            y = y[0]
        y = nn.Upsample(size=outl3.size()[2:], mode='bilinear', align_corners=True)(y)
        g3 = y + outl3

        x0 = g3
        x0 = self.relu(x0)
        # x1 = self.mflow_conv_g3_poolprev_relu_varout_pb1(x0)
        if self.resoff and self.use_cuda_version:
            x1, prev_off[3] = self.mflow_conv_g3_poolprev_relu_varout_pb1(x0, prev_off[3])
        else:
            x1 = self.mflow_conv_g3_poolprev_relu_varout_pb1(x0)
        if not self.use_cuda_version:
            contrastive_loss.append(x1[1])
            x1 = x1[0]
        x1 = self.mflow_conv_g3_pool1(x1)
        # x2 = self.mflow_conv_g3_pool1_outvar_pb2(x1)
        if self.resoff and self.use_cuda_version:
            x2, prev_off[3] = self.mflow_conv_g3_pool1_outvar_pb2(x1, prev_off[3])
        else:
            x2 = self.mflow_conv_g3_pool1_outvar_pb2(x1)
        if not self.use_cuda_version:
            contrastive_loss.append(x2[1])
            x2 = x2[0]
        x2 = self.mflow_conv_g3_pool2(x2)
        x = x0 + x1 + x2
        for i in range(1, 4):
            y = F.relu(x)
            # y = self.__getattr__("mflow_conv_g3_b{}_conv".format(i))(y)
            if self.resoff and self.use_cuda_version:
                y, prev_off[3] = self.__getattr__("mflow_conv_g3_b{}_conv".format(i))(y, prev_off[3])
            else:
                y = self.__getattr__("mflow_conv_g3_b{}_conv".format(i))(y)
            if not self.use_cuda_version:
                contrastive_loss.append(y[1])
                y = y[0]
            y = self.relu(y)
            # y = self.__getattr__("mflow_conv_g3_b{}_conv_relu_varout_dimred".format(i))(y)
            if self.resoff and self.use_cuda_version:
                y, prev_off[3] = self.__getattr__("mflow_conv_g3_b{}_conv_relu_varout_dimred".format(i))(y, prev_off[3])
            else:
                y = self.__getattr__("mflow_conv_g3_b{}_conv_relu_varout_dimred".format(i))(y)
            if not self.use_cuda_version:
                contrastive_loss.append(y[1])
                y = y[0]
            y += x
            x = y
        # y = self.mflow_conv_g3_b3_joint_varout_dimred(y)
        if self.resoff and self.use_cuda_version:
            y, prev_off[3] = self.mflow_conv_g3_b3_joint_varout_dimred(y, prev_off[3])
        else:
            y = self.mflow_conv_g3_b3_joint_varout_dimred(y)
        if not self.use_cuda_version:
            contrastive_loss.append(y[1])
            y = y[0]
        y = nn.Upsample(size=outl4.size()[2:], mode='bilinear', align_corners=True)(y)
        g4 = y + outl4

        x0 = g4
        x0 = self.relu(x0)
        x1 = self.mflow_conv_g4_poolprev_relu_varout_pb1(x0)
        x1 = self.mflow_conv_g4_pool1(x1)
        x2 = self.mflow_conv_g4_pool1_outvar_pb2(x1)
        x2 = self.mflow_conv_g4_pool2(x2)
        x = x0 + x1 + x2
        for i in range(1, 4):
            y = F.relu(x)
            y = self.__getattr__("mflow_conv_g4_b{}_conv".format(i))(y)
            y = self.relu(y)
            y = self.__getattr__("mflow_conv_g4_b{}_conv_relu_varout_dimred".format(i))(y)
            y += x
            x = y
        y = self.mflow_conv_g4_b3_joint_varout_dimred(y)
        y = nn.Upsample(size=outl5.size()[2:], mode='bilinear', align_corners=True)(y)
        g5 = y + outl5

        x0 = g5
        x0 = self.relu(x0)
        x1 = self.mflow_conv_g5_poolprev_relu_varout_pb1(x0)
        x1 = self.mflow_conv_g5_pool1(x1)
        x2 = self.mflow_conv_g5_pool1_outvar_pb2(x1)
        x2 = self.mflow_conv_g5_pool2(x2)
        x3 = self.mflow_conv_g5_pool3_outvar_pb3(x2)
        x3 = self.mflow_conv_g5_pool3(x3)
        x4 = self.mflow_conv_g5_pool4_outvar_pb4(x3)
        x4 = self.mflow_conv_g5_pool4(x4)
        x = x0 + x1 + x2 + x3 + x4
        for i in range(1, 4):
            y = F.relu(x)
            y = self.__getattr__("mflow_conv_g5_b{}_conv".format(i))(y)
            y = self.relu(y)
            y = self.__getattr__("mflow_conv_g5_b{}_conv_relu_varout_dimred".format(i))(y)
            y += x
            x = y
        y = self.mflow_conv_g5_b3_joint_varout_dimred(y)
        y = nn.Upsample(size=outl6.size()[2:], mode='bilinear', align_corners=True)(y)
        g6 = y + outl6

        x0 = g6
        x0 = self.relu(x0)
        x1 = self.mflow_conv_g6_poolprev_relu_varout_pb1(x0)
        x1 = self.mflow_conv_g6_pool1(x1)
        x2 = self.mflow_conv_g6_pool1_outvar_pb2(x1)
        x2 = self.mflow_conv_g6_pool2(x2)
        x3 = self.mflow_conv_g6_pool3_outvar_pb3(x2)
        x3 = self.mflow_conv_g6_pool3(x3)
        x4 = self.mflow_conv_g6_pool4_outvar_pb4(x3)
        x4 = self.mflow_conv_g6_pool4(x4)
        x = x0 + x1 + x2 + x3 + x4
        for i in range(1, 4):
            y = F.relu(x)
            y = self.__getattr__("mflow_conv_g6_b{}_conv".format(i))(y)
            y = self.relu(y)
            y = self.__getattr__("mflow_conv_g6_b{}_conv_relu_varout_dimred".format(i))(y)
            y += x
            x = y
        out = self.drop(y)
        out = self.mflow_conv_g6_b3_joint_drop_conv_new_2(out)
        return out, contrastive_loss
