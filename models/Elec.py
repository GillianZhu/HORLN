import torch
import torch.nn as nn
from collections import OrderedDict


class GroupFC(nn.Module):
    """Define a Resnet block"""

    def __init__(self, input_shape, output_nc, group_num=4, view=True):
        super(GroupFC, self).__init__()
        self.view = view

        input_nc = input_shape[0]
        h = input_shape[1]
        w = input_shape[2]

        self.groupFC = nn.Conv2d(input_nc, output_nc, groups=group_num, kernel_size=(h, w), stride=(h, w), padding=0)

    def forward(self, x):
        out = self.groupFC(x)
        if self.view:
            return out.view([x.shape[0], -1])
        else:
            return out


class IIWBlock(nn.Module):
    def __init__(self, input_nc, output_nc, h=148, w=7, kernel_size=3, activation='relu', use_dropout=False):
        super(IIWBlock, self).__init__()
        # average the number of output channels
        temp = output_nc // 2
        output_nc1 = output_nc2 = temp // 3
        output_nc3 = temp - output_nc1 * 2

        # relationship between kernel size(k), dilation rate(d), padding(p) to keep feature map size(h) unchanged after convolution
        # k_after_d = k + (k-1)*(d-1),  h + 2p - k_after_d + 1 = h
        # -> 2p = (k-1)*d

        # Inter-Week Convolutions
        self.first_conv = nn.Conv2d(input_nc, output_nc1,
                                    kernel_size=(kernel_size, 1), dilation=1, padding=((kernel_size - 1) // 2, 0),
                                    stride=1, bias=True)

        self.second_conv = nn.Conv2d(input_nc, output_nc2,
                                     kernel_size=(kernel_size, 1), dilation=2, padding=(kernel_size - 1, 0),
                                     stride=1, bias=True)

        self.third_conv = nn.Conv2d(input_nc, output_nc3,
                                    kernel_size=(kernel_size, 1), dilation=3, padding=((3 * kernel_size - 3) // 2, 0),
                                    stride=1, bias=True)

        # Intra-Week Convolutions
        self.first_conv2 = nn.Conv2d(input_nc, output_nc1,
                                     kernel_size=(1, kernel_size), dilation=1, padding=(0, (kernel_size - 1) // 2),
                                     stride=1, bias=True)

        self.second_conv2 = nn.Conv2d(input_nc, output_nc2,
                                      kernel_size=(1, kernel_size), dilation=2, padding=(0, kernel_size - 1),
                                      stride=1, bias=True)

        self.third_conv2 = nn.Conv2d(input_nc, output_nc3,
                                     kernel_size=(1, kernel_size), dilation=3, padding=(0, (3 * kernel_size - 3) // 2),
                                     stride=1, bias=True)

        activate_block = [nn.BatchNorm2d((output_nc))]
        if activation == 'prelu':
            activate_block += [nn.PReLU()]
        elif activation == 'relu':
            activate_block += [nn.ReLU(True)]

        if use_dropout:
            activate_block += [nn.Dropout(0.5)]

        self.activate_block = nn.Sequential(*activate_block)

    def forward(self, x):
        o1 = self.first_conv(x)
        o2 = self.second_conv(x)
        o3 = self.third_conv(x)
        o4 = self.first_conv2(x)
        o5 = self.second_conv2(x)
        o6 = self.third_conv2(x)
        conv_feature = torch.cat((o1, o2, o3, o4, o5, o6), dim=1)

        return self.activate_block(conv_feature)


class SDM(nn.Module):

    def __init__(self, h, w, output_dim, day_head=4, day_groups=2):
        super(SDM, self).__init__()

        assert (h == 148 and w == 7)
        self.day_head = day_head

        self.day_sd_layer = nn.Sequential(OrderedDict([
            ('dense1', GroupFC((self.day_head,
                                w,
                                w), output_dim,
                               group_num=day_groups)),
            ('norm1', nn.BatchNorm1d(output_dim)),
            ('prelu1', nn.PReLU()),
            ('drop1', nn.Dropout(p=0.7))
        ]))

    def forward(self, x):
        N, C, H, W = x.shape

        H_pad_num = self.day_head - H % self.day_head
        if H_pad_num > 0:
            pad = nn.ReplicationPad2d(padding=(0, 0, H_pad_num, 0))
            x = pad(x)
            H = H + H_pad_num

        # N x W x C*H -> N x W x head x (C*H/4) -> N x head x W x (C*H/4)
        day_input = x.permute(0, 3, 1, 2).reshape(N,
                                                  W,
                                                  self.day_head,
                                                  (C * H) // self.day_head).permute(0, 2, 1,
                                                                                    3).contiguous()
        day_sd = torch.einsum("bnqd,bnkd->bnqk", day_input, day_input)
        day_sd_fea = self.day_sd_layer(day_sd)

        return day_sd_fea


class Elec_HORLN_Model(nn.Module):
    def __init__(self, middle_nc=32, elec_nc=2):
        super(Elec_HORLN_Model, self).__init__()

        h = 148  # 148 weeks
        w = 7  # 7 days in a week

        self.ConvBlock1_1 = IIWBlock(elec_nc, middle_nc, activation='prelu', use_dropout=True)
        self.ConvBlock3_1 = IIWBlock(middle_nc, middle_nc // 2, activation='prelu', use_dropout=True)

        self.dense_layer = nn.Sequential(OrderedDict([
            ('dense1', GroupFC((middle_nc // 2, h, w), 200)),
            ('norm1', nn.BatchNorm1d(200)),
            ('prelu1', nn.PReLU()),
            ('drop1', nn.Dropout(p=0.7))
        ]))

        self.sdm = SDM(h=h, w=w, output_dim=300, day_head=4, day_groups=1)

        self.fusion_layer = nn.Sequential(OrderedDict([
            ('dense1', nn.Linear(200 + 300, 100)),
            ('norm1', nn.BatchNorm1d(100)),
            ('prelu1', nn.PReLU()),
            ('dropout1', nn.Dropout(p=0.6)),

            ('dense', nn.Linear(100, 1))
        ]))

        self.sigmoid = nn.Sigmoid()

    def forward(self, elec):
        # first order information
        convblock1_1 = self.ConvBlock1_1(elec)
        convblock3_1 = self.ConvBlock3_1(convblock1_1)
        first_order_info = self.dense_layer(convblock3_1)

        # second order information
        second_order_info = self.sdm(convblock3_1)

        pred = self.fusion_layer(torch.cat((first_order_info, second_order_info), dim=1))

        # when training, we use BCEWithLogitsLoss = Sigmoid + BCELoss,
        # So Sigmoid is not used here when training.
        if self.training == False:
            pred = self.sigmoid(pred)

        return pred


