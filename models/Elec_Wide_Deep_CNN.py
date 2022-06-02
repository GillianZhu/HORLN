import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

# The compared method WDCNNModel is derived from
# https://github.com/neuralmind-ai/electricity-theft-detection-with-self-attention/blob/master/CNN_model.py


def kernel_fn(kernel, channel_in, channel_out, device):
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(channel_out, channel_in, 1, 1).float()
    weight = nn.Parameter(data=kernel, requires_grad=False)

    return weight.to(device)

# random_seed = 123
# torch.manual_seed(random_seed)
class KernelConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, stride=1, padding=0, bias=False):
        super(KernelConv2d, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.padding = padding

        self.g1_kernel = [[0.0, -1.0, 0.0],
                     [0.0, 2.0, 0.0],
                     [0.0, -1.0, 0.0]]
        self.g2_kernel = [[0.0, 0.0, 0.0],
                     [-1.0, 2.0, -1.0],
                     [0.0, 0.0, 0.0]]

        self.bias = bias

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        g1_kernel = kernel_fn(self.g1_kernel, self.channel_in, self.channel_out, x.device) # [channel_out, channel_in, kernel, kernel]
        g2_kernel = kernel_fn(self.g2_kernel, self.channel_in, self.channel_out, x.device)
        if self.bias:
            out_g1 = F.conv2d(x, g1_kernel, stride=self.stride, padding=self.padding, bias=torch.rand(self.channel_out).to(x.device))
            out_g2 = F.conv2d(x, g2_kernel, stride=self.stride, padding=self.padding, bias=torch.rand(self.channel_out).to(x.device))
        else:
            out_g1 = F.conv2d(x, g1_kernel, stride=self.stride, padding=self.padding)
            out_g2 = F.conv2d(x, g2_kernel, stride=self.stride, padding=self.padding)
        out = torch.tanh(out_g1+out_g2)
        return out


class WDCNNModel(nn.Module):
    def __init__(self):
        super(WDCNNModel, self).__init__()

        # self.cnn_nc = 16
        # self.wide_fc_nc = 50
        # self.deep_fc_nc = 60
        self.cnn_nc = 60
        self.wide_fc_nc = 90
        self.deep_fc_nc = 90

        self.wide_net = nn.Sequential(OrderedDict([
            ('wide_fc', nn.Linear(148 * 7*2, self.wide_fc_nc)),
            ('wide_fc_relu', nn.ReLU()),
        ]))

        self.deep_net = nn.Sequential(OrderedDict([
            ('conv1', KernelConv2d(2, self.cnn_nc, stride=1, padding=1, bias=True)),
            ('relu1', nn.ReLU()),

            ('conv2', KernelConv2d(self.cnn_nc, self.cnn_nc, stride=1, padding=1, bias=True)),
            ('relu2', nn.ReLU()),

            ('conv3', KernelConv2d(self.cnn_nc, self.cnn_nc, stride=1, padding=1, bias=True)),
            ('relu3', nn.ReLU()),

            ('conv4', KernelConv2d(self.cnn_nc, self.cnn_nc, stride=1, padding=1, bias=True)),
            ('relu4', nn.ReLU()),

            ('conv5', KernelConv2d(self.cnn_nc, self.cnn_nc, stride=1, padding=1, bias=True)),
            ('relu5', nn.ReLU()),

            ('maxpool', nn.MaxPool2d((1, 7), stride=(1, 7))),
        ]))

        self.deep_net_fc = nn.Sequential(OrderedDict([
            ('deep_fc', nn.Linear(self.cnn_nc * 148, self.deep_fc_nc)),
            ('deep_fc_relu', nn.ReLU()),
        ]))

        self.fusion_fc = nn.Linear(self.wide_fc_nc + self.deep_fc_nc, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # wide&deep model does not use the mask map
        # x = x[:, 0:1, :, :]

        wide_output = self.wide_net(x.view(x.shape[0], -1))

        deep_output = self.deep_net(x)
        deep_output = self.deep_net_fc(deep_output.view(deep_output.shape[0], -1))

        output = self.fusion_fc(torch.cat((wide_output, deep_output), 1))
        if self.training == False:
            output = self.sigmoid(output)

        return output


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


class WDCNNModel_g1g2_mask_699_hyorder(nn.Module):
    def __init__(self):
        super(WDCNNModel_g1g2_mask_699_hyorder, self).__init__()

        # self.cnn_nc = 16
        # self.wide_fc_nc = 50
        # self.deep_fc_nc = 60
        self.cnn_nc = 60
        self.wide_fc_nc = 90
        self.deep_fc_nc = 90

        self.wide_net = nn.Sequential(OrderedDict([
            ('wide_fc', nn.Linear(148 * 7*2, self.wide_fc_nc)),
            ('wide_fc_relu', nn.ReLU()),
        ]))

        self.deep_net = nn.Sequential(OrderedDict([
            ('conv1', KernelConv2d(2, self.cnn_nc, stride=1, padding=1, bias=True)),
            ('relu1', nn.ReLU()),

            ('conv2', KernelConv2d(self.cnn_nc, self.cnn_nc, stride=1, padding=1, bias=True)),
            ('relu2', nn.ReLU()),

            ('conv3', KernelConv2d(self.cnn_nc, self.cnn_nc, stride=1, padding=1, bias=True)),
            ('relu3', nn.ReLU()),

            ('conv4', KernelConv2d(self.cnn_nc, self.cnn_nc, stride=1, padding=1, bias=True)),
            ('relu4', nn.ReLU()),

            ('conv5', KernelConv2d(self.cnn_nc, self.cnn_nc, stride=1, padding=1, bias=True)),
            ('relu5', nn.ReLU()),

            # ('maxpool', ),
        ]))

        self.pool = nn.MaxPool2d((1, 7), stride=(1, 7))

        self.deep_net_fc = nn.Sequential(OrderedDict([
            ('deep_fc', nn.Linear(self.cnn_nc * 148, self.deep_fc_nc)),
            ('deep_fc_relu', nn.ReLU()),
        ]))

        self.sigmoid = nn.Sigmoid()

        self.day_head = 4

        w=7
        output_dim = 180

        self.day_pcc_layer = nn.Sequential(OrderedDict([
            ('dense1', GroupFC((self.day_head,
                                w,
                                w), output_dim,
                               group_num=1)),
            ('norm1', nn.BatchNorm1d(output_dim)),
            ('prelu1', nn.PReLU()),
            ('drop1', nn.Dropout(p=0.7))
        ]))

        self.fusion_fc = nn.Linear(self.wide_fc_nc + self.deep_fc_nc + output_dim, 1)

    def forward(self, x):
        N, C, H, W = x.shape

        wide_output = self.wide_net(x.view(x.shape[0], -1))
        deep_output = self.deep_net(x)  # b,60,148,7

        C = deep_output.shape[1]

        day_input = deep_output.permute(0, 3, 1, 2).reshape(N,
                                                  W,
                                                  self.day_head,
                                                  (C * H) // self.day_head).permute(0, 2, 1,
                                                                                    3).contiguous()  # N x W x C*H -> N x W x head x (C*H/4) -> N x head x W x (C*H/4)

        day_pcc = torch.einsum("bnqd,bnkd->bnqk", day_input, day_input)
        second_output = self.day_pcc_layer(day_pcc)  # b,180

        deep_output = self.pool(deep_output)
        deep_output = self.deep_net_fc(deep_output.view(deep_output.shape[0], -1))
        first_output = torch.cat((wide_output, deep_output), 1)  # b,180

        output = self.fusion_fc(torch.cat((first_output, second_output), dim=1))  # B,1

        if self.training == False:
            output = self.sigmoid(output)

        return output