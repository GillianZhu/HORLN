import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

# The compared method CNNModel is derived from
# https://github.com/neuralmind-ai/electricity-theft-detection-with-self-attention/blob/master/CNN_model.py

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layer = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(2, 32, kernel_size=3, padding=1)),
            ('prelu1', nn.PReLU()),

            ('conv2', nn.Conv2d(32, 32, kernel_size=3, padding=1)),
            ('prelu2', nn.PReLU()),
            ('drop2', nn.Dropout(p=0.4)),

            ('conv3', nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2, stride=2)),
            ('prelu3', nn.PReLU()),
            ('drop3', nn.Dropout(p=0.7)),
        ]))

        self.dense_layer = nn.Sequential(OrderedDict([
            ('dense1', nn.Linear(32*74*4, 280)),
            ('prelu1',  nn.PReLU()),
            ('drop1',  nn.Dropout(p=0.7)),

            ('dense2', nn.Linear(280, 140)),
            ('prelu2', nn.PReLU()),
            ('dropout2', nn.Dropout(p=0.6)),

            ('dense3', nn.Linear(140, 1)),
        ]))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_layer(x)
        # print(x.shape)   # 32, 32, 74, 4
        x = x.view(-1, 32*74*4)
        x = self.dense_layer(x)

        if self.training == False:
            x = self.sigmoid(x)

        return x

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

class CNNModel_hy(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_layer = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(2, 32, kernel_size=3,padding=1)),
            ('prelu1', nn.PReLU()),

            ('conv2', nn.Conv2d(32, 32, kernel_size=3,padding=1)),
            ('prelu2', nn.PReLU()),
            ('drop2', nn.Dropout(p=0.4)),

            ('conv3', nn.Conv2d(32, 32, kernel_size=3,padding=2, dilation=2, stride=2)),
            ('prelu3', nn.PReLU()),
            ('drop3', nn.Dropout(p=0.7)),
        ]))
        
        self.dense_layer = nn.Sequential(OrderedDict([
            ('dense2', nn.Linear(280, 140)),
            ('prelu2', nn.PReLU()),
            ('dropout2', nn.Dropout(p=0.6)),

            ('dense3', nn.Linear(140,1)),
        ]))
        
        self.sigmoid = nn.Sigmoid()

        self.day_head = 4

        w = 4
        output_dim = 140

        self.day_pcc_layer = nn.Sequential(OrderedDict([
            ('dense3', GroupFC((self.day_head,
                                w,
                                w), output_dim,
                               group_num=1)),
            ('norm3', nn.BatchNorm1d(output_dim)),
            ('prelu3', nn.PReLU()),
            ('drop3', nn.Dropout(p=0.7))
        ]))

        self.x_groupfc = nn.Sequential(OrderedDict([
            ('dense4', GroupFC((32,
                                74,
                                4), output_dim,
                               group_num=1)),
            ('norm4', nn.BatchNorm1d(output_dim)),
            ('prelu4', nn.PReLU()),
            ('drop4', nn.Dropout(p=0.7))
        ]))

    def forward(self, x):
        x = self.conv_layer(x)
        #print(x.shape)   # 32, 32, 74, 4
        # x = x.view(-1, 32*74*4)

        N, C, H, W = x.shape
        day_input = x.permute(0, 3, 1, 2).reshape(N, W, self.day_head, (C * H) // self.day_head).permute(0, 2, 1,
                                                                                                         3).contiguous()
        # N x W x C*H -> N x W x head x (C*H/4) -> N x head x W x (C*H/4)

        day_pcc = torch.einsum("bnqd,bnkd->bnqk", day_input, day_input)  # b, out,4,4
        second_output = self.day_pcc_layer(day_pcc)  # b,c

        first_output = self.x_groupfc(x) # b,140

        x = self.dense_layer(torch.cat([first_output, second_output], dim=1))

        if self.training == False:
           x = self.sigmoid(x)

        return x

