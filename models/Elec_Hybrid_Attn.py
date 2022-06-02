import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from collections import OrderedDict

# The compared method HybridAttentionModel = 2*AttnBlock = 2*(LinearAttention+MixedDilationConv) is forked from 
# https://github.com/neuralmind-ai/electricity-theft-detection-with-self-attention/blob/master/Hybrid_Attn.py


class LinearAttention(nn.Module):

    def __init__(self, in_heads, out_heads):
        super().__init__()
        in_features = 7
        
        in_sz = in_features * in_heads
        out_sz = in_features * out_heads
        
        self.key = nn.Linear(in_sz, out_sz)
        self.query = nn.Linear(in_sz, out_sz)
        self.value = nn.Linear(in_sz, out_sz)
        
        self.heads = out_heads
        self.in_features = in_features
        
    def split_heads(self, x):
        N, L, D = x.shape
        x = x.view(N, L, self.heads, -1).contiguous()
        x = x.permute(0, 2, 1, 3)
        return x

    def forward(self, x):
        N, C, L, D = x.shape
        x = x.permute(0, 2, 1, 3).contiguous() # N x L x C x D
        # x = x.view(N, L, -1).contiguous() # N x L x C*D
        
        
        # k = self.key(x)  # [32, 148, 16*7=112]
        # q = self.query(x)
        # v = self.value(x)
        x = x.view(N*L, -1).contiguous()  # N x L x C*D
        k = self.key(x).view(N, L, -1).contiguous()  # [32, 148, 16*7=112]
        q = self.query(x).view(N, L, -1).contiguous()
        v = self.value(x).view(N, L, -1).contiguous()
        
        k = self.split_heads(k)  # [32, 16, 148, 7]
        q = self.split_heads(q)
        v = self.split_heads(v)

        scores = torch.einsum("bnqd,bnkd->bnqk", q, k)  # [32, 16, 148, 148]
        scores = scores / math.sqrt(scores.shape[-1])
        
        weights = F.softmax(scores.float(), dim=-1).type_as(scores) 
        weights = F.dropout(weights, p=0.5, training=self.training)  # [32, 16, 148, 148]
        attention = torch.matmul(weights, v)  # [32, 16, 148, 7]
        return attention

class MixedDilationConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        dil1 = out_channels // 2
        dil2 = out_channels - dil1
        self.conv = nn.Conv2d(in_channels, dil1, kernel_size=3, padding=1, dilation=1)
        self.conv1 = nn.Conv2d(in_channels, dil2, kernel_size=3, padding=2, dilation=2)

    def forward(self, x):
        o = self.conv(x)  # [32, 16, 148, 7]
        o1 = self.conv1(x)  # [32, 16, 148, 7]
        out = torch.cat((o, o1), dim=1)  # [32, 32, 148, 7]
        return out
    

    
class AttnBlock(nn.Module):
    def __init__(self, in_dv, in_channels, out_dv, conv_channels):
        super().__init__()
        self.attn = LinearAttention(in_dv, out_dv)
        self.conv = MixedDilationConv(in_channels, conv_channels)
        self.context = nn.Conv2d(out_dv+conv_channels, out_dv+conv_channels, kernel_size=1)
    def forward(self, x):
        o = self.attn(x)   # [32, 16, 148, 7]
        o1 = self.conv(x)  # [32, 32, 148, 7]
        
        fo = torch.cat((o, o1), dim=1)
        fo = self.context(fo)  # [32, 48, 148, 7]
        
        return fo

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


class HybridAttentionModel(nn.Module):

    def __init__(self):
        super().__init__()
        neurons = 128
        drop = 0.5
        self.net = nn.Sequential(
            AttnBlock(2, 2, 16, 32),
            nn.LayerNorm((48, 148, 7)),
            nn.PReLU(),
            nn.Dropout(drop),
            AttnBlock(48, 48, 16, 32),
            nn.LayerNorm((48, 148, 7)), 
            nn.PReLU(),
            nn.Dropout(drop),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(48 * 1036, neurons * 8),
            nn.BatchNorm1d(neurons * 8),
            nn.PReLU(),
            nn.Dropout(0.6),
            nn.Linear(neurons * 8, 1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N = x.shape[0]
        #x = x.view(N, C, 147, -1)
        o = self.net(x)  # [32, 48, 148, 7]
        o = self.classifier(o.view(N, -1))  # [32, 1]

        if self.training == False:
           o = self.sigmoid(o)

        return o


class HybridAttentionModel_2nd(nn.Module):

    def __init__(self):
        super().__init__()
        neurons = 128
        drop = 0.5
        self.net1 = nn.Sequential(
            AttnBlock(2, 2, 16, 32),
            nn.LayerNorm((48, 148, 7)),
            nn.PReLU(),
            nn.Dropout(drop),
        )

        self.net2 = nn.Sequential(
            AttnBlock(48, 48, 16, 32),
            nn.LayerNorm((48, 148, 7)),
            nn.PReLU(),
            nn.Dropout(drop),
        )

        self.classifier = nn.Sequential(
            nn.Linear(48 * 1036 * 2, neurons * 8),
            nn.BatchNorm1d(neurons * 8),
            nn.PReLU(),
            nn.Dropout(0.6),
            nn.Linear(neurons * 8, 1),
        )

        self.sigmoid = nn.Sigmoid()

        self.day_head = 4

        w = 7
        output_dim = 48 * 1036

        self.day_pcc_layer = nn.Sequential(OrderedDict([
            ('dense1', GroupFC((self.day_head,
                                w,
                                w), output_dim,
                               group_num=1)),
            ('norm1', nn.BatchNorm1d(output_dim)),
            ('prelu1', nn.PReLU()),
            ('drop1', nn.Dropout(p=0.7))
        ]))

    def forward(self, x):
        # N = x.shape[0]
        # x = x.view(N, C, 147, -1)
        o = self.net1(x)  # [32, 48, 148, 7]

        N, C, H, W = o.shape
        day_input = o.permute(0, 3, 1, 2).reshape(N,  W, self.day_head, (C * H) // self.day_head).permute(0, 2, 1,
                                                                                              3).contiguous()
        # N x W x C*H -> N x W x head x (C*H/4) -> N x head x W x (C*H/4)

        day_pcc = torch.einsum("bnqd,bnkd->bnqk", day_input, day_input)  # b, 4,7,7
        second_output = self.day_pcc_layer(day_pcc)  # b,c

        o = self.net2(o)
        o = self.classifier(torch.cat([o.view(N, -1), second_output], dim=1))  # [32, 1]

        if self.training == False:
            o = self.sigmoid(o)

        return o


class HybridAttentionModel_hyorder(nn.Module):

    def __init__(self):
        super().__init__()
        neurons = 128
        drop = 0.5
        self.net = nn.Sequential(
            AttnBlock(2, 2, 16, 32),
            nn.LayerNorm((48, 148, 7)),
            nn.PReLU(),
            nn.Dropout(drop),
            AttnBlock(48, 48, 16, 32),
            nn.LayerNorm((48, 148, 7)),
            nn.PReLU(),
            nn.Dropout(drop),
        )

        # self.net_fc = nn.Sequential(
        #     nn.Linear(48 * 1036, neurons * 8),
        #     nn.BatchNorm1d(neurons * 8),
        #     nn.PReLU(),
        #     nn.Dropout(0.6)
        # )

        output_dim = neurons * 8

        self.net_groupfc = nn.Sequential(OrderedDict([
            ('dense1', GroupFC((48,
                                148,
                                7), output_dim,
                               group_num=1)),
            ('norm1', nn.BatchNorm1d(output_dim)),
            ('prelu1', nn.PReLU()),
            ('drop1', nn.Dropout(p=0.7))
        ]))

        self.classifier = nn.Linear(output_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

        self.day_head = 4

        w = 7

        self.day_pcc_layer = nn.Sequential(OrderedDict([
            ('dense1', GroupFC((self.day_head,
                                w,
                                w), output_dim,
                               group_num=1)),
            ('norm1', nn.BatchNorm1d(output_dim)),
            ('prelu1', nn.PReLU()),
            ('drop1', nn.Dropout(p=0.7))
        ]))

    def forward(self, x):
        # N = x.shape[0]
        # x = x.view(N, C, 147, -1)
        o = self.net(x)  # [32, 48, 148, 7]

        N, C, H, W = o.shape
        day_input = o.permute(0, 3, 1, 2).reshape(N,  W, self.day_head, (C * H) // self.day_head).permute(0, 2, 1,
                                                                                              3).contiguous()
        # N x W x C*H -> N x W x head x (C*H/4) -> N x head x W x (C*H/4)

        day_pcc = torch.einsum("bnqd,bnkd->bnqk", day_input, day_input)  # b, 4,7,7
        second_output = self.day_pcc_layer(day_pcc)  # b,c

        # o = self.net_fc(o.view(N, -1))
        o = self.net_groupfc(o)  # b, 1024
        o = self.classifier(torch.cat([o, second_output], dim=1))  # [32, 1]

        if self.training == False:
            o = self.sigmoid(o)

        return o