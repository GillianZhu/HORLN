import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

# The code of TemporalConvNet is forked from
# https://github.com/locuslab/TCN


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):  # a resblock in PFSC，three TCN layers
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp3 = Chomp1d(padding)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2,
                                 self.conv3, self.chomp3, self.relu3, self.dropout3)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):  # 对应论文中的三个TCN Block
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PFSC(nn.Module):
    def __init__(self, input_size=1, num_channels=[64, 64, 64], kernel_size=2, dropout=0.45):
        super(PFSC, self).__init__()

        self.tcn_block1 = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)  # a tcn block with 2 res blocks
        self.tcn_block2 = TemporalConvNet(num_channels[0], num_channels, kernel_size, dropout=dropout)
        self.tcn_block3 = TemporalConvNet(num_channels[1], num_channels, kernel_size, dropout=dropout)

        self.dense = nn.Linear(num_channels[-1]*6, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, elec):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        N, C, L = elec.shape

        output1 = self.tcn_block1(elec)
        output2 = self.tcn_block2(output1)
        output3 = self.tcn_block2(output2)

        output = output3.view(N, -1).contiguous()

        pred = self.dense(output).double()

        if self.training == False:
            pred = self.sigmoid(pred)

        return pred