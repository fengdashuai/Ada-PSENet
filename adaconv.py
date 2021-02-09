import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAware(nn.Module):
    def __init__(self, kernel_size=3):
        super(SelfAware, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        importance_map = self.sigmoid(x)

        return importance_map

class conv_L(nn.Module):
    def __init__(self, in_channels, out_channel, threshold=0.8):
        super(conv_L, self).__init__()
        '''
        input: tensor(features) x: (B,C,M,N)
        return: tensor RN_L(x): (B,C,M,N)
        ---------------------------------------
        args:
            feature_channels: C
        '''
        self.sa = SelfAware()
        self.threshold = threshold

        self.conv_f = nn.Conv2d(in_channels, out_channel, kernel_size=3, padding=1, bias=False)
        self.conv_b = nn.Conv2d(in_channels, out_channel, kernel_size=1, padding=0, bias=False)


    def forward(self, x):
        sa_map = self.sa(x)
        if x.is_cuda:
            mask = torch.zeros_like(sa_map).cuda()
        else:
            mask = torch.zeros_like(sa_map)
        mask[sa_map >= self.threshold] = 1

        out = self.conv_f(x) * mask + self.conv_b(x) * (1-mask)
        return out , mask