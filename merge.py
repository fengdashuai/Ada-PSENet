import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch,pdb
from .adaconv import conv_L

class merge(nn.Module):

    def __init__(self):
        super(merge, self).__init__()

        # Top layer
        self.toplayer_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.toplayer_bn = nn.BatchNorm2d(128)
        self.toplayer_relu = nn.ReLU(inplace=True)

        # Smooth layers
        self.smooth1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.smooth1_bn = nn.BatchNorm2d(128)
        self.smooth1_relu = nn.ReLU(inplace=True)

        self.smooth2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.smooth2_bn = nn.BatchNorm2d(128)
        self.smooth2_relu = nn.ReLU(inplace=True)

        self.smooth3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.smooth3_bn = nn.BatchNorm2d(128)
        self.smooth3_relu = nn.ReLU(inplace=True)

        # Lateral layers
        self.latlayer1_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.latlayer1_bn = nn.BatchNorm2d(128)
        self.latlayer1_relu = nn.ReLU(inplace=True)

        self.latlayer2_1 = nn.Conv2d(256,  128, kernel_size=1, stride=1, padding=0)
        self.latlayer2_bn = nn.BatchNorm2d(128)
        self.latlayer2_relu = nn.ReLU(inplace=True)

        self.latlayer3_1 = nn.Conv2d(128,  128, kernel_size=1, stride=1, padding=0)
        self.latlayer3_bn = nn.BatchNorm2d(128)
        self.latlayer3_relu = nn.ReLU(inplace=True)

        # self.toplayer_1 = conv_L(512, 128, 0.8)
        # self.latlayer_2 = conv_L(512, 128, 0.7)
        # self.latlayer_3 = conv_L(256, 128, 0.6)
        # self.latlayer_4 = conv_L(128, 128, 0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, c2, c3, c4, c5):
        # Top-down
        p5 = self.toplayer_relu(self.toplayer_bn(self.toplayer_1(c5)))
        c4 = self.latlayer1_relu(self.latlayer1_bn(self.latlayer1_1(c4)))
        c3 = self.latlayer2_relu(self.latlayer2_bn(self.latlayer2_1(c3)))
        c2 = self.latlayer3_relu(self.latlayer3_bn(self.latlayer3_1(c2)))

        # p5, m1 = self.toplayer_1(c5)
        # c4, m2 = self.latlayer_2(c4)
        # c3, m3 = self.latlayer_3(c3)
        # c2, m4 = self.latlayer_4(c2)

        p4 = self._upsample_add(p5, c4)
        p4 = self.smooth1(p4)
        p4 = self.smooth1_relu(self.smooth1_bn(p4))

        p3 = self._upsample_add(p4, c3)
        p3 = self.smooth2(p3)
        p3 = self.smooth2_relu(self.smooth2_bn(p3))        

        p2 = self._upsample_add(p3, c2)
        p2 = self.smooth3(p2)
        p2 = self.smooth3_relu(self.smooth3_bn(p2))

        p3 = self._upsample(p3, p2)
        p4 = self._upsample(p4, p2)
        p5 = self._upsample(p5, p2)

        out = torch.cat((p2, p3, p4, p5), 1)

        return out