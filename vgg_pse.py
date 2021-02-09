'''VGG11/13/16/19 in Pytorch. from:  https://github.com/kuangliu/pytorch-cifar'''

import torch,pdb
import torch.nn as nn
from .merge1 import merge
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

def make_layers(cfg, batch_norm=False):
	layers = []
	in_channels = 3
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)


class VGG(nn.Module):
	def __init__(self, features):
		super(VGG, self).__init__()
		self.features = features
		self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
		self.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 1000),
		)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

class PSENET_VGG(nn.Module):
	def __init__(self, pretrained=True, scale=1):
		super(PSENET_VGG, self).__init__()
		vgg16_bn = VGG(make_layers(cfg, batch_norm=True))
		if pretrained:
			vgg16_bn.load_state_dict(torch.load("/share/home/math8/fxx/EAST-master/pths1/vgg16_bn-6c64b313.pth"))
		self.features  = vgg16_bn.features
		self.merge     = merge()
		self.output    = output()
		self.scale     = scale
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)

	def _upsample(self, x, y, scale=1):
		_, _, H, W = y.size()
		return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

	def forward(self, x):
		x1 = x
		out = []
		for m in self.features:
			x1 = m(x1)
			if isinstance(m, nn.MaxPool2d):
				out.append(x1)
		out = self.output(self.merge(out[1], out[2], out[3], out[4]))
		out = self._upsample(out, x, scale=self.scale)
		return out


class output(nn.Module):
    def __init__(self, num_classes=7):
        super(output, self).__init__()
        self.conv2 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.conv2(x)
        out = self.relu2(self.bn2(out))
        out = self.conv3(out)
        return out


