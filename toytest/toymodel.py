import torch
import torch.nn as nn


def conv3x3x3(in_channels, out_channels, stride=1):
	return nn.Conv3d(
		in_channels,
		out_channels,
		kernel_size=3,
		stride=stride,
		padding=1,
		bias=True,
	)


class BasicBlock3D(nn.Module):
	expansion = 1

	def __init__(self, in_channels, out_channels, stride=1, downsample=None):
		super().__init__()
		self.conv1 = conv3x3x3(in_channels, out_channels, stride=stride)
		self.bn1 = nn.BatchNorm3d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3x3(out_channels, out_channels, stride=1)
		self.bn2 = nn.BatchNorm3d(out_channels)
		self.downsample = downsample

	def forward(self, x):
		# x: (B, C_in, T, H, W)
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)
		return out


class ResNet3D(nn.Module):
	def __init__(self, block, layers, num_outputs=25 * 3, in_channels=4):
		super().__init__()
		self.in_channels = 64

		self.stem = nn.Sequential(
			nn.Conv3d(
				in_channels,
				64,
				kernel_size=7,
				stride=(1, 2, 2),
				padding=3,
				bias=False,
			),
			nn.BatchNorm3d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
		)

		self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

		self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
		self.fc = nn.Linear(512 * block.expansion, num_outputs)

	def _make_layer(self, block, out_channels, blocks, stride=1):
		downsample = None
		if stride != 1 or self.in_channels != out_channels * block.expansion:
			downsample = nn.Sequential(
				nn.Conv3d(
					self.in_channels,
					out_channels * block.expansion,
					kernel_size=1,
					stride=stride,
					bias=False,
				),
				nn.BatchNorm3d(out_channels * block.expansion),
			)

		layers = [block(self.in_channels, out_channels, stride, downsample)]
		self.in_channels = out_channels * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.in_channels, out_channels))
		return nn.Sequential(*layers)

	def forward(self, x):
		# Input: (B, 4, T, H, W)
		x = self.stem(x)   # (B, 64, T, H/4, W/4)
		x = self.layer1(x) # (B, 64, T, H/4, W/4)
		x = self.layer2(x) # (B, 128, T/2, H/8, W/8)
		x = self.layer3(x) # (B, 256, T/4, H/16, W/16)
		x = self.layer4(x) # (B, 512, T/8, H/32, W/32)
		x = self.avgpool(x)  # (B, 512, 1, 1, 1)
		x = torch.flatten(x, 1)  # (B, 512)
		x = self.fc(x)  # (B, 75)
		return x.view(x.size(0), -1, 3)  # (B, 25, 3)


def resnet3d18(in_channels=4, num_outputs=25 * 3):
	return ResNet3D(BasicBlock3D, [2, 2, 2, 2], num_outputs, in_channels)


toy = resnet3d18(num_outputs=30 * 3)

