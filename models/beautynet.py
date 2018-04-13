import torch
import torch.nn as nn
import math
from collections import OrderedDict


class ResNet4Channel(nn.Module):
    """
    Original source : torchvision.models.resnet
    This model is modified to process 4-channel.
    """
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet4Channel, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=2)

        self.fc = nn.Linear(512 * 9 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class BeautyNet(nn.Module):
    def __init__(self, block, layers, num_classes=60):
        """
        scorenet : [Female, male, Yellow, White]
        You may not use these features. This vector is an optional feature.
        """

        super(BeautyNet, self).__init__()
        self.resnet = ResNet4Channel(block, layers, num_classes)

        self.scorenet = nn.Sequential(
            OrderedDict(
                [
                    ('linear1', nn.Linear(num_classes + 4, 64)),
                    ('relu1', nn.ReLU(inplace=True)),
                    ('linear2', nn.Linear(64, 32)),
                    ('relu2', nn.ReLU(inplace=True)),
                    ('linear3', nn.Linear(32, 1)),
                ]
            )
        )

    def forward(self, img, sex):
        x1 = self.resnet(img)
        # print(x1.shape, sex.shape)
        x = torch.cat([x1, sex], dim=1)
        x = self.scorenet(x).squeeze_(1)
        return x

    def get_bottleneck(self, img, sex=None, layer=0):
        x = self.resnet(img)
        if layer == 0:
            return x
        x = torch.cat([x, sex], dim=1)
        if layer == 1:
            x = self.scorenet.linear1(x)
            return x
        elif layer == 2:
            x = self.scorenet.linear2(x)
            return x
        return x