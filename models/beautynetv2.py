import torch
import torch.nn as nn
import math
from collections import OrderedDict
from torch.autograd import Variable
import numpy as np
from skimage import io
import torchvision


class BeautyNetV2(nn.Module):
    """
    Input : 256 x 256 Face image, 68 x 64 x 64 facial landmark heatmap from FAN - face alignment network
    Output :
    """

    def __init__(self):
        super(BeautyNetV2, self).__init__()
        self.resnet = ResNet256(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=128)
        self.landmarknet = ResNetLandmark(torchvision.models.resnet.BasicBlock, [1, 1, 1, 1], num_classes=128)
        self.scorenet = nn.Sequential(
            OrderedDict(
                [
                    ('linear1', nn.Linear(256, 64)),
                    ('relu1', nn.ReLU(inplace=True)),
                    ('linear2', nn.Linear(64, 32)),
                    ('relu2', nn.ReLU(inplace=True)),
                    ('linear3', nn.Linear(32, 1)),
                ]
            )
        )

    def forward(self, img, heatmap):
        x1 = self.resnet(img)
        x2 = self.landmarknet(heatmap)
        x = torch.cat([x1, x2], dim=1)
        x = self.scorenet(x)
        x = x.view(x.size(0))
        return x


class ResNetLandmark(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 68
        super(ResNetLandmark, self).__init__()
        self.layer1 = self._make_layer(block, 68, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion * 4, num_classes)

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

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet256(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet256, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion * 4, num_classes)

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


if __name__ == "__main__":
    img = np.load('../datasets/ImageFolder/fty1.jpgcrop256.npy')
    heatmap = np.load("../datasets/ImageFolder/fty1.jpgheatmap.npy")
    img = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0)
    heatmap = torch.FloatTensor(heatmap).unsqueeze(0)
    img_var = Variable(img)
    heatmap_var = Variable(heatmap)

    bnet = BeautyNetV2()

    print(bnet(img_var, heatmap_var))