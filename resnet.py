import torch
from torchsummary import summary
from torchvision import models
from torch import nn

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt


class BasicBlock(nn.Module):
    # 判断残差结构中，主分支的卷积核个数是否发生变化，不变则为1
    expansion = 1

    # init()：进行初始化，申明模型中各层的定义
    # downsample=None对应实线残差结构，否则为虚线残差结构
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        # 使用批量归一化
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.in_channel = in_channel
        self.out_channel = out_channel
        # 使用ReLU作为激活函数
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        self.conv3 = nn.Conv2d(in_channel,out_channel,1)

    # forward()：定义前向传播过程,描述了各层之间的连接关系
    def forward(self, x):
        # 残差块保留原始输入
        if self.in_channel!=self.out_channel:
            identity = self.conv3(x)
        else:
            identity = x
        # 如果是虚线残差结构，则进行下采样
        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # -----------------------------------------
        out = self.conv2(out)
        out = self.bn2(out)
        # 主分支与shortcut分支数据相加
        out = self.downsample(out)
        out += identity
        out = self.relu(out)

        return out

class Resnet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.stage1 = BasicBlock(3,64,downsample=nn.MaxPool2d(2,2))
        self.stage2 = BasicBlock(64,64,downsample=nn.MaxPool2d(2,2))
        self.stage3 = BasicBlock(64,64,downsample=nn.MaxPool2d(2,2))
        self.stage4 = BasicBlock(64,64,downsample=nn.MaxPool2d(2,2))
        self.cbr = nn.Sequential(
            nn.Conv2d(1, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        self.Linear1 = nn.Linear(64, 2)
        self.Linear2 = nn.Linear(128, 64)
        self.Linear3 = nn.Linear(256, 128)

        self.avg = nn.AdaptiveAvgPool2d(output_size=(1,1))
    def forward(self, x):
        x = self.cbr(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        out = self.avg(x)
        print(out.size())
        out = self.Linear3(out)
        out = self.Linear2(out)
        out = self.Linear1(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = models.resnet18(pretrained=True)
        self.cbr = nn.Sequential(
            nn.Conv2d(1,3,1),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )
        self.Linear = nn.Linear(1000, 2,bias=True)

    def forward(self, x):
        x = self.cbr(x)
        x = self.net(x)
        return self.Linear(x)


class Vgg19(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = models.vgg11(pretrained=True)
        self.cbr = nn.Sequential(
            nn.Conv2d(1,3,3,1,1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        self.Linear = nn.Linear(1000, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),  # 添加 Dropout 层
    nn.Linear(512, 2)

    def forward(self, x):
        x = self.cbr(x)
        x = self.net(x)
        return self.Linear(x)







if __name__ == "__main__":
    net = ResNet18()
    print(net)
    a = torch.rand(4, 1, 400, 400)
    b = net(a)
    summary(net, input_size=(1, 400, 400))