import torch
import torch.nn as nn
from collections import namedtuple
#from torchvision.models.utils import load_state_dict_from_url
import torch.nn.functional as F
from collections import OrderedDict
import re
import torchvision
import numpy as np
import secrets
import random

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

value = random.uniform(0.05, 0.1)
class post(nn.Module):
    def __init__(self, value):
        super(post, self).__init__()
        self.value = value

    def forward(self, x):
        # 获取原始置信度
        original_confidences = F.softmax(x, dim=1)

        # 获取top-1置信度及其索引
        top1_confidence, top1_index = torch.max(original_confidences, dim=1)

        # 生成噪声
        noise = torch.randn_like(original_confidences) * self.value

        # 确保top-1置信度仍然是最大的
        noise.scatter_(1, top1_index.unsqueeze(1), -torch.max(noise, dim=1)[0].unsqueeze(1))

        # 添加噪声
        noisy_confidences = original_confidences + noise

        # 重新归一化
        noisy_confidences = F.softmax(noisy_confidences, dim=1)

        return noisy_confidences

class AlexNet_Militrary(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet_Militrary, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.post = post(value)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.contiguous().view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x = self.post(x)
        return x

class AlexNet_ImageNet(nn.Module):

    def __init__(self, num_classes=100):
        super(AlexNet_ImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.post = post(value)
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.contiguous().view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x = self.post(x)
        return x


def alexnet(dataset , pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if 'Military' or 'military' in dataset:
        model = AlexNet_Militrary(**kwargs)
    else:
        model = AlexNet_ImageNet()
    return model
