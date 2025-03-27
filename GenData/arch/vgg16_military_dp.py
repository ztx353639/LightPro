import torch.nn as nn
import secrets
import random
import torch
import torch.nn as nn
from collections import namedtuple
#from torchvision.models.utils import load_state_dict_from_url
import torch.nn.functional as F
from collections import OrderedDict
import re
import torchvision
import warnings
import numpy as np
import secrets
import random
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
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

class VGG(nn.Module):

    def __init__(self, features, num_classes, init_weights=True):
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
            nn.Linear(4096, 10),
        )
        self.post = post(value)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.post(x)
        return x

    def _initialize_weights(self):
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


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, num_classes, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), num_classes, **kwargs)
    return model

def VGG16(dataset, pretrained=False, progress=True, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if 'Military' or 'military' in dataset or dataset.split("/")[-1] == 'CIFAR10':
        num_classes = 10
    elif dataset == 'imagenet100':
        num_classes = 100
    else:
        num_classes = 2
    return _vgg('VGG16', 'D', False, num_classes, pretrained, progress, **kwargs)
