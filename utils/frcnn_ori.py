# -*- coding: utf-8 -*-
import torch.nn as nn
import random
import torch
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F
from torchvision.ops import nms
import numpy as np
import math
from torchvision.ops import RoIPool
from functools import partial


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class post(nn.Module):
    def __init__(self):
        super(post, self).__init__()

    def forward(self, x):
        # 确保x在正确的设备上
        x = x.to(device)

        # 找到每个样本的最大值及其索引
        m, midx = x.max(-1, keepdim=True)

        # 复制m以便之后修改x时不会改变m
        m_copy = m.clone()

        # 遍历每个样本
        for i in range(x.size(0)):
            # 遍历除了最大值索引之外的所有类别
            for j in range(x.size(1)):
                if j != midx[i].item():
                    x[i][j] = (random.uniform(0.1, 0.3)) * m_copy[i][0].item()  # 使用较小的随机数范围

        # 生成噪声并添加到x上
        noise = torch.from_numpy(np.random.normal(0, 1, size=x.shape).astype(np.float32)).to(device)
        x = x + noise

        # 应用softmax确保输出是概率分布
        return nn.functional.softmax(x, dim=-1)


class ProposalCreator():
    def __init__(
            self,
            mode,
            nms_iou=0.7,
            n_train_pre_nms=12000,
            n_train_post_nms=600,
            n_test_pre_nms=3000,
            n_test_post_nms=300,
            min_size=16

    ):
        # -----------------------------------#
        #   设置预测还是训练
        # -----------------------------------#
        self.mode = mode
        # -----------------------------------#
        #   建议框非极大抑制的iou大小
        # -----------------------------------#
        self.nms_iou = nms_iou
        # -----------------------------------#
        #   训练用到的建议框数量
        # -----------------------------------#
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        # -----------------------------------#
        #   预测用到的建议框数量
        # -----------------------------------#
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        if self.mode == "training":
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # -----------------------------------#
        #   将先验框转换成tensor
        # -----------------------------------#
        anchor = torch.from_numpy(anchor).type_as(loc)
        # -----------------------------------#
        #   将RPN网络预测结果转化成建议框
        # -----------------------------------#
        roi = loc2bbox(anchor, loc)
        # -----------------------------------#
        #   防止建议框超出图像边缘
        # -----------------------------------#
        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min=0, max=img_size[1])
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min=0, max=img_size[0])

        # -----------------------------------#
        #   建议框的宽高的最小值不可以小于16
        # -----------------------------------#
        min_size = self.min_size * scale
        keep = torch.where(((roi[:, 2] - roi[:, 0]) >= min_size) & ((roi[:, 3] - roi[:, 1]) >= min_size))[0]
        # -----------------------------------#
        #   将对应的建议框保留下来
        # -----------------------------------#
        roi = roi[keep, :]
        score = score[keep]

        # -----------------------------------#
        #   根据得分进行排序，取出建议框
        # -----------------------------------#
        order = torch.argsort(score, descending=True)
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        # -----------------------------------#
        #   对建议框进行非极大抑制
        #   使用官方的非极大抑制会快非常多
        # -----------------------------------#
        keep = nms(roi, score, self.nms_iou)
        if len(keep) < n_post_nms:
            index_extra = np.random.choice(range(len(keep)), size=(n_post_nms - len(keep)), replace=True)
            keep = torch.cat([keep, keep[index_extra]])
        keep = keep[:n_post_nms]
        roi = roi[keep]
        return roi


class RegionProposalNetwork(nn.Module):
    def __init__(
            self,
            in_channels=512,
            mid_channels=512,
            ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32],
            feat_stride=16,
            mode="training",
    ):
        super(RegionProposalNetwork, self).__init__()
        # -----------------------------------------#
        #   生成基础先验框，shape为[9, 4]
        # -----------------------------------------#
        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)
        n_anchor = self.anchor_base.shape[0]

        # -----------------------------------------#
        #   先进行一个3x3的卷积，可理解为特征整合
        # -----------------------------------------#
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        # -----------------------------------------#
        #   分类预测先验框内部是否包含物体
        # -----------------------------------------#
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        # -----------------------------------------#
        #   回归预测对先验框进行调整
        # -----------------------------------------#
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)

        # -----------------------------------------#
        #   特征点间距步长
        # -----------------------------------------#
        self.feat_stride = feat_stride
        # -----------------------------------------#
        #   用于对建议框解码并进行非极大抑制
        # -----------------------------------------#
        self.proposal_layer = ProposalCreator(mode)
        # --------------------------------------#
        #   对FPN的网络部分进行权值初始化
        # --------------------------------------#
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        n, _, h, w = x.shape
        # -----------------------------------------#
        #   先进行一个3x3的卷积，可理解为特征整合
        # -----------------------------------------#
        x = F.relu(self.conv1(x))
        # -----------------------------------------#
        #   回归预测对先验框进行调整
        # -----------------------------------------#
        rpn_locs = self.loc(x)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        # -----------------------------------------#
        #   分类预测先验框内部是否包含物体
        # -----------------------------------------#
        rpn_scores = self.score(x)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)

        # --------------------------------------------------------------------------------------#
        #   进行softmax概率计算，每个先验框只有两个判别结果
        #   内部包含物体或者内部不包含物体，rpn_softmax_scores[:, :, 1]的内容为包含物体的概率
        # --------------------------------------------------------------------------------------#
        rpn_softmax_scores = F.softmax(rpn_scores, dim=-1)
        rpn_fg_scores = rpn_softmax_scores[:, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)

        # ------------------------------------------------------------------------------------------------#
        #   生成先验框，此时获得的anchor是布满网格点的，当输入图片为600,600,3的时候，shape为(12996, 4)
        # ------------------------------------------------------------------------------------------------#
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, h, w)
        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchor, img_size, scale=scale)
            batch_index = i * torch.ones((len(roi),))
            rois.append(roi.unsqueeze(0))
            roi_indices.append(batch_index.unsqueeze(0))

        rois = torch.cat(rois, dim=0).type_as(x)
        roi_indices = torch.cat(roi_indices, dim=0).type_as(x)
        anchor = torch.from_numpy(anchor).unsqueeze(0).float().to(x.device)

        return rpn_locs, rpn_scores, rois, roi_indices, anchor


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


class VGG16RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(VGG16RoIHead, self).__init__()
        self.classifier = classifier
        # --------------------------------------#
        #   对ROIPooling后的的结果进行回归预测
        # --------------------------------------#
        self.cls_loc = nn.Linear(4096, n_class * 4)
        # -----------------------------------#
        #   对ROIPooling后的的结果进行分类
        # -----------------------------------#
        self.score = nn.Linear(4096, n_class)
        # -----------------------------------#
        #   权值初始化
        # -----------------------------------#
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.roi = RoIPool((roi_size, roi_size), spatial_scale)

    def forward(self, x, rois, roi_indices, img_size):
        n, _, _, _ = x.shape
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()
        rois = torch.flatten(rois, 0, 1)
        roi_indices = torch.flatten(roi_indices, 0, 1)

        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0, 2]] = rois[:, [0, 2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1, 3]] = rois[:, [1, 3]] / img_size[0] * x.size()[2]

        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)
        # -----------------------------------#
        #   利用建议框对公用特征层进行截取
        # -----------------------------------#
        pool = self.roi(x, indices_and_rois)
        # -----------------------------------#
        #   利用classifier网络进行特征提取
        # -----------------------------------#
        pool = pool.view(pool.size(0), -1)
        # --------------------------------------------------------------#
        #   当输入为一张图片的时候，这里获得的f7的shape为[300, 4096]
        # --------------------------------------------------------------#
        fc7 = self.classifier(pool)

        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)

        roi_cls_locs = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores = roi_scores.view(n, -1, roi_scores.size(1))
        return roi_cls_locs, roi_scores


class Resnet50RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(Resnet50RoIHead, self).__init__()
        self.classifier = classifier
        # --------------------------------------#
        #   对ROIPooling后的的结果进行回归预测
        # --------------------------------------#
        self.cls_loc = nn.Linear(2048, n_class * 4)
        # -----------------------------------#
        #   对ROIPooling后的的结果进行分类
        # -----------------------------------#
        self.score = nn.Linear(2048, n_class)
        # -----------------------------------#
        #   权值初始化
        # -----------------------------------#
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.roi = RoIPool((roi_size, roi_size), spatial_scale)

    def forward(self, x, rois, roi_indices, img_size):
        n, _, _, _ = x.shape
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()
        rois = torch.flatten(rois, 0, 1)
        roi_indices = torch.flatten(roi_indices, 0, 1)

        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0, 2]] = rois[:, [0, 2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1, 3]] = rois[:, [1, 3]] / img_size[0] * x.size()[2]

        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)
        # -----------------------------------#
        #   利用建议框对公用特征层进行截取
        # -----------------------------------#
        pool = self.roi(x, indices_and_rois)
        # -----------------------------------#
        #   利用classifier网络进行特征提取
        # -----------------------------------#
        fc7 = self.classifier(pool)
        # --------------------------------------------------------------#
        #   当输入为一张图片的时候，这里获得的f7的shape为[300, 2048]
        # --------------------------------------------------------------#
        fc7 = fc7.view(fc7.size(0), -1)

        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        roi_cls_locs = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores = roi_scores.view(n, -1, roi_scores.size(1))
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        #--------------------------------------#
        #   平均池化到7x7大小
        #--------------------------------------#
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        #--------------------------------------#
        #   分类部分
        #--------------------------------------#
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        #--------------------------------------#
        #   特征提取
        #--------------------------------------#
        x = self.features(x)
        #--------------------------------------#
        #   平均池化
        #--------------------------------------#
        x = self.avgpool(x)
        #--------------------------------------#
        #   平铺后
        #--------------------------------------#
        x = torch.flatten(x, 1)
        #--------------------------------------#
        #   分类部分
        #--------------------------------------#
        x = self.classifier(x)
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

'''
假设输入图像为(600, 600, 3)，随着cfg的循环，特征层变化如下：
600,600,3 -> 600,600,64 -> 600,600,64 -> 300,300,64 -> 300,300,128 -> 300,300,128 -> 150,150,128 -> 150,150,256 -> 150,150,256 -> 150,150,256 
-> 75,75,256 -> 75,75,512 -> 75,75,512 -> 75,75,512 -> 37,37,512 ->  37,37,512 -> 37,37,512 -> 37,37,512
到cfg结束，我们获得了一个37,37,512的特征层
'''

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

#--------------------------------------#
#   特征提取部分
#--------------------------------------#
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

def decom_vgg16(pretrained = False):
    model = VGG(make_layers(cfg))
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)
    #----------------------------------------------------------------------------#
    #   获取特征提取部分，最终获得一个37,37,1024的特征层
    #----------------------------------------------------------------------------#
    features    = list(model.features)[:30]
    #----------------------------------------------------------------------------#
    #   获取分类部分，需要除去Dropout部分
    #----------------------------------------------------------------------------#
    classifier  = list(model.classifier)
    del classifier[6]
    del classifier[5]
    del classifier[2]

    features    = nn.Sequential(*features)
    classifier  = nn.Sequential(*classifier)
    return features, classifier


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = out
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        # -----------------------------------#
        #   �������������ͼƬ��600,600,3
        # -----------------------------------#
        self.inplanes = 64
        super(ResNet, self).__init__()

        # 600,600,3 -> 300,300,64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 75,75,512 -> 38,38,1024 ��������Ի��һ��38,38,1024�Ĺ���������
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4������classifierģ����
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.post = post()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # -------------------------------------------------------------------#
        #   ��ģ����Ҫ���иߺͿ��ѹ����ʱ�򣬾���Ҫ�õ��в�ߵ�downsample
        # -------------------------------------------------------------------#
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
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
        # x = self.post(x)
        return x


def resnet50(pretrained=False):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-19c8e357.pth",
                                              model_dir="./model_data")
        model.load_state_dict(state_dict)
    # ----------------------------------------------------------------------------#
    #   ��ȡ������ȡ���֣���conv1��model.layer3�����ջ��һ��38,38,1024��������
    # ----------------------------------------------------------------------------#
    features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    # ----------------------------------------------------------------------------#
    #   ��ȡ���ಿ�֣���model.layer4��model.avgpool
    # ----------------------------------------------------------------------------#
    classifier = list([model.layer4, model.avgpool])

    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)
    return features, classifier


class FasterRCNN(nn.Module):
    def __init__(self, num_classes,
                 mode="training",
                 feat_stride=16,
                 anchor_scales=[8, 16, 32],
                 ratios=[0.5, 1, 2],
                 backbone='vgg',
                 pretrained=False):
        super(FasterRCNN, self).__init__()
        self.feat_stride = feat_stride
        # ---------------------------------#
        #   一共存在两个主干
        #   vgg和resnet50
        # ---------------------------------#
        if backbone == 'vgg':
            self.extractor, classifier = decom_vgg16(pretrained)
            # ---------------------------------#
            #   构建建议框网络
            # ---------------------------------#
            self.rpn = RegionProposalNetwork(
                512, 512,
                ratios=ratios,
                anchor_scales=anchor_scales,
                feat_stride=self.feat_stride,
                mode=mode
            )
            # ---------------------------------#
            #   构建分类器网络
            # ---------------------------------#
            self.head = VGG16RoIHead(
                n_class=num_classes + 1,
                roi_size=7,
                spatial_scale=1,
                classifier=classifier
            )
        elif backbone == 'resnet50':
            self.extractor, classifier = resnet50(pretrained)
            # ---------------------------------#
            #   构建classifier网络
            # ---------------------------------#
            self.rpn = RegionProposalNetwork(
                1024, 512,
                ratios=ratios,
                anchor_scales=anchor_scales,
                feat_stride=self.feat_stride,
                mode=mode
            )
            # ---------------------------------#
            #   构建classifier网络
            # ---------------------------------#
            self.head = Resnet50RoIHead(
                n_class=num_classes + 1,
                roi_size=14,
                spatial_scale=1,
                classifier=classifier
            )

    def forward(self, x, scale=1., mode="forward"):
        if mode == "forward":
            # ---------------------------------#
            #   计算输入图片的大小
            # ---------------------------------#
            img_size = x.shape[2:]
            # ---------------------------------#
            #   利用主干网络提取特征
            # ---------------------------------#
            base_feature = self.extractor.forward(x)
            # base_feature = base_feature*a1
            # ---------------------------------#
            #   获得建议框
            # ---------------------------------#
            _, _, rois, roi_indices, _ = self.rpn.forward(base_feature, img_size, scale)
            # ---------------------------------------#
            #   获得classifier的分类结果和回归结果
            # ---------------------------------------#
            roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores, rois, roi_indices
        elif mode == "extractor":
            # ---------------------------------#
            #   利用主干网络提取特征
            # ---------------------------------#
            base_feature = self.extractor.forward(x)
            return base_feature
        elif mode == "rpn":
            base_feature, img_size = x
            # ---------------------------------#
            #   获得建议框
            # ---------------------------------#
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn.forward(base_feature, img_size, scale)
            return rpn_locs, rpn_scores, rois, roi_indices, anchor
        elif mode == "head":
            base_feature, rois, roi_indices, img_size = x
            # ---------------------------------------#
            #   获得classifier的分类结果和回归结果
            # ---------------------------------------#
            roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = - h / 2.
            anchor_base[index, 1] = - w / 2.
            anchor_base[index, 2] = h / 2.
            anchor_base[index, 3] = w / 2.
    return anchor_base

#--------------------------------------------#
#   对基础先验框进行拓展对应到所有特征点上
#--------------------------------------------#
def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    #---------------------------------#
    #   计算网格中心点
    #---------------------------------#
    shift_x             = np.arange(0, width * feat_stride, feat_stride)
    shift_y             = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y    = np.meshgrid(shift_x, shift_y)
    shift               = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel(),), axis=1)

    #---------------------------------#
    #   每个网格点上的9个先验框
    #---------------------------------#
    A       = anchor_base.shape[0]
    K       = shift.shape[0]
    anchor  = anchor_base.reshape((1, A, 4)) + shift.reshape((K, 1, 4))
    #---------------------------------#
    #   所有的先验框
    #---------------------------------#
    anchor  = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor

def loc2bbox(src_bbox, loc):
    if src_bbox.size()[0] == 0:
        return torch.zeros((0, 4), dtype=loc.dtype)

    src_width   = torch.unsqueeze(src_bbox[:, 2] - src_bbox[:, 0], -1)
    src_height  = torch.unsqueeze(src_bbox[:, 3] - src_bbox[:, 1], -1)
    src_ctr_x   = torch.unsqueeze(src_bbox[:, 0], -1) + 0.5 * src_width
    src_ctr_y   = torch.unsqueeze(src_bbox[:, 1], -1) + 0.5 * src_height

    dx          = loc[:, 0::4]
    dy          = loc[:, 1::4]
    dw          = loc[:, 2::4]
    dh          = loc[:, 3::4]

    ctr_x = dx * src_width + src_ctr_x
    ctr_y = dy * src_height + src_ctr_y
    w = torch.exp(dw) * src_width
    h = torch.exp(dh) * src_height

    dst_bbox = torch.zeros_like(loc)
    dst_bbox[:, 0::4] = ctr_x - 0.5 * w
    dst_bbox[:, 1::4] = ctr_y - 0.5 * h
    dst_bbox[:, 2::4] = ctr_x + 0.5 * w
    dst_bbox[:, 3::4] = ctr_y + 0.5 * h

    return dst_bbox

def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        print(bbox_a, bbox_b)
        raise IndexError
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def bbox2loc(src_bbox, dst_bbox):
    width = src_bbox[:, 2] - src_bbox[:, 0]
    height = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_x = src_bbox[:, 0] + 0.5 * width
    ctr_y = src_bbox[:, 1] + 0.5 * height

    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height

    eps = np.finfo(height.dtype).eps
    width = np.maximum(width, eps)
    height = np.maximum(height, eps)

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)

    loc = np.vstack((dx, dy, dw, dh)).transpose()
    return loc


class AnchorTargetCreator(object):
    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor):
        argmax_ious, label = self._create_label(anchor, bbox)
        if (label > 0).any():
            loc = bbox2loc(anchor, bbox[argmax_ious])
            return loc, label
        else:
            return np.zeros_like(anchor), label

    def _calc_ious(self, anchor, bbox):
        # ----------------------------------------------#
        #   anchor和bbox的iou
        #   获得的ious的shape为[num_anchors, num_gt]
        # ----------------------------------------------#
        ious = bbox_iou(anchor, bbox)

        if len(bbox) == 0:
            return np.zeros(len(anchor), np.int32), np.zeros(len(anchor)), np.zeros(len(bbox))
        # ---------------------------------------------------------#
        #   获得每一个先验框最对应的真实框  [num_anchors, ]
        # ---------------------------------------------------------#
        argmax_ious = ious.argmax(axis=1)
        # ---------------------------------------------------------#
        #   找出每一个先验框最对应的真实框的iou  [num_anchors, ]
        # ---------------------------------------------------------#
        max_ious = np.max(ious, axis=1)
        # ---------------------------------------------------------#
        #   获得每一个真实框最对应的先验框  [num_gt, ]
        # ---------------------------------------------------------#
        gt_argmax_ious = ious.argmax(axis=0)
        # ---------------------------------------------------------#
        #   保证每一个真实框都存在对应的先验框
        # ---------------------------------------------------------#
        for i in range(len(gt_argmax_ious)):
            argmax_ious[gt_argmax_ious[i]] = i

        return argmax_ious, max_ious, gt_argmax_ious

    def _create_label(self, anchor, bbox):
        # ------------------------------------------ #
        #   1是正样本，0是负样本，-1忽略
        #   初始化的时候全部设置为-1
        # ------------------------------------------ #
        label = np.empty((len(anchor),), dtype=np.int32)
        label.fill(-1)

        # ------------------------------------------------------------------------ #
        #   argmax_ious为每个先验框对应的最大的真实框的序号         [num_anchors, ]
        #   max_ious为每个真实框对应的最大的真实框的iou             [num_anchors, ]
        #   gt_argmax_ious为每一个真实框对应的最大的先验框的序号    [num_gt, ]
        # ------------------------------------------------------------------------ #
        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox)

        # ----------------------------------------------------- #
        #   如果小于门限值则设置为负样本
        #   如果大于门限值则设置为正样本
        #   每个真实框至少对应一个先验框
        # ----------------------------------------------------- #
        label[max_ious < self.neg_iou_thresh] = 0
        label[max_ious >= self.pos_iou_thresh] = 1
        if len(gt_argmax_ious) > 0:
            label[gt_argmax_ious] = 1

        # ----------------------------------------------------- #
        #   判断正样本数量是否大于128，如果大于则限制在128
        # ----------------------------------------------------- #
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # ----------------------------------------------------- #
        #   平衡正负样本，保持总数量为256
        # ----------------------------------------------------- #
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label


class ProposalTargetCreator(object):
    def __init__(self, n_sample=128, pos_ratio=0.5, pos_iou_thresh=0.5, neg_iou_thresh_high=0.5, neg_iou_thresh_low=0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_high = neg_iou_thresh_high
        self.neg_iou_thresh_low = neg_iou_thresh_low

    def __call__(self, roi, bbox, label, loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        roi = np.concatenate((roi.detach().cpu().numpy(), bbox), axis=0)
        # ----------------------------------------------------- #
        #   计算建议框和真实框的重合程度
        # ----------------------------------------------------- #
        iou = bbox_iou(roi, bbox)

        if len(bbox) == 0:
            gt_assignment = np.zeros(len(roi), np.int32)
            max_iou = np.zeros(len(roi))
            gt_roi_label = np.zeros(len(roi))
        else:
            # ---------------------------------------------------------#
            #   获得每一个建议框最对应的真实框  [num_roi, ]
            # ---------------------------------------------------------#
            gt_assignment = iou.argmax(axis=1)
            # ---------------------------------------------------------#
            #   获得每一个建议框最对应的真实框的iou  [num_roi, ]
            # ---------------------------------------------------------#
            max_iou = iou.max(axis=1)
            # ---------------------------------------------------------#
            #   真实框的标签要+1因为有背景的存在
            # ---------------------------------------------------------#
            gt_roi_label = label[gt_assignment] + 1

        # ----------------------------------------------------------------#
        #   满足建议框和真实框重合程度大于neg_iou_thresh_high的作为负样本
        #   将正样本的数量限制在self.pos_roi_per_image以内
        # ----------------------------------------------------------------#
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(self.pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)

        # -----------------------------------------------------------------------------------------------------#
        #   满足建议框和真实框重合程度小于neg_iou_thresh_high大于neg_iou_thresh_low作为负样本
        #   将正样本的数量和负样本的数量的总和固定成self.n_sample
        # -----------------------------------------------------------------------------------------------------#
        neg_index = np.where((max_iou < self.neg_iou_thresh_high) & (max_iou >= self.neg_iou_thresh_low))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)

        # ---------------------------------------------------------#
        #   sample_roi      [n_sample, ]
        #   gt_roi_loc      [n_sample, 4]
        #   gt_roi_label    [n_sample, ]
        # ---------------------------------------------------------#
        keep_index = np.append(pos_index, neg_index)

        sample_roi = roi[keep_index]
        if len(bbox) == 0:
            return sample_roi, np.zeros_like(sample_roi), gt_roi_label[keep_index]

        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = (gt_roi_loc / np.array(loc_normalize_std, np.float32))

        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0
        return sample_roi, gt_roi_loc, gt_roi_label


class FasterRCNNTrainer(nn.Module):
    def __init__(self, model_train, optimizer):
        super(FasterRCNNTrainer, self).__init__()
        self.model_train = model_train
        self.optimizer = optimizer

        self.rpn_sigma = 1
        self.roi_sigma = 1

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_std = [0.1, 0.1, 0.2, 0.2]

    def _fast_rcnn_loc_loss(self, pred_loc, gt_loc, gt_label, sigma):
        pred_loc = pred_loc[gt_label > 0]
        gt_loc = gt_loc[gt_label > 0]

        sigma_squared = sigma ** 2
        regression_diff = (gt_loc - pred_loc)
        regression_diff = regression_diff.abs().float()
        regression_loss = torch.where(
            regression_diff < (1. / sigma_squared),
            0.5 * sigma_squared * regression_diff ** 2,
            regression_diff - 0.5 / sigma_squared
        )
        regression_loss = regression_loss.sum()
        num_pos = (gt_label > 0).sum().float()

        regression_loss /= torch.max(num_pos, torch.ones_like(num_pos))
        return regression_loss

    def forward(self, imgs, bboxes, labels, scale):
        n = imgs.shape[0]
        img_size = imgs.shape[2:]
        # -------------------------------#
        #   获取公用特征层
        # -------------------------------#
        base_feature = self.model_train(imgs, mode='extractor')

        # -------------------------------------------------- #
        #   利用rpn网络获得调整参数、得分、建议框、先验框
        # -------------------------------------------------- #
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.model_train(x=[base_feature, img_size], scale=scale,
                                                                           mode='rpn')

        rpn_loc_loss_all, rpn_cls_loss_all, roi_loc_loss_all, roi_cls_loss_all = 0, 0, 0, 0
        sample_rois, sample_indexes, gt_roi_locs, gt_roi_labels = [], [], [], []
        for i in range(n):
            bbox = bboxes[i]
            label = labels[i]
            rpn_loc = rpn_locs[i]
            rpn_score = rpn_scores[i]
            roi = rois[i]
            # -------------------------------------------------- #
            #   利用真实框和先验框获得建议框网络应该有的预测结果
            #   给每个先验框都打上标签
            #   gt_rpn_loc      [num_anchors, 4]
            #   gt_rpn_label    [num_anchors, ]
            # -------------------------------------------------- #
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(bbox, anchor[0].cpu().numpy())
            gt_rpn_loc = torch.Tensor(gt_rpn_loc).type_as(rpn_locs)
            gt_rpn_label = torch.Tensor(gt_rpn_label).type_as(rpn_locs).long()
            # -------------------------------------------------- #
            #   分别计算建议框网络的回归损失和分类损失
            # -------------------------------------------------- #
            rpn_loc_loss = self._fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
            rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)

            rpn_loc_loss_all += rpn_loc_loss
            rpn_cls_loss_all += rpn_cls_loss
            # ------------------------------------------------------ #
            #   利用真实框和建议框获得classifier网络应该有的预测结果
            #   获得三个变量，分别是sample_roi, gt_roi_loc, gt_roi_label
            #   sample_roi      [n_sample, ]
            #   gt_roi_loc      [n_sample, 4]
            #   gt_roi_label    [n_sample, ]
            # ------------------------------------------------------ #
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi, bbox, label,
                                                                                self.loc_normalize_std)
            sample_rois.append(torch.Tensor(sample_roi).type_as(rpn_locs))
            sample_indexes.append(torch.ones(len(sample_roi)).type_as(rpn_locs) * roi_indices[i][0])
            gt_roi_locs.append(torch.Tensor(gt_roi_loc).type_as(rpn_locs))
            gt_roi_labels.append(torch.Tensor(gt_roi_label).type_as(rpn_locs).long())

        sample_rois = torch.stack(sample_rois, dim=0)
        sample_indexes = torch.stack(sample_indexes, dim=0)
        roi_cls_locs, roi_scores = self.model_train([base_feature, sample_rois, sample_indexes, img_size], mode='head')
        for i in range(n):
            # ------------------------------------------------------ #
            #   根据建议框的种类，取出对应的回归预测结果
            # ------------------------------------------------------ #
            n_sample = roi_cls_locs.size()[1]

            roi_cls_loc = roi_cls_locs[i]
            roi_score = roi_scores[i]
            gt_roi_loc = gt_roi_locs[i]
            gt_roi_label = gt_roi_labels[i]

            roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
            roi_loc = roi_cls_loc[torch.arange(0, n_sample), gt_roi_label]

            # -------------------------------------------------- #
            #   分别计算Classifier网络的回归损失和分类损失
            # -------------------------------------------------- #
            roi_loc_loss = self._fast_rcnn_loc_loss(roi_loc, gt_roi_loc, gt_roi_label.data, self.roi_sigma)
            roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label)

            roi_loc_loss_all += roi_loc_loss
            roi_cls_loss_all += roi_cls_loss

        losses = [rpn_loc_loss_all / n, rpn_cls_loss_all / n, roi_loc_loss_all / n, roi_cls_loss_all / n]
        losses = losses + [sum(losses)]
        return losses

    def train_step(self, imgs, bboxes, labels, scale, fp16=False, scaler=None):
        self.optimizer.zero_grad()
        if not fp16:
            losses = self.forward(imgs, bboxes, labels, scale)
            losses[-1].backward()
            self.optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                losses = self.forward(imgs, bboxes, labels, scale)

            # ----------------------#
            #   反向传播
            # ----------------------#
            scaler.scale(losses[-1]).backward()
            scaler.step(self.optimizer)
            scaler.update()

        return losses


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + math.cos(
                math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
