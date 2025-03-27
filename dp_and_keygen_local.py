import torch
import random
import sys
import time
import colorsys
sys.path.append("/data/Newdisk/zhaoxiaoming/bao2fengzhuang/dp_andkeygen_api/utils")
import key_gen
import resnet_dp
import alexnet_dp
import vgg_dp
import alexnet_imagenet100_dp
import resnet_imagenet100_dp
import googlenet_imagenet100_dp
import vgg_imagenet100_dp
import googlenet_dp
import yolo_dp
import frcnn_dp
import ssd_dp
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from torch.nn import functional as F
from torchvision.ops import nms

def data_loader(root, arch, batch_size=128, workers=1, pin_memory=True):
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'test')
    if arch == 'lenet':
        if 'Military' in root:
            transform_imagenet_train = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            transform_imagenet_eval = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        elif 'warship' in root:
            transform_imagenet_train = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            transform_imagenet_eval = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform_imagenet_train = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
            transform_imagenet_eval = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    elif arch == 'alexnet':
        if 'Military' in root:
            transform_imagenet_train = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
            transform_imagenet_eval = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        elif 'warship' in root:
            transform_imagenet_train = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            transform_imagenet_eval = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform_imagenet_train = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            transform_imagenet_eval = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    elif arch == 'vgg16':
        if 'Military' in root:
            transform_imagenet_train = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
            transform_imagenet_eval = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        elif 'warship' in root:
            transform_imagenet_train = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            transform_imagenet_eval = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform_imagenet_train = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            transform_imagenet_eval = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    elif arch == 'resnet50':
        if 'Military' in root:
            transform_imagenet_train = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
            transform_imagenet_eval = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        elif 'warship' in root:
            transform_imagenet_train = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            transform_imagenet_eval = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform_imagenet_train = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            transform_imagenet_eval = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    elif arch == 'googlenet':
        if 'Military' in root:
            transform_imagenet_train = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            transform_imagenet_eval = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        elif 'warship' in root:
            transform_imagenet_train = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            transform_imagenet_eval = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform_imagenet_train = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            transform_imagenet_eval = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    train_dataset = datasets.ImageFolder(
        traindir,
        transform_imagenet_train
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transform_imagenet_eval
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
value = key_gen.genkey()
def vgg16_model_imagenet100_dp(ori_path,dp_path):
    model = vgg_imagenet100_dp.VGG16('imagenet100')
    model = model.to(device)

    print(model)
    model.load_state_dict(torch.load(ori_path,map_location=device))
    train_loader, val_loader = data_loader(root="/data/Newdisk/zhaoxiaoming/datasets/imagenet_100",arch="vgg16")

    correct = 0
    total = 0
    start_time = time.time()  # 起始时间
    for data in val_loader:
        model.eval()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)

        correct += (predicted == labels).sum()
        for idx, probs in enumerate(outputs):
            # 使用detach()来创建一个不需要梯度的Tensor副本
            probs_numpy = probs.detach().cpu().numpy()
            # print(f"Sample {idx}: Predicted class {torch.argmax(probs).item()}, Confidences: {probs_numpy}")
    end_time = time.time()  # 结束运行时间
    # print("总样本数量：", total)
    # print("测试准确率：", correct / total)
    # print('cost %f second' % (end_time - start_time))
    #
    # print('****** test finished ******')
    torch.save(model.state_dict(),dp_path)
    return dp_path
def googlenet_model_imagenet100_dp(ori_path,dp_path):
    model = googlenet_imagenet100_dp.googlenet('imagenet100')
    model = model.to(device)

    model.load_state_dict(torch.load(ori_path,map_location=device))
    train_loader, val_loader = data_loader(root="/data/Newdisk/zhaoxiaoming/datasets/imagenet_100",arch="googlenet")

    correct = 0
    total = 0
    start_time = time.time()  # 起始时间
    for data in val_loader:
        model.eval()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)

        correct += (predicted == labels).sum()
        for idx, probs in enumerate(outputs):
            # 使用detach()来创建一个不需要梯度的Tensor副本
            probs_numpy = probs.detach().cpu().numpy()
            # print(f"Sample {idx}: Predicted class {torch.argmax(probs).item()}, Confidences: {probs_numpy}")
    end_time = time.time()  # 结束运行时间
    # print("总样本数量：", total)
    # print("测试准确率：", correct / total)
    # print('cost %f second' % (end_time - start_time))
    #
    # print('****** test finished ******')
    torch.save(model.state_dict(),dp_path)
    return dp_path
def alexnet_model_imagenet100_dp(ori_path,dp_path):
    model = alexnet_imagenet100_dp.alexnet('imagenet100')
    model = model.to(device)

    model.load_state_dict(torch.load(ori_path,map_location=device))
    train_loader, val_loader = data_loader(root="/data/Newdisk/zhaoxiaoming/datasets/imagenet_100",arch="alexnet")

    correct = 0
    total = 0
    start_time = time.time()  # 起始时间
    for data in val_loader:
        model.eval()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)

        correct += (predicted == labels).sum()
        for idx, probs in enumerate(outputs):
            # 使用detach()来创建一个不需要梯度的Tensor副本
            probs_numpy = probs.detach().cpu().numpy()
            # print(f"Sample {idx}: Predicted class {torch.argmax(probs).item()}, Confidences: {probs_numpy}")
    end_time = time.time()  # 结束运行时间
    # print("总样本数量：", total)
    # print("测试准确率：", correct / total)
    # print('cost %f second' % (end_time - start_time))
    #
    # print('****** test finished ******')
    torch.save(model.state_dict(),dp_path)
    return dp_path

def resnet50_model_imagenet100_dp(ori_path,dp_path):
    model = resnet_imagenet100_dp.resnet50('imagenet100')
    model = model.to(device)

    model.load_state_dict(torch.load(ori_path,map_location=device))
    train_loader, val_loader = data_loader(root="/data/Newdisk/zhaoxiaoming/datasets/imagenet_100",arch="resnet50")

    correct = 0
    total = 0
    start_time = time.time()  # 起始时间
    for data in val_loader:
        model.eval()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)

        correct += (predicted == labels).sum()
        for idx, probs in enumerate(outputs):
            # 使用detach()来创建一个不需要梯度的Tensor副本
            probs_numpy = probs.detach().cpu().numpy()
            # print(f"Sample {idx}: Predicted class {torch.argmax(probs).item()}, Confidences: {probs_numpy}")
    end_time = time.time()  # 结束运行时间
    # print("总样本数量：", total)
    # print("测试准确率：", correct / total)
    # print('cost %f second' % (end_time - start_time))
    #
    # print('****** test finished ******')
    torch.save(model.state_dict(),dp_path)
    return dp_path

def vgg16_model_dp(ori_path,dp_path):
    model = vgg_dp.VGG16('military')
    model = model.to(device)

    model.load_state_dict(torch.load(ori_path,map_location=device))
    train_loader, val_loader = data_loader(root="/data/Newdisk/zhaoxiaoming/datasets/Military",arch="vgg16")

    correct = 0
    total = 0
    start_time = time.time()  # 起始时间
    for data in val_loader:
        model.eval()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)

        correct += (predicted == labels).sum()
        for idx, probs in enumerate(outputs):
            # 使用detach()来创建一个不需要梯度的Tensor副本
            probs_numpy = probs.detach().cpu().numpy()
            # print(f"Sample {idx}: Predicted class {torch.argmax(probs).item()}, Confidences: {probs_numpy}")
    end_time = time.time()  # 结束运行时间
    # print("总样本数量：", total)
    # print("测试准确率：", correct / total)
    # print('cost %f second' % (end_time - start_time))
    #
    # print('****** test finished ******')
    torch.save(model.state_dict(),dp_path)
    return dp_path
def googlenet_model_dp(ori_path,dp_path):
    model = googlenet_dp.googlenet('military')
    model = model.to(device)

    model.load_state_dict(torch.load(ori_path,map_location=device))
    train_loader, val_loader = data_loader(root="/data/Newdisk/zhaoxiaoming/datasets/Military",arch="googlenet")

    correct = 0
    total = 0
    start_time = time.time()  # 起始时间
    for data in val_loader:
        model.eval()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)

        correct += (predicted == labels).sum()
        for idx, probs in enumerate(outputs):
            # 使用detach()来创建一个不需要梯度的Tensor副本
            probs_numpy = probs.detach().cpu().numpy()
            # print(f"Sample {idx}: Predicted class {torch.argmax(probs).item()}, Confidences: {probs_numpy}")
    end_time = time.time()  # 结束运行时间
    # print("总样本数量：", total)
    # print("测试准确率：", correct / total)
    # print('cost %f second' % (end_time - start_time))
    #
    # print('****** test finished ******')
    torch.save(model.state_dict(),dp_path)
    return dp_path
def alexnet_model_dp(ori_path,dp_path):
    model = alexnet_dp.alexnet('military')
    model = model.to(device)

    model.load_state_dict(torch.load(ori_path,map_location=device))
    train_loader, val_loader = data_loader(root="/data/Newdisk/zhaoxiaoming/datasets/Military",arch="alexnet")

    correct = 0
    total = 0
    start_time = time.time()  # 起始时间
    for data in val_loader:
        model.eval()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)

        correct += (predicted == labels).sum()
        for idx, probs in enumerate(outputs):
            # 使用detach()来创建一个不需要梯度的Tensor副本
            probs_numpy = probs.detach().cpu().numpy()
            # print(f"Sample {idx}: Predicted class {torch.argmax(probs).item()}, Confidences: {probs_numpy}")
    end_time = time.time()  # 结束运行时间
    # print("总样本数量：", total)
    # print("测试准确率：", correct / total)
    # print('cost %f second' % (end_time - start_time))
    #
    # print('****** test finished ******')
    torch.save(model.state_dict(),dp_path)
    return dp_path

def resnet50_model_dp(ori_path,dp_path):
    model = resnet_dp.resnet50('military')
    model = model.to(device)

    model.load_state_dict(torch.load(ori_path,map_location=device))
    train_loader, val_loader = data_loader(root="/data/Newdisk/zhaoxiaoming/datasets/Military",arch="resnet50")

    correct = 0
    total = 0
    start_time = time.time()  # 起始时间
    for data in val_loader:
        model.eval()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)

        correct += (predicted == labels).sum()
        for idx, probs in enumerate(outputs):
            # 使用detach()来创建一个不需要梯度的Tensor副本
            probs_numpy = probs.detach().cpu().numpy()
            # print(f"Sample {idx}: Predicted class {torch.argmax(probs).item()}, Confidences: {probs_numpy}")
    end_time = time.time()  # 结束运行时间
    # print("总样本数量：", total)
    # print("测试准确率：", correct / total)
    # print('cost %f second' % (end_time - start_time))
    #
    # print('****** test finished ******')
    torch.save(model.state_dict(),dp_path)
    return dp_path


class YOLO(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        # "model_path"        : '/data/Newdisk/zhaoxiaoming/steal/sjs/yolox-pytorch-main/model_data/yolox_weights_new_11.pt',
        "classes_path"      : '/data/Newdisk/zhaoxiaoming/bao2fengzhuang/object_detection/yolo/model_data/voc_classes.txt',
        #---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        #---------------------------------------------------------------------#
        "input_shape"       : [640, 640],
        #---------------------------------------------------------------------#
        #   所使用的YoloX的版本。nano、tiny、s、m、l、x
        #---------------------------------------------------------------------#
        "phi"               : 's',
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True,

    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化YOLO
    # ---------------------------------------------------#
    def __init__(self, ori_path, **kwargs):

        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value

            # ---------------------------------------------------#
        #   获得种类和先验框的数量
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)

        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate(ori_path)

        show_config(**self._defaults)

    # ---------------------------------------------------#
    #   生成模型
    # ---------------------------------------------------#
    def generate(self, ori_path, onnx=False):
        self.net = yolo_dp.YoloBody(self.num_classes, self.phi)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(ori_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(ori_path))
        if not onnx:
            # print(1)
            if self.cuda:
                # print(1)
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

class YOLO1(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        # "model_path"        : '/data/Newdisk/zhaoxiaoming/steal/sjs/yolox-pytorch-main/model_data/yolox_weights_new_11.pt',
        "classes_path"      : '/data/Newdisk/zhaoxiaoming/bao2fengzhuang/object_detection_warship/yolo/model_data/ship.txt',
        #---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        #---------------------------------------------------------------------#
        "input_shape"       : [640, 640],
        #---------------------------------------------------------------------#
        #   所使用的YoloX的版本。nano、tiny、s、m、l、x
        #---------------------------------------------------------------------#
        "phi"               : 's',
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True,

    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化YOLO
    # ---------------------------------------------------#
    def __init__(self, ori_path, **kwargs):

        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value

            # ---------------------------------------------------#
        #   获得种类和先验框的数量
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)

        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate(ori_path)

        show_config(**self._defaults)

    # ---------------------------------------------------#
    #   生成模型
    # ---------------------------------------------------#
    def generate(self, ori_path, onnx=False):
        self.net = yolo_dp.YoloBody(self.num_classes, self.phi)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(ori_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(ori_path))
        if not onnx:
            # print(1)
            if self.cuda:
                # print(1)
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

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

class DecodeBox():
    def __init__(self, std, num_classes):
        self.std            = std
        self.num_classes    = num_classes + 1

    def frcnn_correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        #-----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        #-----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        box_mins    = box_yx - (box_hw / 2.)
        box_maxes   = box_yx + (box_hw / 2.)
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def forward(self, roi_cls_locs, roi_scores, rois, image_shape, input_shape, nms_iou = 0.3, confidence = 0.5):
        results = []
        bs      = len(roi_cls_locs)
        #--------------------------------#
        #   batch_size, num_rois, 4
        #--------------------------------#
        rois    = rois.view((bs, -1, 4))
        #----------------------------------------------------------------------------------------------------------------#
        #   对每一张图片进行处理，由于在predict.py的时候，我们只输入一张图片，所以for i in range(len(mbox_loc))只进行一次
        #----------------------------------------------------------------------------------------------------------------#
        for i in range(bs):
            #----------------------------------------------------------#
            #   对回归参数进行reshape
            #----------------------------------------------------------#
            roi_cls_loc = roi_cls_locs[i] * self.std
            #----------------------------------------------------------#
            #   第一维度是建议框的数量，第二维度是每个种类
            #   第三维度是对应种类的调整参数
            #----------------------------------------------------------#
            roi_cls_loc = roi_cls_loc.view([-1, self.num_classes, 4])

            #-------------------------------------------------------------#
            #   利用classifier网络的预测结果对建议框进行调整获得预测框
            #   num_rois, 4 -> num_rois, 1, 4 -> num_rois, num_classes, 4
            #-------------------------------------------------------------#
            roi         = rois[i].view((-1, 1, 4)).expand_as(roi_cls_loc)
            cls_bbox    = loc2bbox(roi.contiguous().view((-1, 4)), roi_cls_loc.contiguous().view((-1, 4)))
            cls_bbox    = cls_bbox.view([-1, (self.num_classes), 4])
            #-------------------------------------------------------------#
            #   对预测框进行归一化，调整到0-1之间
            #-------------------------------------------------------------#
            cls_bbox[..., [0, 2]] = (cls_bbox[..., [0, 2]]) / input_shape[1]
            cls_bbox[..., [1, 3]] = (cls_bbox[..., [1, 3]]) / input_shape[0]

            roi_score   = roi_scores[i]
            prob        = F.softmax(roi_score, dim=-1)

            results.append([])
            for c in range(1, self.num_classes):
                #--------------------------------#
                #   取出属于该类的所有框的置信度
                #   判断是否大于门限
                #--------------------------------#
                c_confs     = prob[:, c]
                c_confs_m   = c_confs > confidence

                if len(c_confs[c_confs_m]) > 0:
                    #-----------------------------------------#
                    #   取出得分高于confidence的框
                    #-----------------------------------------#
                    boxes_to_process = cls_bbox[c_confs_m, c]
                    confs_to_process = c_confs[c_confs_m]

                    keep = nms(
                        boxes_to_process,
                        confs_to_process,
                        nms_iou
                    )
                    #-----------------------------------------#
                    #   取出在非极大抑制中效果较好的内容
                    #-----------------------------------------#
                    good_boxes  = boxes_to_process[keep]
                    confs       = confs_to_process[keep][:, None]
                    labels      = (c - 1) * torch.ones((len(keep), 1)).cuda() if confs.is_cuda else (c - 1) * torch.ones((len(keep), 1))
                    #-----------------------------------------#
                    #   将label、置信度、框的位置进行堆叠。
                    #-----------------------------------------#
                    c_pred      = torch.cat((good_boxes, confs, labels), dim=1).cpu().numpy()
                    # 添加进result里
                    results[-1].extend(c_pred)

            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                box_xy, box_wh = (results[-1][:, 0:2] + results[-1][:, 2:4])/2, results[-1][:, 2:4] - results[-1][:, 0:2]
                results[-1][:, :4] = self.frcnn_correct_boxes(box_xy, box_wh, input_shape, image_shape)

        return results

class FRCNN(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        # "model_path": '/data/Newdisk/zhaoxiaoming/steal/sjs/frcnn-pytorch-main/model_data/voc_weights_resnet.pth',
        "classes_path": '/data/Newdisk/zhaoxiaoming/bao2fengzhuang/object_detection/frcnn/model_data/voc_classes.txt',
        "input_shape": [600, 600],
        # ---------------------------------------------------------------------#
        #   网络的主干特征提取网络，resnet50或者vgg
        # ---------------------------------------------------------------------#
        "backbone": "resnet50",
        # ---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        # ---------------------------------------------------------------------#
        "confidence": 0.5,
        # ---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        # ---------------------------------------------------------------------#
        "nms_iou": 0.3,
        # ---------------------------------------------------------------------#
        #   用于指定先验框的大小
        # ---------------------------------------------------------------------#
        'anchors_size': [8, 16, 32],
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": True,

    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化YOLO
    # ---------------------------------------------------#
    def __init__(self, ori_path, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value
            # ---------------------------------------------------#
        #   获得种类和先验框的数量
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)

        self.std = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(self.num_classes + 1)[None]
        if self.cuda:
            self.std = self.std.cuda()
        print(self.std.shape, self.num_classes)
        self.bbox_util = DecodeBox(self.std, self.num_classes)

        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate(ori_path)

        show_config(**self._defaults)

    # ---------------------------------------------------#
    #   生成模型
    # ---------------------------------------------------#
    def generate(self, ori_path):
        # -------------------------------#
        #   载入模型与权值
        # -------------------------------#
        self.net = frcnn_dp.FasterRCNN(self.num_classes, "predict", anchor_scales=self.anchors_size, backbone=self.backbone)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(ori_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(ori_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

class FRCNN1(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        # "model_path": '/data/Newdisk/zhaoxiaoming/steal/sjs/frcnn-pytorch-main/model_data/voc_weights_resnet.pth',
        "classes_path": '/data/Newdisk/zhaoxiaoming/bao2fengzhuang/object_detection/frcnn/model_data/ship.txt',
        "input_shape": [600, 600],
        # ---------------------------------------------------------------------#
        #   网络的主干特征提取网络，resnet50或者vgg
        # ---------------------------------------------------------------------#
        "backbone": "resnet50",
        # ---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        # ---------------------------------------------------------------------#
        "confidence": 0.5,
        # ---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        # ---------------------------------------------------------------------#
        "nms_iou": 0.3,
        # ---------------------------------------------------------------------#
        #   用于指定先验框的大小
        # ---------------------------------------------------------------------#
        'anchors_size': [8, 16, 32],
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": True,

    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化YOLO
    # ---------------------------------------------------#
    def __init__(self, ori_path, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value
            # ---------------------------------------------------#
        #   获得种类和先验框的数量
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)

        self.std = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(self.num_classes + 1)[None]
        if self.cuda:
            self.std = self.std.cuda()
        print(self.std.shape, self.num_classes)
        self.bbox_util = DecodeBox(self.std, self.num_classes)

        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate(ori_path)

        show_config(**self._defaults)

    # ---------------------------------------------------#
    #   生成模型
    # ---------------------------------------------------#
    def generate(self, ori_path):
        # -------------------------------#
        #   载入模型与权值
        # -------------------------------#
        self.net = frcnn_dp.FasterRCNN(self.num_classes, "predict", anchor_scales=self.anchors_size, backbone=self.backbone)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(ori_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(ori_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

class AnchorBox():
    def __init__(self, input_shape, min_size, max_size=None, aspect_ratios=None, flip=True):
        self.input_shape = input_shape

        self.min_size = min_size
        self.max_size = max_size

        self.aspect_ratios = []
        for ar in aspect_ratios:
            self.aspect_ratios.append(ar)
            self.aspect_ratios.append(1.0 / ar)

    def call(self, layer_shape, mask=None):
        # --------------------------------- #
        #   获取输入进来的特征层的宽和高
        #   比如38x38
        # --------------------------------- #
        layer_height = layer_shape[0]
        layer_width = layer_shape[1]
        # --------------------------------- #
        #   获取输入进来的图片的宽和高
        #   比如300x300
        # --------------------------------- #
        img_height = self.input_shape[0]
        img_width = self.input_shape[1]

        box_widths = []
        box_heights = []
        # --------------------------------- #
        #   self.aspect_ratios一般有两个值
        #   [1, 1, 2, 1/2]
        #   [1, 1, 2, 1/2, 3, 1/3]
        # --------------------------------- #
        for ar in self.aspect_ratios:
            # 首先添加一个较小的正方形
            if ar == 1 and len(box_widths) == 0:
                box_widths.append(self.min_size)
                box_heights.append(self.min_size)
            # 然后添加一个较大的正方形
            elif ar == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(self.min_size * self.max_size))
                box_heights.append(np.sqrt(self.min_size * self.max_size))
            # 然后添加长方形
            elif ar != 1:
                box_widths.append(self.min_size * np.sqrt(ar))
                box_heights.append(self.min_size / np.sqrt(ar))

        # --------------------------------- #
        #   获得所有先验框的宽高1/2
        # --------------------------------- #
        box_widths = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)

        # --------------------------------- #
        #   每一个特征层对应的步长
        # --------------------------------- #
        step_x = img_width / layer_width
        step_y = img_height / layer_height

        # --------------------------------- #
        #   生成网格中心
        # --------------------------------- #
        linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x,
                           layer_width)
        liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y,
                           layer_height)
        centers_x, centers_y = np.meshgrid(linx, liny)
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)

        # 每一个先验框需要两个(centers_x, centers_y)，前一个用来计算左上角，后一个计算右下角
        num_anchors_ = len(self.aspect_ratios)
        anchor_boxes = np.concatenate((centers_x, centers_y), axis=1)
        anchor_boxes = np.tile(anchor_boxes, (1, 2 * num_anchors_))
        # 获得先验框的左上角和右下角
        anchor_boxes[:, ::4] -= box_widths
        anchor_boxes[:, 1::4] -= box_heights
        anchor_boxes[:, 2::4] += box_widths
        anchor_boxes[:, 3::4] += box_heights

        # --------------------------------- #
        #   将先验框变成小数的形式
        #   归一化
        # --------------------------------- #
        anchor_boxes[:, ::2] /= img_width
        anchor_boxes[:, 1::2] /= img_height
        anchor_boxes = anchor_boxes.reshape(-1, 4)

        anchor_boxes = np.minimum(np.maximum(anchor_boxes, 0.0), 1.0)
        return anchor_boxes


# ---------------------------------------------------#
#   用于计算共享特征层的大小
# ---------------------------------------------------#
def get_vgg_output_length(height, width):
    filter_sizes = [3, 3, 3, 3, 3, 3, 3, 3]
    padding = [1, 1, 1, 1, 1, 1, 0, 0]
    stride = [2, 2, 2, 2, 2, 2, 1, 1]
    feature_heights = []
    feature_widths = []

    for i in range(len(filter_sizes)):
        height = (height + 2 * padding[i] - filter_sizes[i]) // stride[i] + 1
        width = (width + 2 * padding[i] - filter_sizes[i]) // stride[i] + 1
        feature_heights.append(height)
        feature_widths.append(width)
    return np.array(feature_heights)[-6:], np.array(feature_widths)[-6:]


def get_mobilenet_output_length(height, width):
    filter_sizes = [3, 3, 3, 3, 3, 3, 3, 3, 3]
    padding = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    stride = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    feature_heights = []
    feature_widths = []

    for i in range(len(filter_sizes)):
        height = (height + 2 * padding[i] - filter_sizes[i]) // stride[i] + 1
        width = (width + 2 * padding[i] - filter_sizes[i]) // stride[i] + 1
        feature_heights.append(height)
        feature_widths.append(width)
    return np.array(feature_heights)[-6:], np.array(feature_widths)[-6:]

def get_anchors(input_shape=[300, 300], anchors_size=[30, 60, 111, 162, 213, 264, 315], backbone='vgg'):
    if backbone == 'vgg' or backbone == 'resnet50':
        feature_heights, feature_widths = get_vgg_output_length(input_shape[0], input_shape[1])
        aspect_ratios = [[1, 2], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2], [1, 2]]
    else:
        feature_heights, feature_widths = get_mobilenet_output_length(input_shape[0], input_shape[1])
        aspect_ratios = [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]

    anchors = []
    for i in range(len(feature_heights)):
        anchor_boxes = AnchorBox(input_shape, anchors_size[i], max_size=anchors_size[i + 1],
                                 aspect_ratios=aspect_ratios[i]).call([feature_heights[i], feature_widths[i]])
        anchors.append(anchor_boxes)

    anchors = np.concatenate(anchors, axis=0)
    return anchors


class BBoxUtility(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def ssd_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        # -----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        # -----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            # -----------------------------------------------------------------#
            #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            #   new_shape指的是宽高缩放情况
            # -----------------------------------------------------------------#
            new_shape = np.round(image_shape * np.min(input_shape / image_shape))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape

            box_yx = (box_yx - offset) * scale
            box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]],
                               axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def decode_boxes(self, mbox_loc, anchors, variances):
        # 获得先验框的宽与高
        anchor_width = anchors[:, 2] - anchors[:, 0]
        anchor_height = anchors[:, 3] - anchors[:, 1]
        # 获得先验框的中心点
        anchor_center_x = 0.5 * (anchors[:, 2] + anchors[:, 0])
        anchor_center_y = 0.5 * (anchors[:, 3] + anchors[:, 1])

        # 真实框距离先验框中心的xy轴偏移情况
        decode_bbox_center_x = mbox_loc[:, 0] * anchor_width * variances[0]
        decode_bbox_center_x += anchor_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * anchor_height * variances[0]
        decode_bbox_center_y += anchor_center_y

        # 真实框的宽与高的求取
        decode_bbox_width = torch.exp(mbox_loc[:, 2] * variances[1])
        decode_bbox_width *= anchor_width
        decode_bbox_height = torch.exp(mbox_loc[:, 3] * variances[1])
        decode_bbox_height *= anchor_height

        # 获取真实框的左上角与右下角
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

        # 真实框的左上角与右下角进行堆叠
        decode_bbox = torch.cat((decode_bbox_xmin[:, None],
                                 decode_bbox_ymin[:, None],
                                 decode_bbox_xmax[:, None],
                                 decode_bbox_ymax[:, None]), dim=-1)
        # 防止超出0与1
        decode_bbox = torch.min(torch.max(decode_bbox, torch.zeros_like(decode_bbox)), torch.ones_like(decode_bbox))
        return decode_bbox

    def decode_box(self, predictions, anchors, image_shape, input_shape, letterbox_image, variances=[0.1, 0.2],
                   nms_iou=0.3, confidence=0.5):
        # ---------------------------------------------------#
        #   :4是回归预测结果
        # ---------------------------------------------------#
        mbox_loc = predictions[0]
        # ---------------------------------------------------#
        #   获得种类的置信度
        # ---------------------------------------------------#
        mbox_conf = nn.Softmax(-1)(predictions[1])

        results = []
        # ----------------------------------------------------------------------------------------------------------------#
        #   对每一张图片进行处理，由于在predict.py的时候，我们只输入一张图片，所以for i in range(len(mbox_loc))只进行一次
        # ----------------------------------------------------------------------------------------------------------------#
        for i in range(len(mbox_loc)):
            results.append([])
            # --------------------------------#
            #   利用回归结果对先验框进行解码
            # --------------------------------#
            decode_bbox = self.decode_boxes(mbox_loc[i], anchors, variances)

            for c in range(1, self.num_classes):
                # --------------------------------#
                #   取出属于该类的所有框的置信度
                #   判断是否大于门限
                # --------------------------------#
                c_confs = mbox_conf[i, :, c]
                c_confs_m = c_confs > confidence
                if len(c_confs[c_confs_m]) > 0:
                    # -----------------------------------------#
                    #   取出得分高于confidence的框
                    # -----------------------------------------#
                    boxes_to_process = decode_bbox[c_confs_m]
                    confs_to_process = c_confs[c_confs_m]

                    keep = nms(
                        boxes_to_process,
                        confs_to_process,
                        nms_iou
                    )
                    # -----------------------------------------#
                    #   取出在非极大抑制中效果较好的内容
                    # -----------------------------------------#
                    good_boxes = boxes_to_process[keep]
                    confs = confs_to_process[keep][:, None]
                    labels = (c - 1) * torch.ones((len(keep), 1)).cuda() if confs.is_cuda else (c - 1) * torch.ones(
                        (len(keep), 1))
                    # -----------------------------------------#
                    #   将label、置信度、框的位置进行堆叠。
                    # -----------------------------------------#
                    c_pred = torch.cat((good_boxes, labels, confs), dim=1).cpu().numpy()
                    # 添加进result里
                    results[-1].extend(c_pred)

            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                box_xy, box_wh = (results[-1][:, 0:2] + results[-1][:, 2:4]) / 2, results[-1][:, 2:4] - results[-1][:,
                                                                                                        0:2]
                results[-1][:, :4] = self.ssd_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)

        return results

class SSD(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        # "model_path"        : '/data/Newdisk/zhaoxiaoming/zxm/object_detection/weights/ssd_weights.pth',
        "classes_path"      : '/data/Newdisk/zhaoxiaoming/bao2fengzhuang/object_detection/ssd/model_data/voc_classes.txt',
        #---------------------------------------------------------------------#
        #   用于预测的图像大小，和train时使用同一个即可
        #---------------------------------------------------------------------#
        "input_shape"       : [300, 300],
        #-------------------------------#
        #   主干网络的选择
        #   vgg或者mobilenetv2或者resnet50
        #-------------------------------#
        "backbone"          : "vgg",
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #---------------------------------------------------------------------#
        #   用于指定先验框的大小
        #---------------------------------------------------------------------#
        'anchors_size'      : [30, 60, 111, 162, 213, 264, 315],
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化ssd
    # ---------------------------------------------------#
    def __init__(self, ori_path, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        # ---------------------------------------------------#
        #   计算总的类的数量
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors = torch.from_numpy(get_anchors(self.input_shape, self.anchors_size, self.backbone)).type(
            torch.FloatTensor)
        if self.cuda:
            self.anchors = self.anchors.cuda()
        self.num_classes = self.num_classes + 1

        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.bbox_util = BBoxUtility(self.num_classes)
        self.generate(ori_path)

        show_config(**self._defaults)

    # ---------------------------------------------------#
    #   生成模型
    # ---------------------------------------------------#
    def generate(self, ori_path, onnx=False):
        # -------------------------------#
        #   载入模型与权值
        # -------------------------------#
        self.net = ssd_dp.SSD300(self.num_classes, self.backbone)
        device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(ori_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(ori_path))
        if not onnx:
            if self.cuda:
                #self.net = torch.nn.DataParallel(self.net)
                self.net = self.net.cuda()

class SSD1(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        # "model_path"        : '/data/Newdisk/zhaoxiaoming/zxm/object_detection/weights/ssd_weights.pth',
        "classes_path"      : '/data/Newdisk/zhaoxiaoming/bao2fengzhuang/object_detection/ssd/model_data/ship.txt',
        #---------------------------------------------------------------------#
        #   用于预测的图像大小，和train时使用同一个即可
        #---------------------------------------------------------------------#
        "input_shape"       : [300, 300],
        #-------------------------------#
        #   主干网络的选择
        #   vgg或者mobilenetv2或者resnet50
        #-------------------------------#
        "backbone"          : "vgg",
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #---------------------------------------------------------------------#
        #   用于指定先验框的大小
        #---------------------------------------------------------------------#
        'anchors_size'      : [30, 60, 111, 162, 213, 264, 315],
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化ssd
    # ---------------------------------------------------#
    def __init__(self, ori_path, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        # ---------------------------------------------------#
        #   计算总的类的数量
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors = torch.from_numpy(get_anchors(self.input_shape, self.anchors_size, self.backbone)).type(
            torch.FloatTensor)
        if self.cuda:
            self.anchors = self.anchors.cuda()
        self.num_classes = self.num_classes + 1

        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.bbox_util = BBoxUtility(self.num_classes)
        self.generate(ori_path)

        show_config(**self._defaults)

    # ---------------------------------------------------#
    #   生成模型
    # ---------------------------------------------------#
    def generate(self, ori_path, onnx=False):
        # -------------------------------#
        #   载入模型与权值
        # -------------------------------#
        self.net = ssd_dp.SSD300(self.num_classes, self.backbone)
        device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(ori_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(ori_path))
        if not onnx:
            if self.cuda:
                #self.net = torch.nn.DataParallel(self.net)
                self.net = self.net.cuda()

def dp_test(model):
    start_time = time.time()
    class_names, _ = get_classes("/data/Newdisk/zhaoxiaoming/bao2fengzhuang/object_detection/yolo/model_data/voc_classes.txt")



def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def yolo_model_dp(ori_path, dp_path):
    print("Load model.")
    yolo = YOLO(ori_path)
    print("Load model done.")
    print("start test")
    # dp_test(yolo.net)
    torch.save(yolo.net.state_dict(), dp_path)
    print("Dp model done.")
    return dp_path

def yolo_model_warship_dp(ori_path, dp_path):
    print("Load model.")
    yolo = YOLO1(ori_path)
    print("Load model done.")
    print("start test")
    # dp_test(yolo.net)
    torch.save(yolo.net.state_dict(), dp_path)
    print("Dp model done.")
    return dp_path

def frcnn_model_dp(ori_path, dp_path):
    print("Load model.")
    frcnn = FRCNN(ori_path)
    print("Load model done.")
    print("start test")
    # dp_test(frcnn.net)
    torch.save(frcnn.net.state_dict(), dp_path)
    print("Dp model done.")
    return dp_path

def frcnn_model_warship_dp(ori_path, dp_path):
    print("Load model.")
    frcnn = FRCNN1(ori_path)
    print("Load model done.")
    print("start test")
    # dp_test(frcnn.net)
    torch.save(frcnn.net.state_dict(), dp_path)
    print("Dp model done.")
    return dp_path

def ssd_model_dp(ori_path, dp_path):
    print("Load model.")
    ssd = SSD(ori_path)
    print("Load model done.")
    print("start test")
    # dp_test(ssd.net)
    torch.save(ssd.net.state_dict(), dp_path)
    print("Dp model done.")
    return dp_path

def ssd_model_warship_dp(ori_path, dp_path):
    print("Load model.")
    ssd = SSD1(ori_path)
    print("Load model done.")
    print("start test")
    # dp_test(ssd.net)
    torch.save(ssd.net.state_dict(), dp_path)
    print("Dp model done.")
    return dp_path

# if __name__ == '__main__':
#     dp_path= "./GenModelPth/"
#     if not os.path.exists(dp_path):
#         os.makedirs(dp_path)
#     dp_path = "./GenModelPth/resnet50_military_dp.pth"
#     ori_path= "/data/Newdisk/zhaoxiaoming/bao2fengzhuang/classification/ResNet50/ori_model/ResNet50_Military_91.30.pth"
#     resnet50_model_dp(ori_path,dp_path)