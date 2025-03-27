import requests
import os
import shutil
def vgg16_imagenet100_dp():
    url = "http://127.0.0.1:8001/VGG16_imagenet100_dp"
    pth_path = './GenData/pth/'
    if not os.path.exists(pth_path):
        os.makedirs(pth_path)
    pth_path = pth_path + 'vgg16_imagenet100_dp.pth'
    arch_path = './GenData/arch/'
    if not os.path.exists(arch_path):
        os.makedirs(arch_path)
    arch_path = arch_path + 'vgg16_imagenet100_dp.py'
    source_file = './utils/vgg_imagenet100_dp.py'
    destination_file = arch_path
    shutil.copy(source_file, destination_file)
    data = {
        "MODEL_NAME": "vgg16",
        "ORI_MODEL_PATH": "/data/Newdisk/zhaoxiaoming/bao2fengzhuang/classfication_imagenet_100/VGG16/ori_model/VGG16_imagenet_100.pth",
        "DP_MODEL_PTH_PATH": pth_path,
        "DP_MODEL_ARCH_PATH": arch_path
    }
    response = requests.post(url, json=data)

    print(response.json())

def alexnet_imagenet100_dp():
    url = "http://127.0.0.1:8001/AlexNet_imagenet100_dp"
    pth_path = './GenData/pth/'
    if not os.path.exists(pth_path):
        os.makedirs(pth_path)
    pth_path = pth_path + 'alexnet_imagenet100_dp.pth'
    arch_path = './GenData/arch/'
    if not os.path.exists(arch_path):
        os.makedirs(arch_path)
    arch_path = arch_path + 'alexnet_imagenet100_dp.py'
    source_file = './utils/alexnet_imagenet100_dp.py'
    destination_file = arch_path
    shutil.copy(source_file, destination_file)
    data = {
        "MODEL_NAME": "alexnet",
        "ORI_MODEL_PATH": "/data/Newdisk/zhaoxiaoming/bao2fengzhuang/classfication_imagenet_100/AlexNet/ori_model/AlexNet_imagenet_100_54.60%.pth",
        "DP_MODEL_PTH_PATH": pth_path,
        "DP_MODEL_ARCH_PATH": arch_path
    }
    response = requests.post(url, json=data)

    print(response.json())

def googlenet_imagenet100_dp():
    url = "http://127.0.0.1:8001/GoogleNet_imagenet100_dp"
    pth_path = './GenData/pth/'
    if not os.path.exists(pth_path):
        os.makedirs(pth_path)
    pth_path = pth_path + 'googlenet_imagenet100_dp.pth'
    arch_path = './GenData/arch/'
    if not os.path.exists(arch_path):
        os.makedirs(arch_path)
    arch_path = arch_path + 'googlenet_imagenet100_dp.py'
    source_file = './utils/googlenet_imagenet100_dp.py'
    destination_file = arch_path
    shutil.copy(source_file, destination_file)
    data = {
        "MODEL_NAME": "googlenet",
        "ORI_MODEL_PATH": "/data/Newdisk/zhaoxiaoming/bao2fengzhuang/classfication_imagenet_100/GoogleNet/ori_model/GoogleNet_imagenet_100_65.20%.pth",
        "DP_MODEL_PTH_PATH": pth_path,
        "DP_MODEL_ARCH_PATH": arch_path
    }
    response = requests.post(url, json=data)

    print(response.json())


def resnet50_imagenet100_dp():
    url = "http://127.0.0.1:8001/ResNet50_imagenet100_dp"
    pth_path = './GenData/pth/'
    if not os.path.exists(pth_path):
        os.makedirs(pth_path)
    pth_path = pth_path + 'resnet50_imagenet100_dp.pth'
    arch_path = './GenData/arch/'
    if not os.path.exists(arch_path):
        os.makedirs(arch_path)
    arch_path = arch_path + 'resnet50_imagenet100_dp.py'
    source_file = './utils/resnet_imagenet100_dp.py'
    destination_file = arch_path
    shutil.copy(source_file, destination_file)
    data = {
        "MODEL_NAME": "resnet50",
        "ORI_MODEL_PATH": "/data/Newdisk/zhaoxiaoming/bao2fengzhuang/classfication_imagenet_100/ResNet50/ori_model/ResNet50_imagenet_100_64.26%.pth",
        "DP_MODEL_PTH_PATH": pth_path,
        "DP_MODEL_ARCH_PATH": arch_path
    }
    response = requests.post(url, json=data)

    print(response.json())
def vgg16_dp():
    url = "http://127.0.0.1:8001/VGG16_dp"
    pth_path = './GenData/pth/'
    if not os.path.exists(pth_path):
        os.makedirs(pth_path)
    pth_path = pth_path + 'vgg16_military_dp.pth'
    arch_path = './GenData/arch/'
    if not os.path.exists(arch_path):
        os.makedirs(arch_path)
    arch_path = arch_path + 'vgg16_military_dp.py'
    source_file = './utils/vgg_dp.py'
    destination_file = arch_path
    shutil.copy(source_file, destination_file)
    data = {
        "MODEL_NAME": "vgg16",
        "ORI_MODEL_PATH": "/data/Newdisk/zhaoxiaoming/bao2fengzhuang/classification/VGG16/model/ori_model/VGG16_Military.pth",
        "DP_MODEL_PTH_PATH": pth_path,
        "DP_MODEL_ARCH_PATH": arch_path
    }
    response = requests.post(url, json=data)

    print(response.json())

def alexnet_dp():
    url = "http://127.0.0.1:8001/AlexNet_dp"
    pth_path = './GenData/pth/'
    if not os.path.exists(pth_path):
        os.makedirs(pth_path)
    pth_path = pth_path + 'alexnet_military_dp.pth'
    arch_path = './GenData/arch/'
    if not os.path.exists(arch_path):
        os.makedirs(arch_path)
    arch_path = arch_path + 'alexnet_military_dp.py'
    source_file = './utils/alexnet_dp.py'
    destination_file = arch_path
    shutil.copy(source_file, destination_file)
    data = {
        "MODEL_NAME": "alexnet",
        "ORI_MODEL_PATH": "/data/Newdisk/zhaoxiaoming/bao2fengzhuang/classification/AlexNet/model/ori_model/AlexNet_Military.pth",
        "DP_MODEL_PTH_PATH": pth_path,
        "DP_MODEL_ARCH_PATH": arch_path
    }
    response = requests.post(url, json=data)

    print(response.json())

def googlenet_dp():
    url = "http://127.0.0.1:8001/GoogleNet_dp"
    pth_path = './GenData/pth/'
    if not os.path.exists(pth_path):
        os.makedirs(pth_path)
    pth_path = pth_path + 'googlenet_military_dp.pth'
    arch_path = './GenData/arch/'
    if not os.path.exists(arch_path):
        os.makedirs(arch_path)
    arch_path = arch_path + 'googlenet_military_dp.py'
    source_file = './utils/googlenet_dp.py'
    destination_file = arch_path
    shutil.copy(source_file, destination_file)
    data = {
        "MODEL_NAME": "googlenet",
        "ORI_MODEL_PATH": "/data/Newdisk/zhaoxiaoming/bao2fengzhuang/classification/GoogleNet/model/ori_model/GoogleNet_Military_76.80%.pth",
        "DP_MODEL_PTH_PATH": pth_path,
        "DP_MODEL_ARCH_PATH": arch_path
    }
    response = requests.post(url, json=data)

    print(response.json())


def resnet50_dp():
    url = "http://127.0.0.1:8001/ResNet50_dp"
    pth_path = './GenData/pth/'
    if not os.path.exists(pth_path):
        os.makedirs(pth_path)
    pth_path = pth_path + 'resnet50_military_dp.pth'
    arch_path = './GenData/arch/'
    if not os.path.exists(arch_path):
        os.makedirs(arch_path)
    arch_path = arch_path + 'resnet50_military_dp.py'
    source_file = './utils/resnet_dp.py'
    destination_file = arch_path
    shutil.copy(source_file, destination_file)
    data = {
        "MODEL_NAME": "resnet50",
        "ORI_MODEL_PATH": "/data/Newdisk/zhaoxiaoming/bao2fengzhuang/classification/ResNet50/model/ori_model/ResNet50_Military_91.30.pth",
        "DP_MODEL_PTH_PATH": pth_path,
        "DP_MODEL_ARCH_PATH": arch_path
    }
    response = requests.post(url, json=data)

    print(response.json())

def yolo_dp():
    url = "http://127.0.0.1:8001/YOLO_dp"
    pth_path = './GenData/pth/'
    if not os.path.exists(pth_path):
        os.makedirs(pth_path)
    pth_path = pth_path + 'yolo_voc_dp.pth'
    arch_path = './GenData/arch/'
    if not os.path.exists(arch_path):
        os.makedirs(arch_path)
    arch_path = arch_path + 'yolo_voc_dp.py'
    source_file = './utils/yolo_dp.py'
    destination_file = arch_path
    shutil.copy(source_file, destination_file)
    data = {
        "MODEL_NAME": "yolo",
        "ORI_MODEL_PATH": "/data/Newdisk/zhaoxiaoming/bao2fengzhuang/object_detection/yolo/ori_model/yolox_weights_new_11.pt",
        "DP_MODEL_PTH_PATH": pth_path,
        "DP_MODEL_ARCH_PATH": arch_path
    }
    response = requests.post(url, json=data)

    print(response.json())

def yolo_warship_dp():
    url = "http://127.0.0.1:8002/YOLO_warship_dp"
    pth_path = './GenData/pth/'
    if not os.path.exists(pth_path):
        os.makedirs(pth_path)
    pth_path = pth_path + 'yolo_warship_dp_test.pth'
    arch_path = './GenData/arch/'
    if not os.path.exists(arch_path):
        os.makedirs(arch_path)
    arch_path = arch_path + 'yolo_warship_dp.py'
    source_file = './utils/yolo_dp.py'
    destination_file = arch_path
    shutil.copy(source_file, destination_file)
    data = {
        "MODEL_NAME": "yolo",
        "ORI_MODEL_PATH": "/data/Newdisk/zhaoxiaoming/bao2fengzhuang/object_detection_warship/yolo/ori_model/yolox_ship.pth",
        "DP_MODEL_PTH_PATH": pth_path,
        "DP_MODEL_ARCH_PATH": arch_path
    }
    response = requests.post(url, json=data)

    print(response.json())

def frcnn_dp():
    url = "http://127.0.0.1:8001/FRCNN_dp"
    pth_path = './GenData/pth/'
    if not os.path.exists(pth_path):
        os.makedirs(pth_path)
    pth_path = pth_path + 'frcnn_voc_dp.pth'
    arch_path = './GenData/arch/'
    if not os.path.exists(arch_path):
        os.makedirs(arch_path)
    arch_path = arch_path + 'frcnn_voc_dp.py'
    source_file = './utils/frcnn_dp.py'
    destination_file = arch_path
    shutil.copy(source_file, destination_file)
    data = {
        "MODEL_NAME": "frcnn",
        "ORI_MODEL_PATH": "/data/Newdisk/zhaoxiaoming/bao2fengzhuang/object_detection/frcnn/ori_model/voc_weights_resnet.pth",
        "DP_MODEL_PTH_PATH": pth_path,
        "DP_MODEL_ARCH_PATH": arch_path
    }
    response = requests.post(url, json=data)

    print(response.json())

def frcnn_warship_dp():
    url = "http://127.0.0.1:8002/FRCNN_warship_dp"
    pth_path = './GenData/pth/'
    if not os.path.exists(pth_path):
        os.makedirs(pth_path)
    pth_path = pth_path + 'frcnn_warship_dp.pth'
    arch_path = './GenData/arch/'
    if not os.path.exists(arch_path):
        os.makedirs(arch_path)
    arch_path = arch_path + 'frcnn_warship_dp.py'
    source_file = './utils/frcnn_dp.py'
    destination_file = arch_path
    shutil.copy(source_file, destination_file)
    data = {
        "MODEL_NAME": "frcnn",
        "ORI_MODEL_PATH": "/data/Newdisk/zhaoxiaoming/bao2fengzhuang/object_detection_warship/frcnn/ori_model/frcnn_ship.pth",
        "DP_MODEL_PTH_PATH": pth_path,
        "DP_MODEL_ARCH_PATH": arch_path
    }
    response = requests.post(url, json=data)

    print(response.json())

def ssd_dp():
    url = "http://127.0.0.1:8001/SSD_dp"
    pth_path = './GenData/pth/'
    if not os.path.exists(pth_path):
        os.makedirs(pth_path)
    pth_path = pth_path + 'ssd_voc_dp.pth'
    arch_path = './GenData/arch/'
    if not os.path.exists(arch_path):
        os.makedirs(arch_path)
    arch_path = arch_path + 'ssd_voc_dp.py'
    source_file = './utils/ssd_dp.py'
    destination_file = arch_path
    shutil.copy(source_file, destination_file)
    data = {
        "MODEL_NAME": "ssd",
        "ORI_MODEL_PATH": "/data/Newdisk/zhaoxiaoming/bao2fengzhuang/object_detection/ssd/ori_model/ssd_weights.pth",
        "DP_MODEL_PTH_PATH": pth_path,
        "DP_MODEL_ARCH_PATH": arch_path
    }
    response = requests.post(url, json=data)

    print(response.json())

def ssd_warship_dp():
    url = "http://127.0.0.1:8002/SSD_warship_dp"
    pth_path = './GenData/pth/'
    if not os.path.exists(pth_path):
        os.makedirs(pth_path)
    pth_path = pth_path + 'ssd_warship_dp.pth'
    arch_path = './GenData/arch/'
    if not os.path.exists(arch_path):
        os.makedirs(arch_path)
    arch_path = arch_path + 'ssd_warship_dp.py'
    source_file = './utils/ssd_dp.py'
    destination_file = arch_path
    shutil.copy(source_file, destination_file)
    data = {
        "MODEL_NAME": "ssd",
        "ORI_MODEL_PATH": "/data/Newdisk/zhaoxiaoming/bao2fengzhuang/object_detection_warship/ssd/ori_model/best_epoch_weights.pth",
        "DP_MODEL_PTH_PATH": pth_path,
        "DP_MODEL_ARCH_PATH": arch_path
    }
    response = requests.post(url, json=data)

    print(response.json())

if __name__ == '__main__':
    # vgg16_imagenet100_dp()
    # alexnet_imagenet100_dp()
    # resnet50_imagenet100_dp()
    # googlenet_imagenet100_dp()
    # vgg16_dp()
    # alexnet_dp()
    # resnet50_dp()
    # googlenet_dp()
    # yolo_dp()
    yolo_warship_dp()
    # frcnn_dp()
    # frcnn_warship_dp()
    # ssd_dp()
    # ssd_warship_dp()