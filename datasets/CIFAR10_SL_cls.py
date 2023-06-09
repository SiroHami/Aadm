"""
CIFAR10 full dataset
"""

import os
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from .dataset_info import DatasetInfo

class CIFAR10Fine(CIFAR10):
    """
    CIFAR10 dataset which returns the label as well
    """
    def __init__(self, 
                 root = os.path.join("~",".torch", "datasets", "cifar10"), 
                 mode = "train",
                 transform=None):
        super(CIFAR10Fine, self).__init__(
            root=root, train=(mode == "train"),
            transform=transform,
            download=True)
    
class CIFAR10_SL_Info(DatasetInfo):
    def __init__(self):
        super(CIFAR10_SL_Info, self).__init__()
        self.label = "CIFAR10_SL"
        self.short_label = "cifar"
        self.root_dir = "CIFAR10_SL"
        self.dataset_class = CIFAR10Fine
        self.num_labels = 50000
        self.num_train_samples = 50000
        self.in_channels = 3
        self.num_classes = 10
        self.input_image_size = (32, 32)
        self.train_metric_capts = ["Train.Err"]
        self.train_metric_names = ["Top1Error"]
        self.train_metric_extra_kwargs = [{"name": "err"}]
        self.val_metric_capts = ["Val.Err"]
        self.val_metric_names = ["Top1Error"]
        self.val_metric_extra_kwargs = [{"name": "err"}]
        self.saver_acc_ind = 0
        self.train_transform = cifar10_train_transform
        self.train_unlabel_transform = cifar10_train_transform
        self.val_transform = cifar10_val_transform
        self.test_transform = cifar10_val_transform
        self.m1_type = "sirohami code by imgcls"
        self.loss_name = "CrossEntropyLoss"


def cifar10_train_transform(ds_metainfo,
                            mean_rgb=(0.4914, 0.4822, 0.4465),
                            std_rgb=(0.2023, 0.1994, 0.2010),
                            jitter_param=0):
    assert (ds_metainfo is not None)
    assert (ds_metainfo.input_image_size[0] == 32)
    return transforms.Compose([
        transforms.RandomCrop(
            size=32,
            padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=jitter_param,
            contrast=jitter_param,
            saturation=jitter_param),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb)
    ])

def cifar10_val_transform(ds_metainfo,
                          mean_rgb=(0.4914, 0.4822, 0.4465),
                          std_rgb=(0.2023, 0.1994, 0.2010)):
    assert (ds_metainfo is not None)
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb)
    ])