"""
    CIFAR-10 classification dataset.
    based by https://github.com/osmr/imgclsmob/blob/master/pytorch/datasets/cifar10_cls_dataset.py 
    Add unsupervised 
"""

import os

import numpy as np

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from .dataset_metainfo import DatasetMetaInfo, train_val_split
from utils import transpose, normalize

class CIFAR10Load(CIFAR10):
    """
    CIFAR-10 image classification dataset.


    Parameters:
    ----------
    root : str, default '~/.torch/datasets/cifar10'
        Path to temp folder for storing data.
    mode : str, default 'train'
        'train', 'val', or 'test'.
    transform : function, default None
        A function that takes data and label and transforms them.
    """
    def __init__(self, 
                 root=os.path.join("~", ".torch", "datasets", "cifar10"),
                 mode="train",
                 transform=None):
      super(CIFAR10, self).__init__(
          root=root,
          train=(mode == "train"),
          transfrom=transform,
          download=True
      )

class CIFAR10MetaInfo(DatasetMetaInfo):
  def __init(self):
    super(CIFAR10MetaInfo, self).__init__()
    self.label = "CIFAR10"
    self.short_label = "cifar"
    self.root_dir_name = "cifar10"
    self.dataset_clas = CIFAR10Load
    self.num_training_sample = 50000
    self.in_channels = 3
    self.input_image_size = (32, 32)
    self.labeled_train_metric_capts = ["labeled_Train.Err"]
    self.labeled_train_metric_names = ["Top1Error"]
    self.labeled_train_metric_extra_kwargs = [{"name": "err"}]
    self.unlabeled_train_metric_capts = ["unlabeled_Train.Err"]
    self.unlabeled_train_metric_names = ["Top1Error"]
    self.unlabeled_train_metric_extra_kwargs = [{"name": "err"}]
    self.val_metric_capts = ["Val.Err"]
    self.val_metric_names = ["Top1Error"]
    self.val_metric_extra_kwargs = [{"name": "err"}]
    self.saver_acc_ind = 0
    self.labeled_train_transform = cifar10_labeled_train_transform(DatasetMetaInfo)
    self.unlabeled_train_transform = cifar10_unlabeled_train_transform(DatasetMetaInfo)
    self.val_tansform = cifar10_val_transform(DatasetMetaInfo)
    self.test_transsform = cifar10_val_transform(DatasetMetaInfo)
    self.get_cifar10 = get_cifar10()
    self.ml_type = "SiroHami, based on omsr'code "


class CIFAR10_labeled(CIFAR10):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_labeled, self).__init__(root, train=train,
                                              transform=transform, target_transform=target_transform,
                                              download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = transpose(normalize(self.data, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR10_unlabeled(CIFAR10_labeled):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_unlabeled, self).__init__(root, indexs, train=train,
                                                transform=transform, target_transform=target_transform,
                                                download=download)
        self.targets = np.array([-1 for i in range(len(self.targets))])


def get_cifar10(root, n_labeled,
                transform_labeled_train=None, transform_unlabeled_train=None, transform_val=None,
                download=True):

    base_dataset = CIFAR10(root, train=True, download=download)
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(
        base_dataset.targets, int(n_labeled/10))

    train_labeled_dataset = CIFAR10_labeled(
        root, train_labeled_idxs, train=True, transform=transform_labeled_train)
    train_unlabeled_dataset = CIFAR10_unlabeled(
        root, train_unlabeled_idxs, train=True, transform=transform_unlabeled_train)
    val_dataset = CIFAR10_labeled(
        root, val_idxs, train=True, transform=transform_val, download=True)
    test_dataset = CIFAR10_labeled(
        root, train=False, transform=transform_val, download=True)

    print(
        f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)}")
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset

def cifar10_labeled_train_transform(ds_metainfo,
                            mean_rgb=(0.4914, 0.4822, 0.4465),
                            std_rgb=(0.2023, 0.1994, 0.2010),
                            ):
    assert (ds_metainfo is not None)
    assert (ds_metainfo.input_image_size[0] == 32)
    
    labeled_train_cifar10 = transforms.Compose([
        transforms.RandomCrop(
            size=32,
            padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb)
    ])
    return labeled_train_cifar10


def cifar10_unlabeled_train_transform(ds_metainfo,
                            mean_rgb=(0.4914, 0.4822, 0.4465),
                            std_rgb=(0.2023, 0.1994, 0.2010),
                            ):
    assert (ds_metainfo is not None)
    assert (ds_metainfo.input_image_size[0] == 32)
    unlabeled_train_cifar10 = transforms.Compose([
        transforms.RandomCrop(
            size=32,
            padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb)
    ])
    return unlabeled_train_cifar10 


def cifar10_val_transform(ds_metainfo,
                          mean_rgb=(0.4914, 0.4822, 0.4465),
                          std_rgb=(0.2023, 0.1994, 0.2010)):
    assert (ds_metainfo is not None)
    val_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb)
    ])
    return val_cifar10