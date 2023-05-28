"""
    CIFAR-10 classification dataset.
    based by https://github.com/osmr/imgclsmob/blob/master/pytorch/datasets/cifar10_cls_dataset.py 
    Add unsupervised 
"""

import os
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from .dataset_metainfo import DatasetMetaInfo


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
    self.labeled_train_transform = cifar10_labeled_train_transform
    self.unlabeled_train_transform = cifar10_unlabeled_train_transform
    self.val_tansform = cifar10_val_transform
    self.test_transsform = cifar10_val_transform
    self.ml_type = "SiroHami, based on imgcls"


def cifar10_labeled_train_transform(ds_metainfo,
                            mean_rgb=(0.4914, 0.4822, 0.4465),
                            std_rgb=(0.2023, 0.1994, 0.2010),
                            jitter_param=0.4):
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

def cifar10_unlabeled_train_transform(ds_metainfo,
                            mean_rgb=(0.4914, 0.4822, 0.4465),
                            std_rgb=(0.2023, 0.1994, 0.2010),
                            jitter_param=0.4):
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