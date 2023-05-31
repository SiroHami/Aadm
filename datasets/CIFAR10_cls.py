"""
CIFAR10 dataset
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
                 root = os.path.join(".torch", "datasets", "cifar10"), 
                 mode = "train",
                 transform=None):
        super(CIFAR10Fine, self).__init__(
            root=root, train=(mode == "train"),
            transform=transform,
            download=True)
    
    def __getitem__(self, index):
        img, target = super(CIFAR10Fine, self).__getitem__(index)
        return img, target, index
    
class CIFAR10Info(DatasetInfo):
    def __init__(self):
        super(CIFAR10Info, self).__init__()
        self.label = "CIFAR10"
        self.num_labels = None
        self.root_dir = "cifar10"
        self.dataset_class = CIFAR10Fine
        self.num_train_samples = 50000
        self.in_channels = 3
        self.num_classes = 10
        self.input_size = (32, 32)
        self.train_metric = "Train.Error"
        self.train_metric_name = "Top1Error"
        self.val_metric = "Val.Error"
        self.val_metric_name = "Top1Error"
        self.saver_acc_ind = 0
        self.train_transfom = None
        self.val_transfom = None
        self.test_transfom = None