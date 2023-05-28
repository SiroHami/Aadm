"""
    Dataset routines.
    based by https://github.com/osmr/imgclsmob/blob/master/pytorch/datasets/cifar10_cls_dataset.py 
    Add unsupervised 
"""

__all__ = ['get_dataset_metainfo', 'get_labeled_train_data_source',
           'get_labeled_train_data_source', 'get_val_data_source', 'get_test_data_source']

from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision
from .datasets.cifar10_dataset import CIFAR10MetaInfo




def get_dataset_metainfo(dataset_name):
    """
    Get dataset metainfo by name of dataset.

    Parameters:
    ----------
    dataset_name : str
        Dataset name.

    Returns:
    -------
    DatasetMetaInfo
        Dataset metainfo.
    """
    dataset_metainfo_map = {
        "CIFAR10": CIFAR10MetaInfo,

    }
    if dataset_name in dataset_metainfo_map.keys():
        return dataset_metainfo_map[dataset_name]()
    else:
        raise Exception("Unrecognized dataset: {}".format(dataset_name))

def baseline_data(root):
  base_dataset = torchvision.datasets.CIFAR10(root, train=True, download=True) #CIFAR10
  return base_dataset


def get_labeled_train_data_source(ds_metainfo,
                          batch_size,
                          num_workers):
    """
    Get data source for training subset.

    Parameters:
    ----------
    ds_metainfo : DatasetMetaInfo
        Dataset metainfo.
    batch_size : int
        Batch size.
    num_workers : int
        Number of background workers.

    Returns:
    -------
    DataLoader
        Data source.
    """
    transform_labeled_train = ds_metainfo.labeled_train_transform(ds_metainfo=ds_metainfo)
    kwargs = ds_metainfo.dataset_class_extra_kwargs if ds_metainfo.dataset_class_extra_kwargs is not None else {}
    dataset = ds_metainfo.dataset_class(
        root=ds_metainfo.root_dir_path,
        mode="train",
        transform=transform_labeled_train,
        **kwargs)
    ds_metainfo.update_from_dataset(dataset)
    if not ds_metainfo.train_use_weighted_sampler:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True)
    else:
        sampler = WeightedRandomSampler(
            weights=dataset.sample_weights,
            num_samples=len(dataset))
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            # shuffle=True,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True)

def get_unlabeled_train_data_source(ds_metainfo,
                          batch_size,
                          num_workers):
    """
    Get data source for training subset.

    Parameters:
    ----------
    ds_metainfo : DatasetMetaInfo
        Dataset metainfo.
    batch_size : int
        Batch size.
    num_workers : int
        Number of background workers.

    Returns:
    -------
    DataLoader
        Data source.
    """
    transform_unlabeled_train = ds_metainfo.unlabeled_train_transform(ds_metainfo=ds_metainfo)
    kwargs = ds_metainfo.dataset_class_extra_kwargs if ds_metainfo.dataset_class_extra_kwargs is not None else {}
    dataset = ds_metainfo.dataset_class(
        root=ds_metainfo.root_dir_path,
        mode="train",
        transform=transform_unlabeled_train,
        **kwargs)
    ds_metainfo.update_from_dataset(dataset)
    if not ds_metainfo.train_use_weighted_sampler:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True)
    else:
        sampler = WeightedRandomSampler(
            weights=dataset.sample_weights,
            num_samples=len(dataset))
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            # shuffle=True,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True)


def get_val_data_source(ds_metainfo,
                        batch_size,
                        num_workers):
    """
    Get data source for validation subset.

    Parameters:
    ----------
    ds_metainfo : DatasetMetaInfo
        Dataset metainfo.
    batch_size : int
        Batch size.
    num_workers : int
        Number of background workers.

    Returns:
    -------
    DataLoader
        Data source.
    """
    transform_val = ds_metainfo.val_transform(ds_metainfo=ds_metainfo)
    kwargs = ds_metainfo.dataset_class_extra_kwargs if ds_metainfo.dataset_class_extra_kwargs is not None else {}
    dataset = ds_metainfo.dataset_class(
        root=ds_metainfo.root_dir_path,
        mode="val",
        transform=transform_val,
        **kwargs)
    ds_metainfo.update_from_dataset(dataset)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)


def get_test_data_source(ds_metainfo,
                         batch_size,
                         num_workers):
    """
    Get data source for testing subset.

    Parameters:
    ----------
    ds_metainfo : DatasetMetaInfo
        Dataset metainfo.
    batch_size : int
        Batch size.
    num_workers : int
        Number of background workers.

    Returns:
    -------
    DataLoader
        Data source.
    """
    transform_test = ds_metainfo.test_transform(ds_metainfo=ds_metainfo)
    kwargs = ds_metainfo.dataset_class_extra_kwargs if ds_metainfo.dataset_class_extra_kwargs is not None else {}
    dataset = ds_metainfo.dataset_class(
        root=ds_metainfo.root_dir_path,
        mode="test",
        transform=transform_test,
        **kwargs)
    ds_metainfo.update_from_dataset(dataset)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)