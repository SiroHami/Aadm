"""
    utils for dataset

    code from https://github.com/osmr/imgclsmob/blob/master/pytorch/dataset_utils.py
"""

__all__ = ['get_dataset_info', 'get_train_data_info', 'get_val_data_info', 'get_test_data_info']


from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from datasets.CIFAR10_cls import *


def get_dataset_info(dataset_name):
    """
    Get dataset info.
    
    Parameters:
    ----------
    dataset_name : str
        Name of dataset.
    
    Returns:
    -------
    DatasetInfo
        Dataset info.
    """
    dataset_info_dict  = {
        "CIFAR10": CIFAR10Info
    }

    if dataset_name not in dataset_info_dict:
        raise ValueError("Unsupported dataset: {}".format(dataset_name))
    
    return dataset_info_dict[dataset_name]()

def get_train_data_info(dataset_info, batch_size, num_worker):
    """
    Get train data info.

    Parameters:
    ----------
    dataset_info : DatasetInfo
        Dataset info.
    batch_size : int
        Batch size.
    num_worker : int
        Number of workers.

    Returns:
    -------
    DataLoader
        Train data Information.
    """
    tranform_train = dataset_info.get_train_transform(dataset_info)
    kwargs = dataset_info.get_train_loader_kwargs
    dataset = dataset_info.dataset_class(root=dataset_info.root_dir, mode="train", trainsform=tranform_train, **kwargs)
    dataset_info.update_from_dataset(dataset)
    if not dataset_info.train_use_weighted_sampler:
        train_data_info = DataLoader(dataset=dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=num_worker,
                                        pin_memory=True)
        return train_data_info
    else:
        sampler = WeightedRandomSampler(weights=dataset.sample_weights,
                                        num_samples=len(dataset))
        train_data_info = DataLoader(dataset=dataset,
                                        batch_size=batch_size,
                                        sampler=sampler,
                                        num_workers=num_worker,
                                        pin_memory=True)
        return train_data_info

def get_val_data_info(dataset_info, batch_size, num_worker):
    """
    Get val data info.

    Parameters:
    ----------
    dataset_info : DatasetInfo
        Dataset info.
    batch_size : int
        Batch size.
    num_worker : int
        Number of workers.

    Returns:
    -------
    DataLoader
        Val data Information.
    """
    tranform_val = dataset_info.get_val_transform(dataset_info)
    kwargs = dataset_info.get_val_loader_kwargs
    dataset = dataset_info.dataset_class(root=dataset_info.root_dir, mode="val", trainsform=tranform_val, **kwargs)
    dataset_info.update_from_dataset(dataset)
    val_data_info = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_worker,
                                pin_memory=True)
    return val_data_info

def get_test_data_info(dataset_info, batch_size, num_worker):
    """
    Get test data info.

    Parameters:
    ----------
    dataset_info : DatasetInfo
        Dataset info.
    batch_size : int
        Batch size.
    num_worker : int
        Number of workers.

    Returns:
    -------
    DataLoader
        Test data Information.
    """
    tranform_test = dataset_info.get_test_transform(dataset_info)
    kwargs = dataset_info.get_test_loader_kwargs
    dataset = dataset_info.dataset_class(root=dataset_info.root_dir, mode="test", trainsform=tranform_test, **kwargs)
    dataset_info.update_from_dataset(dataset)
    test_data_info = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_worker,
                                pin_memory=True)
    return test_data_info