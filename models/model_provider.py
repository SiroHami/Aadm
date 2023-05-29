"""
model provider

this code is from osmr's github
https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/model_provider.py#L178

"""
__all__ = ['get_model', 'get_model_list']

from models.LeNet import LeNet
from models.AlexNet import AlexNet
from models.ResNet import resnet18, resnet34, resnet50, resnet101, resnet152

_models = {
    'LeNet': LeNet,

    'AlexNet': AlexNet,
    
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    
}

def get_model(name, **kwargs):
    """
    Get supported model.

    Parameters:
    ----------
    name : str
        Name of model.

    Returns:
    -------
    Module
        Resulted model.
    """
    name = name.lower()
    if name not in _models:
        raise ValueError("Unsupported model: {}".format(name))
    net = _models[name](**kwargs)
    return net