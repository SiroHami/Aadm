"""
model provider

this code is from osmr's github
https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/model_provider.py#L178

"""
__all__ = ['get_model', 'get_model_list']

from .CNN.LeNet import *
from .CNN.AlexNet import *
from .CNN.ResNet import *
from .CNN.WideResNet import *
from .SSL.MixMatch import *

_models = {
    #CNN
    'LeNet': LeNet,

    'AlexNet': AlexNet,
    
    'ResNet': ResNet,
    'resnet10': resnet10,
    'resnet12': resnet12,
    'resnet14': resnet14,
    'resnet14b': resnetbc14b,
    'resnet16': resnet16,
    'resnet18_wd4': resnet18_wd4,
    'resnet18_wd2': resnet18_wd2,
    'resnet18_w3d4': resnet18_w3d4,
    'resnet18': resnet18,
    'resnet26': resnet26,
    'resnetbc26b': resnetbc26b,
    'resnet34': resnet34,
    'resnetbc38b': resnetbc38b,
    'resnet50' : resnet50,
    'resnet50b' : resnet50b,
    'resnt101' : resnet101,
    'resnet101b' : resnet101b,
    'resnet152' : resnet152,
    'resnet152b' : resnet152b,
    'resnet200' : resnet200,
    'resnet200b' : resnet200b,
    'ResBlock' : ResBlock,
    'ResBottleneck' : ResBottleneck,
    'ResUnit' : ResUnit,
    'ResInitBlock' : ResInitBlock,
    
    'wrn50_2' : wrn50_2,


    #SSL
    'MixMatch' : MixMatch
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

