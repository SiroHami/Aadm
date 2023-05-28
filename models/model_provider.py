"""
model provider

this code is from osmr's github
https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/model_provider.py#L178

"""
import lenet as LeNet
import AlexNet as AlexNet
import ResNet

__all__ = ['get_model', 'get_model_list']

_models = {
    'LeNet': LeNet,

    'AlexNet': AlexNet,
    
    'resnet18': ResNet.resnet18,
    'resnet34': ResNet.resnet34,
    'resnet50': ResNet.resnet50,
    'resnet101': ResNet.resnet101,
    'resnet152': ResNet.resnet152,
    
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