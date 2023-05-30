import os
import numpy as np

import torch
import torchvision.transforms as transforms

#one hot label function
def onehot(label, n_classes):
    return torch.zeros(label.size(0), n_classes).scatter_(1, label.view(-1, 1), 1)

#shrpening function
def sharpen(x, T):
    x = x**(1/T)
    return x / x.sum(dim=1, keepdim=True)

#label guessing function
def label_guessing(model, u, K):
    for i in range(K):
        u[i] = model(u[i])
    return u

#mixup function
def mixup(data, targets, alpha, num_classes):
    indices = torch.randperm(data.size(0))
    n_data = data[indices]
    targets2 = targets[indices]

    targets = onehot(targets, num_classes)
    targets2 = onehot(targets2, num_classes)

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)])
    data = data * lam + n_data * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)

    return data, targets

#agumentation function
def mixmatch_augmenter():
    img =transforms.Comopse(
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(size=(0, 0.1)),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=23, sigma=(0, 0.5))], 
            transforms.ColorJitter(contrast=(0.75, 1.5), p=1),
            transforms.GaussianNoise(mean=0, std=(0.0, 0.05), p=1),
            transforms.RandomAffine(
                degrees=(-25, 25),
                translate=(0.2, 0.2),
                scale=(0.8, 1.2),
                shear=(-8, 8)),
                  p=1.0))
    def agument(img):
        imgs = []
        for i in range(img.shape[0]):
            imgs.append(img[i,:,:,:])

        img = img
        return img
    return agument

#MixMatch function
def MixMatch(x, u, T, K,num_classes,model, batch_size):
    for i in range(batch_size):
        x[i] = mixmatch_augmenter(x[i])
        for i in range(K):
            u[i] = mixmatch_augmenter(u[i])
        qb= torch.mean(label_guessing(model, u, K))
        q = sharpen(qb, T)
    p = onehot(x.target, num_classes)
    xhat = torch.cat((x, p), dim=0)
    uhat = torch.cat((u, q), dim=0)
    sxhat = torch.randperm(xhat)
    suhat = torch.randperm(uhat)
    w = torch.cat((sxhat, suhat), dim=0)
    xdot = mixup(xhat, w)
    udot = mixup(uhat, w)
    return xdot, udot

class MixMatch():
    def __init__(self,
                 model,
                 num_labels,
                 K=2,
                 T=0.5):
        super(MixMatch, self).__init__()
        self.model = 'WRN'
        self.num_labels = num_labels
        self.K = K
        self.T = T


def get_mixmatch(model_name=None,
                pretrained=False,
                root=os.path.join('~', '.torch', 'models'),
                **kwargs):
    """
    Create MixMatch model with specific parameters

    Parameters
    ----------
    model_name : str, default
        Model name
    pretrained : bool, default False
        If True, returns a model pre-trained
    root : str, default '~/.torch/models'
        Location for keeping the model parameters

    Returns
    -------
    Model class
    """
    net = MixMatch(
        model=model_name,
        **kwargs)
    
    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from model_provider import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net

def mixmatch(**kwargs):
    """
    MixMatch model from paper MixMatch: A Holistic Approach to Semi-Supervised Learning
    https://arxiv.org/abs/1905.02249

    Parameters
    ----------
    pretrained : bool, default = False
        If True, returns a CNN model pre-trained on CIFAR-10
    root : str, default "~/.torch/models"
        Location for keeping the model parameters.
    """
    return get_mixmatch(model_name='mixmatch', **kwargs)
    