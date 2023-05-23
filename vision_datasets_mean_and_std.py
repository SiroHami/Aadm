import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets

def calculate_mean(trainset):
    imgs = [item[0] for item in trainset]  # item[0] and item[1] are image and its label
    imgs = torch.stack(imgs, dim=0).numpy()

    # calculate mean over each channel (r,g,b)
    mean_r = imgs[:, 0, :, :].mean()
    mean_g = imgs[:, 1, :, :].mean()
    mean_b = imgs[:, 2, :, :].mean()
    mean_rgb = (mean_r, mean_g, mean_b)
    return mean_rgb

def calculate_std(trainset):
  imgs = [item[0] for item in trainset]  # item[0] and item[1] are image and its label
  imgs = torch.stack(imgs, dim=0).numpy()

  # calculate std over each channel (r,g,b)
  std_r = imgs[:, 0, :, :].std()
  std_g = imgs[:, 1, :, :].std()
  std_b = imgs[:, 2, :, :].std()
  std_rgb = (std_r, std_g, std_b)
  return std_rgb