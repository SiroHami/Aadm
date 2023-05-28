import random
import torch
import numpy as np

def fix_seed(seed: int) -> None:
  torch.manual_seed(seed) #fix seed started with torch.~~
  torch.cuda.manual_seed(seed) #fix seed started with torch.cuda.~~
  torch.cuda.manual_seed_all(seed) #fix seed when you use multi_gpu
  torch.backends.cudnn.deterministic = True #Pytorch ise cudnn as backend # this can slow down speed
  torch.backends.cudnn.benchmark = False #when model operate convolution, use optimized algorithm 
                                         #if input image size is too different, this is not good for performance of model
  np.random.seed(seed) #fix numpy seed
  random.seed(seed) #fix python seed