import torch
import numpy as np

def mixup(images, bboxes, areas, alpha=1.0):
    """
    Randomly mixes the given list if images with each other
    
    :param images: The images to be mixed up
    :param bboxes: The bounding boxes (labels)
    :param areas: The list of area of all the bboxes
    :param alpha: Required to generate image wieghts (lambda) using beta distribution. In this case we'll use alpha=1, which is same as uniform distribution
    """
    # Generate random indices to shuffle the images
    indices = torch.randperm(len(images))
    shuffled_images = images[indices]
    shuffled_bboxes = bboxes[indices]
    shuffled_areas = areas[indices]
    
    # Generate image weight (minimum 0.4 and maximum 0.6)
    lam = np.clip(np.random.beta(alpha, alpha), 0.4, 0.6)
    print(f'lambda: {lam}')
    
    # Weighted Mixup
    mixedup_images = lam*images + (1 - lam)*shuffled_images
    
    mixedup_bboxes, mixedup_areas = [], []
    for bbox, s_bbox, area, s_area in zip(bboxes, shuffled_bboxes, areas, shuffled_areas):
        mixedup_bboxes.append(bbox + s_bbox)
        mixedup_areas.append(area + s_area)
    
    return mixedup_images, mixedup_bboxes, mixedup_areas, indices.numpy()