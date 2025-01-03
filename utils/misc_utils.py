import random

import numpy as np
import torch
from skimage import filters


def to_output_format(image):
    image = (image + 1.0) / 2.0  # (C,H,W), -1,1 -> 0,1
    image = (image.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return image


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def gaussian_smooth(x):
    x = filters.gaussian(x, 10)
    x -= x.min()
    x /= x.max()
    return x
