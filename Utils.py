import torch
import numpy as np
from Args import Args

def mm_normalize(data, mask):
    data_min = data[mask].min()
    data_max = data[mask].max()
    data = np.multiply((data - data_min)/(data_max-data_min), mask.astype(np.float32))
    return data
