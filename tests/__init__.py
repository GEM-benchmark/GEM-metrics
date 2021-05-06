import random
import numpy as np
random.seed(0)
np.random.seed(0)

try:
    import torch
    torch.manual_seed(0)
except ImportError as e:
    pass
