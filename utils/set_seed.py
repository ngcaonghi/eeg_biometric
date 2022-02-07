import torch, random, os
import numpy as np

def set_seed(seed):
    '''
    Set a seed to enforce reproducibility.

    Argument:
        - seed (int)
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)