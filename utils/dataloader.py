import torch
import numpy as np
from torch.utils.data import Dataset, Sampler, RandomSampler, BatchSampler, DataLoader
from torch.autograd import Variable
from torch import nn
from collections import defaultdict

class TripletLossDataset(Dataset):
    '''
    Subclass of torch.utils.data.Dataset for returning triplets of data for 
    triplet loss function.

    Methods:
        - __init__(data, targets)
        - __getitem__(index)
        - __len__()

    Attributes:
        - data (np.array): a numpy array of shape (n_samples, ...)
        - labels (array-like): an array-like object containing labels
        - label_ids (defaultdict(list)): a defaultdict whose keys are the labels
        and whose values are lists of those label's indices in the self.labels
        array.
    '''
    def __init__(self, data, labels):
        '''
        Init method.

        Arguments:
            - data (np.array): a numpy array of shape (n_samples, ...)
            - labels (array-like): an array-like object containing labels
        '''
        self.data = data
        self.labels = labels
        label_ids = defaultdict(list)
        for i, y in enumerate(self.labels):
            label_ids[y].append(i)
        self.label_ids = label_ids
  
    def __getitem__(self, index):
        '''
        Overwrites torch.utils.data.Dataset's __getitem__ method.
        '''
        # get anchor
        anchor, y1 = self.data[index], self.labels[index]
        # get positive
        id2 = np.random.choice(self.label_ids[y1])
        while (id2 == index):
            id2 = np.random.choice(self.label_ids[y1])
        pos = self.data[id2]
        # get negative
        y3 = np.random.choice(self.labels)
        while (y3 == y1):
            y3 = np.random.choice(self.labels)
        neg = self.data[np.random.choice(self.label_ids[y3])]

        return anchor, pos, neg
  
    def __len__(self):
        '''
        Overwrites torch.utils.data.Dataloader's __len__ method.
        '''
        return len(self.data)

class _HardSampler(Sampler):
    '''
    Subclass of torch.utils.data.Sampler for sampling data that contain a 
    proportion of "hard" examples. 
    
    More specifically, each round of sampling is composed of 2 subroutines: the 
    first round samples n=batch_size triplets and chooses (n * hard_ratio) 
    triplets that produce the highest losses; the second round fills the rest of 
    the batch with randomly sampled triplets.
    '''
    def __init__(self, dataset, snn, batch_size, hard_ratio, device='cuda'):
        self.snn = snn
        self.dataset = dataset
        self.batch_size = batch_size
        self.hard_ratio = hard_ratio
        self.device = device
        self.criterion = nn.TripletMarginLoss(reduce=False)
        self.random_sampler = RandomSampler(dataset, replacement=True, 
                              num_samples=batch_size)
    
    def get_loss(self, anchor, pos, neg):
        a_embed, p_embed, n_embed = self.snn.forward(anchor, pos, neg)
        return self.criterion(a_embed, p_embed, n_embed)
    
    def __iter__(self):
        '''
        Overwrites torch.utils.data.Sampler's __iter__ method.
        '''
        device = self.device
        num_batches = len(self.dataset) // self.batch_size
        for b in range(num_batches):
            idx = list(self.random_sampler.__iter__())
            a_tensor = []
            p_tensor = []
            n_tensor = []
            for i in idx:
                a, p, n = self.dataset[i]
                a, p, n = a[None, ...], p[None, ...], n[None, ...]
                a_tensor.append(Variable(torch.from_numpy(a).float().to(device)))
                p_tensor.append(Variable(torch.from_numpy(p).float().to(device)))
                n_tensor.append(Variable(torch.from_numpy(n).float().to(device)))
            a_tensor = torch.cat(a_tensor, axis=0)
            p_tensor = torch.cat(p_tensor, axis=0)
            n_tensor = torch.cat(n_tensor, axis=0)
            losses = self.get_loss(a_tensor, p_tensor, n_tensor)
            sorted_idx = [i for l, i in sorted(zip(losses, idx), reverse=True)]
            hard_idx = sorted_idx[:int(self.batch_size * self.hard_ratio)]
            random_idx = list(self.random_sampler.__iter__())
            random_idx = random_idx[int(self.batch_size * self.hard_ratio):]
            for i in hard_idx + random_idx:
                yield i

    def __len__(self):
        '''
        Overwrites torch.utils.data.Sampler's __len__ method.
        '''
        return len(self.dataset)

class TripletLossDataLoader(DataLoader):
    '''
    Subclass of torch.utils.data.DataLoader for loading the TripletLossDataset.
    All methods are inherited from torch.utils.data.DataLoader except for
    __init__ which is overwritten.
    '''
    def __init__(self, dataset, snn, batch_size, hard_ratio, device='cuda', 
                **kwargs):
        '''
        Overwites torch.utils.data.DataLoader's __init__method.

        Arguments:
            - dataset (TripletLossDataset)
            - snn (nn.Module): the siamese network
            - batch_size (int)
            - hard_ratio (float): the proportion of "hard" examples within a
            batch. Each round of sampling is composed of 2 subroutines: the 
            first round samples n=batch_size triplets and chooses 
            (n * hard_ratio) triplets that produce the highest losses; the 
            second round fills the rest of the batch with randomly sampled 
            triplets.
            - device (str): either 'cuda' or 'cpu'.
        '''
        sampler = _HardSampler(
            dataset, snn, batch_size, hard_ratio, device
        )
        batch_sampler = BatchSampler(
            sampler=sampler, batch_size=batch_size, drop_last=True
        )
        super(TripletLossDataLoader, self).__init__(
            dataset=dataset, batch_sampler=batch_sampler, **kwargs
        )