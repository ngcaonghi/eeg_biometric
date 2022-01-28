import torch
import numpy as np
from torch.utils.data import Dataset, Sampler, RandomSampler, BatchSampler, DataLoader
from torch.autograd import Variable
from torch import nn

class TripleLossDataset(Dataset):
    def __init__(self, data, targets, device='cuda'):
        self.data = data
        self.labels = targets
        label_ids = defaultdict(list)
        for i, y in enumerate(self.labels):
            label_ids[y].append(i)
        self.label_ids = label_ids
        self.device = device
  
    def __getitem__(self, index):
        # get anchor
        anchor, y1 = self.data[index], self.labels[index]
        anchor = Variable(torch.from_numpy(anchor).float().to(self.device))
        # get positive
        id2 = np.random.choice(self.label_ids[y1])
        while (id2 == index):
            id2 = np.random.choice(self.label_ids[y1])
        pos = self.data[id2]
        pos = Variable(torch.from_numpy(pos).float().to(self.device))
        # get negative
        y3 = np.random.choice(self.labels)
        while (y3 == y1):
            y3 = np.random.choice(self.labels)
        neg = self.data[np.random.choice(self.label_ids[y3])]
        neg = Variable(torch.from_numpy(neg).float().to(self.device))

        return anchor, pos, neg
  
    def __len__(self):
        return len(self.data)

class _HardSampler(Sampler):
    def __init__(self, dataset, snn, batch_size, hard_ratio):
        self.snn = snn
        self.dataset = dataset
        self.batch_size = batch_size
        self.hard_ratio = hard_ratio
        self.criterion = nn.TripletMarginLoss(reduce=False)
        self.random_sampler = RandomSampler(dataset, replacement=True, 
                              num_samples=batch_size)
    
    def get_loss(self, anchor, pos, neg):
        a_embed, p_embed, n_embed = self.snn.forward(anchor, pos, neg)
        return self.criterion(a_embed, p_embed, n_embed)
    
    def __iter__(self):
        num_batches = len(self.dataset) // self.batch_size
        for b in range(num_batches):
            idx = list(self.random_sampler.__iter__())
            a_tensor = []
            p_tensor = []
            n_tensor = []
            for i in idx:
                a, p, n = self.dataset[i]
                a_tensor.append(a[None, ...])
                p_tensor.append(p[None, ...])
                n_tensor.append(n[None, ...])
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
        return len(self.dataset)

class TripletLossDataLoader(DataLoader):
    def __init__(self, dataset, snn, batch_size, hard_ratio, **kwargs):
        sampler = _HardSampler(
            dataset, snn, batch_size, hard_ratio
        )
        batch_sampler = BatchSampler(
            sampler=sampler, batch_size=batch_size, drop_last=True
        )
        super(TripletLossDataLoader, self).__init__(
            dataset=dataset, batch_sampler=batch_sampler, **kwargs
        )