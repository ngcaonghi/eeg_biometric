import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam, RMSprop
from torch_optimizer import AdaBelief, Yogi
from torch.optim.lr_scheduler import OneCycleLR
from collections import defaultdict

# methods and classes in this repo
from set_seed import set_seed
from dataloader import TripletLossDataLoader

def _var_data(triplet, device):
    '''
    Helper method to return torch.autograd.Variables from a triplet of tensors. 
    '''
    anchor, pos, neg = triplet
    anchor = Variable(anchor.float().to(device))
    pos = Variable(pos.float().to(device))
    neg = Variable(neg.float().to(device))
    return anchor, pos, neg

def train(model, train_dataset, val_dataset, epochs, batch_size, 
          hard_ratio=0.2, 
          optimizer='RMSprop', 
          lr=5e-4,
          weight_decay=1e-5,
          max_lr=1e-3,
          seed=1024):
    '''
    Train a siamese network and return its training and validation losses.

    Arguments:
        - model (nn.Module): the siamese network.
        - train_dataset (TripletLossDataset, Subset): a 
        dataloader.TripletLossDataset or its torch.utils.data.Subset containing 
        data for training.
        - val_dataset (TripletLossDataset, Subset): a 
        dataloader.TripletLossDataset or its torch.utils.data.Subset containing 
        data for validation.
        - epochs (int): number of epochs.
        - batch_size (int)
        - hard_ratio (float): the proportion of "hard" examples within a batch. 
        Each round of sampling is composed of 2 subroutines: the first round 
        samples n=batch_size triplets and chooses (n * hard_ratio) triplets that
        produce the highest losses; the second round fills the rest of the batch
        with randomly sampled triplets.
        - optimizer (str): name of the optimizer. Must be one of the following:
        Adam, AdaBelief, Yogi, and RMSprop. Default at 'RMSprop'.
        - lr (float): the initial learning rate of the optimizer. Default at 
        5e-4.
        - weight_decay (float): the weight decay of the optimizer. Default at
        1e-5.
        - max_lr (float): the maximum learning rate that the 1cycle scheduler
        can reach. Default at 1e-3.
        See the documentation for torch.optim.lr_scheduler.OneCycleLR for more
        details.
        - seed (int): manual seed for reproducibility.

    Return: 
        - (train_losses, val_losses): a tuple of lists containing the training 
        losses and the validation losses, respectively.
    '''
    set_seed(seed)
    train_losses = []
    val_losses = []

    # set device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    # declare loss function
    criterion = nn.TripletMarginLoss().to(device)
    
    # declare optimizer
    if optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'AdaBelief':
        optimizer = AdaBelief(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'Yogi':
        optimizer = Yogi(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'RMSprop':
        optimizer = RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError('Optimizer must be one of the following: '
                        +'Adam, AdaBelief, Yogi, and RMSprop.')

    # use the 1-cycle policy on the optimizer
    scheduler = OneCycleLR(optimizer, 
                           max_lr=max_lr, 
                           epochs=epochs, 
                           steps_per_epoch=len(train_dataset) // batch_size)

    # creates dataloader
    train_dataloader = TripletLossDataLoader(
        train_dataset, snn=model, batch_size=batch_size, hard_ratio=hard_ratio, 
        device=device, shuffle=False
    )
    val_dataloader = TripletLossDataLoader(
        val_dataset, snn=model, batch_size=batch_size*2, hard_ratio=0, 
        device=device, shuffle=False
    )

    # training routine
    for epoch in range(epochs):
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(train_dataloader, 0):
            anchor, pos, neg = _var_data(data, device)
            optimizer.zero_grad()
            x1, x2, x3 = model(anchor, pos, neg)
            loss = criterion(x1, x2, x3)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            epoch_steps += 1

        print("[%d] training loss: %.3f" % (epoch + 1,
                                  running_loss / epoch_steps))
        train_losses.append(running_loss/epoch_steps)
        running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0

        for i, data in enumerate(val_dataloader, 0):
            with torch.no_grad():
                anchor, pos, neg = _var_data(data, device)
                x1, x2, x3 = model(anchor, pos, neg)
                loss = criterion(x1, x2, x3)
                val_loss += loss.cpu().numpy()
                val_steps += 1
        running_val_loss = val_loss / val_steps
        print("[%d] validation loss: %.3f" % (epoch + 1, running_val_loss))
        val_losses.append(running_val_loss)
    
    return train_losses, val_losses