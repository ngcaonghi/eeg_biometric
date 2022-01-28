import torch
from torch import nn

# Convolutional models.

class Conv2dNet(nn.Module):
    '''
    nn.Module for 2D convolutional network. Convolution direction is performed 
    along 2 axes, which are EEG channels and time. Input EEG data must have the 
    shape of (n_batch, 1, n_eeg_channels, time_bins).

    Attributes:
        - cnn (nn.Sequential): nn layers
        - depth (int): number of convolutional layers 
    
    Methods:
        - __init__(self, depth)
        - forward(self, s)
    '''
    def __init__(self, depth=3, kernel_size=(5, 5, 3)):
        '''
        Init method.

        Parameters:
            - depth (int):  number of convolutional layers. Must be >= 1.
            - kernel_size(int, list, tuple, np.array): convolutional kernel 
            sizes. If int, the same kernel size is applied to all layers. 
            Otherwise, the object size must be equal to depth, and the kernel 
            sizes will be assigned in the order of the layers. 
        '''
        super(SiameseNetwork, self).__init__()
        self.depth = depth

        # check if depth value is valid
        if depth < 1:
            raise ValueError('Depth must be >=1.')

        # check if kernel_size is valid
        if (not isinstance(kernel_size, int)):
            if len(kernel_size) != depth:
                m = 'If kernel_size is a list, a tuple, or an np' + \
                    '.array, its length must be equal to depth.'
                raise ValueError(m)
            else:
                self.kernel_sizes = kernel_size
        else:
            self.kernel_sizes = [kernel_size] * depth
        
        # build layers
        layers = []
        layers.append(nn.Conv2d(1, 16, kernel_size=kernel_sizes[0], 
                                       padding='same'))
        for i in range(depth-1):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))
            layers.append(nn.Conv2d(2**(i+4), 2**(i+5), 
                          kernel_size=kernel_sizes[i+1], 
                          padding='same'))

        self.cnn = nn.Sequential(*layers)

    def forward(self, s):
        '''
        Forward method.

        Parameter:
            - s (torch.Tensor): EEG sequence. Must be of shape 
            (n_batch, 1, n_eeg_channels, time_bins)
        '''
        x = self.cnn(s)
        x = x.view(x.size()[0], -1)
        return x

class Conv1dNet(nn.Module):
    