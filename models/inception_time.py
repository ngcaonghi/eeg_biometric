import torch
import torch.nn.functional as F
from torch import nn

class InceptionTime(nn.Module):
    '''
    Pytorch implementation of the InceptionTime architecture based on 
    https://github.com/hfawaz/InceptionTime, with modifications specific to the
    problem of personal verification using EEG.

    Attributes:
        - incept_depth (int): number of inception modules
        - incept_kernels (tuple(int)): a 3-tuple of integers representing the 3
        kernel sizes of conv1d layers within each inception module.
        - avg_pool_kernel (int): the kernel size of the average pooling layer
        - output_size (int): the length of the output, i.e. the dimensionality
        of the embedding space
        - conv_1, conv_2, conv_3, res, ap1d, fc1, fc2 (nn.Module): torch neural 
        net modules.
        - 
    
    Methods:
        - __init__(incept_depth, incept_kernels, avg_pool_kernel, fc1_input_size, 
                   num_eeg_channels)
        - _inception_module
        - _shortcut
        - forward(s)
    '''
    def __init__(self, incept_depth=12, incept_kernels=(129, 65, 33, 17), 
                avg_pool_kernel=64, fc=True, output_size=50):
        '''
        Init method.
        Paramaters:
            - incept_depth (int): number of inception modules. Default at 9.
            - incept_kernels (tuple(int)): a 4-tuple of integers representing 
            the 4 kernel sizes of conv1d layers within each inception module.
            Should be odd numbers sampled from a logarithmic sequence and be in
            a decreasing order. Default at (129, 65, 33, 17). 
            - avg_pool_kernel (int): the kernel size of the average pooling 
            layer. Default at 64.
            - fc (bool): whether the model includes fully connected layers at
            the end. Default at True.
            - output_size (int): the length of the output, i.e. the 
            dimensionality of the embedding space. Default at 50.
        '''
        super(InceptionTime, self).__init__()

        # default attributes or values specific to the EEG data
        self._NUM_CHANNELS = 14
        self._SFREQ = 256
        self._EPOCH_LEN = 5
        self._DELTA_NUM_FILTERS = 4
        self._THETA_NUM_FILTERS = 3
        self._ALPHA_NUM_FILTERS = 4
        self._BETA_NUM_FILTERS = 3

        # atrributes from init arguments
        self.incept_depth = incept_depth
        self.incept_kernels = incept_kernels
        self.avg_pool_kernel = avg_pool_kernel
        self.fc = fc
        self.output_size = output_size
        k_1, k_2, k_3, k_4 = incept_kernels

        # convolutional layers for inception modules
        self.conv_1 = nn.Sequential(
            nn.Conv1d(self.num_eeg_channels, self._DELTA_NUM_FILTERS, 
                      kernel_size=1),
            nn.Conv1d(self._DELTA_NUM_FILTERS, self._DELTA_NUM_FILTERS, 
                      kernel_size=k_1, padding=k_1//2)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(self.num_eeg_channels, self._THETA_NUM_FILTERS, 
                      kernel_size=1),
            nn.Conv1d(self._THETA_NUM_FILTERS, self._THETA_NUM_FILTERS,
                      kernel_size=k_2, padding=k_2//2)
        )      
        self.conv_3 = nn.Sequential(
            nn.Conv1d(self.num_eeg_channels, self._ALPHA_NUM_FILTERS, 
                      kernel_size=1),
            nn.Conv1d(self._ALPHA_NUM_FILTERS, self._ALPHA_NUM_FILTERS, 
                                kernel_size=k_3, padding=k_3//2)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv1d(self.num_eeg_channels, self._BETA_NUM_FILTERS, 
                      kernel_size=1),
            nn.Conv1d(self._BETA_NUM_FILTERS, self._BETA_NUM_FILTERS, 
                                kernel_size=k_4, padding=k_4//2)
        )

        # res layers
        self.res = nn.Conv1d(num_eeg_channels, num_eeg_channels, kernel_size=1)

        # avg pool
        self.ap1d = nn.AvgPool1d(avg_pool_kernel)

        # fully connected
        fc1_input_size = (self._SFREQ * self._EPOCH_LEN * self._NUM_CHANNELS) \
                        // avg_pool_kernel
        self.fc1 = nn.Linear(fc1_input_size, output_size * 5)
        self.fc2 = nn.Linear(output_size * 5, output_size)

  
    def _inception_module(self, input):
        '''
        Define an inception module.
        '''
        conv_layers = []
        conv_layers.append(self.conv_1(input))
        conv_layers.append(self.conv_2(input))
        conv_layers.append(self.conv_3(input))
        conv_layers.append(self.conv_4(input))
        x = torch.cat(conv_layers, 1)
        x = F.relu(x)
        return x
  
    def _shortcut(self, input, output):
        '''
        Define skip connections (shortcuts) for residual blocks.
        '''
        shortcut_y = self.res(input)
        x = torch.add(shortcut_y, output)
        x = F.relu(x)
        return x                  

    def forward(self, s):
        '''
        Forward function.
        
        Parameter:
            - s (torch.Tensor): a torch tensor of size (num_batch, num_channels,
            length) representing an EEG epoch.

        Return: a torch tensor of size (num_batch, output_size) representing the
        lower-dimensional embedding of the EEG epoch.
        '''
        x = s
        input_res = s

        for d in range(self.incept_depth):
            x = self._inception_module(x)
            if d % 3 == 2:
                x = self._shortcut(input_res, x)
                input_res = x
        x = self.ap1d(x)
        x = torch.flatten(x, 1)
        if self.fc:
            x = self.fc1(x)
            x = self.fc2(x)
        x = x.view(x.size()[0], -1)
        return x