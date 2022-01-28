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
        - fc1_input_size (int): the size of the first fully connected layer
        - num_eeg_channels (int): number of EEG channels
        - output_size (int): the length of the output, i.e. the dimensionality
        of the embedding space
        - conv_0_1, conv_0_2, conv_1, conv_2, conv_3, res, ap1d, fc1, fc2 
        (nn.Module): torch neural net modules.
    
    Methods:
        - __init__(incept_depth, incept_kernels, avg_pool_kernel, fc1_input_size, 
                   num_eeg_channels)
        - _inception_module
        - _shortcut
        - forward(s)
    '''
    def __init__(self, incept_depth=9, incept_kernels=(33, 17, 9), 
                avg_pool_kernel=64, fc1_input_size=280, num_eeg_channels=14,
                output_size=14):
        '''
        Init method.
        Paramaters:
            - incept_depth (int): number of inception modules. Default at 9.
            - incept_kernels (tuple(int)): a 3-tuple of integers representing 
            the 3 kernel sizes of conv1d layers within each inception module.
            Should be odd numbers. Default at (33, 17, 9). 
            - avg_pool_kernel (int): the kernel size of the average pooling 
            layer. Default at 64.
            - fc1_input_size (int): the size of the first fully connected layer.
            Default at 280.
            - num_eeg_channels (int): number of EEG channels. Default at 14.
            - output_size (int): the length of the output, i.e. the 
            dimensionality of the embedding space. Default at 14.
        '''
        super(InceptionTime, self).__init__()
        self.incept_depth = incept_depth
        self.incept_kernels = incept_k
        self.avg_pool_kernel = avg_pool_kernel
        self.fc1_input_size = fc1_input_size
        self.num_eeg_channels = num_eeg_channels
        self.output_size = output_size
        k_1, k_2, k_3 = incept_k
        num_ch_1 = num_eeg_channels // 3
        num_ch_2 = (num_eeg_channels - num_ch_1) // 2
        num_ch_3 = num_eeg_channels - num_ch_2 - num_ch_1

        # inception layers
        self.conv_0_1 = nn.Conv1d(self.num_eeg_channels, 4, kernel_size=1)
        self.conv_0_2 = nn.Conv1d(self.num_eeg_channels, 5, kernel_size=1)
        self.conv_1 = nn.Conv1d(num_ch_1, num_ch_1, kernel_size=k_1, padding=k_1//2)
        self.conv_2 = nn.Conv1d(num_ch_2, num_ch_2, kernel_size=k_2, padding=k_2//2)
        self.conv_3 = nn.Conv1d(num_ch_3, num_ch_3, kernel_size=k_3, padding=k_3//2)

        # res layers
        self.res = nn.Conv1d(num_eeg_channels, num_eeg_channels, kernel_size=1)

        # avg pool
        self.ap1d = nn.AvgPool1d(avg_pool_kernel)

        # fully connected
        self.fc1 = nn.Linear(fc1_input_size, output_size * 5)
        self.fc2 = nn.Linear(output_size * 5, output_size)

  
    def _inception_module(self, input):
        '''
        Define an inception module.
        '''
        input_1 = self.conv_0_1(input)
        input_2 = self.conv_0_2(input)
        conv_layers = []
        conv_layers.append(self.conv_1(input_1))
        conv_layers.append(self.conv_2(input_2))
        conv_layers.append(self.conv_3(input_2))
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
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(x.size()[0], -1)
        return x