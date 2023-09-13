# The conv stride is fixed to 1 pixel. 
# input of ConNets is 224 x 224 RGB image.
'''
Preprocessing: Subtracting the mean of RGB value, computed on the training set, from each pixel.
Filters: 3 x 3
- We also use 1 x 1 conv filters for linear transformation of the input channel.
The spatial padding to control the the size of the output feature maps and maintain the spatial information
Max-Pooling 2 x 2 window with stride 2

16 weight layers. the resolution stays the same. so, same convolution
'''


""" VGG - 16 Architecture """ # Go to en of the code for architecture

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Architecture
VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
# Then flatten and 4096, 4096, 1000 linear layers
# Then, Softmax

class VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG_net, self).__init__()
        # generalization
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG16) # padding the strides and MaPool are same

        '''
        for 5 MaxPool Layers, so divide by 5

        512*7*7
        # 224 is the image size
        224/(2**5)   --- 5 M layers, we used 2x2 size kernels and 2x2 strides

        512*7*7 ---> 25088 
        '''
        self.fcs = nn.Sequential(
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )


    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1) # flatenning
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels # initially 3

        for x in architecture:
            if type(x) == int: # conv layers
                out_channels = x 
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                     kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                     nn.BatchNorm2d(x),
                                     nn.ReLU()]
                in_channels = x # for next layer 
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
        
        return nn.Sequential(*layers) # unpacking all the layers we stored in the list






model = VGG_net(in_channels=3, num_classes=1000)

''' Add batchnorm and relu to each conv layer
16 weight layers

conv3-64 x2
--MaxPool

conv3-128 x2
--MaxPool

conv3-256 x3
--MaxPool

conv3-512 x3
--MaxPool

conv3-512 x3
--MaxPool

FC-4096
FC-4096
FC-1000
Softmax  # Multiclass
'''