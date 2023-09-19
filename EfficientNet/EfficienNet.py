import torch
import torch.nn as nn
from math import ceil


"""
## Architecture:
           --[HxW]-- --#Layers--
Conv 3x3   [224x224]  (1)

MBConv1 k3x3 - expand ration - 1, channels - 16 [112x112] (1)
MBConv6 k3x3 - expand ration - 6, channels - 24 [112x112] (2)

MBConv6 k5x5 - expand ration - 6, channels - 40 [56x56] (2)
 
MBConv6 k3x3 - expand ration - 6, channels - 80 [28x28] (3)

MBConv6 k5x5 - expand ration - 6, channels - 112 [14x14] (3)
MBConv6 k5x5 - expand ration - 6, channels - 192 [14x14] (4)

MBConv6 k3x3 - expand ration - 6, channels - 320 [7x7] (1)

Conv 1x1 & Pooling & FC - expand ration - 6, channels - 1280  [7x7] (1)

""" # using this, we implement the below
base_model = [
    # expand_ration, channels, repeats, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

 
phi_values = {
    '''tuple of: (phi_value, resolution, drop_rate)'''
    # depth  = alpha  ** phi, 
    # width = beta ** phi,
    # resolution = gamma ** phi

    "b0" : (0, 224, 0.2), # alhpa, beta, gamma -- values
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}

# conv block
class CNNBlock(nn.Module): # conv, bn, relu
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups  # depth wise convolution - to reduce the computational complexity and improve the efficieny of the model
            # each channel is processed independently with its set of filters
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU() # SiLU <--> Swish\
    
    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))
    

# if groups = 1, default, normal conv
# if groups = in_channels, depthwise conv
class SqueezeExcitation(nn.Module): # compute attention scores for each of the channel
    def __init__(self, in_channels, reduced_dim): # reduced_dim at middle
        super(SqueezeExcitation, self).__init__()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # C x H x W ---> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, kernel_size=1), # bring back the same dimension
            nn.Sigmoid(),         
        )
    
    def forward(self, x): # attention scores 
        return x * self.se(x) # each channel is multiplied by the values comes out from se() --- how much the channels is prioritized

class InvertedResidualBlock(nn.Module): # takes the input and expands to higher number of channels,  and then brings back to the same no of channel shtat was previously 
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expand_ratio, reduction=4, survival_prob=0.8): # expand_Ration --- sqeeze exitation
        # survival_prob -- for stochastic depth
        super().__init__(InvertedResidualBlock, self).__init__()

        self.use_residual = in_channels == out_channels and stride == 1

        hidden = in_channels * expand_ratio
        self.expand = in_channels != h

class EfficientNet(nn.Module):
    pass









'''
Config
Structure
CNN Block
Squeeze excitation
Inverted Residual Block (w. Stochastic depth)
EfficientNet
'''