''''
After certain depth of the N, thee accurray goes down. So, to solve this, we make some residual connections, it solves the issue. 

It learns the complex features, and using skip connections / Identity mapping, so, it can remember what it learn before. It won't forget what it learns before.
It never gets worser 
'''

import torch
import torch.nn as nn

"""
ResNet-18
ResNet-34
          ---
ResNet-50    |
ResNet-101   |   --- We are implementing these
ResNet-152   |
          ---
"""
## Implementation for ResNet_50, 101, 152 layers

class block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4 # no of channels after a block is always four times what it was when it entered.
        '''in_channels and out_channels remains the same for all layers in a block''' # as kernel size is constantly 1
        # conv block :1 
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=intermediate_channels, 
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        # conv block :1 

        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        # conv block :1  # 4 times the number of channels

        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels*self.expansion, 
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(intermediate_channels*self.expansion) # 64 * 4 = 256

        self.relu = nn.ReLU()

        self.identity_downsample = identity_downsample # conv layer for identity mapping  
        self.stride = stride

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        # Identity 
        if self.identity_downsample is not None:   # we use the identity down sample if we need to change the shape 
            identity = self.identity_downsample(identity)
        
        # only for 1st block, doing this without downsample, logically
        x += identity # making the residual connections
        x = self.relu(x)
        return x
    
class ResNet(nn.Module): # block - class, layers - list (how many times we need to use the block)
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        # initial layers
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=64, 
                               kernel_size=7, stride=2, padding=3, bias=False) 
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        '''34, 101, 152 has the same block, but the number of layers differs'''
    # ResNet-34 === [3, 4, 6, 3]
    # ResNet-101 === [3, 4, 23, 3]
    # ResNet-152 === [3, 8, 36, 3]

        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1) # for rest of the layers, we use stride of 2
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)  # at the end, it will be 128*4 = 512
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2) # 256 * 4 = 1024
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2) # 512 * 4 = 2048 channels at the end


        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # we defing the output size 
        self.fc = nn.Linear(512*4, num_classes) # 2048 --->  num_classes
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    # makes multiple block layers
    def _make_layer(self, block, num_residual_blocks, out_channels, stride): # num_residual_blocks - no of times we use the class 'block', # out_channels - no of output channels in each layer
        # stride will be 1 or 2
        identify_downsample = None
        layers = []
        '''eithier, we change the input size or the identity'''
        # when we are using the conv layers to change the identity, when we are using the identiity_downsamples
        if stride != 1 or self.in_channels != out_channels * 4:  # if the input_size is 56, 28, so we need to change the identity. here the stride will be 2, in order to reducet the channels to 56 -> 28 and 28 -> 14
            identify_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

        # in_channels, out_channels, 
        # layer 1: in-64, out-256 | 
        # layer 2: in-256, out-512 | 
        # layer 3: in-512, out-1024 | 
        # layer 4: in-1024, out-2048  ------ channels
        layers.append(
            block(self.in_channels, out_channels, identify_downsample, stride)
        ) # changes the no of channels 

        self.in_channels = out_channels * 4 # 256  - 64*4 
        # output is the 256

        # so, next, we have to map 256 layers to the 64
        """ we map 256 ->  64 and then the output of that block will be  64 * 4 -> 256, so again we have to map to 64"""
        # the stride will be same and num_channels will be in 256
        for i in range(num_residual_blocks - 1): # we already computed 1 res block
            layers.append(block(self.in_channels, out_channels)) # in-256, out-64 
        
        return nn.Sequential(*layers) # it will unpack the list and make as the layers
    


def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)




def test():
    BATCH_SIZE = 4
    net = ResNet101(img_channel=3, num_classes=1000)
    y = net(torch.randn(BATCH_SIZE, 3, 224, 224))
    assert y.size() == torch.Size([BATCH_SIZE, 1000])
    print(y.size()) # batch_Size, num_classes


if __name__ == "__main__":
    test()
