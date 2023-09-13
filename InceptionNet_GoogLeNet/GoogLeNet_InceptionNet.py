'''
Paper: Going Deeper with Convolutions - 2014 - SOTA 

Abstract:
---------
    For classification and detection.
    Improved utilization of the computing resources inside the network.
    Increasing the depth and width of the network while keeping the computational budget constant.
    Architectural decision based on the Hebbian principal and the inituition of multi-scale processing.
    22 layers Deep network

Filter size: 1x1, 3x3, 5x5

a) Inception Module, Naive Version. --- 5x5 is expensive
b) Inception Module with dimention reductions --- 1x1 conv reduces the dim before give to 5x5 conv, so that 5x5 will take place on less dimension, therefore, reducing the filters 
'''
import torch 
import torch.nn as nn

"""
# Inception Block

Prev_block:
        1x1 conv 
        1x1 conv - 3x3 conv --- kernel 3x3, stride = 1, paddiing = 1
        1x1 conv - 5x5 conv --- kernel 5x5, stride = 1, padding = 2
        3x3 MP - 1x1 conv

concat all layers --> next block
"""
# conv block for inceptions block 
# only performs the single conv layers with activation and batchnorm
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs): # to make it general, we use kernel size, stride, for all these we use **kwargs
        super(conv_block, self).__init__()

        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs) # kernel size=(1,1), (2,2), (3,3)
        self.batchnorm = nn.BatchNorm2d(out_channels) # not invented at that time, but we add

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))


class Inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool): # all these are filters
        # output of the 1x1 conv 
        # reduction to 3x3 | # output of the 3x3
        # reduction to 5x5 | # output of the 5x5
        # output of 1x1_pool, output after the max pooling
        super(Inception_block, self).__init__()
        ### 4 branches
        # for 1x1 conv
        self.branch1 = conv_block(in_channels=in_channels,
                                  out_channels=out_1x1,
                                  kernel_size = 1)
        # for 3x3 branch
        self.branch2 = nn.Sequential( # we combine two conv_blocks as  a layer
            # reduction
            conv_block(in_channels, red_3x3, kernel_size=1),
            conv_block(red_3x3, out_3x3, kernel_size=3, padding=1)
        )

        # for branch 3
        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=1), # adding=0, stride=1, default
            conv_block(red_5x5, out_5x5, kernel_size=5, padding=2)
        )

        # for branch 4
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out_1x1pool, kernel_size=1) # 
        )
    # we didn't the change the input size, 28x28, but no of filters have changed
    def forward(self, x): # we concat all the filters
        # N - no of images
        # N * filters * 28x28 # so, we concat along the dim=1, filters
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1) # along the 1st dim 


class InceptionAux(nn.Module): # aux block
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = conv_block(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3, aux_logits=True, num_classes = 1000):
        super(GoogLeNet, self).__init__()
        assert aux_logits == True or aux_logits == False

        self.aux_logits = aux_logits

        ## Block:1 
        self.conv1 = conv_block(in_channels=in_channels, out_channels=64, 
                                kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_block(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ## Block:2, (Inception 3a, 3b)
        # Order: [ in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool ]
        self.inception3a = Inception_block(in_channels=192, out_1x1=64, red_3x3=96, out_3x3=128, red_5x5=16, out_5x5=32, out_1x1pool=32)
        self.inception3b = Inception_block(in_channels=256, out_1x1=128, red_3x3=128, out_3x3=192, red_5x5=32, out_5x5=96, out_1x1pool=64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ## Block:3, (Inception 4a, 4b, 4c, 4d, 4e)
        self.inception4a = Inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ## Block:4, (Inception 5a, 5b)
        self.inception5a = Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_block(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(in_features=1024, out_features=1000)

        # if we need aux logits
        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = self.aux2 = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)

        # <Auxilary output>
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        # <Auxilary output>
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1) # flatenning
        x = self.dropout(x)        
        x = self.fc1(x)

        if self.aux_logits and self.training:
            return aux1, aux2, x
        else:
            return x
        
"""
# Architecture -- we also use Batchnorm here

Conv 
MP - Max Pooling
Conv
MP

Inception 3a , 3b

MP

Inception 4a, 4b, 4c, 4d, 4e

MP

Inception 5a, 5b

AP - Average Pooling

Dropout (0.4)

Linear
softmax
"""

if __name__ == "__main__":
    x = torch.randn(3, 3, 224, 224) # 3 images, 3 channels, h, w
    model = GoogLeNet()
    # print(model(x).shape) # should output 3x1000
    print(len(model(x)))
    print(f"\n {model(x)}")