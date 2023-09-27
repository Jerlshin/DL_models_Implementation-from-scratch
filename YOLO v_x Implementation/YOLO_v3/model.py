import torch
import torch.nn as nn

# Taken from official paper mam
# conv is same throughout
"""Architecture is given below"""

# https://arxiv.org/pdf/1804.02767.pdf

config = [
    (32, 3, 1), # filters, kernel_size, stride
    (64, 3, 2),
    ["B", 1], # B - residual blocks followed by no of repeats 
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S", # scale prediction block and computing YOLO loss. we have 3 S in architecture
    (256, 1, 1),
    "U", # Upsampling the feature map and concatenating with a previous layer
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]
# S - in architecture, we have 4 conv, but, in implementation, we have 2 conv -- that's what it means 

# The network predicts 4 coordinates for each bbox
# t_x, t_y, t_w, t_h
# If the cells is offset from the top left corner of the image by (C_x, C_y) and the bbox box prior has width and height (p_w, p_h)



'''

Convolutional - 32F, 3x3, 256x256
Convolutional - 64F, 3x3/2, 128x128

Convolutional 32F, 1x1 --
Convolutional 64F, 3x3   | -- x1 -- 128x128
Residual               __|

Convolutional 128F, 3x3/2 -- 64x64

Convolutional 64F, 1x1 --
Convolutional 128F, 3x3  | -- x2 -- 64x64
Residual               __|


Convolutional 256F, 3x3/2 -- 32x32

Convolutional 128F, 1x1 --
Convolutional 256F, 3x3  | -- x8 -- 32x32
Residual               __|


Convolutional 512F, 3x3/2 -- 16x16

Convolutional 256, 1x1 --
Convolutional 512, 3x3   | -- x8 -- 16x16
Residual               __|


Convolutional 1024F, 3x3/2 -- 8x8

Convolutional 512F, 1x1  --
Convolutional 1024F, 3x3   | -- x4 -- 8x8
Residual                 __|


Avgpool - Global
Connected - 1000
Softmax

'''

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs ): # batch_norm and activate
        super(CNNBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              bias=not bn_act, **kwargs)
        # if we use batch norm and activation in the block, then we dont want to use Bias

        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act
    
    def forward(self, x):
        if self.use_bn_act:
            return self.leaky_relu(self.bn(self.conv(x)))
        else:
            return self.conv(x)
        
class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual_True, num_repeats=1):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList() # list of the modules

        for _ in range(num_repeats):
            self.layers += [
                nn.Sequential( # we add sequential to run through both of these
                    CNNBlock(channels, channels//2, kernel_size=1, padding=0), # 1x1 conv
                    CNNBlock(channels//2, channels, kernel_size=3, padding=1) # 3x3 conv 
                )
            ]
        self.use_residual = use_residual_True
        self.num_repeats = num_repeats
    
    def forward(self, x):
        for layer in self.layers:
            if self.use_residual: # if we want to use skip connection
                x = layer(x) + x # after running through conv layers, we add the input x to it
            else:
                x = layer(x)
            # x = layer(x) + x if self.use_residual else layer(x)
        return x

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ScalePrediction, self).__init__()

        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2*in_channels, kernel_size=3, padding=1),
            CNNBlock(2*in_channels, 3 * (num_classes + 5), bn_act=False, kernel_size=1) # for every anchor box, we need one node for eac class that we want to predict. 
            # we have 3 anchors per cell -- 3 * (4 + 1) - 4 bbox offsets, 1 objectnedd prediction 
        )
        self.num_classes = num_classes
    
    def forward(self, x):
        return (
            self.pred(x) # splitting the long vector to dim, unsqueezing, 
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            # 3-anchor box, num_classes + 5 - as another dim, so splitting the 1st dim into two dim
            .permute(0, 1, 3, 4, 2) # as num_classes is common, but that to the last dim, and all the calculations will be concatenated in the last dim 
        ) 
        '''
        # N - num of examples, each example has 3 anchors, 13x13 grid for each grid, we have num_classes + 5 ouput 
        # N, 3, 13, 13, num_classes + 5 --- 13, 26, 52
        '''

class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20): # 20 for PASCAL VOC dataset, 80 for COCO dataset
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()
    
    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        # check if it is tuple, list or a string
        for module in config:
            if isinstance(module, tuple): # conv blocks
                out_channels, kernel_size, stride = module # extract all
                # we have two conv, 1x1, and 3x3
                ''' padding is 1 is kernel size is 3 and 0 for 1x1 conv'''
                layers.append(
                    CNNBlock(in_channels, out_channels,
                             kernel_size=kernel_size, stride=stride,
                             padding=1 if kernel_size == 3 else 0) 
                )
                in_channels = out_channels # out_channels of the prev block is the in_channels of the nest block
            
            elif isinstance(module, list): # residual blocks
                num_repeats = module[1]
                layers.append(
                    ResidualBlock(in_channels, use_residual_True=True, num_repeats=num_repeats)
                ) # we use skip connections here
            
            elif isinstance(module, str): # upsampling or scale prediction
                if module == "S": # scale prediction
                    '''Architecture of the Scale Prediction'''
                    layers += [ # 4 layer - 1, 1, 2
                        ResidualBlock(in_channels, use_residual_True=False, num_repeats=1), # we dont use skip connections here, just the conv block
                        CNNBlock(in_channels, in_channels//2, kernel_size=1), # 1x1 conv 
                        ScalePrediction(in_channels//2, num_classes=self.num_classes)
                    ]
                    # we dont want to conine from the scale prediction block
                    # we want to continue from the CNN block
                    in_channels = in_channels // 2

                elif module == "U": # upsampling, we use Upsample layers
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3 # after upsampling, scale_factor=2
                    ''' After upsampling, unlike in architecture, we just concatenate, there will be no conv layer in between the upsampling layers and the concat followed by the conv block'''
                    '''We want to concatenate the one that is last before the Scale prediction'''
        return layers

    def forward(self, x):
        outputs = [] # for scale pred
        route_connections = [] # for concatenating the channels 

        for layer in self.layers:
            # when it is scale prediction, we want to append the output
            '''but, we want to continue from the 1st layer of the ScalePrediction as in the architecture'''
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x)) 
                # dont want to continue from the detection layer
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                # we want to add the route connection
                route_connections.append(x) # save the weights 
            
            elif isinstance(layer, nn.Upsample):
                # concat to the last route connection
                x = torch.cat([x, route_connections[-1]], dim=1) # along the channels, in pytorch 
                route_connections.pop() # remove the last connection after concatenating
        
        return outputs
    
# YOLOv1 - 448
# YOLOv3 - 416 (multi scale prediction)
