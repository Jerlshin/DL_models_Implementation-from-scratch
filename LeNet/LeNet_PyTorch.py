import torch
import torch.nn as nn

# for handwritten digit recognition

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # It uses tanh and sigmoid activation function, but we can use ReLU, ReLu was not invented atthat time 
        # we use another weight initialization.
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        # for grayscale image, only one channel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10) # for MNIST dataset
    
    def forward(self, x):
        self.relu(self.conv1(x))
        self.pool(x)
        self.relu(self.conv2(x))
        self.pool(x)
        self.relu(self.conv3(x)) # num_examples x 120 x 1 x 1 ---> num_examples x 120
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x) # no avtivation at the last layer

