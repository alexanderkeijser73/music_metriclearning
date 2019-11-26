import torch.nn as nn
from torch.nn.functional import relu

class ResBlockIdentity(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=(3, 1)):
        super(ResBlockIdentity, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv1 = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=self.out_channels,
                               kernel_size=self.kernel_size,
                               padding=self.padding)
        self.conv2 = nn.Conv2d(in_channels=self.out_channels,
                               out_channels=self.out_channels,
                               kernel_size=self.kernel_size,
                               padding=self.padding)

    def forward(self, x):
        F = relu(self.conv1(x))
        F = self.conv2(F)
        # shortcut
        sc = x
        out = relu(F + sc)
        return out


class ResBlockConvShortcut(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 1)):
        super(ResBlockConvShortcut, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv1 = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=self.out_channels,
                               kernel_size=self.kernel_size,
                               stride=(2, 1),
                               padding=self.padding)
        self.conv2 = nn.Conv2d(in_channels=self.out_channels,
                               out_channels=self.out_channels,
                               kernel_size=self.kernel_size,
                               padding=self.padding)
        self.shortcut = nn.Conv2d(in_channels=self.in_channels,
                                  out_channels=self.out_channels,
                                  kernel_size=self.kernel_size,
                                  stride=(2, 1),
                                  padding=self.padding)

    def forward(self, x):
        F = relu(self.conv1(x))
        F = self.conv2(F)
        # shortcut
        sc = self.shortcut(x)
        out = relu(F + sc)
        return out

