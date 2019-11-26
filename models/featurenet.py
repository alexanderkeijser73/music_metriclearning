from torch import sigmoid
from torch.nn.functional import relu

from models.featurenet_modules import *


class FeatureNet(nn.Module):

    def __init__(self, n_freq_bins=80, in_channels=3, n_outputs=512, sigmoid=False):
        super(FeatureNet, self).__init__()
        self.conv0 = nn.Conv1d(in_channels=in_channels,
                               out_channels=64,
                               kernel_size=(3, n_freq_bins),
                               padding=(1, 0))
        self.block1 = ResBlockIdentity(64, 64)
        self.block2 = ResBlockConvShortcut(64, 128)
        self.block3 = ResBlockConvShortcut(128, 256)
        self.block4 = ResBlockConvShortcut(256, n_outputs)
        self.sigmoid = sigmoid
        self.output_dim = n_outputs

        # placeholder for the gradients
        self.gradients = None
        self.activations = None

    def forward(self, x):
        x = self.get_activations(x)

        if x.requires_grad:
            h = x.register_hook(self.save_gradients)

        # Global pooling
        x = torch.mean(x, dim=(2, 3))

        if self.sigmoid:
            x = sigmoid(x)

        return x

    # hook for the gradients of the activations
    def save_gradients(self, grad):
        self.gradients = grad

    # method for the gradient extraction
    def get_gradients(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        # First Conv layer
        x = relu(self.conv0(x))

        # ResNet blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x