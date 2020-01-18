import torch.nn as nn

class MetricNet(nn.Module):

    def __init__(self, in_features=2*512, n_nodes=512):
        super(MetricNet, self).__init__()
        self.in_features = in_features
        self.n_nodes = n_nodes
        self.net = nn.Sequential(
            nn.Linear(self.in_features, self.n_nodes),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(self.n_nodes, self.n_nodes),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(self.n_nodes, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
