import torch 
import numpy as np
import torch.nn as nn 
dtype = torch.float32

import torch
import torch.nn as nn
import torch.nn.functional as F

dtype = torch.float32  # 确保 dtype 已定义

class ConvolutionalNetwork(nn.Module):
    """
    Defining the model that classifies the images
    """

    def __init__(self):
        super().__init__()
        self.model = self.network()

    def __str__(self):
        return f"ConvNet structure: {self.model}"

    def network(self):

        CONV = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, stride=1, dtype=dtype),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3),

            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, dtype=dtype),
            nn.ReLU()
        )

        FC = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=96, out_features=48, dtype=dtype),
            nn.Sigmoid(),
            nn.Linear(in_features=48, out_features=10, dtype=dtype),
            nn.LogSoftmax(dim=1)
        )

        return nn.Sequential(CONV, FC)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = ConvolutionalNetwork()
    testdata = torch.randn(64, 1, 28, 28, dtype=dtype)
    out = model(testdata)
    print(out.shape)  # 应该是 [64, 10]
