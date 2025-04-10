import torch.nn as nn
import torch.nn.functional as F
import torch

class EEGNet(nn.Module):
    def __init__(self, n_channels=21, n_samples=1250, num_classes=2, dropout_rate=0.5):
        super(EEGNet, self).__init__()

        # Temporal convolution: learn temporal filters across time dimension
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 64), padding=(0, 32), bias=False),  # shape: (B, 8, C, T)
            nn.BatchNorm2d(8)
        )

        # Depthwise spatial convolution: one spatial filter per temporal filter
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(n_channels, 1), groups=8, bias=False),  # shape: (B, 16, 1, T)
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout_rate)
        )

        # Separable convolution: combines temporal filters again
        self.separableConv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout_rate)
        )

        # Dynamically compute the flattened feature size after conv layers
        dummy_input = torch.zeros(1, 1, n_channels, n_samples)
        with torch.no_grad():
            x = self.firstconv(dummy_input)
            x = self.depthwiseConv(x)
            x = self.separableConv(x)
            flattened_size = x.reshape(1, -1).shape[1]  # dynamically computed

        # Final classification layer
        self.classifier = nn.Linear(flattened_size, num_classes)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.reshape(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x