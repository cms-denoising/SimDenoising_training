"""
Adapted from https://github.com/SaoYan/DnCNN-PyTorch/blob/master/models.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as f

class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=9):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 0
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        
        return out

if __name__=="__main__":
    net = DnCNN()
    print(net)
    input = torch.randn(1, 1, 6, 6)
    out = net(input)
    print(out)
