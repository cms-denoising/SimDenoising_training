"""
Adapted from https://github.com/SaoYan/DnCNN-PyTorch/blob/master/models.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable

class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=9):
        super(DnCNN, self).__init__()
        kernel_size = 1
        padding = 0
        features = 100
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out

class PatchLoss(nn.Module):
    def __initII(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(PatchLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, output, target, patch_size):
        avg_loss = 0
        for i in range(len(output)):
            # split output and target images into patches
            output_patches = output[i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            target_patches = target[i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            max_patch_loss = 0
            # calculate loss for each patch of the image
            for i in range(list(output_patches.size())[0]):
                for j in range(list(output_patches.size())[1]):
                    max_patch_loss = max(max_patch_loss, f.l1_loss(output_patches[i][j], target_patches[i][j]))
            avg_loss+=max_patch_loss
        avg_loss/=len(output)
        return avg_loss;


if __name__=="__main__":
    criterion = PatchLoss()
    dtype = torch.FloatTensor
    
    x = Variable(torch.randn(100, 100).type(dtype), requires_grad=False)
    y = Variable(torch.randn(100, 100).type(dtype), requires_grad=False)
    loss = criterion(x, y, 10)
    net = DnCNN()
    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    
