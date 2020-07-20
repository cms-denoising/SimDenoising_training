import torch 
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as f
from dataset import *
import glob
import uproot
from torch.utils.data import DataLoader
import numpy as np 
from models import DnCNN
import torch.optim as optim



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
        avg_loss/=(list(output_patches.size())[0] * (list(output_patches.size())[1]))
        return avg_loss;


class WeightedPatchLoss(nn.Module):
    def __initII(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(PatchLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, output, target, patch_size):
        avg_loss = 0
        for i in range(len(output)):
            # split output and target images into patches
            output_patches = output[i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            target_patches = target[i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            weighted_loss = 0
            # calculate loss for each patch of the image
            for i in range(list(output_patches.size())[0]):
                for j in range(list(output_patches.size())[1]):
                    weighted_loss += f.l1_loss(output_patches[i][j],target_patches[i][j]) * torch.mean(target_patches[i][j])
            avg_loss+=weighted_loss
        avg_loss/=(list(output_patches.size())[0] * (list(output_patches.size())[1]))
        return avg_loss;



"""
class WeightedPatchLoss(nn.Module):
    def __initII(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(PatchLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, output, target, patch_size):
        num_true=0
        num_false = 0
        for i in range(len(output)):
            #print("Histogram #" + str(i))
            # split output and target images into patches
            output_patches = output[i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            target_patches = target[i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            losses = np.zeros((list(output_patches.size())[0] * (list(output_patches.size())[1])))
            means = np.zeros((list(output_patches.size())[0] * (list(output_patches.size())[1])))
            count = 0
            # calculate loss for each patch of the image
            for i in range(list(output_patches.size())[0]):
                for j in range(list(output_patches.size())[1]):
                    losses[count] = f.l1_loss(output_patches[i][j], target_patches[i][j])
                    means[count] = torch.mean(target_patches[i][j])
                    count+=1
            #print(losses)
            #print(means)
            if (np.argmax(losses) == np.argmax(means)):
                #print("True")
                num_true+=1
            else:
                #print("False")
                num_false+=1
        return num_true, num_false
"""

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

if __name__ == "__main__":
    dataset = RootDataset(root_file='test.root', sigma = 15)
    loader = DataLoader(dataset=dataset, batch_size=100)
    model = DnCNN(channels = 1, num_of_layers=9)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    criterion = PatchLoss()
    for i, data in enumerate(loader, 0):
        model.train()
        model.zero_grad()
        optimizer.zero_grad()
        truth, noise = data
        output = model(noise.unsqueeze(1).float())
        num_true, num_false = criterion(output.squeeze(1), truth, 50)    
        print("Correct:   " + str(num_true))
        print("Incorrect: " + str(num_false))
        
        
