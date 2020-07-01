import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as f



# returns numpy array with loss for each image patch
def patch_based_losses(output, target, patch_size):
    # split output and target images into patches
    output_patches = output.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    target_patches = target.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    losses = np.zeros(list(output_patches.size())[1])
    # calculate loss for each patch of the image
    for i in range(list(output_patches.size())[1]):
        losses[i] = f.l1_loss(output_patches[0][i], target_patches[0][i])
    return losses;


#example
if __name__=="__main__":
    dtype = torch.FloatTensor
    x = Variable(torch.randn(100, 100).type(dtype))
    y = Variable(torch.randn(100, 100).type(dtype))
    print(patch_based_losses(x, y, 5))


