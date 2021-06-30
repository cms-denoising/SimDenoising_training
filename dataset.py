import numpy as np
import uproot as up
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import torch.utils.data as udata
import torch
import math

def get_all_histograms(file_path):
    file = up.open(file_path)
    tree = file["g4SimHits/tree"]
    branch = tree["bin_weights"].array()
    return branch;

def get_bin_weights(branch, n):
    data = np.zeros((100,100))
    count = 0
    for y in range(100):
        for x in range(100):
            data[99-x][y]=branch[n][count]
            #if (data[99-x][y] != 0):
                #data[99-x][y] = math.log10(data[99-x][y])
            count+=1
    # do random rotation/flips
    flipx = random.randint(0, 1)
    flipy = random.randint(0, 1)
    rot = random.randint(0, 3)
    if (flipx):
        data = np.fliplr(data)
    if flipy:
        data = np.flipud(data)
    for i in range(rot):
        data = np.rot90(data)
    return data;

def add_noise(data, sigma):
    return np.clip(data + np.random.normal(loc=0.0,scale=sigma, size=[100,100]), a_min=0, a_max=None);

class RootDataset(udata.Dataset):
    allowed_transforms = ["none","normalize","log10"]
    def __init__(self, root_file, sigma, transform="none"):
        self.root_file = root_file
        self.sigma = sigma
        self.histograms = get_all_histograms(root_file)
        self.transform = transform
        if self.transform not in self.allowed_transforms:
            raise ValueError("Unknown transform: {}".format(self.transform))

    def __len__(self):
        return len(self.histograms)

    def __getitem__(self, idx):
        truth_np = get_bin_weights(self.histograms, idx).copy()
        noisy_np = add_noise(truth_np, self.sigma).copy()

        if self.transform=="log10":
            truth_np = np.log10(truth_np, where=truth_np>0)
            noisy_np = np.log10(noisy_np, where=noisy_np>0)
        elif self.transform=="normalize":
            means = np.mean(truth_np)
            stdevs = np.std(truth_np)
            truth_np -= means
            truth_np /= stdevs
            noisy_np -= means
            noisy_np /= stdevs

        truth = torch.from_numpy(truth_np)
        noisy = torch.from_numpy(noisy_np)
        return truth, noisy

if __name__=="__main__":
    dataset = RootDataset("test.root", 1)
    truth, noise = dataset.__getitem__(0)
    plt.imshow(truth.numpy())
    plt.colorbar()
    plt.savefig("truth.png")
    plt.close()
    plt.imshow(noise.numpy())
    plt.colorbar()
    plt.savefig("noise.png")
