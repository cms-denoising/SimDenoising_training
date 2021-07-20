import numpy as np
import uproot as up
import matplotlib.pyplot as plt
import matplotlib as mpl
import random 
import torch.utils.data as udata
import torch 
import math

def get_branch(file_paths):
    file_path = file_paths.split(' ')
    branch = []
    for i, elem in enumerate(file_path):
        file = up.open(file_path[i])
        tree = file["g4SimHits/tree"]
        events = tree["bin_weights"].array()
        branch = np.concatenate((branch, events))
    return branch;

def get_flips():
    flipx = random.randint(0, 1)
    flipy = random.randint(0, 1)
    rot = random.randint(0, 3)
    return flipx, flipy, rot
    

def get_bin_weights(branch, n, flipx, flipy, rot):
    data = np.zeros((100,100))
    count = 0
    for y in range(100):
        for x in range(100):
            data[99-x][y]=branch[n][count]
            #if (data[99-x][y] != 0):
                #data[99-x][y] = math.log10(data[99-x][y])
            count+=1
    # do random rotation/flips
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
    def __init__(self, fuzzy_root, sharp_root, transform='none'):
        self.sharp_root = sharp_root
        self.fuzzy_root = fuzzy_root
        #self.sigma = sigma
        self.sharp_branch = get_branch(sharp_root)
        self.fuzzy_branch = get_branch(fuzzy_root)
        self.transform = transform
        if self.transform not in self.allowed_transforms:
            raise ValueError("Unknown transform: {}".format(self.transform))
        
    def __len__(self):
        if len(self.sharp_branch) == len(self.fuzzy_branch):
            return len(self.sharp_branch)
        else:
            print("Sharp and fuzzy dataset lengths do not match")

    def __getitem__(self, idx):
        flipx, flipy, rot = get_flips()
        sharp_np = get_bin_weights(self.sharp_branch, idx, flipx, flipy, rot).copy()
        fuzzy_np = get_bin_weights(self.fuzzy_branch, idx, flipx, flipy, rot).copy()
                    
        if self.transform=="log10":
            sharp_np = np.log10(sharp_np, where=sharp_np>0)
            fuzzy_np = np.log10(fuzzy_np, where=fuzzy_np>0)
        elif self.transform=="normalize":
            means = np.mean(sharp_np)
            stdevs = np.std(sharp_np)
            sharp_np -= means
            sharp_np /= stdevs
            fuzzy_np -= means
            fuzzy_np /= stdevs
        
        sharp = torch.from_numpy(sharp_np)
        fuzzy = torch.from_numpy(fuzzy_np)
        return sharp, fuzzy 

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
