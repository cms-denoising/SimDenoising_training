import numpy as np
import uproot as up
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import torch.utils.data as udata
import torch
import math

def get_flips():
    flipx = random.randint(0, 1)
    flipy = random.randint(0, 1)
    rot = random.randint(0, 3)
    return flipx, flipy, rot
    

def get_bin_weights(branch, n, x_bins, y_bins, flipx = None, flipy=None, rot=None):
    data = np.zeros((x_bins,y_bins))
    count = 0
    for y in range(y_bins):
        for x in range(x_bins):
            data[(x_bins-1)-x][y]=branch[n][count]
            #if (data[99-x][y] != 0):
                #data[99-x][y] = math.log10(data[99-x][y])
            count+=1
    # do random rotation/flips
    if (flipx):
        data = np.fliplr(data)
    if flipy:
        data = np.flipud(data)
    if rot:
        for i in range(rot):
            data = np.rot90(data)
    return data;

def add_noise(data, sigma):
    return np.clip(data + np.random.normal(loc=0.0,scale=sigma, size=[100,100]), a_min=0, a_max=None);

def get_tree(file_path):
    file = up.open(file_path)
    tree = file["g4SimHits/tree"]
    return tree;

def get_branch(file_paths):
    file_path = file_paths.split(' ')
    branch = []
    for i, elem in enumerate(file_path):
        tree = get_tree(file_path[i])
        events = tree["bin_weights"].array()
        branch = np.concatenate((branch, events))
    return branch;


#check that all functions called in RootBasic are added to dataset.py
class RootBasic(udata.Dataset):
    allowed_transforms = ["none","normalize","log10"]
    def __init__(self, fuzzy_root, sharp_root, transform='none'):
        self.sharp_root = sharp_root
        self.fuzzy_root = fuzzy_root
        self.sharp_branch = get_branch(sharp_root)
        self.fuzzy_branch = get_branch(fuzzy_root)
        self.sharp_tree = get_tree(self.sharp_root)
        self.fuzzy_tree = get_tree(self.fuzzy_root)
        #assumes bin configuration is the same for all files
        self.x_bins = self.sharp_tree["xbins"].array().to_numpy()[0]
        self.y_bins = self.sharp_tree["ybins"].array().to_numpy()[0]
        self.x_min = self.sharp_tree["xmin"].array().to_numpy()[0]
        self.y_min = self.sharp_tree["ymin"].array().to_numpy()[0]
        self.x_max = self.sharp_tree["xmax"].array().to_numpy()[0]
        self.y_max = self.sharp_tree["ymax"].array().to_numpy()[0]
        self.transform = transform
        self.means = None
        self.stdevs = None
        if self.transform not in self.allowed_transforms:
            raise ValueError("Unknown transform: {}".format(self.transform))

    def __len__(self):
        if len(self.sharp_branch) == len(self.fuzzy_branch):
            return len(self.sharp_branch)
        else:
            raise RuntimeError("Sharp and fuzzy dataset lengths do not match")
            
    def __getitem__(self, idx):
        flipx, flipy, rot = get_flips()
        x_bins = self.x_bins
        y_bins = self.y_bins
        sharp_np = get_bin_weights(self.sharp_branch, idx, x_bins, y_bins, flipx, flipy, rot).copy()
        fuzzy_np = get_bin_weights(self.fuzzy_branch, idx, x_bins, y_bins, flipx, flipy, rot).copy()
        return sharp_np, fuzzy_np
    
    #can only be called after __getitem__ has run in RootDataset
    def unnormalize(self,array):
        array *= self.stdevs
        array += self.means 
        return array

class RootDataset(RootBasic):
    def __getitem__(self, idx):
        sharp_np, fuzzy_np = super().__getitem__(idx)
                    
        if self.transform=="log10":
            sharp_np = np.log10(sharp_np, where=sharp_np>0)
            fuzzy_np = np.log10(fuzzy_np, where=fuzzy_np>0)
        elif self.transform=="normalize":
            self.means = np.mean(sharp_np)
            self.stdevs = np.std(sharp_np)
            sharp_np -= self.means
            sharp_np /= self.stdevs
            fuzzy_np -= self.means
            fuzzy_np /= self.stdevs
        
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
