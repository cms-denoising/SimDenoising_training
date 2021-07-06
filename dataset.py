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
    def __init__(self, fuzzy_root, sharp_root):
        self.sharp_root = sharp_root
        self.fuzzy_root = fuzzy_root
        #self.sigma = sigma
        self.sharp_histograms = get_all_histograms(sharp_root)
        self.fuzzy_histograms = get_all_histograms(fuzzy_root)
        
    def __len__(self):
        if len(self.sharp_histograms) == len(self.fuzzy_histograms):
            return len(self.sharp_histograms)
        else:
            print("Sharp and fuzzy dataset lengths do not match")

    def __getitem__(self, idx):
        sharp_np = get_bin_weights(self.sharp_histograms, idx).copy()
        fuzzy_np = get_bin_weights(self.fuzzy_histograms, idx).copy()
        
        for ix in range(sharp_np.shape[0]):
            for iy in range(sharp_np.shape[1]):
                if (sharp_np[ix, iy] != 0):
                    sharp_np[ix, iy] = math.log10(sharp_np[ix, iy])
                if (fuzzy_np[ix, iy] != 0):
                    fuzzy_np[ix, iy] = math.log10(fuzzy_np[ix, iy])
        
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
