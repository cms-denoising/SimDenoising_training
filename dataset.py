import sys
import numpy as np
import uproot as up
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import torch.utils.data as udata
import torch

def get_flips():
    flipx = random.randint(0, 1)
    flipy = random.randint(0, 1)
    rot = random.randint(0, 3)
    return flipx, flipy, rot

def get_tree(file_path):
    file = up.open(file_path)
    tree = file["g4SimHits/tree"]
    return tree

def get_branch(file_paths):
    branch = []
    for i, elem in enumerate(file_paths):
        tree = get_tree(file_paths[i])
        events = tree["bin_weights"].array()
        branch = np.concatenate((branch, events))
    branch = np.asarray(branch)
    return branch

class RootDataset(udata.Dataset):
    allowed_transforms = ["none","normalize","normalizeSharp","log10"]
    nfeatures = 1
    def __init__(self, fuzzy_root, sharp_root, transform='none', shuffle=True, output=False):
        # assume bin configuration is the same for all files
        sharp_tree = get_tree(sharp_root[0])
        self.xbins = sharp_tree["xbins"].array().to_numpy()[0]
        self.ybins = sharp_tree["ybins"].array().to_numpy()[0]
        self.xmin = sharp_tree["xmin"].array().to_numpy()[0]
        self.ymin = sharp_tree["ymin"].array().to_numpy()[0]
        self.xmax = sharp_tree["xmax"].array().to_numpy()[0]
        self.ymax = sharp_tree["ymax"].array().to_numpy()[0]

        # other member variables
        self.transform = transform
        self.means = None
        self.stdevs = None
        self.do_unnormalize = False
        self.output = output
        if self.transform not in self.allowed_transforms:
            raise ValueError("Unknown transform: {}".format(self.transform))

        # get data in np format
        self.sharp_branch = get_branch(sharp_root)
        self.fuzzy_branch = get_branch(fuzzy_root)
        # reshape to image tensor
        self.sharp_branch = self.sharp_branch.reshape((self.sharp_branch.shape[0],self.nfeatures,self.xbins,self.ybins))
        self.fuzzy_branch = self.fuzzy_branch.reshape((self.fuzzy_branch.shape[0],self.nfeatures,self.xbins,self.ybins))
        # apply transform if any
        if self.transform=="log10":
            self.sharp_branch = np.log10(self.sharp_branch, where=self.sharp_branch>0)
            self.fuzzy_branch = np.log10(self.fuzzy_branch, where=self.fuzzy_branch>0)
        elif self.transform.startswith("normalize"):
            norm_branch = self.sharp_branch if self.transform=="normalizeSharp" else self.fuzzy_branch
            self.means = np.average(norm_branch, axis=(1,2,3))[:,None,None,None]
            self.stdevs = np.std(norm_branch, axis=(1,2,3))[:,None,None,None]
            self.sharp_branch = np.divide(self.sharp_branch-self.means,self.stdevs,where=self.stdevs!=0)
            self.fuzzy_branch = np.divide(self.fuzzy_branch-self.means,self.stdevs,where=self.stdevs!=0)
        # apply random rotation/flips consistently for both datasets
        for idx in range(self.sharp_branch.shape[0]):
            flipx, flipy, rot = get_flips()
            def do_flips(branch,idx,flipx,flipy,rot):
                if flipx: branch[idx] = np.fliplr(branch[idx])
                if flipy: branch[idx] = np.flipud(branch[idx])
                if rot: branch[idx] = np.rot90(branch[idx], rot, (1,2))
            do_flips(self.sharp_branch,idx,flipx,flipy,rot)
            do_flips(self.fuzzy_branch,idx,flipx,flipy,rot)
        # fix shapes
        self.sharp_branch = self.sharp_branch.squeeze(1)
        self.fuzzy_branch = self.fuzzy_branch.squeeze(1)
        if self.means is not None:
            self.means = np.squeeze(self.means,1)
        if self.stdevs is not None:
            self.stdevs = np.squeeze(self.stdevs,1)

    def __len__(self):
        if len(self.sharp_branch) == len(self.fuzzy_branch):
            return len(self.sharp_branch)
        else:
            raise RuntimeError("Sharp and fuzzy dataset lengths do not match")

    def __getitem__(self, idx):
        if self.do_unnormalize:
            return self.unnormalize(self.sharp_branch[idx],idx=idx).squeeze(), \
                   self.unnormalize(self.fuzzy_branch[idx],idx=idx).squeeze()
        else:
            if self.output and self.transform.startswith("normalize"):
                return self.sharp_branch[idx], self.fuzzy_branch[idx], self.means[idx], self.stdevs[idx]
            else:
                return self.sharp_branch[idx], self.fuzzy_branch[idx]

    # assumes array is same size as inputs
    def unnormalize(self,array,idx=None,means=None,stdevs=None):
        if means is None: means = self.means
        else: means = np.asarray(means)
        if stdevs is None: stdevs = self.stdevs
        else: stdevs = np.asarray(stdevs)

        if self.transform=="log10":
            array = np.power(array,10)
        elif self.transform.startswith("normalize"):
            if idx==None:
                array = array*stdevs+means
            else:
                array = array*stdevs[idx].squeeze()+means[idx].squeeze()
        return array

if __name__=="__main__":
    torch.manual_seed(0)
    dataset = RootDataset([sys.argv[1]], [sys.argv[2]])
    truth, noise = dataset.__getitem__(0)
    print(truth,truth.shape)
    print(noise,noise.shape)
    torch.manual_seed(0)
    dataset = RootDataset([sys.argv[1]], [sys.argv[2]], "normalize")
    truth, noise = dataset.__getitem__(0)
    print(truth,truth.shape)
    print(noise,noise.shape)
    print(dataset.means[0],dataset.means.shape)
    print(dataset.stdevs[0],dataset.stdevs.shape)
    dataset.do_unnormalize = True
    truth, noise = dataset.__getitem__(0)
    print(truth,truth.shape)
    print(noise,noise.shape)
