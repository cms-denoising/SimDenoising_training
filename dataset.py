import sys
import numpy as np
import uproot as up
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import torch.utils.data as udata
import torch
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation

# full set of order 4 rotations/flips
# using random transforms deterministically, and then applying to random subsets of data...
flips = [
    RandomRotation((0,0)), # identity
    RandomRotation((90,90)),
    RandomRotation((180,180)),
    RandomRotation((270,270)),
    RandomHorizontalFlip(1.0),
    Compose([
        RandomRotation((90,90)),
        RandomHorizontalFlip(1.0),
    ]),
    Compose([
        RandomRotation((180,180)),
        RandomHorizontalFlip(1.0),
    ]),
    Compose([
        RandomRotation((270,270)),
        RandomHorizontalFlip(1.0),
    ]),
]

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
    def __init__(self, fuzzy_root, sharp_root, transform='none', shuffle=True):
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
        # combine on feature axis to apply random rotation/flips consistently for both datasets
        combined_branch = torch.from_numpy(np.concatenate([self.sharp_branch,self.fuzzy_branch],axis=1))
        # split into 8 subsets, apply different flip/rotation to each one, recombine, shuffle randomly
        combined_branches = torch.tensor_split(combined_branch,len(flips))
        combined_branches = [flips[i](combined_branches[i]) for i in range(len(flips))]
        combined_branch = torch.cat(combined_branches)
        if shuffle:
            idx = torch.randperm(combined_branch.shape[0])
            combined_branch = combined_branch[idx]
        # restore original split
        self.sharp_branch, self.fuzzy_branch = torch.split(combined_branch,1,dim=1)
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
            return self.sharp_branch[idx], self.fuzzy_branch[idx]

    # assumes array is same size as inputs
    def unnormalize(self,array,idx=None):
        if self.transform=="log10":
            array = np.power(array,10)
        elif self.transform.startswith("normalize"):
            if idx==None:
                array = array*self.stdevs+self.means
            else:
                array = array*self.stdevs[idx].squeeze()+self.means[idx].squeeze()
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
