import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as f
import os
import argparse
from models import DnCNN, PatchLoss, WeightedPatchLoss
from dataset import *

import glob
import torch.optim as optim
import uproot
import matplotlib.pyplot as plt
from magiconfig import ArgumentParser, MagiConfigOptions
from torch.utils.data import DataLoader
import math

parser = ArgumentParser(description="DnCNN", config_options=MagiConfigOptions())
parser.add_argument("--datafile", nargs="?", type=str, default="./data/training"
, help='data path')
parser.add_argument("--model", type=str, default=None, help="Existing model")
parser.add_argument("--batchSize", type=int, default=100, help="batch size")
parser.add_argument("--sigma", type=int, default=10, help="noise")
parser.add_argument("--num_of_layers", type=int, default=9, help="layers")
parser.add_argument("--outf", type=str, default="", help="output director path")
parser.add_argument("--features", type=str, default=100, help="num features used")
parser.add_argument("--patchSize", type=int, default=50, help="patch size")
args = parser.parse_args()
model = DnCNN(channels=1, num_of_layers=args.num_of_layers)
model.load_state_dict(torch.load(args.model))
model.eval()
#model.to('cuda')
criterion = PatchLoss()
#criterion.to('cuda')
num = 1000
energy_ratios = np.zeros(num)
noisy_energy_ratios = np.zeros(num)
total_energy_ratio = np.zeros(num)
total_noisy_energy_ratio = np.zeros(num)

branch = get_all_histograms(args.datafile)
for image in range(num):
    count = 0
    data = get_bin_weights(branch, image).copy()
    means = np.mean(data)
    stdevs = np.std(data)
    noisy = add_noise(data, args.sigma).copy()
    data_norm = (data-means)/stdevs
    noisy_norm = (noisy-means)/stdevs
    data_norm = torch.from_numpy(data_norm)
    noisy_norm = torch.from_numpy(noisy_norm)
    noisy_norm = noisy_norm.unsqueeze(0)
    noisy_norm = noisy_norm.unsqueeze(1)
    output_norm = model(noisy_norm.float()).squeeze(0).squeeze(0).detach().numpy()
    output = (output_norm * stdevs) + means
    
    total_energy_ratio[image] = np.sum(output) / np.sum(data)
    total_noisy_energy_ratio[image] = np.sum(noisy) / np.sum(data)
    sum_ratio = 0
    noise_sum_ratio = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if (data[i][j] != 0):
                sum_ratio += output[i][j] / data[i][j]
                noise_sum_ratio += noisy[i][j] / data[i][j]
                count+=1
    energy_ratios[image] = sum_ratio / count
    noisy_energy_ratios[image] = noise_sum_ratio / count

plt.hist(total_energy_ratio, histtype='step', bins=100, color='b', label='reconstructed', range=(0,10))
plt.hist(total_noisy_energy_ratio, histtype='step', bins=100, color= 'r', label='noisy', range=(0,10))
plt.legend()
plt.savefig("total_energy_ratio.png")
plt.close()

plt.hist2d(total_energy_ratio, total_noisy_energy_ratio, bins=50, range=[[0, 2], [0,10]])
#plt.hist2d(energy_ratios, noisy_energy_ratios, bins=50, range = [[0, 25], [0, 25]])
plt.colorbar()
plt.savefig("2dtotal__energy_ratios.png")
plt.close()
#print(energy_ratios)
#print(noisy_energy_ratios)
