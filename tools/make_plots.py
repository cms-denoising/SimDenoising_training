import sys
sys.path.append(".local/lib/python3.8/site-packages")

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
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from magiconfig import ArgumentParser, MagiConfigOptions
from torch.utils.data import DataLoader
import math

parser = ArgumentParser(description="DnCNN", config_options=MagiConfigOptions())
parser.add_argument("--datafile", nargs="?", type=str, default="./data/training", help='data path')
parser.add_argument("--model", type=str, default=None, help="Existing model")
parser.add_argument("--batchSize", type=int, default=100, help="batch size")
parser.add_argument("--noiselevel", type=int, default=10, help="noise")
parser.add_argument("--num_of_layers", type=int, default=9, help="layers")
parser.add_argument("--outf", type=str, default="", help="output director path")
parser.add_argument("--patchSize", type=int, default=50, help="patch size")
args = parser.parse_args()

dataset = RootDataset(root_file=args.datafile, sigma = args.noiselevel)
loader = DataLoader(dataset=dataset, batch_size=args.batchSize)
model = DnCNN(channels=1, num_of_layers=args.num_of_layers, features=100)
model.load_state_dict(torch.load(args.model))
model.eval()
model.to('cuda')
criterion = PatchLoss()
criterion.to('cuda')

losses = np.zeros(len(loader))
truth_energy = np.zeros(len(loader))
noise_energy = np.zeros(len(loader))
recon_energy = np.zeros(len(loader))
scale_diff = np.zeros(len(loader))
max_truth = np.zeros(len(loader))
max_noise = np.zeros(len(loader))
max_recon = np.zeros(len(loader))
energy_ratio = np.zeros(len(loader))
noise_energy_ratio = np.zeros(len(loader))
max_energy_ratio_recongen = np.zeros(len(loader))
max_energy_ratio_noisegen = np.zeros(len(loader))
for i, data in enumerate(loader, 0):
            truth, noise = data
            noise = noise.unsqueeze(1)
            output = model((noise.float().to('cuda')))
            noise.squeeze(0)
            output = output.squeeze(1)
            batch_loss = criterion(output.to('cuda'), truth.to('cuda'), patch_size=args.patchSize)
            losses[i] = batch_loss.item()
            truth_energy[i] = truth.sum()/args.batchSize
            noise_energy[i] = noise.sum()/args.batchSize
            recon_energy[i] = output.sum()/args.batchSize
            energy_ratio[i] = recon_energy[i]/truth_energy[i]
            noise_energy_ratio[i] = noise_energy[i]/truth_energy[i]
            max_truth[i] = truth.max()
            max_noise[i] = noise.max()
            max_recon[i] = output.max()
            max_energy_ratio_recongen[i] = max_recon[i]/max_truth[i]
            max_energy_ratio_noisegen[i] = max_noise[i]/max_truth[i]
            
info_file = open(args.outf+"calculations.txt", "w")
info_file.write("Average loss/image: " + str(np.mean(losses)))
info_file.write("\nAvg tot energy ratio recon/input: " + str(np.mean(energy_ratio)))
info_file.write("\nAvg tot energy ratio noise/input: " + str(np.mean(noise_energy_ratio)))
info_file.write("\nAvg max energy ratio recon/input: " + str(np.mean(max_energy_ratio_recongen)))
info_file.write("\nAvg max energy ratio noise/input: " + str(np.mean(max_energy_ratio_noisegen)))
info_file.close() 

plt.hist(energy_ratio, histtype='step', bins=100, color='b', label='reconstructed', range=(0.25, 2.5))
plt.hist(noise_energy_ratio, histtype='step', bins=100, color='r', label='noisy', range=(0.25, 2.5))
plt.legend()
plt.savefig(args.outf+"combined_total_energy_ratios.png")
plt.close()

plt.hist(max_energy_ratio_recongen, histtype='step', bins=100, color='b', label='reconstructed', range=(0.5,2.25))
plt.hist(max_energy_ratio_noisegen, histtype='step', bins=100, color='r', label='noisy', range=(0.5,2.25))
plt.legend()
plt.savefig(args.outf+"combined_max_energy_ratios.png")
plt.close()



#print(energy_ratio)
#print(noise_energy_ratio)
print(np.mean(losses))

print(np.mean(energy_ratio))
print(np.mean(noise_energy_ratio))
      
print(np.mean(max_energy_ratio_recongen))
print(np.mean(max_energy_ratio_noisegen)) 
