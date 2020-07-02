# modified from github.com/SaoYan/DnCNN-PyTorch/blob/master/train.py
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as f
import os
import argparse
from models import DnCNN, PatchLoss
from dataset import *
import glob
import torch.optim as optim
import uproot
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# parse arguments
parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("training_path", nargs="?", type=str, default="./data/training", help='path of .root data set to be used for training')
parser.add_argument("validation_path", nargs="?", type=str, default="./data/validation", help='path of .root data set to be used for validation')
parser.add_argument("--num_of_layers", type=int, default=9, help="Number of total layers")
parser.add_argument("--sigma", type=float, default=25, help='noise level; ignored when mode=B')

opt = parser.parse_args()

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

if __name__=="__main__":
    model = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    model.apply(init_weights)
    criterion = PatchLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    training_files = glob.glob(os.path.join(opt.training_path, '*.root'))
    for training_file in training_files:
        branch = get_all_histograms(training_file)
        for i in range(1):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            data = get_bin_weights(branch, 0).copy()
            noisy = add_noise(data, opt.sigma).copy()
            data = torch.from_numpy(data)
            noisy = torch.from_numpy(noisy)
            noisy = noisy.unsqueeze(0)
            noisy = noisy.unsqueeze(1)
            out_train = model(noisy.float())
            print(out_train.squeeze(0).squeeze(0).size())
            loss = criterion(out_train.squeeze(0).squeeze(0), data, 10)
            loss.backward()
            optimizer.step()
            model.eval()



