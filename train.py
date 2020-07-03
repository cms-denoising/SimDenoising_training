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
from tensorboardX import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# parse arguments
parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("training_path", nargs="?", type=str, default="./data/training", help='path of .root data set to be used for training')
parser.add_argument("validation_path", nargs="?", type=str, default="./data/validation", help='path of .root data set to be used for validation')
parser.add_argument("--num_of_layers", type=int, default=9, help="Number of total layers")
parser.add_argument("--sigma", type=float, default=25, help='noise level')
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
opt = parser.parse_args()

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def main():
    model = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    model.apply(init_weights)
    criterion = PatchLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    writer = SummaryWriter(opt.outf)
    # train the net
    training_files = glob.glob(os.path.join(opt.training_path, '*.root'))
    for training_file in training_files:
        print("Opened file " + training_file)
        branch = get_all_histograms(training_file)
        length = np.size(branch)
        for i in range(1):
            print("Begin processing event " + str(i) + " of " + str(length) + " in file") 
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            # get data (ground truth)
            data = get_bin_weights(branch, 0).copy()
            # add noise
            noisy = add_noise(data, opt.sigma).copy()
            # convert to tensor
            data = torch.from_numpy(data)
            noisy = torch.from_numpy(noisy)
            noisy = noisy.unsqueeze(0)
            noisy = noisy.unsqueeze(1)
            out_train = model(noisy.float())
            loss = criterion(out_train.squeeze(0).squeeze(0), data, 10)
            loss.backward()
            optimizer.step()
            model.eval()
        model.eval()
    # validation
    validation_files = glob.glob(os.path.join(opt.validation_path, '*root'))
    # peak signal to noise ratio
    loss_val = 0
    for validation_file in validation_files:
        print("Opened file " + validation_file)
        branch = get_all_histograms(validation_file)
        length = np.size(branch)
        for i in range (1):
            # get data (ground truth)
            data = get_bin_weights(branch, 0).copy()
            # add noise
            noisy = add_noise(data, opt.sigma).copy()
            # convert to tensor
            data = torch.from_numpy(data)
            noisy = torch.from_numpy(noisy)
            noisy = noisy.unsqueeze(0)
            noisy = noisy.unsqueeze(1)
            out_train = model(noisy.float())
            loss_val+=criterion(out_train.squeeze(0).squeeze(0), data, 10)
        loss_val/=length
    # save the model
    torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))
        
if __name__ == "__main__":
    main()


