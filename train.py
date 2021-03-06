# modified from github.com/SaoYan/DnCNN-PyTorch/blob/master/train.py

import sys
sys.path.append(".local/lib/python{}.{}/site-packages".format(sys.version_info.major,sys.version_info.minor))

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
from tensorboardX import SummaryWriter
from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter
from torch.utils.data import DataLoader
import math
from tqdm import tqdm
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# parse arguments
parser = ArgumentParser(description="DnCNN", config_options=MagiConfigOptions(), formatter_class=ArgumentDefaultsRawHelpFormatter)

parser.add_argument("--num-layers", type=int, default=9, help="Number of total layers in the CNN")
parser.add_argument("--outf", type=str, required=True, help='Name of folder to be used to store outputs')
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--trainfileSharp", type=str, default=[], nargs='+', help='Path to higher quality .root file for training')
parser.add_argument("--trainfileFuzz", type=str, default=[], nargs='+', help='Path to lower quality .root file for training')
parser.add_argument("--valfileSharp", type=str, default=[], nargs='+', help='Path to higher quality .root file for validation')
parser.add_argument("--valfileFuzz", type=str, default=[], nargs='+', help='Path to lower quality .root file for validation')
parser.add_argument("--batchSize", type=int, default=100, help="Training batch size")
parser.add_argument("--model", type=str, default=None, help="Existing model to continue training, if applicable")
parser.add_argument("--patchSize", type=int, default=20, help="Size of patches to apply in loss function")
parser.add_argument("--kernelSize", type=int, default=3, help="Size of kernel in CNN")
parser.add_argument("--features", type=int, default=9, help="Number of features in CNN layers")
parser.add_argument("--transform", type=str, default=[], nargs='*', choices=RootDataset.allowed_transforms, help="transform(s) for input data")
parser.add_argument("--num-workers", type=int, default=8, help="Number of workers for data loaders")
parser.add_argument("--randomseed", type=int, default=0, help="Initial value for random.seed()")
args = parser.parse_args()

# backward compatibility
if not isinstance(args.transform,list): args.transform = [args.transform]

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def main():
    random.seed(args.randomseed)
    torch.manual_seed(args.randomseed)

    os.makedirs(args.outf)
    parser.write_config(args, args.outf + "/config_out.py")
    # choose cpu or gpu
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        print("Using GPU")
    else:
        args.device = torch.device('cpu')
        print("Using CPU")

    # Load dataset
    print('Loading dataset ...\n')

    dataset_train = RootDataset(sharp_root=args.trainfileSharp, fuzzy_root=args.trainfileFuzz, transform=args.transform)
    loader_train = DataLoader(dataset=dataset_train, batch_size=args.batchSize, num_workers=args.num_workers, shuffle=True)
    dataset_val = RootDataset(sharp_root=args.valfileSharp, fuzzy_root=args.valfileFuzz, transform=args.transform)
    loader_val = DataLoader(dataset=dataset_val, batch_size=args.batchSize, num_workers=args.num_workers)

    xbins = dataset_train.xbins
    ybins = dataset_train.ybins
    xmin = dataset_train.xmin
    xmax = dataset_train.xmax
    ymin = dataset_train.ymin
    ymax = dataset_train.ymax

    # Build model
    model = DnCNN(channels=1, num_of_layers=args.num_layers, kernel_size=args.kernelSize, features=args.features).to(device=args.device)
    if (args.model == None):
        model.apply(init_weights)
        print("Creating new model ")
    else:
        print("Loading model from file " + args.model)
        model.load_state_dict(torch.load(args.model))
        model.eval()

    # Loss function
    criterion = PatchLoss()
    criterion.to(device=args.device)

    #Optimizer
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=10, verbose=True)

    # training and validation
    step = 0
    training_losses = np.zeros(args.epochs)
    validation_losses = np.zeros(args.epochs)
    for epoch in range(args.epochs):
        print("Beginning epoch " + str(epoch))
        # training
        train_loss = 0
        for i, data in tqdm(enumerate(loader_train, 0), unit="batch", total=len(loader_train)):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            sharp, fuzzy = data
            fuzzy = fuzzy.unsqueeze(1)
            output = model((fuzzy.float().to(args.device)))
            batch_loss = criterion(output.squeeze(1).to(args.device), sharp.to(args.device)).to(args.device)
            batch_loss.backward()
            optimizer.step()
            model.eval()
            train_loss+=batch_loss.item()
            del sharp
            del fuzzy
            del output
            del batch_loss
        train_loss = train_loss/len(loader_train)
        training_losses[epoch] = train_loss
        tqdm.write("loss: "+ str(train_loss))

        val_loss = 0
        for i, data in tqdm(enumerate(loader_val, 0), unit="batch", total=len(loader_val)):
            val_sharp, val_fuzzy = data
            val_output = model((val_fuzzy.unsqueeze(1).float().to(args.device)))
            output_loss = criterion(val_output.squeeze(1).to(args.device), val_sharp.to(args.device)).to(args.device)
            val_loss+=output_loss.item()
            del val_sharp
            del val_fuzzy
            del val_output
            del output_loss
        val_loss = val_loss/len(loader_val)
        scheduler.step(torch.tensor([val_loss]))
        validation_losses[epoch] = val_loss
        tqdm.write("val_loss: "+ str(val_loss))

        # save the model
        model.eval()
        torch.save(model.state_dict(), os.path.join(args.outf, 'net.pth'))
        jitmodel = torch.jit.script(model)
        torch.jit.save(jitmodel, os.path.join(args.outf, 'net.jit.pth'))

    # write out training and validataion loss values to text files
    with open(args.outf + "/training_losses.txt","w") as tfileout:
        tfileout.write("\n".join("{}".format(tl) for tl in training_losses)+"\n")
    with open(args.outf + "/validation_losses.txt","w") as vfileout:
        vfileout.write("\n".join("{}".format(vl) for vl in validation_losses)+"\n")

if __name__ == "__main__":
    main()
