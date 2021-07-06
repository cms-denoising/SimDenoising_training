# modified from github.com/SaoYan/DnCNN-PyTorch/blob/master/train.py

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
from tensorboardX import SummaryWriter
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
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
parser.add_argument("--sigma", type=float, default=20, help='Standard deviation of gaussian noise level')
parser.add_argument("--outf", type=str, default="logs", help='Name of folder to be used to store outputs')
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--trainfileSharp", type=str, default="test.root", help='Path to higher quality .root file for training')
parser.add_argument("--trainfileFuzz", type=str, default="test.root", help='Path to lower quality .root file for training')
parser.add_argument("--valfileSharp", type=str, default="test.root", help='Path to higher quality .root file for validation')
parser.add_argument("--valfileFuzz", type=str, default="test.root", help='Path to lower quality .root file for validation')
parser.add_argument("--batchSize", type=int, default=100, help="Training batch size")
parser.add_argument("--model", type=str, default=None, help="Existing model to continue training, if applicable")
parser.add_argument("--patchSize", type=int, default=20, help="Size of patches to apply in loss function")
parser.add_argument("--kernelSize", type=int, default=3, help="Size of kernel in CNN")
parser.add_argument("--features", type=int, default=9, help="Number of features in CNN layers")
parser.add_argument("--transform", type=str, default="none", choices=RootDataset.allowed_transforms, help="transform for input data")
parser.add_argument("--num-workers", type=int, default=8, help="Number of workers for data loaders")
parser.add_argument("--randomseed", type=int, default=0, help="Initial value for random.seed()")
args = parser.parse_args()

# store a file with configuration information in the output directory
def write_info_file():
    info_file = open(args.outf+"/info.txt", "w")
    print("Creating information file")
    if (args.model != None):
        info_file.write("The model used was loaded from " + args.model)
    info_file.write("\nUsing WeightedPatchLoss")
    info_file.write("\nTraining high resolution dataset file: " + args.trainfileSharp)
    info_file.write("\nTraining low resolution dataset file: " + args.trainfileFuzz)
    info_file.write("\nValidation high resolution dataset file: " + args.valfileSharp)
    info_file.write("\nValidation low resolution dataset file: " + args.valfileFuzz)
    info_file.write("\nNoise level (sigma): " + str(args.sigma))
    info_file.write("\nEpochs: " + str(args.epochs))
    info_file.write("\nInitial learning rate: " + str(args.lr))
    info_file.write("\nTraining batch size: " + str(args.batchSize))
    info_file.write("\nLoss function patch size: " + str(args.patchSize))
    info_file.close()

# create and save sharp, fuzzy, and reconstructed data sets and store in text files
def make_sample_images(fuzzy_root, sharp_root, model, transform='none'):
    branch_arrays = RootBasic(fuzzy_root, sharp_root, transform)
    dataset = RootDataset(fuzzy_root, sharp_root, transform)
    model.to('cpu')
    random.seed(args.randomseed) #makes random orientations match those from training
    for event in range(10):
        sharp_norm, fuzzy_norm = dataset[event]
        fuzzy_eval = fuzzy_norm.unsqueeze(0).unsqueeze(1)
        output = model(fuzzy_eval.float()).squeeze(0).squeeze(0).cpu().detach().numpy()
        output_un = dataset.unnormalize(output)
        np.savetxt(args.outf+'/samples/output' + str(event) + '.txt', output_un)
    random.seed(args.randomseed)
    for event in range(10):
        sharp, fuzzy = branch_arrays[event]
        np.savetxt(args.outf+'/samples/sharp' + str(event) + '.txt', sharp)
        np.savetxt(args.outf+'/samples/fuzzy' + str(event) + '.txt', fuzzy)
    model.to('cuda')

#makes histograms given bin weights listed in .txt file    
def make_plots(fin, x_min, x_max, x_bins, y_min, y_max, y_bins):
    binweights = np.loadtxt(fin)
    binarray = []
    for i, elem in enumerate(binweights):
        for j, elem in enumerate(binweights[i]):
            binarray.append(binweights[i][j])
        
    #builds axes for histogram given min/max and bin number
    x_axis = []
    count = 0
    x_start = x_min
    while count < y_bins:
        for i in range(x_bins):
            x = x_start + ((x_max-x_min)/x_bins)*float(i)
            x_axis.append(x)
        count = count + 1

    y_axis = []
    y_start = y_min
    for i in range(y_bins):
        y = y_start + ((y_max-y_min)/y_bins)*float(i)
        count = 0
        while count < x_bins:
            y_axis.append(y)
            count = count + 1
            
    #makes histogram
    fig = plt.subplots(figsize =(10, 7))
    plt.hist2d(x_axis, y_axis, bins=[x_bins,y_bins], weights = binarray)
    plt.title("Energy Deposits Projected on z plane")
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")
    plt.colorbar(label = "Energy (MeV)")
    fin = str(fin).replace(args.outf+'/samples/','')
    plt.savefig(args.outf+'/plots/' + str(fin).replace( '.txt', '.png'))

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def main():
    random.seed(args.randomseed)
    
    os.makedirs(args.outf+'/samples')

    write_info_file()
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
    
    x_bins = dataset_train.x_bins
    y_bins = dataset_train.y_bins
    x_min = dataset_train.x_min
    x_max = dataset_train.x_max
    y_min = dataset_train.y_min
    y_max = dataset_train.y_max

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
            train_loss = train_loss/len(loader_train)
            del sharp
            del fuzzy
            del output
            del batch_loss
        training_losses[epoch] = train_loss
        tqdm.write("loss: "+ str(train_loss))

        val_loss = 0
        for i, data in tqdm(enumerate(loader_val, 0), unit="batch", total=len(loader_val)):
            val_sharp, val_fuzzy = data
            val_output = model((val_fuzzy.unsqueeze(1).float().to(args.device)))
            output_loss = criterion(val_output.squeeze(1).to(args.device), val_sharp.to(args.device)).to(args.device)
            val_loss+=output_loss.item()
            val_loss = val_loss/len(loader_val)
            del val_sharp
            del val_fuzzy
            del val_output
            del output_loss
        scheduler.step(torch.tensor([val_loss]))
        validation_losses[epoch] = val_loss
        tqdm.write("val_loss: "+ str(val_loss))

        # save the model
        model.eval()
        torch.save(model.state_dict(), os.path.join(args.outf, 'net.pth'))
        
    # plot loss/epoch for training and validation sets
    training = plt.plot(training_losses, label='training')
    validation = plt.plot(validation_losses, label='validation')
    plt.legend()
    plt.savefig(args.outf + "/loss_plot.png")
    
    #write out training and validataion loss values to text files
    tfileout = open("training_losses.txt","w")
    vfileout = open("validation_losses.txt","w")
    for i, elem in enumerate(training_losses):
        tfileout.write("%f " % training_losses[i] + "\n")
    for elem in enumerate(validation_losses):
        tfileout.write("%f " % validation_losses[i] + "\n")
        vfileout.write("%f " % validation_losses[i] + "\n")

    make_sample_images(args.valfileFuzz, args.valfileSharp, model, args.transform)
    
    #makes histograms of sample data
    os.makedirs(args.outf+'/plots')
    for fin in os.listdir(args.outf+'/samples'):
        make_plots(args.outf+'/samples/'+fin, x_min, x_max, x_bins, y_min, y_max, y_bins)
    
    

if __name__ == "__main__":
    main()
