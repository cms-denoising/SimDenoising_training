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
import matplotlib.pyplot as plt
from magiconfig import ArgumentParser, MagiConfigOptions
from torch.utils.data import DataLoader

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# parse arguments
parser = ArgumentParser(description="DnCNN", config_options=MagiConfigOptions())
parser.add_argument("training_path", nargs="?", type=str, default="./data/training", help='path of .root data set to be used for training')
parser.add_argument("validation_path", nargs="?", type=str, default="./data/validation", help='path of .root data set to be used for validation')
parser.add_argument("--num_of_layers", type=int, default=9, help="Number of total layers")
parser.add_argument("--sigma", type=float, default=3, help='noise level')
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--trainfile", type=str, default="test.root", help='path of .root file for training')
parser.add_argument("--valfile", type=str, default="test.root", help='path of .root file for validation')
parser.add_argument("--batchSize", type=int, default=50, help="Training batch size")
args = parser.parse_args()

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def main():

    # choose cpu or gpu 
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        print("Using GPU")
    else:
        args.device = torch.device('cpu')
        print("Using CPU")

    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = RootDataset(root_file=args.trainfile, sigma = args.sigma)
    loader_train = DataLoader(dataset=dataset_train, batch_size=args.batchSize)
    dataset_val = RootDataset(root_file=args.valfile, sigma=args.sigma)

    # Build model
    model = DnCNN(channels=1, num_of_layers=args.num_of_layers).to(device=args.device)
    model.apply(init_weights)

    # Loss function
    criterion = PatchLoss()
    criterion.to(device=args.device)

    #Optimizer
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    
    # training and validation
    step = 0
    losses = np.zeros(args.epochs)
    for epoch in range(args.epochs):
        print("Beginning epoch " + str(epoch))
        
        # training
        loss = 0
        for i, data in enumerate(loader_train, 0):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            truth, noise = data
            noise = noise.unsqueeze(1)
            output = model(noise.float().to(args.device))
            loss = criterion(output.squeeze(1).to(args.device), truth.to(args.device), 50).to(args.device)
            print("Batch loss: " + str(loss.item()))
        loss.backward()
        optimizer.step()
        model.eval()
        losses[epoch] = loss.item()
        
        # TODO validation
        for k in range(len(dataset_val)):
            truth_val, noise_val = dataset_val[k]
            noise_val = noise_val.unsqueeze(0).unsqueeze(0)
            output_val = model(noise_val.float().to(args.device))
            diff = output_val.squeeze(1) - truth_val
            max = torch.max(diff)
            #still in progress
    
        # save the model
        torch.save(model.state_dict(), os.path.join(args.outf, 'net.pth'))
    plt.plot(losses)
    plt.savefig("losses.png")
    
    #make some images and store to csv
    branch = get_all_histograms("test.root")
    for image in range(10):
        model.to('cpu')
        data = get_bin_weights(branch, image).copy()
        np.savetxt('logs/truth' + str(image) + '.txt', data)
        noisy = add_noise(data, args.sigma).copy()
        np.savetxt('logs/noisy' + str(image) + '.txt', noisy)
        data = torch.from_numpy(data)
        noisy = torch.from_numpy(noisy)
        noisy = noisy.unsqueeze(0)
        noisy = noisy.unsqueeze(1)
        out_train = model(noisy.float()).squeeze(0).squeeze(0)
        np.savetxt('logs/output' + str(image) + '.txt', out_train.detach().numpy())
    

if __name__ == "__main__":
    main()
