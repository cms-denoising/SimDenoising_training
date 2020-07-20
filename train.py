# modified from github.com/SaoYan/DnCNN-PyTorch/blob/master/train.py
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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# parse arguments
parser = ArgumentParser(description="DnCNN", config_options=MagiConfigOptions())
parser.add_argument("training_path", nargs="?", type=str, default="./data/training", help='path of .root data set to be used for training')
parser.add_argument("validation_path", nargs="?", type=str, default="./data/validation", help='path of .root data set to be used for validation')
parser.add_argument("--num_of_layers", type=int, default=9, help="Number of total layers")
parser.add_argument("--sigma", type=float, default=20, help='noise level')
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--trainfile", type=str, default="test.root", help='path of .root file for training')
parser.add_argument("--valfile", type=str, default="test.root", help='path of .root file for validation')
parser.add_argument("--batchSize", type=int, default=100, help="Training batch size")
parser.add_argument("--model", type=str, default=None, help="Existing model, if applicable")
parser.add_argument("--patchSize", type=int, default=20, help="Size of patches to apply in loss function")
args = parser.parse_args()

# store a file with configuration information in the output directory
def write_info_file():    
    info_file = open(args.outf+"/info.txt", "w")
    print("Creating information file")
    if (args.model != None):
        info_file.write("The model used was loaded from " + args.model)
    info_file.write("\nTraining dataset file: " + args.trainfile)
    info_file.write("\nValidation dataset file: " + args.valfile)
    info_file.write("\nNoise level (sigma): " + str(args.sigma))
    info_file.write("\nEpochs: " + str(args.epochs))
    info_file.write("\nInitial learning rate: " + str(args.lr))
    info_file.write("\nTraining batch size: " + str(args.batchSize))
    info_file.write("\nLoss function patch size: " + str(args.patchSize))
    info_file.close()

# create and save truth, noisy, and reconstructed data sets and store in text files
def make_sample_images(model):
    branch = get_all_histograms("test.root")
    for image in range(10):
        model.to('cpu')
        data = get_bin_weights(branch, image).copy()
        np.savetxt(args.outf+'/samples/truth' + str(image) + '.txt', data)
        noisy = add_noise(data, args.sigma).copy()
        np.savetxt(args.outf+'/samples/noisy' + str(image) + '.txt', noisy)
        data = torch.from_numpy(data)
        noisy = torch.from_numpy(noisy)
        noisy = noisy.unsqueeze(0)
        noisy = noisy.unsqueeze(1)
        output = model(noisy.float()).squeeze(0).squeeze(0).detach().numpy()
        np.savetxt(args.outf+'/samples/output' + str(image) + '.txt', output)
        truth = data.numpy()
        noisy = noisy.numpy()
        diff = output-truth
        np.savetxt(args.outf+'/samples/diff' + str(image) + '.txt', diff)
        del data
        del noisy
        del output
        del diff


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def main():

    write_info_file()

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
    dataset_val = RootDataset(root_file=args.valfile, sigma=math.log(args.sigma))
    val_train = DataLoader(dataset=dataset_val, batch_size=args.batchSize)

    # Build model
    model = DnCNN(channels=1, num_of_layers=args.num_of_layers).to(device=args.device)
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
        for i, data in enumerate(loader_train, 0):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            truth, noise = data
            noise = noise.unsqueeze(1)
            output = model((noise.float().to(args.device)))
            batch_loss = criterion(output.squeeze(1).to(args.device), truth.to(args.device), args.patchSize).to(args.device)
            batch_loss.backward()
            optimizer.step()
            model.eval()
            #print(batch_loss.item())
            train_loss+=batch_loss.item()
            del truth
            del noise
            del output
            del batch_loss
        training_losses[epoch] = train_loss
        print("t: "+ str(train_loss))
        
        val_loss = 0
        for i, data in enumerate(val_train, 0):
            val_truth, val_noise =  data
            val_output = model((val_noise.unsqueeze(1).float().to(args.device)))
            output_loss = criterion(val_output.squeeze(1).to(args.device), val_truth.to(args.device), args.patchSize).to(args.device)
            val_loss+=output_loss.item()
            #print(output_loss.item())
            del val_truth
            del val_noise
            del val_output
            del output_loss
        scheduler.step(torch.tensor([val_loss]))
        validation_losses[epoch] = val_loss
        print("v: "+ str(val_loss))
        
        # save the model
        model.eval()
        torch.save(model.state_dict(), os.path.join(args.outf, 'net.pth'))
    
    # plot loss/epoch for training and validation sets
    training = plt.plot(training_losses, label='training')
    validation = plt.plot(validation_losses, label='validation')
    plt.legend()
    plt.savefig(args.outf + "/loss_plot.png")
    
    make_sample_images(model)
if __name__ == "__main__":
    main()
