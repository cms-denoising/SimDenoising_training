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
parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
args = parser.parse_args()

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def main():
    args.device = torch.device('cpu')
    # check for gpu
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        print("Switched to gpu")
    else:
        print("Using CPU")
    model = DnCNN(channels=1, num_of_layers=args.num_of_layers).to(device=args.device)
    model.apply(init_weights)
    criterion = PatchLoss()
    criterion.to(device=args.device)
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    writer = SummaryWriter(args.outf)
    # numpy arrays for storing stuff.. this is a Very Inefficient Method for doing this but
    # hopefully its just temporary until I get enough space to install tensorboard...
    train_loss = np.empty(0)
    train_maxdiff = np.empty(0)
    validate_loss = np.empty(0)
    validate_maxdiff = np.empty(0)
    train_step_number = np.empty(0)
    validate_step_number = np.empty(0)
    # train the net
    step = 0
    for epoch in range(args.epochs):
        training_files = glob.glob(os.path.join(args.training_path, '*.root'))
        for training_file in training_files:
            print("Opened file " + training_file)
            branch = get_all_histograms(training_file)
            length = np.size(branch)
            for i in range(length):
                model.train()
                model.zero_grad()
                optimizer.zero_grad()
                # get data (ground truth)
                data = get_bin_weights(branch, 0).copy()
                # add noise
                noisy = add_noise(data, args.sigma).copy()
                # convert to tensor
                data = torch.from_numpy(data).to(device=args.device)
                noisy = torch.from_numpy(noisy)
                noisy = noisy.unsqueeze(0)
                noisy = noisy.unsqueeze(1).to(device=args.device)
                out_train = model(noisy.float()).to(device=args.device)
                loss = criterion(out_train.squeeze(0).squeeze(0), data, 10)
                loss.backward()
                optimizer.step()
                model.eval()
                if step % 10 ==0:
                    train_loss = np.append(train_loss, loss.item())
                step+=1
            model.eval()
        # validation
        validation_files = glob.glob(os.path.join(args.validation_path, '*root'))
        # peak signal to noise ratio
        loss_val = 0
        count = 0
        for validation_file in validation_files:
            print("Opened file " + validation_file)
            branch = get_all_histograms(validation_file)
            length = np.size(branch)
            for i in range (length):
                # get data (ground truth)
                data = get_bin_weights(branch, 0).copy()
                # add noise
                noisy = add_noise(data, opt.sigma).copy()
                # convert to tensor
                data = torch.from_numpy(data).to(device=args.device)
                noisy = torch.from_numpy(noisy)
                noisy = noisy.unsqueeze(0)
                noisy = noisy.unsqueeze(1).to(device=args.device)
                out_train = model(noisy.float()).to(device=args.device)
                loss_val+=criterion(out_train.squeeze(0).squeeze(0), data, 10)
            loss_val/=length
            writer.add_scalar('Loss values on validation data', loss_val, count)
            count+=1
        # save the model
        torch.save(model.state_dict(), os.path.join(args.outf, 'net.pth'))
        
if __name__ == "__main__":
    main()


