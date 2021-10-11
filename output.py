import numpy as np

import sys
sys.path.append(".local/lib/python3.8/site-packages")

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as f
from models import DnCNN, PatchLoss, WeightedPatchLoss
import uproot as up
import numpy as np
import torch.utils.data as udata
import dataset as dat
from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_model(trained_model, device):
    model = DnCNN(channels=1, num_of_layers=9, kernel_size=3, features=100).to(device)
    model.load_state_dict(torch.load(trained_model, map_location=device))
    model.eval()
    torch.no_grad()
    return model

def get_output(dataset, event, model, device):
    # necessary to populate stdevs and means
    _, fuzzy = dataset[event]
    fuzzy = fuzzy.unsqueeze(0).unsqueeze(1).float().to(device)
    output = model(fuzzy).squeeze(1).cpu().detach().numpy()
    output = dataset.unnormalize(output)
    del _
    del fuzzy
    return output

def main():
    parser = ArgumentParser(description="DnCNN", config_options=MagiConfigOptions(), formatter_class=ArgumentDefaultsRawHelpFormatter)

    parser.add_argument("--model", type=str, required=True, help='Path to .pth file with saved model')
    parser.add_argument("--numpy", type=str, default="test.npz", help='Name of .npz file with CNN-enhanced low quality (fuzzy) data')
    parser.add_argument("--fileSharp", type=str, default=[], nargs='+', help='Path to higher quality .root file for making plots')
    parser.add_argument("--fileFuzz", type=str, default=[], nargs='+', help='Path to lower quality .root file for making plots')
    parser.add_argument("--randomseed", type=int, default=0, help="Initial value for random.seed()")
    parser.add_argument("--transform", type=str, default="normalize", choices=dat.RootDataset.allowed_transforms, help="transform for input data")
    args = parser.parse_args()

    # choose cpu or gpu
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    random.seed(args.randomseed)
    dataset = dat.RootDataset(args.fileFuzz,args.fileSharp,args.transform)

    model = load_model(args.model, device)

    outputs = []
    for i in range(len(dataset)):
        output = get_output(dataset,i,model,device)
        if i==0: outputs = output
        else: outputs = np.concatenate((outputs,output))
        del output
    np.savez(args.numpy, outputs)

if __name__ == "__main__":
    main()
