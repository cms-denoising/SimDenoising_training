import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as f
from models import DnCNN, PatchLoss, WeightedPatchLoss
import uproot as up
import numpy as np
import torch.utils.data as udata
import dataset as dat
import sys
import random

def load_model(trained_model, dev):
    model = DnCNN(channels=1, num_of_layers=9, kernel_size=3, features=100).to(device=dev)
    model.load_state_dict(torch.load(trained_model, map_location=torch.device('cpu'))) 
    model.eval()
    return model 

def get_output(dataset, event, model):
    #dataset = RootDataset(fuzzy_root, sharp_root, transform)
    model.to('cpu')
    
    #flipx, flipy, rot = get_flips()
    #sharp, fuzzy = branch_arrays[event]
    sharp_norm, fuzzy_norm = dataset[event]
    fuzzy_eval = fuzzy_norm.unsqueeze(0).unsqueeze(1)
    output = model(fuzzy_eval.float()).squeeze(0).squeeze(0).cpu().detach().numpy()
    output_un = dataset.unnormalize(output)
    #model.to('cuda')
    return output_un

def main():
    device = 'cpu'
    
    random.seed(0) #FIX! This needs to come from something
    dataset = dat.RootDataset(sys.argv[1], sys.argv[2], 'normalize')
    basic_dataset = dat.RootBasic(sys.argv[1], sys.argv[2], 'normalize')
    
    model = load_model(sys.argv[4], device)
    
    outputs = []
    for i in range(len(basic_dataset)):
        output = get_output(dataset, i, model)
        outputs.append(output)
    np.savez(sys.argv[3], outputs)
        
        
if __name__ == "__main__":
    main()
