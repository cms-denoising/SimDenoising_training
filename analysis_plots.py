import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as f
from models import DnCNN, PatchLoss, WeightedPatchLoss
import uproot as up
import numpy as np
import torch.utils.data as udata
import matplotlib.pyplot as plt
import random
import dataset as dat

### Functions that probably need to go into dataset.py but live here right now

def choose_device():
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        print("Using GPU")
    else:
        dev = torch.device('cpu')
        print("Using CPU")
    return dev

def load_model(trained_model, dev):
    model = DnCNN(channels=1, num_of_layers=9, kernel_size=3, features=9).to(device=dev)
    model.load_state_dict(torch.load(trained_model, map_location=torch.device('cpu'))) 
    model.eval()
    return model 

def eval_loaders(plotdata):
    dataset = plotdata
    loader = udata.DataLoader(dataset=plotdata)
    return loader

def get_output(dataset, event):
    #dataset = RootDataset(fuzzy_root, sharp_root, transform)
    model.to('cpu')
    #random.seed(args.randomseed) #makes random orientations match those from training
    #flipx, flipy, rot = get_flips()
    #sharp, fuzzy = branch_arrays[event]
    sharp_norm, fuzzy_norm = dataset[event]
    fuzzy_eval = fuzzy_norm.unsqueeze(0).unsqueeze(1)
    output = model(fuzzy_eval.float()).squeeze(0).squeeze(0).cpu().detach().numpy()
    output_un = dataset.unnormalize(output)
    #model.to('cuda')
    return output_un

### Calculate quantities for plots

def calculate_ppe(dataset, basic_dataset, event):
    ppe = []
    sharp, fuzzy = basic_dataset[event]
    sharp_energies = np.ndarray.flatten(sharp)
    fuzzy_energies = np.ndarray.flatten(fuzzy)
    ppe_sharp = np.mean(sharp_energies)
    ppe_fuzzy = np.mean(fuzzy_energies)
    output = get_output(dataset, event)
    ppe_output = np.mean(output)
    ppe.append(ppe_sharp)
    ppe.append(ppe_fuzzy)
    ppe.append(ppe_output)
    return ppe
    
def ppe_plotdata(dataset, basic_dataset):
    ppe_sharp = []
    ppe_fuzzy = []
    ppe_output = []
    for i in range(len(basic_dataset)):
        ppes = calculate_ppe(dataset, basic_dataset, i)
        sharp_ppe = ppes[0]
        fuzzy_ppe = ppes[1]
        output_ppe = ppes[2]
        ppe_sharp.append(sharp_ppe)
        ppe_fuzzy.append(fuzzy_ppe)
        ppe_output.append(output_ppe)
    return ppe_sharp, ppe_fuzzy, ppe_output

def centroid(dataset, basic_dataset, event, event_type):
    if event_type == 'output':
        event_points = []
        output = get_output(dataset,event)
        y_avg = 0
        for y in range(100):
            y_en = y*np.mean(output[y])
            y_avg += y_en
        y_avg = y_avg/100
        x_avg = 0
        for x in range(100):
            for y in range(100):
                x_en = x*output[y][x]
                x_avg += x_en
        x_avg = (x_avg/100)/100
    if event_type == 'fuzzy':
        event_points = []
        sharp, fuzzy = basic_dataset[event]
        y_avg = 0
        for y in range(100):
            y_en = y*np.mean(fuzzy[y])
            y_avg += y_en
        y_avg = y_avg/100
        x_avg = 0
        for x in range(100):
            for y in range(100):
                x_en = x*fuzzy[y][x]
                x_avg += x_en
        x_avg = (x_avg/100)/100
    if event_type == 'sharp':
        event_points = []
        sharp, fuzzy = basic_dataset[event]
        y_avg = 0
        for y in range(100):
            y_en = y*np.mean(sharp[y])
            y_avg += y_en
        y_avg = y_avg/100
        x_avg = 0
        for x in range(100):
            for y in range(100):
                x_en = x*sharp[y][x]
                x_avg += x_en
        x_avg = (x_avg/100)/100
    return x_avg, y_avg
            
def centroid_plotdata(dataset, basic_dataset, event_type, ppe_plot):
    centroids = []
    centroids_x = []
    centroids_y = []
    for i in range(len(basic_dataset)):
        cntr = centroid(dataset, basic_dataset, i, event_type)
        if event_type == 'sharp': j=0 
        if event_type == 'fuzzy': j=1
        if event_type == 'output': j=2
        cntr_x = cntr[0]/(ppe_plot[j][i])
        cntr_y = cntr[1]/(ppe_plot[j][i])
        centroids_x.append(cntr_x)
        centroids_y.append(cntr_y)
    centroids.append(centroids_x)
    centroids.append(centroids_y)
    return centroids

def centroid_rad_data(centroid_plotdata):
    centroid_rads = []
    for i in range(len(centroid_plotdata[0])):
        centroid_rad = np.sqrt(centroid_plotdata[0][i]**2+centroid_plotdata[1][i]**2)
        centroid_rads.append(centroid_rad)
    return centroid_rads

def hits_above_threshold(dataset, basic_dataset, threshold, event, event_type):
    if event_type == 'sharp' or 'fuzzy':
        sharp, fuzzy = basic_dataset[event]
    count = 0
    if event_type == 'sharp':
        for y in range(100):
            for x in range(100):
                if sharp[y][x] >= threshold:
                    count +=1
    if event_type == 'fuzzy':
        for y in range(100):
            for x in range(100):
                if fuzzy[y][x] >= threshold:
                    count +=1
    if event_type == 'output':
        output = get_output(dataset, event)
        for y in range(100):
            for x in range(100):
                if output[y][x] >= threshold:
                    count +=1
    return count

def hits_data(dataset, basic_dataset, threshold, event_type):
    hits = []
    for i in range(len(basic_dataset)):
        hit = hits_above_threshold(dataset, basic_dataset, threshold, i, event_type)
        hits.append(hit)
    return hits

def diff(dataset1, dataset2):
    diffset = []
    for i, elem in enumerate(dataset1):
        diff = dataset1[i] - dataset2[i]
        diffset.append(diff)
    return diffset

### Makes plots

def plot(ppe_data):
    plt.hist(ppe_data)
    plt.show()