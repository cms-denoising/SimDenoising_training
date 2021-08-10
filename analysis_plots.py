import sys
sys.path.append(".local/lib/python3.8/site-packages")

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
import os
import dataset as dat
from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter

parser = ArgumentParser(description="DnCNN", config_options=MagiConfigOptions(), formatter_class=ArgumentDefaultsRawHelpFormatter)

parser.add_argument("--outf", type=str, default="analysis-plots", help='Name of folder to be used to store outputs')
#how do I get rid of the defaults...?
parser.add_argument("--numpy", type=str, default="test.npz", help='Path to .npz file of CNN-enhanced low quality (fuzzy) data')
parser.add_argument("--fileSharp", type=str, default="test.root", help='Path to higher quality .root file for making plots')
parser.add_argument("--fileFuzz", type=str, default="test.root", help='Path to lower quality .root file for making plots')
parser.add_argument("--randomseed", type=int, default=0, help="Initial value for random.seed()")
parser.add_argument("--transform", type=str, default="none", choices=dat.RootDataset.allowed_transforms, help="transform for input data")
args = parser.parse_args()

def calculate_bins(fin):
    upfile = up.open(fin)
    tree = upfile["g4SimHits"]["tree"]
    x_bins = tree["xbins"].array().to_numpy()[0]
    y_bins = tree["ybins"].array().to_numpy()[0]
    return x_bins, y_bins
    

def calculate_ppe(dataset, outputs, event):
    ppe = []
    sharp, fuzzy = dataset[event]
    output = outputs[event]
    sharp_energies = np.ndarray.flatten(sharp)
    fuzzy_energies = np.ndarray.flatten(fuzzy)
    output_energies = np.ndarray.flatten(output)
    ppe_sharp = np.mean(sharp_energies)
    ppe_fuzzy = np.mean(fuzzy_energies)
    ppe_output = np.mean(output_energies)
    ppe.append(ppe_sharp)
    ppe.append(ppe_fuzzy)
    ppe.append(ppe_output)
    return ppe
    
def ppe_plotdata(dataset, outputs):
    random.seed(0)
    ppe_sharp = []
    ppe_fuzzy = []
    ppe_output = []
    for i in range(len(dataset)):
        ppes = calculate_ppe(dataset, outputs, i)
        sharp_ppe = ppes[0]
        fuzzy_ppe = ppes[1]
        output_ppe = ppes[2]
        ppe_sharp.append(sharp_ppe)
        ppe_fuzzy.append(fuzzy_ppe)
        ppe_output.append(output_ppe)
    return ppe_sharp, ppe_fuzzy, ppe_output

def centroid(dataset, outputs, event, event_type, x_bins, y_bins):
    if event_type == 'output':
        event_points = []
        output = outputs[event]
        y_avg = 0
        for y in range(y_bins):
            y_en = y*np.mean(output[y])
            y_avg += y_en
        y_avg = y_avg/y_bins
        x_avg = 0
        for x in range(x_bins):
            for y in range(y_bins):
                x_en = x*output[y][x]
                x_avg += x_en
        x_avg = (x_avg/x_bins)/y_bins
    if event_type == 'fuzzy':
        event_points = []
        sharp, fuzzy = dataset[event]
        y_avg = 0
        for y in range(y_bins):
            y_en = y*np.mean(fuzzy[y])
            y_avg += y_en
        y_avg = y_avg/y_bins
        x_avg = 0
        for x in range(x_bins):
            for y in range(y_bins):
                x_en = x*fuzzy[y][x]
                x_avg += x_en
        x_avg = (x_avg/x_bins)/y_bins
    if event_type == 'sharp':
        event_points = []
        sharp, fuzzy = dataset[event]
        y_avg = 0
        for y in range(y_bins):
            y_en = y*np.mean(sharp[y])
            y_avg += y_en
        y_avg = y_avg/y_bins
        x_avg = 0
        for x in range(x_bins):
            for y in range(y_bins):
                x_en = x*sharp[y][x]
                x_avg += x_en
        x_avg = (x_avg/x_bins)/y_bins
    return x_avg, y_avg
            
def centroid_plotdata(dataset, outputs, event_type, ppe_plot, x_bins, y_bins):
    random.seed(0)
    centroids = []
    centroids_x = []
    centroids_y = []
    for i in range(len(dataset)):
        cntr = centroid(dataset, outputs, i, event_type, x_bins, y_bins)
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
    random.seed(0)
    centroid_rads = []
    for i in range(len(centroid_plotdata[0])):
        centroid_rad = np.sqrt(centroid_plotdata[0][i]**2+centroid_plotdata[1][i]**2)
        centroid_rads.append(centroid_rad)
    return centroid_rads

def hits_above_threshold(dataset, outputs, threshold, event, event_type, x_bins, y_bins):
    if event_type == 'sharp' or 'fuzzy':
        sharp, fuzzy = dataset[event]
    count = 0
    if event_type == 'sharp':
        for y in range(y_bins):
            for x in range(x_bins):
                if sharp[y][x] >= threshold:
                    count +=1
    if event_type == 'fuzzy':
        for y in range(y_bins):
            for x in range(x_bins):
                if fuzzy[y][x] >= threshold:
                    count +=1
    if event_type == 'output':
        output = outputs[event]
        for y in range(y_bins):
            for x in range(x_bins):
                if output[y][x] >= threshold:
                    count +=1
    return count

def hits_data(dataset, outputs, threshold, event_type, x_bins, y_bins):
    random.seed(0)
    hits = []
    for i in range(len(dataset)):
        hit = hits_above_threshold(dataset, outputs, threshold, i, event_type, x_bins, y_bins)
        hits.append(hit)
    return hits

def dist_above_threshold(dataset, outputs, threshold, event, event_type, x_bins, y_bins):
    hits = []
    if event_type == 'sharp' or 'fuzzy':
        sharp, fuzzy = dataset[event]
    if event_type == 'sharp':
        for y in range(y_bins):
            for x in range(x_bins):
                if sharp[y][x] >= threshold:
                    hits.append(sharp[y][x])
    if event_type == 'fuzzy':
        for y in range(y_bins):
            for x in range(x_bins):
                if fuzzy[y][x] >= threshold:
                    hits.append(fuzzy[y][x])
    if event_type == 'output':
        output = outputs[event]
        for y in range(y_bins):
            for x in range(x_bins):
                if output[y][x] >= threshold:
                    hits.append(output[y][x])
    return hits

def dist_hits_data(dataset, outputs, threshold, event_type, x_bins, y_bins):
    dist_hits = []
    for i in range(len(dataset)):
        event_hits = dist_above_threshold(dataset, outputs, threshold, i, event_type, x_bins, y_bins)
        for elem in event_hits:
            dist_hits.append(elem)
    return dist_hits

# def dist_hits_data(dataset, outputs, threshold, event_type):
#     random.seed(0)
#     hits = []
#     length = 0
#     for i in range(len(dataset)):
#         event_hits = dist_above_threshold(dataset, outputs, threshold, i, event_type)
#         hits.append(event_hits)
#         length += len(event_hits)
#     hits_dist = np.zeros(length)
#     for i in range(len(dataset)):
#         np.append(hits_dist, hits[i])
#     return hits_dist

def diff(dataset1, dataset2):
    diffset = []
    for i, elem in enumerate(dataset1):
        diff = dataset1[i] - dataset2[i]
        diffset.append(diff)
    return diffset

def plot_hist(data, plotname, axis_x, axis_y, bins=None, labels=None, plotrange=None):
    plt.hist(data, bins=bins, alpha=0.75, label=labels)
    plt.legend(loc='upper left')
    plt.title(plotname)
    plt.xlabel(axis_x)
    plt.ylabel(axis_y)
    if plotrange:
        plt.xlim(xmin=plotrange[0], xmax=plotrange[1])
    
def plot_scatter(data, data2, plotname, axis_x, axis_y, bins=None, labels=None, plotrange=None):
    plt.scatter(data[0], data[1], label=labels[0])
    if data2:
        plt.scatter(data2[0], data2[1], label=labels[1])
    plt.title(plotname)
    plt.legend(loc='upper left')
    plt.xlabel(axis_x)
    plt.ylabel(axis_y)
    
def main():
    #def main():
    #def main():
    os.makedirs(args.outf+'/analysis-plots/')
    
    x_bins, y_bins = calculate_bins(args.fileSharp)
    
    random.seed(args.randomseed)
    outputs = np.load(args.numpy)['arr_0']
    dataset = dat.RootBasic(args.fileFuzz, args.fileSharp, args.transform)

    ppe_plot = ppe_plotdata(dataset, outputs)
    centroid_data = centroid_plotdata(dataset, outputs, 'sharp', ppe_plot, x_bins, y_bins)
    centroid_data_fuzzy = centroid_plotdata(dataset, outputs, 'fuzzy', ppe_plot, x_bins, y_bins)
    centroid_data_output = centroid_plotdata(dataset, outputs, 'output', ppe_plot, x_bins, y_bins)

    plot_hist([ppe_plot[0],ppe_plot[2],ppe_plot[1],], 'Energy per Pixel', 'Energy', 'Number of Events', bins=None, labels = ['high-quality', 'enhanced','low-quality'])
    plt.savefig(args.outf+'/analysis-plots/energy-per-pixel-hle.png')
    plt.clf()

    plot_hist([ppe_plot[0],ppe_plot[2]], 'Energy per Pixel', 'Energy', 'Number of Events', bins=None, labels = ['high-quality', 'enhanced'])
    plt.savefig(args.outf+'/analysis-plots/energy-per-pixel-he.png')
    plt.clf()

    hits_sharp = dist_hits_data(dataset, outputs, 0.01, 'sharp', x_bins, y_bins)
    hits_fuzzy = dist_hits_data(dataset, outputs, 0.01, 'fuzzy', x_bins, y_bins)
    hits_output = dist_hits_data(dataset, outputs, 0.01, 'output', x_bins, y_bins)

    plot_hist([hits_sharp,hits_output], 'Hits Above Threshold', 'Energy in Pixel', 'Number of Pixels', bins=20, labels = ['high-quality', 'enhanced'], plotrange=(0,2000))
    plt.savefig(args.outf+'/analysis-plots/hits-above-threshold-dist-he.png')
    plt.clf()

    plot_hist([hits_sharp,hits_output,hits_fuzzy], 'Hits Above Threshold', 'Energy in Pixel', 'Number of Pixels', bins=20, labels = ['high-quality', 'enhanced', 'low-quality'], plotrange=(0,2000))
    plt.savefig(args.outf+'/analysis-plots/hits-above-threshold-dist-hle.png')
    plt.clf()

    plot_hist([centroid_rad_data(centroid_data),centroid_rad_data(centroid_data_output),centroid_rad_data(centroid_data_fuzzy)], 'Radius of Energy Centroid', 'Radius', 'Number of Events', bins=None, labels = ['high-quality', 'enhanced','low-quality'])
    plt.savefig(args.outf+'/analysis-plots/rad-centroid-hist-hle.png')
    plt.clf()

    plot_hist([centroid_rad_data(centroid_data),centroid_rad_data(centroid_data_output)], 'Radius of Energy Centroid', 'Radius', 'Number of Events', bins=5, labels = ['high-qualtiy', 'enhanced'], plotrange=(20,60))
    plt.savefig(args.outf+'/analysis-plots/rad-centroid-hist-he.png')
    plt.clf()
    
if __name__ == "__main__":
    main()