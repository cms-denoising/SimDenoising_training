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
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import random
import os, time
import dataset as dat
from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter
from functools import partial
from collections import OrderedDict

def calculate_bins(fin):
    upfile = up.open(fin)
    tree = upfile["g4SimHits"]["tree"]
    bininfo = dict(
        x = {},
        y = {},
    )
    def get_bin_qty(tree,coord,qty):
        return tree["{}{}".format(coord,qty)].array().to_numpy()[0]
    for coord in bininfo:
        bininfo[coord]["nbins"] = get_bin_qty(tree,coord,"bins")
        bininfo[coord]["min"] = get_bin_qty(tree,coord,"min")
        bininfo[coord]["max"] = get_bin_qty(tree,coord,"max")
        bininfo[coord]["size"] = (bininfo[coord]["max"]-bininfo[coord]["min"])/bininfo[coord]["nbins"]

    return bininfo

def freeze_dataset(dataset):
    return dataset.sharp_branch, dataset.fuzzy_branch

def ppe(data):
    return np.mean(data, axis=(1,2))

def centroid(data, bininfo):
    # assume evenly spaced bins
    width = 0.5
    # need same bin centers for each event: concatenate is the fastest way to do this
    xcenters = np.arange(data.shape[1])+width
    xcenters = np.concatenate([xcenters[np.newaxis,...]]*data.shape[0],axis=0)
    ycenters = np.arange(data.shape[2])+width
    ycenters = np.concatenate([ycenters[np.newaxis,...]]*data.shape[0],axis=0)
    xenergies = np.sum(data, axis=2)
    yenergies = np.sum(data, axis=1)
    xavg = np.sum(xcenters*xenergies,axis=1)/np.sum(xenergies,axis=1)*bininfo["x"]["size"]+bininfo["x"]["min"]
    yavg = np.sum(ycenters*yenergies,axis=1)/np.sum(yenergies,axis=1)*bininfo["y"]["size"]+bininfo["y"]["min"]
    rad = np.sqrt(xavg**2+yavg**2)
    return rad

def hits_above_threshold(data, threshold):
    return np.sum(data>threshold,axis=(1,2))

def dist_above_threshold(data, threshold):
    tmp = data[data>threshold]
    return tmp.flatten()

def plot_hist(data, plotname, axis_x, axis_y, bins=None, labels=None, plotrange=None, path=None):
    plt.hist(data, bins=bins, alpha=0.75, label=labels)
    plt.legend(loc='upper left')
    plt.title(plotname)
    plt.xlabel(axis_x)
    plt.ylabel(axis_y)
    if plotrange:
        plt.xlim(xmin=plotrange[0], xmax=plotrange[1])
    if path:
        plt.savefig(path)
    plt.clf()

def plot_scatter(data, data2, plotname, axis_x, axis_y, bins=None, labels=None, plotrange=None, plotline=True, path=None):
    plt.scatter(data[0], data[1], label=labels[0])
    min_x = min(data[0])
    max_x = max(data[0])
    if data2:
        plt.scatter(data2[0], data2[1], label=labels[1])
        min_x = min(min(data2[0]), min_x)
        max_x = max(max(data2[0]), max_x)
    if plotline == True:
        x=[min_x, max_x]
        y=x
        plt.plot(x, y)
    plt.title(plotname)
    plt.legend(loc='upper left')
    plt.xlabel(axis_x)
    plt.ylabel(axis_y)
    if path:
        plt.savefig(path)
    plt.clf()

def print_time(do_print,qty,t1,operation="Computed"):
    t2 = time.time()
    if do_print: print("{} {} ({} s)".format(operation,qty,t2-t1))
    return t2

def main():
    parser = ArgumentParser(description="DnCNN", config_options=MagiConfigOptions(), formatter_class=ArgumentDefaultsRawHelpFormatter)

    parser.add_argument("--outf", type=str, default="analysis-plots", help='Name of folder to be used to store outputs')
    parser.add_argument("--numpy", type=str, default="test.npz", help='Path to .npz file of CNN-enhanced low quality (fuzzy) data')
    parser.add_argument("--fileSharp", type=str, default=[], nargs='+', help='Path to higher quality .root file for making plots')
    parser.add_argument("--fileFuzz", type=str, default=[], nargs='+', help='Path to lower quality .root file for making plots')
    parser.add_argument("--randomseed", type=int, default=0, help="Initial value for random.seed()")
    parser.add_argument("--verbose", default=False, action="store_true", help="enable verbose printouts")
    args = parser.parse_args()

    os.makedirs(args.outf+'/analysis-plots/',exist_ok=True)

    t1 = time.time()
    if args.verbose: print("Started")
    bininfo = calculate_bins(args.fileSharp[0])

    outputs = np.load(args.numpy)['arr_0']
    random.seed(args.randomseed)
    torch.manual_seed(args.randomseed)
    sharp, fuzzy = freeze_dataset(dat.RootDataset(args.fileFuzz, args.fileSharp, 'none'))
    dataset = dict(
        sharp = dict(data=sharp),
        fuzzy = dict(data=fuzzy),
        outputs = dict(data=outputs),
    )

    threshold = 0.01
    qtys = OrderedDict([
        ("ppe",ppe),
        ("centroid",partial(centroid,bininfo=bininfo)),
        ("nhits",partial(hits_above_threshold,threshold=threshold)),
        ("hits",partial(dist_above_threshold,threshold=threshold)),
    ])

    t2 = print_time(args.verbose, "datasets", t1, "Loaded")

    t3 = t2
    for qty,fn in qtys.items():
        for dataname,datadict in dataset.items():
            datadict[qty] = fn(datadict["data"])
        t3 = print_time(args.verbose, qty, t3)

    plot_hist([dataset["sharp"]["ppe"],dataset["outputs"]["ppe"],dataset["fuzzy"]["ppe"]], 'Energy per Pixel', 'Energy (MeV)', 'Number of Events', bins=None, labels = ['high-quality', 'enhanced','low-quality'], path=args.outf+'/analysis-plots/energy-per-pixel-hle.png')

    plot_hist([dataset["sharp"]["ppe"],dataset["outputs"]["ppe"]], 'Energy per Pixel', 'Energy (MeV)', 'Number of Events', bins=None, labels = ['high-quality', 'enhanced'], path=args.outf+'/analysis-plots/energy-per-pixel-he.png')

    plot_hist([dataset["sharp"]["hits"],dataset["outputs"]["hits"],dataset["fuzzy"]["hits"]], 'Hits Above Threshold', 'Energy in Pixel (MeV)', 'Number of Pixels', bins=20, labels = ['high-quality', 'enhanced', 'low-quality'], plotrange=(0,2000), path=args.outf+'/analysis-plots/hits-above-threshold-dist-hle.png')

    plot_hist([dataset["sharp"]["hits"],dataset["outputs"]["hits"]], 'Hits Above Threshold', 'Energy in Pixel (MeV)', 'Number of Pixels', bins=None, labels = ['high-quality', 'enhanced'], plotrange=(0,2000), path=args.outf+'/analysis-plots/hits-above-threshold-dist-he.png')

    plot_hist([dataset["sharp"]["centroid"],dataset["outputs"]["centroid"],dataset["fuzzy"]["centroid"]], 'Radius of Energy Centroid', 'Radius (pixels)', 'Number of Events', bins=5, labels = ['high-quality', 'enhanced'], path=args.outf+'/analysis-plots/rad-centroid-hist-hle.png')

    plot_hist([dataset["sharp"]["centroid"],dataset["outputs"]["centroid"]], 'Radius of Energy Centroid', 'Radius (pixels)', 'Number of Events', bins=5, labels = ['high-quality', 'enhanced'], path=args.outf+'/analysis-plots/rad-centroid-hist-he.png')

    plot_hist([dataset["sharp"]["nhits"],dataset["outputs"]["nhits"],dataset["fuzzy"]["nhits"]], 'Number of Hits', "", 'Number of Events', bins=10, labels = ['high-quality', 'enhanced','low-quality'], path=args.outf+'/analysis-plots/hit-number-hle.png')

    plot_hist([dataset["sharp"]["nhits"],dataset["outputs"]["nhits"]], 'Number of Hits', "", 'Number of Events', bins=10, labels = ['high-quality', 'enhanced'], path=args.outf+'/analysis-plots/hit-number-he.png')

    plot_scatter([dataset["sharp"]["centroid"],dataset["outputs"]["centroid"]],[dataset["sharp"]["centroid"],dataset["fuzzy"]["centroid"]], 'Energy Centroid vs. High Quality Energy Centroid', 'Radius (high-quality) (pixels)', 'Radius (pixels)', labels=['enhanced', 'low-quality'], plotrange=None, path=args.outf+'/analysis-plots/rad-centroid-scatter.png')

    plot_scatter([dataset["sharp"]["nhits"],dataset["outputs"]["nhits"]], [dataset["sharp"]["nhits"],dataset["fuzzy"]["nhits"]], 'Hits vs. Hits', 'Hits(high-quality)', 'Hits(low-quality)', labels=['enhanced', 'low-quality'], plotrange=None, plotline=True, path=args.outf+'/analysis-plots/hit-scatter.png')

    plot_scatter([dataset["sharp"]["ppe"],dataset["outputs"]["ppe"]],[dataset["sharp"]["ppe"],dataset["fuzzy"]["ppe"]], 'Energy vs. Energy', 'Energy per Pixel(high-quality) (MeV)', 'Energy per Pixel(low-quality) (MeV)', labels=['enhanced', 'low-quality'], plotrange=None, plotline=True, path=args.outf+'/analysis-plots/energy-scatter.png')

if __name__ == "__main__":
    main()
