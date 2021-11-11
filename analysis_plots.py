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
    # to get real units, multiply by bin size and add axis minimum value
    xavg = np.sum(xcenters*xenergies,axis=1)/np.sum(xenergies,axis=1)
    yavg = np.sum(ycenters*yenergies,axis=1)/np.sum(yenergies,axis=1)
    rad = np.sqrt(xavg**2+yavg**2)
    return rad

def hits_above_threshold(data, threshold):
    return np.sum(data>threshold,axis=(1,2))

def dist_above_threshold(data, threshold):
    tmp = data[data>threshold]
    return tmp.flatten()

def plot_hist(data, axis_x, axis_y, bins=10, labels=None, plotrange=None, path=None, logx=False, logy=False):
    if logx:
        bins = np.logspace(np.log10(min(np.concatenate(data))),np.log10(max(np.concatenate(data))),bins)
    plt.hist(data, bins=bins, alpha=0.75, label=labels)
    plt.legend(loc='upper left')
    plt.xlabel(axis_x)
    plt.ylabel(axis_y)
    if logx: plt.xscale("log")
    if logy: plt.yscale("log")
    if plotrange:
        plt.xlim(xmin=plotrange[0], xmax=plotrange[1])
    if path:
        plt.savefig(path)
    plt.clf()

def plot_scatter(data, axis_x, axis_y, bins=None, labels=None, plotline=True, path=None):
    xmin = min(np.concatenate([pair[0] for pair in data]))
    xmax = max(np.concatenate([pair[0] for pair in data]))
    for pair,label in zip(data,labels):
        if plotline:
            label = "{} ({:.2f})".format(label, np.corrcoef(pair[0],pair[1])[0][1])
        plt.scatter(pair[0], pair[1], label=label)
    if plotline == True:
        x=[xmin,xmax]
        y=x
        plt.plot(x, y)
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

    plot_hist([dataset["sharp"]["ppe"],dataset["outputs"]["ppe"],dataset["fuzzy"]["ppe"]], r'$\langle$Energy/pixel$\rangle$ [MeV]', 'Number of events', bins=None, labels = ['high-quality', 'enhanced','low-quality'], path=args.outf+'/analysis-plots/energy-per-pixel-hle.png')

    plot_hist([dataset["sharp"]["ppe"],dataset["outputs"]["ppe"]], r'$\langle$Energy/pixel$\rangle$ [MeV]', 'Number of events', bins=None, labels = ['high-quality', 'enhanced'], path=args.outf+'/analysis-plots/energy-per-pixel-he.png')

    plot_hist([dataset["sharp"]["hits"],dataset["outputs"]["hits"],dataset["fuzzy"]["hits"]], 'Pixel energy [MeV]', 'Number of pixels', bins=20, logx=True, logy=True, labels = ['high-quality', 'enhanced', 'low-quality'], path=args.outf+'/analysis-plots/hits-above-threshold-dist-hle.png')

    plot_hist([dataset["sharp"]["hits"],dataset["outputs"]["hits"]], 'Pixel energy [MeV]', 'Number of pixels', bins=20, logx=True, logy=True, labels = ['high-quality', 'enhanced'], path=args.outf+'/analysis-plots/hits-above-threshold-dist-he.png')

    plot_hist([dataset["sharp"]["centroid"],dataset["outputs"]["centroid"],dataset["fuzzy"]["centroid"]], 'Centroid [pixels]', 'Number of events', bins=5, labels = ['high-quality', 'enhanced'], path=args.outf+'/analysis-plots/rad-centroid-hist-hle.png')

    plot_hist([dataset["sharp"]["centroid"],dataset["outputs"]["centroid"]], 'Centroid [pixels]', 'Number of events', bins=5, labels = ['high-quality', 'enhanced'], path=args.outf+'/analysis-plots/rad-centroid-hist-he.png')

    plot_hist([dataset["sharp"]["nhits"],dataset["outputs"]["nhits"],dataset["fuzzy"]["nhits"]], 'Number of hits', 'Number of events', bins=10, labels = ['high-quality', 'enhanced','low-quality'], path=args.outf+'/analysis-plots/hit-number-hle.png')

    plot_hist([dataset["sharp"]["nhits"],dataset["outputs"]["nhits"]], 'Number of hits', 'Number of events', bins=10, labels = ['high-quality', 'enhanced'], path=args.outf+'/analysis-plots/hit-number-he.png')

    plot_scatter([[dataset["sharp"]["centroid"],dataset["outputs"]["centroid"]],[dataset["sharp"]["centroid"],dataset["fuzzy"]["centroid"]]], 'Centroid (high-quality) [pixels]', 'Centroid [pixels]', labels=['enhanced', 'low-quality'], path=args.outf+'/analysis-plots/rad-centroid-scatter.png')

    plot_scatter([[dataset["sharp"]["nhits"],dataset["outputs"]["nhits"]], [dataset["sharp"]["nhits"],dataset["fuzzy"]["nhits"]]], 'Number of hits (high-quality)', 'Number of hits', labels=['enhanced', 'low-quality'], path=args.outf+'/analysis-plots/hit-scatter.png')

    plot_scatter([[dataset["sharp"]["ppe"],dataset["outputs"]["ppe"]],[dataset["sharp"]["ppe"],dataset["fuzzy"]["ppe"]]], r'$\langle$Energy/pixel$\rangle$ (high-quality) [MeV]', r'$\langle$Energy/pixel$\rangle$ [MeV]', labels=['enhanced', 'low-quality'], path=args.outf+'/analysis-plots/energy-scatter.png')

if __name__ == "__main__":
    main()
