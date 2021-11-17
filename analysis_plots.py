import sys
sys.path.append(".local/lib/python{}.{}/site-packages".format(sys.version_info.major,sys.version_info.minor))

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
import mplhep as hep

hep.style.use("CMS")
plt.style.use('default.mplstyle')
# based on https://github.com/mpetroff/accessible-color-cycles
# gray, red, blue, mauve, orange, purple
colors = ["#9c9ca1", "#e42536", "#5790fc", "#964a8b", "#f89c20", "#7a21dd"]
styles = ['--','-',':','-.']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

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

def ppe(data, threshold=None):
    if threshold is not None: data = np.ma.masked_array(data,mask=data<=threshold)
    return np.mean(data, axis=(1,2))

# energy-weighted, so threshold is unneeded
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

def pixel_energy(data, threshold=None):
    if threshold is not None: data = data[data>threshold]
    return data.flatten()

def plot_hist(data, axis_x, axis_y, bins=20, labels=None, plotrange=None, path=None, logx=False, logy=True, loc='best', threshold_line=None):
    if plotrange is None:
        data_concat = np.concatenate(data)
        if logx: data_concat = np.ma.masked_array(data_concat, mask=data_concat<=0.0)
        plotrange = [np.min(data_concat), np.max(data_concat)]
    print(path,plotrange)
    if logx:
        bins = np.logspace(np.log10(plotrange[0]),np.log10(plotrange[1]),bins)
    else:
        bins = np.linspace(plotrange[0],plotrange[1],bins)
    hists = [np.histogram(d,bins=bins) for d in data]
    hep.histplot(hists[0],label=labels[0],histtype='fill')
    for i,hist in enumerate(hists[1:]):
        hep.histplot(hist,label=labels[i+1],linestyle=styles[i])
    plt.xlabel(axis_x)
    plt.ylabel(axis_y)
    if logx: plt.xscale("log")
    if logy: plt.yscale("log")
    if plotrange:
        plt.xlim(xmin=plotrange[0], xmax=plotrange[1])
    if threshold_line is not None:
        plt.axvline(x=threshold_line, color='k')
    plt.legend(loc=loc)
    hep.cms.label(data=False,label="Preliminary",rlabel="")
    if path:
        plt.savefig(path)
    plt.clf()

def plot_scatter(data, axis_x, axis_y, bins=None, labels=None, plotline=True, path=None, loc='best'):
    xmin = min(np.concatenate([pair[0] for pair in data]))
    xmax = max(np.concatenate([pair[0] for pair in data]))
    for i,pair in enumerate(data):
        if plotline:
            labels[i] = "{} ({:.2f})".format(labels[i], np.corrcoef(pair[0],pair[1])[0][1])
        sc = plt.scatter(pair[0], pair[1], label=labels[i], facecolor='none', edgecolor=colors[i+1])
    if plotline:
        x=[xmin,xmax]
        y=x
        plt.plot(x, y)
    plt.legend(loc=loc)
    plt.xlabel(axis_x)
    plt.ylabel(axis_y)
    hep.cms.label(data=False,label="Preliminary",rlabel="")
    if path:
        plt.savefig(path)
    plt.clf()

def print_time(do_print,qty,t1,operation="Computed"):
    t2 = time.time()
    if do_print: print("{} {} ({} s)".format(operation,qty,t2-t1))
    return t2

def main():
    parser = ArgumentParser(description="DnCNN", config_options=MagiConfigOptions(), formatter_class=ArgumentDefaultsRawHelpFormatter)

    parser.add_argument("--outf", type=str, required=True, help='Name of folder to be used to store output folder')
    parser.add_argument("--folder", type=str, default="analysis-plots", help='Name of folder to be used to store output plots')
    parser.add_argument("--numpy", type=str, default="test.npz", help='Path to .npz file of CNN-enhanced low quality (fuzzy) data')
    parser.add_argument("--fileSharp", type=str, default=[], nargs='+', help='Path to higher quality .root file for making plots')
    parser.add_argument("--fileFuzz", type=str, default=[], nargs='+', help='Path to lower quality .root file for making plots')
    parser.add_argument("--randomseed", type=int, default=0, help="Initial value for random.seed()")
    parser.add_argument("--verbose", default=False, action="store_true", help="enable verbose printouts")
    args = parser.parse_args()

    os.makedirs(args.outf+'/'+args.folder,exist_ok=True)

    t1 = time.time()
    if args.verbose: print("Started")
    bininfo = calculate_bins(args.fileSharp[0])

    outputs = np.load(args.numpy)['arr_0']
    random.seed(args.randomseed)
    torch.manual_seed(args.randomseed)
    sharp, fuzzy = freeze_dataset(dat.RootDataset(args.fileFuzz, args.fileSharp))
    dataset = dict(
        sharp = dict(data=sharp),
        fuzzy = dict(data=fuzzy),
        outputs = dict(data=outputs),
    )

    threshold = 0.1
    qtys = OrderedDict([
        ("ppe",ppe),
        ("centroid",partial(centroid,bininfo=bininfo)),
        ("ppe_threshold",partial(ppe,threshold=threshold)),
        ("nhits",partial(hits_above_threshold,threshold=threshold)),
        ("hits",partial(pixel_energy,threshold=0.01)), # different threshold
    ])

    t2 = print_time(args.verbose, "datasets", t1, "Loaded")

    t3 = t2
    for qty,fn in qtys.items():
        for dataname,datadict in dataset.items():
            datadict[qty] = fn(datadict["data"])
        t3 = print_time(args.verbose, qty, t3)

    plot_hist([dataset["sharp"]["ppe"],dataset["fuzzy"]["ppe"],dataset["outputs"]["ppe"]], r'$\langle$Energy/pixel$\rangle$ [MeV]', 'Number of events', loc = 'upper left', labels = ['high-quality', 'low-quality','enhanced'], path=args.outf+'/'+args.folder+'/energy-per-pixel-hle.png')

    plot_hist([dataset["sharp"]["ppe_threshold"],dataset["fuzzy"]["ppe_threshold"],dataset["outputs"]["ppe_threshold"]], r'$\langle$Energy/pixel$\rangle$ [MeV]', 'Number of events', labels = ['high-quality', 'low-quality','enhanced'], path=args.outf+'/'+args.folder+'/energy-per-pixel-threshold-hle.png')

    plot_hist([dataset["sharp"]["hits"],dataset["fuzzy"]["hits"],dataset["outputs"]["hits"]], 'Pixel energy [MeV]', 'Number of pixels', logx=True, loc = 'upper left', threshold_line = threshold, labels = ['high-quality', 'low-quality', 'enhanced'], path=args.outf+'/'+args.folder+'/hits-above-threshold-dist-hle.png')

    plot_hist([dataset["sharp"]["centroid"],dataset["fuzzy"]["centroid"],dataset["outputs"]["centroid"]], 'Centroid [pixels]', 'Number of events', labels = ['high-quality', 'low-quality', 'enhanced'], path=args.outf+'/'+args.folder+'/rad-centroid-hist-hle.png')

    plot_hist([dataset["sharp"]["nhits"],dataset["fuzzy"]["nhits"],dataset["outputs"]["nhits"]], 'Number of hits', 'Number of events', loc = 'upper left', labels = ['high-quality', 'low-quality','enhanced'], path=args.outf+'/'+args.folder+'/hit-number-hle.png')

    plot_scatter([[dataset["sharp"]["centroid"],dataset["fuzzy"]["centroid"]],[dataset["sharp"]["centroid"],dataset["outputs"]["centroid"]]], 'Centroid (high-quality) [pixels]', 'Centroid [pixels]', labels=['low-quality', 'enhanced'], path=args.outf+'/'+args.folder+'/rad-centroid-scatter.png')

    plot_scatter([[dataset["sharp"]["nhits"],dataset["fuzzy"]["nhits"]], [dataset["sharp"]["nhits"],dataset["outputs"]["nhits"]]], 'Number of hits (high-quality)', 'Number of hits', labels=['low-quality', 'enhanced'], path=args.outf+'/'+args.folder+'/hit-scatter.png')

    plot_scatter([[dataset["sharp"]["ppe"],dataset["fuzzy"]["ppe"]],[dataset["sharp"]["ppe"],dataset["outputs"]["ppe"]]], r'$\langle$Energy/pixel$\rangle$ (high-quality) [MeV]', r'$\langle$Energy/pixel$\rangle$ [MeV]', labels=['low-quality', 'enhanced'], path=args.outf+'/'+args.folder+'/energy-scatter.png')

    plot_scatter([[dataset["sharp"]["ppe_threshold"],dataset["fuzzy"]["ppe_threshold"]],[dataset["sharp"]["ppe_threshold"],dataset["outputs"]["ppe_threshold"]]], r'$\langle$Energy/pixel$\rangle$ (high-quality) [MeV]', r'$\langle$Energy/pixel$\rangle$ [MeV]', labels=['low-quality', 'enhanced'], path=args.outf+'/'+args.folder+'/energy-scatter-threshold.png')

    t4 = print_time(args.verbose, "plots", t3, "Made")

if __name__ == "__main__":
    main()
