import sys
sys.path.append(".local/lib/python{}.{}/site-packages".format(sys.version_info.major,sys.version_info.minor))

import uproot as up
import numpy as np
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
lumitext = r'photon, $E=850\,\mathrm{GeV}$, $\eta=0.5$, $\phi=0$'
data_labels = {'sharp': 'Geant4', 'fuzzy': r'Modified', 'outputs': 'CNN'}

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
def centroid(data, bininfo, stdev=False):
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
    def avg(centers,energies, stdev=False):
        avgs = np.average(centers,weights=energies,axis=1)
        if stdev:
            return np.sqrt(avg((centers-avgs[:,None])**2,energies))
        else:
            return avgs
    xavg = avg(xcenters,xenergies,stdev)
    yavg = avg(ycenters,yenergies,stdev)
    rad = np.sqrt(xavg**2+yavg**2)
    return rad

def hits_above_threshold(data, threshold):
    return np.sum(data>threshold,axis=(1,2))

def pixel_energy(data, threshold=None):
    if threshold is not None: data = data[data>threshold]
    return data.flatten()

def plot_hist(dataset, samples, qty, axis_x, axis_y, bins=20, plotrange=None, path=None, logx=False, logy=True, loc='best', threshold_line=None):
    data = [dataset[sample][qty] for sample in samples]
    labels = [data_labels[sample] for sample in samples]
    if plotrange is None:
        data_concat = np.concatenate(data)
        if logx: data_concat = np.ma.masked_array(data_concat, mask=data_concat<=0.0)
        plotrange = [np.min(data_concat), np.max(data_concat)]
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
    hep.cms.lumitext(lumitext,fontsize=mpl.rcParams["font.size"]*0.5)
    if path:
        plt.savefig(path)
    plt.close()

def plot_scatter(dataset, samples, qty, axis_x, axis_y, bins=None, labels=None, plotline=True, path=None, loc='best'):
    data = [[dataset[sample[0]][qty],dataset[sample[1]][qty]] for sample in samples]
    labels = [data_labels[sample[1]] for sample in samples]
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
    hep.cms.lumitext(lumitext,fontsize=mpl.rcParams["font.size"]*0.5)
    if path:
        plt.savefig(path)
    plt.close()

def print_time(do_print,qty,t1,operation="Computed"):
    t2 = time.time()
    if do_print: print("{} {} ({} s)".format(operation,qty,t2-t1))
    return t2

# create and save sharp, fuzzy, and reconstructed data sets and store in text files
def make_sample_images(dataset, bininfo, path):
    # build axes for histogram given min/max and bin number
    def make_axis(info):
        axis = [info["min"]+info["size"]*i for i in range(info["nbins"]+1)]
        return axis
    xaxis = make_axis(bininfo['x'])
    yaxis = make_axis(bininfo['y'])

    nimgs = 10
    for name,data in dataset.items():
        for event in range(nimgs):
            # make histogram
            fig = plt.subplots(figsize=(10, 7))
            artist = hep.hist2dplot(data["data"][event], xaxis, yaxis)
            plt.xlabel("x [mm]")
            plt.ylabel("y [mm]")
            artist.cbar.set_label("Energy [MeV]")
            hep.cms.label(data=False,label="Preliminary",rlabel=lumitext)
            # add text indicating sample type
            # based on mplhep lumitext()
            ax = plt.gca()
            ax.text(
                x = 0.97,
                y = 0.97,
                s = data_labels[name],
                transform = ax.transAxes,
                ha = "right",
                va = "top",
                fontsize = mpl.rcParams["font.size"]*1.25,
                fontweight = "normal",
                fontname = "TeX Gyre Heros",
                color = "w",
            )
            plt.savefig(path+'/'+name+str(event)+'.png')
            plt.close()

def main():
    parser = ArgumentParser(description="DnCNN", config_options=MagiConfigOptions(), formatter_class=ArgumentDefaultsRawHelpFormatter)

    parser.add_argument("--outf", type=str, required=True, help='Name of folder to be used to store output folder')
    parser.add_argument("--folder", type=str, default="analysis-plots", help='Name of folder to be used to store output plots')
    parser.add_argument("--sample", type=str, default="sample-images", help='Name of folder to be used to store sample image plots')
    parser.add_argument("--numpy", type=str, default="test.npz", help='Path to .npz file of CNN-enhanced low quality (fuzzy) data')
    parser.add_argument("--fileSharp", type=str, default=[], nargs='+', help='Path to higher quality .root file for making plots')
    parser.add_argument("--fileFuzz", type=str, default=[], nargs='+', help='Path to lower quality .root file for making plots')
    parser.add_argument("--randomseed", type=int, default=0, help="Initial value for random.seed()")
    parser.add_argument("--verbose", default=False, action="store_true", help="enable verbose printouts")
    args = parser.parse_args()

    ana_path = args.outf+'/'+args.folder
    os.makedirs(ana_path,exist_ok=True)
    sample_path = args.outf+'/'+args.sample
    os.makedirs(sample_path,exist_ok=True)

    t1 = time.time()
    if args.verbose: print("Started")
    bininfo = calculate_bins(args.fileSharp[0])

    outputs = np.load(args.numpy)['arr_0']
    random.seed(args.randomseed)
    sharp, fuzzy = freeze_dataset(dat.RootDataset(args.fileFuzz, args.fileSharp))
    dataset = dict(
        sharp = dict(data=sharp),
        fuzzy = dict(data=fuzzy),
        outputs = dict(data=outputs),
    )

    threshold = 0.1
    qtys = OrderedDict([
        ("ppe",ppe),
        ("ppe_threshold",partial(ppe,threshold=threshold)),
        ("centroid",partial(centroid,bininfo=bininfo)),
        ("centroid_stdev",partial(centroid,bininfo=bininfo,stdev=True)),
        ("nhits",partial(hits_above_threshold,threshold=threshold)),
        ("hits",partial(pixel_energy,threshold=0.01)), # different threshold
    ])

    t2 = print_time(args.verbose, "datasets", t1, "Loaded")

    make_sample_images(dataset, bininfo, sample_path)

    t2b = print_time(args.verbose, "sample images", t2, "Made")

    t3 = t2b
    for qty,fn in qtys.items():
        for dataname,datadict in dataset.items():
            datadict[qty] = fn(datadict["data"])
        t3 = print_time(args.verbose, qty, t3)

    plot_hist(dataset, ["sharp","fuzzy","outputs"], "ppe", r'$\langle$Energy/pixel$\rangle$ [MeV]', 'Number of events', loc = 'upper left', path=ana_path+'/energy-per-pixel-hle.png')

    plot_hist(dataset, ["sharp","fuzzy","outputs"], "ppe_threshold", r'$\langle$Energy/pixel$\rangle$ [MeV]', 'Number of events', path=ana_path+'/energy-per-pixel-threshold-hle.png')

    plot_hist(dataset, ["sharp","fuzzy","outputs"], "hits", 'Pixel energy [MeV]', 'Number of pixels', logx=True, loc = 'upper left', threshold_line = threshold, path=ana_path+'/hits-above-threshold-dist-hle.png')

    plot_hist(dataset, ["sharp","fuzzy","outputs"], "centroid", 'Centroid [pixels]', 'Number of events', path=ana_path+'/rad-centroid-hist-hle.png')

    plot_hist(dataset, ["sharp","fuzzy","outputs"], "centroid_stdev", r'$\sigma_{\mathrm{centroid}}$ [pixels]', 'Number of events', path=ana_path+'/rad-centroid-stdev-hist-hle.png')

    plot_hist(dataset, ["sharp","fuzzy","outputs"], "nhits", 'Number of hits', 'Number of events', loc = 'upper left', path=ana_path+'/hit-number-hle.png')

    plot_scatter(dataset, [["sharp","fuzzy"],["sharp","outputs"]], "centroid", 'Centroid ({}) [pixels]'.format(data_labels["sharp"]), 'Centroid [pixels]', path=ana_path+'/rad-centroid-scatter.png')

    plot_scatter(dataset, [["sharp","fuzzy"],["sharp","outputs"]], "centroid_stdev", r'{} ({}) [pixels]'.format(r'$\sigma_{\mathrm{centroid}}$', data_labels["sharp"]), r'$\sigma_{\mathrm{centroid}}$ [pixels]', path=ana_path+'/rad-centroid-stdev-scatter.png')

    plot_scatter(dataset, [["sharp","fuzzy"],["sharp","outputs"]], "nhits", 'Number of hits ({})'.format(data_labels["sharp"]), 'Number of hits', path=ana_path+'/hit-scatter.png')

    plot_scatter(dataset, [["sharp","fuzzy"],["sharp","outputs"]], "ppe", r'$\langle$Energy/pixel$\rangle$ ({}) [MeV]'.format(data_labels["sharp"]), r'$\langle$Energy/pixel$\rangle$ [MeV]', path=ana_path+'/energy-scatter.png')

    plot_scatter(dataset, [["sharp","fuzzy"],["sharp","outputs"]], "ppe_threshold", r'$\langle$Energy/pixel$\rangle$ ({}) [MeV]'.format(data_labels["sharp"]), r'$\langle$Energy/pixel$\rangle$ [MeV]', path=ana_path+'/energy-scatter-threshold.png')

    t4 = print_time(args.verbose, "plots", t3, "Made")

if __name__ == "__main__":
    main()
