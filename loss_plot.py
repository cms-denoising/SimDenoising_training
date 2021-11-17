import sys
sys.path.append(".local/lib/python{}.{}/site-packages".format(sys.version_info.major,sys.version_info.minor))

from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use("CMS")
plt.style.use('default.mplstyle')
# based on https://github.com/mpetroff/accessible-color-cycles
# blue, red, gray, mauve, orange, purple
colors = ["#5790fc", "#e42536", "#9c9ca1", "#964a8b", "#f89c20", "#7a21dd"]
styles = ['--','-',':','-.']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

# parse arguments
parser = ArgumentParser(description="DnCNN", config_options=MagiConfigOptions(), formatter_class=ArgumentDefaultsRawHelpFormatter)
parser.add_argument("--outf", type=str, default="logs", help='Name of folder to be used to store outputs')
args = parser.parse_args()

# plot loss/epoch for training and validation sets
for name in ["training","validation"]:
    data = np.loadtxt("{}/{}_losses.txt".format(args.outf,name))
    plt.plot(data, label=name)
plt.xlabel("Epoch")
plt.ylabel("Loss")
hep.cms.label(data=False,label="Preliminary",rlabel="")
plt.legend()
plt.savefig(args.outf + "/loss_plot.png")
