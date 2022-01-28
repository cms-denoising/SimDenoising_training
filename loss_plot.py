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
printformats = []
printargs = {
    "png": {"dpi":100},
    "pdf": {"bbox_inches":"tight"},
}
def save_figs(path):
    for pf in printformats:
        fargs = printargs[pf] if pf in printargs else {}
        plt.savefig(path+'.'+pf,**fargs)

# parse arguments
parser = ArgumentParser(description="DnCNN", config_options=MagiConfigOptions(), formatter_class=ArgumentDefaultsRawHelpFormatter)
parser.add_argument("--zero", default=False, action="store_true", help='Set y axis minimum to 0')
parser.add_argument("--outf", type=str, required=True, help='Name of folder to be used to store outputs')
parser.add_argument("--printformats", type=str, default=["png"], nargs='+', help="print formats")
args = parser.parse_args()
printformats = args.printformats

# plot loss/epoch for training and validation sets
for name in ["training","validation"]:
    data = np.loadtxt("{}/{}_losses.txt".format(args.outf,name))
    plt.plot(data, label=name)
plt.xlabel("Epoch")
plt.ylabel("Loss")
if args.zero:
    plt.ylim(bottom=0)
hep.cms.label(data=False,label="Preliminary",rlabel="")
plt.legend()
save_figs(args.outf + "/loss_plot")
