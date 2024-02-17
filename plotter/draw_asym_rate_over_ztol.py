from matplotlib import pyplot as plt
import numpy as np
import os, sys

sys.path.append('.')
from common_func.plotting_helper import *

######### Plotting settings #########
# FIG_SIZE = (16, 9)      # for aspect ratio 16:9
FIG_SIZE = (12, 9)    # for aspect ratio 4:3
DPI = 200
SUBPLOT_PARAM = {'left': 0.15, 'right': 0.98, 'bottom': 0.11, 'top': 0.95, 'wspace': 0.28}

### Set font sizes and line width
titlesize = 30
ticksize = 28
legendsize = 26
linewidth = 3

### Set matplotlib plotting params
mplParams = plot_settings(title = titlesize, tick = ticksize, legend = legendsize,
                          linewidth = linewidth)
mplParams["xtick.major.pad"] = 10

plt.rcParams.update(mplParams)

plt.figure(figsize=FIG_SIZE, dpi=DPI)
plt.subplots_adjust(**SUBPLOT_PARAM)

DATADIR = 'twoPartyRandomness/data/asymp_rate'
DATAFILE = 'tpr_ztoltest-3b-xy_01-wtol_1e-04-M_12.csv'
DATAPATH = os.path.join(DATADIR, DATAFILE)

data = np.genfromtxt(DATAPATH, delimiter=",", skip_header = 1).T
data_ztol = data[0]
data_entropy = data[3]
print(data_ztol)
print(data_entropy)

# plt.scatter(data_ztol, data_entropy)
plt.plot(data_ztol, data_entropy, linestyle='', marker='o', markersize=12)
plt.xscale("log")

X_TITLE = r'$\displaystyle \delta_{z}$' + 'zero tolerance'
plt.xlabel(X_TITLE)
plt.ylabel(r"$\displaystyle H(AB|XYE')$")

OUTFILE = 'asymp_tpr_over_ztol-3b.png'
plt.savefig(OUTFILE)

# plt.show()