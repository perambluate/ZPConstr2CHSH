"""
    Plot the finite rate of blind randomness for the delta-family Bell inequalities
    in (PRL. vol. 129, no. 15, 5, 150403)
"""

from matplotlib import pyplot as plt, ticker as mticker
from functools import partial
from joblib import Parallel, delayed
import numpy as np
import math
import os, sys

### Add current directory to Python path
# (Note: this works when running the command in the dir. 'blindRandomness')
sys.path.append('..')
from common_func.plotting_helper import *

### Parallel settings
N_JOB = 8

### Constant parameters
CHSH_W_Q = (2 + math.sqrt(2))/4    # CHSH win prob
EPSILON = 1e-12                    # Smoothness of smooth min entropy (related to secrecy)
WIN_TOL = 1e-5                     # Tolerant error for win prob
GAMMA = 1e-2                       # Testing ratio
INP_DIST = [1/4]*4

######### Plotting settings #########
# FIG_SIZE = (16, 9)      # for aspect ratio 16:9
FIG_SIZE = (12, 9)    # for aspect ratio 4:3
DPI = 200
SUBPLOT_PARAM = {'left': 0.085, 'right': 0.98, 'bottom': 0.11, 'top': 0.95, 'wspace': 0.28}

### Set font sizes and line width
titlesize = 30
ticksize = 28
legendsize = 26
linewidth = 3

### Set matplotlib plotting params
mplParams = plot_settings(title = titlesize, tick = ticksize, legend = legendsize,
                          linewidth = linewidth)
mplParams["xtick.major.pad"] = 10

#### Colors for different classes
plt.rcParams.update(mplParams)

### Num of rounds
N_RANGE = (1e6, 1e14)
N_SLICE = 100 #200
Ns = np.geomspace(*N_RANGE, num=N_SLICE)

### Freely tunable params
#### Param related to alpha-Renyi entropy
BETA_SLICE = 50
BETAs = np.geomspace(1e-4, 1e-11, num=BETA_SLICE)

#### Testing ratio
GAMMA_SLICE = 50
GAMMAs = 1/np.geomspace(10, 10000, num=GAMMA_SLICE)

#### Another tunable param
NU_PRIME_SLICE = 50
NU_PRIMEs = np.linspace(1, 0.1, num=NU_PRIME_SLICE)

#### Plot for max win prob
fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)

### Get data to compute finite rate
DATAPATH = './data/asymp_br_wbc.csv'
data_mtf = np.genfromtxt(DATAPATH, delimiter=",", skip_header = 1)

counter = 1
for delta, entropy, qbound, lambda_, c_lambda in data_mtf[:5]:
    fr_func = partial(fin_rate_testing, asym_rate = entropy, lambda_ = lambda_,
                      c_lambda = c_lambda, max_p_win = qbound)
    
    FRs = Parallel(n_jobs=N_JOB, verbose = 0)(
          delayed(opt_all)(n, beta_arr = BETAs, nup_arr = NU_PRIMEs, gam_arr = GAMMAs,
                           inp_dist = INP_DIST, fin_rate_func = fr_func) for n in Ns)
    print(FRs)
    print(counter)
    counter += 1

    label = r'$\displaystyle \delta={:.4g}$'.format(delta)
    plt.plot(Ns, FRs, label = label)

################################ Save figure ################################
YLABEL = r'$\displaystyle r$'+' (bit per round)'
XLABEL = r'$\displaystyle n$'+' (number of rounds)'    

plt.xscale("log")
plt.ylabel(YLABEL)
plt.xlabel(XLABEL)
plt.legend(prop={"weight":"bold"})

### Apply the graphic settings
plt.subplots_adjust(**SUBPLOT_PARAM)

### Save file
COM = 'br-qbound-wooltoron'
EPS = f'eps_{EPSILON:.0e}'
WTOL = f'wtol_{WIN_TOL:.0e}'
QUAD = 'M_12'
TAIL = 'test'
FORMAT = 'png'
# OUT_NAME = f'{COM}-{EPS}-{WTOL}-{GAM}-{QUAD}'
OUT_NAME = f'{COM}-{EPS}-{WTOL}-{QUAD}'
if TAIL:
    OUT_NAME += f'-{TAIL}'
OUT_FILE = f'{OUT_NAME}.{FORMAT}'
OUT_PATH = os.path.join(OUT_FILE)
plt.savefig(OUT_PATH, format = FORMAT)

plt.show()