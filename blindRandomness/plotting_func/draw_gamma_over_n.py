"""
    This script generate a plot for optimal gammas over number of rounds.
"""
from matplotlib import pyplot as plt
from cycler import cycler
import numpy as np
import os, sys

### Add directory to Python path
# (Note: this works when running the command in the dir. 'blindRandomness')
sys.path.append('.')
from common_func.plotting_helper import *

CLASS_MAX_WIN = {'CHSH': 0.8535, '1': 0.8294, '2a': 0.8125,
                 '2b': 0.8125, '2b_swap': 0.8125, '2c': 0.8039,
                 '3a': 0.7951, '3b': 0.7837}

SAVE = True         # To save figure or not
SHOW = False        # To show figure or not

######### Plotting settings #########
FIG_SIZE = (12, 9)    # aspect ratio
DPI = 200
SUBPLOT_PARAM = {'left': 0.115, 'right': 0.98, 'bottom': 0.11, 'top': 0.95, 'wspace': 0.28}

titlesize = 30
ticksize = 28
legendsize = 26
linewidth = 3

### Set matplotlib plotting params
mplParams = plot_settings(title = titlesize, tick = ticksize, legend = legendsize, linewidth = linewidth)
mplParams["xtick.major.pad"] = 10
plt.rcParams.update(mplParams)
MARKERS = ['.', 'x', '*', '+']
prop_cycler = cycler(color = 'bgrk') + cycler(marker = MARKERS)
plt.rc('axes', prop_cycle = prop_cycler)

EPSILON = 1e-12             # Smoothness of smooth min entropy (related to secrecy)
WIN_TOL = 1e-4              # Tolerant error for win prob
ZERO_TOL = 1e-9             # Tolerant error for zero-prob constraints

EPS = f'eps_{EPSILON:.0e}'
WTOL = f'wtol_{WIN_TOL:.0e}'
ZTOL = f'ztol_{ZERO_TOL:.0e}'
QUAD = 'M_12'

DATA_DIR = './data/opt_gamma'
OUT_DIR = './figures/corrected_FER/gamma_test'

CLASSES = ['CHSH', '1', '2c', '3b']
fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
counter = 0

for class_ in CLASSES:
    # fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
    
    CLS = f'class_{class_}' if class_ != 'CHSH' else 'CHSH'
    max_win = CLASS_MAX_WIN[class_]
    WEXP = f'w_{max_win*10000:.0f}'.rstrip('0')

    DATA_FILE = f'opt_gam-{CLS}-{WEXP}-{EPS}-{WTOL}-{ZTOL}-{QUAD}.csv'
    DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)
    data = np.genfromtxt(DATA_PATH, delimiter=",", skip_header = 1).T
    # print(data)

    Ns = data[0]
    GAMs = data[2]
    alpha = 0.5 if (counter % 4 == 2) else 1    # Decrease opacity for star-like marker
    plt.plot(Ns, GAMs, label = class_, linestyle='', markersize=12, alpha=alpha)
    counter += 1

plt.ylabel(r'$\displaystyle \gamma$'+' (testing ratio)')
plt.xlabel(r'$\displaystyle n$'+' (number of rounds)')
plt.xscale("log")
plt.yscale("log")
plt.legend(prop={"weight":"bold"}, loc='best')

plt.grid()

plt.subplots_adjust(**SUBPLOT_PARAM)

if SAVE:
    ### General File Name Settings
    COM = 'gamma_over_n'
    WEXP = 'QBOUND'
    TAIL = '0'
    FORMAT = 'png'
    # OUT_NAME = f'{COM}-{CLS}-{WEXP}-{EPS}-{WTOL}-{ZTOL}-{QUAD}'
    OUT_NAME = f'{COM}-all_cls-{WEXP}'
    if TAIL:
        OUT_NAME += f'-{TAIL}'
    out_path = os.path.join(OUT_DIR, f'{OUT_NAME}.{FORMAT}')
    plt.savefig(out_path, format = FORMAT)
if SHOW:
    plt.show()