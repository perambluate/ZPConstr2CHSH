"""
    A script to plot the asymptotic rate of blind randomness with biased input distribution.
"""
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import os, sys

### Add current directory to Python path
# (Note: this works when running the command in the dir. 'blindRandomness')
sys.path.append('.')
from common_func.plotting_helper import *

### To save file or not
SAVE = True
### To show figure or not
SHOW = False
### Set true to normalize winning probability in the superclassical-quantum range for all classes
X_NORMALIZED = False
### NPA hierachy level
LEVEL = 2
### Protocol related params
WIN_TOL = 1e-4
ZER_TOL = 1e-9
N_QUAD = 12

### Directory where data saved
DATA_DIR = './data/bff21_zero'
### Directory to save figures
OUT_DIR = './figures'

### Class: 1, 2a, 2b, 2b_swap, 2c, 3a, 3b
CLASSES = ['2a']
### Tolerance error for zero-probability constraints
#ERRORS = ['1e-05', '1e-04', '1e-03', '1e-02', '1e-01']

CLASS_INPUT_MAP = {'CHSH': '00', '1': '01', '2a': '11',
                   '2b': '01', '2b_swap': '11', '2c': '10',
                   '3a': '11', '3b': '10'}

######### Plotting settings #########
FIG_SIZE = (12, 9)
DPI = 200
SUBPLOT_PARAM = {'left': 0.1, 'right': 0.95, 'bottom': 0.095, 'top': 0.95}

### Default font size
titlesize = 30
ticksize = 28
legendsize = 28
linewidth = 3

### Set matplotlib plotting params
mplParams = plot_settings(title = titlesize, tick = ticksize, legend = legendsize, linewidth = linewidth)
mplParams["axes.prop_cycle"] = \
                mpl.cycler(color = ['blue','forestgreen','firebrick','gray','darkcyan', 'olive'])
plt.rcParams.update(mplParams)

### COMMON FILE NAME
COM = 'biased_inp'
QUAD = f'M_{N_QUAD}'
WTOL = f'wtol_{WIN_TOL:.0e}'
ZTOL = f'ztol_{ZER_TOL:.0e}'

for class_ in CLASSES:
    plt.figure(figsize=FIG_SIZE, dpi=DPI)
    plt.subplots_adjust(**SUBPLOT_PARAM)

    input_ = CLASS_INPUT_MAP[class_]
    CLS = f'cls_{class_}'
    INP = f'xy_{input_}'
    
    GAMMAs = [0.25, 0.4, 0.5, 0.6, 0.7, 0.8]
    # if class_ == '1' or class_ == '2a':
    #     GAMMAs += [0.75, 0.8]
    for gamma in GAMMAs:
        GAM = f'gam_{gamma*100:.0f}'
        DATA_FILE = f'{COM}-{CLS}-{INP}-{GAM}-{WTOL}-{ZTOL}-{QUAD}.csv'
        DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)
        data = np.genfromtxt(DATA_PATH, delimiter=",", skip_header = 3)
        data = data.transpose()[:2]
        
        line = '-'
        marker = 'o'
        label = r'$\displaystyle \gamma =$' + f'{gamma:g}'

        plt.plot(*data, label = label, linestyle=line, marker=marker)

    #lgd = plt.legend(ncol=2, bbox_to_anchor=(0.9, 1.2))
    plt.legend(loc='best')

    X_TITLE = r'$\displaystyle w_{exp}$'+' (winning probability)'
    if X_NORMALIZED:
        X_TITLE += ' (normalized with quantum bound)'
    plt.xlabel(X_TITLE, labelpad = 0)
    plt.ylabel(r"$\displaystyle H(A|BXYE')$")

    #SAVE_ARGS = {"bbox_extra_artists": (lgd,), "bbox_inches": 'tight'}
    SAVE_ARGS = {}

    if SAVE:
        TAIL = 'test'
        FORMAT = 'png'
        OUT_NAME = f'{COM}-{CLS}-{INP}-{WTOL}-{ZTOL}-{QUAD}'
        if TAIL:
            OUT_NAME += f'-{TAIL}'
        OUT_PATH = os.path.join(OUT_DIR, f'{OUT_NAME}.{FORMAT}')
        
        plt.savefig(OUT_PATH, **SAVE_ARGS)
    if SHOW:
        plt.show()