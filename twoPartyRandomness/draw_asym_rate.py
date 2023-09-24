from matplotlib import pyplot as plt
from matplotlib import rc, rcParams
import numpy as np
import os, sys
import re

### Add current directory to Python path
# (Note: this works when running the command in the dir. 'blindRandomness')
sys.path.append('..')
from common_func.plotting_helper import *

TOP = './'
DATA_DIR = os.path.join(TOP,'data/asymp_rate')
X_NORMALIZED = False
# ZERO_CLASS = ['chsh','1','2a','2b','2c','3a','3b']
ZERO_CLASS = ['chsh','1','2a','3b']
DATA_COM = 'tpr'

OUT_DIR = './figures'
SORT = False
PUT_MARKER = False
SAVE = True     # To save file or not
SHOW = False    # To show figure or not

### Figure related
titlesize = 32
ticksize = 30
legendsize = 28
linewidth = 3

mplParams = plot_settings(title = titlesize, tick = ticksize, legend = legendsize, linewidth = linewidth)
plt.rcParams.update(mplParams)

FIG_SIZE = (12, 9)
DPI = 100
SUBPLOT_PARAM = {'left': 0.11, 'right': 0.95, 'bottom': 0.105, 'top': 0.95}

def sort_data(data):
    order = np.argsort(data[0])
    sorted_data = [d[order] for d in data]
    return sorted_data

plt.figure(figsize=FIG_SIZE, dpi=DPI)
plt.subplots_adjust(**SUBPLOT_PARAM)

### Average/max over different inputs
for cls in ZERO_CLASS:
    if cls == 'chsh':
        file_list = [f'{DATA_COM}-{cls}-xy_{x}{y}-M_12-wtol_1e-04-ztol_1e-09.csv' \
                     for x in range(2) for y in range(2)]
    else:
        file_list = [f'{DATA_COM}-{cls}-xy_{x}{y}-M_12-wtol_1e-04-ztol_1e-09.csv' \
                     for x in range(2) for y in range(2)]

    data_list = []
    for i in range(len(file_list)):
        file_ = file_list[i]
        data = np.genfromtxt(os.path.join(DATA_DIR, file_), delimiter=',', skip_header=3).T
        data = data[:2,:]
        if SORT:
            data = sort_data(data)
        data_list.append(data)
        
    ### Maximize
    max_input = np.argmax(np.array(data_list)[:,1][:,0])
    data = data_list[max_input]
    # label = r'{} $\displaystyle x^*y^*={:0>2b}$'.format(class_name, max_input)
    label = 'CHSH'
   
    color='gray'
    if '1' in cls:
        color = 'blue'
        label += '+ 1 zero'
    elif '2' in cls:
        color = 'forestgreen'
        label += '+ 2 zeros'
    elif '3' in cls:
        color = 'firebrick'
        label += '+ 3 zeros'
    line = 'solid'
    # if re.match("[2-3]b", cls):
    #     line = 'dashed'
    # elif re.match("[2-3]c", cls):
    #     line = 'dashdot'

    marker = '' if not PUT_MARKER else 'x'
    
    plt.plot(*data, label = label, color=color, linestyle=line, marker=marker)

# lgd = plt.legend(bbox_to_anchor=(1.02, 1))
plt.legend(loc='best')
X_TITLE = r'$\displaystyle w_{exp}$' #+' (winning probability)'
if X_NORMALIZED:
    X_TITLE += ' (normalized with quantum bound)'
plt.xlabel(X_TITLE)
plt.ylabel(r"$\displaystyle H(AB|XYE')$")
OUT_COM = 'global-asymp'
TAIL = ''
OUT_NAME = f'{OUT_COM}-best_cls_zero_num'
FORMAT = 'png'
if TAIL:
    OUT_NAME = f'{OUT_NAME}-{TAIL}'
OUT_FILE = f'{OUT_NAME}.{FORMAT}'
OUT_PATH = os.path.join(OUT_DIR, OUT_FILE)
# SAVE_ARGS = {"bbox_extra_artists": (lgd,), "bbox_inches": 'tight'}
SAVE_ARGS = {}

if SAVE:
    plt.savefig(OUT_PATH, **SAVE_ARGS)

if SHOW:
    plt.show()
