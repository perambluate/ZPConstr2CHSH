import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import itertools
import os
import re

TOP = './'
DATA_DIR = os.path.join(TOP,'data')
X_NORMALIZED = False
ZERO_CLASSes = ['CHSH','1','2a','2b','2b_swap','2c','3a','3b']
#ERRORS = ['1e-05', '1e-04', '1e-03', '1e-02', '1e-01']
DATA_COMMON = 'diqkd_bff21'

OUT_DIR = './figures'
SORT = False
PUT_MARKER = False
SAVE = True             # Show without save if not true; otherwise save the fig directly.

### Figure related
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.sans-serif'] = ['Latin Modern Roman']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color = \
                                    ['b','green','red','c','m', 'yellowgreen', 'gray', 'purple'])
MARKERS = itertools.cycle(('o','+','v','x','d'))
FIG_SIZE = (12, 9)
DPI = 100
PLOT_OPTION = {"linewidth": 3}
SUBPLOT_PARAM = {'left': 0.1, 'right': 0.95, 'bottom': 0.095, 'top': 0.95}

def sort_data(data):
    order = np.argsort(data[0])
    sorted_data = [d[order] for d in data]
    return sorted_data

plt.figure(figsize=FIG_SIZE, dpi=DPI)
plt.subplots_adjust(**SUBPLOT_PARAM)

### Average/max over different inputs
for cls in ZERO_CLASSes:
    if cls == 'CHSH':
        file_list = [f'{DATA_COMMON}-CHSH-x_{x}-M_12-wtol_1e-04-ztol_1e-09.csv' for x in range(2)]
    else:
        file_list = [f'{DATA_COMMON}-class_{cls}-x_{x}-M_12-wtol_1e-04-ztol_1e-09.csv' \
                     for x in range(2)]
    #plt.figure(figsize=FIG_SIZE, dpi=DPI)
    #plt.subplots_adjust(**SUBPLOT_PARAM)

    class_name = f'class {cls}' if cls != 'CHSH' else 'CHSH'
    data_list = []
    for i in range(len(file_list)):
        file_ = file_list[i]
        data = np.genfromtxt(os.path.join(DATA_DIR, file_), delimiter=',', skip_header=3).T
        data = data[:2,:]
        if SORT:
            data = sort_data(data)
        data_list.append(data)
    
    ### Average
    #data = np.average(data_list, axis=0)
    #label = f'{class_name}'
    
    ### Maximize
    max_input = np.argmax(np.array(data_list)[:,1][:,0])
    data = data_list[max_input]
    label = f'{class_name} x={max_input}'
    
    #label = f'x={i}'
    color='gray'
    if '1' in cls:
        color = 'blue'
    elif '2' in cls:
        color = 'forestgreen'
    elif '3' in cls:
        color = 'firebrick'
    line = 'solid'
    if re.match("[2-3]b", cls):
        line = 'dashed'
        if 'swap' in cls:
            line = 'dotted'
    elif re.match("[2-3]c", cls):
        line = 'dashdot'

    marker = '' if not PUT_MARKER else next(MARKERS)
    
    plt.plot(*data, label = label, color=color, linestyle=line, marker=marker, **PLOT_OPTION)

lgd = plt.legend(fontsize=22, bbox_to_anchor=(1.02, 1))
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
X_TITLE = 'CHSH winning probability'
if X_NORMALIZED:
    X_TITLE += ' (normalized with quantum bound)'
plt.xlabel(X_TITLE, fontsize=24)
plt.ylabel('randomness (bit)', fontsize=24)
#CLASS = f'class_{cls}' if cls != 'CHSH' else 'CHSH'
OUT_COMMON = 'one_party_randomness'
TAIL = '0'
FORMAT = 'png'
OUT_NAME = f'{OUT_COMMON}-all_cls'
if TAIL:
    OUT_NAME = f'{OUT_NAME}-{TAIL}'
OUT_FILE = f'{OUT_NAME}.{FORMAT}'
OUT_PATH = os.path.join(OUT_DIR, OUT_FILE)
SAVE_ARGS = {"bbox_extra_artists": (lgd,), "bbox_inches": 'tight'}

if not SAVE:
    plt.show()
elif os.path.exists(OUT_PATH):
    ans = input("File exists, do u want to replace it? [Y/y]")
    if ans == 'y' or ans == 'Y':
        plt.savefig(OUT_PATH, **SAVE_ARGS)
    else:
        print(f'Current file name is "{OUT_FILE}"')
        print('Change the file name to save.')
else:
    plt.savefig(OUT_PATH, **SAVE_ARGS)

