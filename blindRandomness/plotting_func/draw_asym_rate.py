"""
    A script to plot the asymptotic rate of blind randomness in the following scenario:
        (i) sequential scenario: FM18(PRA 97, 032324 (2018))
                                 BBS20(Quantum 4, 344 (2020))
        (ii) Brown-Fawzi-Fawzi: BFF21(arXiv:2106.13692v2)

    Different code for NPA hierachy:
        (i) Pei-Sheng's Matlab code: scenario name with 'PS'
        (ii) Peter Wittek's Python code

    Different choice of inputs for optimization (specify the Bell function with different p(x,y)):
        (i)  opt_avg: plot H(A|BXYE) optimized over averaged inputs
        (ii)  max_in: plot H(A|BXYE) with optimal fixed inputs (the one gives highest rate)
        (iii) avg_in: plot averaged H(A|BXYE) for all inputs with even weight
        (iv)  min_in: plot H(A|BXYE) with worst fixed inputs (the one gives lowest rate)
"""
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import os, sys
import re

### Add current directory to Python path
# (Note: this works when running the command in the dir. 'blindRandomness')
sys.path.append('.')
from common_func.plotting_helper import *

### Choose 'opt_avg', or 'max_in', or 'min_in', or 'avg_in'
OPT = 'max_in'
### To save file or not
SAVE = True
### To show figure or not
SHOW = False
### Set true to normalize winning probability in the superclassical-quantum range for all classes
X_NORMALIZED = False
### Sort rate of randomness for non-ordered data
SORT = False
### Turn the guessing probability to min-entropy
TAKE_LOG = False
### NPA hierachy level
LEVEL = 2
### Whether to make the plot small
SMALL = False
### Wheter to display all classes in the plot (only display standard CHSH and class 1, 2c, and 3b if false)
ALL_CLS = True

### Top directory where data saved
COMMON_TOP = '/home/chunyo/Documents'
MATLAB_TOP = os.path.join(COMMON_TOP, 'MATLAB/LocalRandomness')
PYTHON_TOP = os.path.join(COMMON_TOP, 'python/ZPConstr2CHSH/blindRandomness')
### Directory to save figures
OUT_DIR = './figures'

### Scenario: FM18, BBS20, BFF21
SCENARIO = ['BFF21']
### Class: 1, 2a, 2b, 2b_swap, 2c, 3a, 3b (standard CHSH is added automatically)
if ALL_CLS:
    CLASSES = ['CHSH','1','2a','2b','2b_swap','2c','3a','3b']
    #CLASSES = ['1','2a','2b','2c','3a','3b']
else:
    # CLASSES = ['1','2c','3b']
    CLASSES = ['CHSH','1','3b']

### Tolerance error for zero-probability constraints
#ERRORS = ['1e-05', '1e-04', '1e-03', '1e-02', '1e-01']

######### Plotting settings #########
if SMALL:
    FIG_SIZE = (2.6, 2.8)
else:
    FIG_SIZE = (12, 9)
DPI = 200
if SMALL:
    SUBPLOT_PARAM = {'left': 0.15, 'right': 0.95, 'bottom': 0.18, 'top': 0.95}
else:
    SUBPLOT_PARAM = {'left': 0.1, 'right': 0.95, 'bottom': 0.095, 'top': 0.95}

### Default font size
if SMALL:
    titlesize = 14
    ticksize = 12
    legendsize = ticksize
    linewidth = 1
else:
    titlesize = 30
    ticksize = 28
    legendsize = 28
    linewidth = 3

### Set matplotlib plotting params
mplParams = plot_settings(title = titlesize, tick = ticksize, legend = legendsize, linewidth = linewidth)

#### Line colors
if ALL_CLS:
    mplParams["axes.prop_cycle"] = \
        mpl.cycler(color = ['b','green','red','c','m', 'olive', 'purple', 'gray'])
else:
    mplParams["axes.prop_cycle"] = \
                      mpl.cycler(color = ['blue','forestgreen','firebrick','gray'])

plt.rcParams.update(mplParams)

def sort_data(data):
    order = np.argsort(data[0])
    sorted_data = [d[order] for d in data]
    return sorted_data

for scenario in SCENARIO:
    plt.figure(figsize=FIG_SIZE, dpi=DPI)
    plt.subplots_adjust(**SUBPLOT_PARAM)
    
    TAKE_LOG = bool(scenario != 'BFF21')

    if 'PS' in scenario:
        DATA_DIR = os.path.join(MATLAB_TOP,'Data',scenario.rstrip('-PS'))
    else:
        DATA_DIR = os.path.join(PYTHON_TOP,'data',scenario)

    DATA_COMMON = 'local_randomness'
    
    for cls in CLASSES:
        data_list = []
        class_name = f'class {cls}' if cls != 'CHSH' else 'CHSH'
        
        if OPT == 'opt_avg':
            if cls == 'CHSH':
                file_ = f'{DATA_COMMON}-CHSH-avg_in-{scenario}.csv'
            else:
                file_ = f'{DATA_COMMON}-class_{cls}-avg_in-{scenario}.csv'
            data = np.genfromtxt(os.path.join(DATA_DIR, file_), delimiter=',').T
            if SORT:
                data = sort_data(data)
            if TAKE_LOG:
                data = [data[0], -np.log2(data[1])]
            label = f'{class_name}'
            
        else:
            file_list = []
            if cls == 'CHSH':
                file_list += [f'{DATA_COMMON}-CHSH-xy_{x}{y}-M_12-dev_1e-05-{scenario}.csv' \
                            for x in range(2) for y in range(2)]
            else:
                file_list += [f'{DATA_COMMON}-class_{cls}-xy_{x}{y}-M_12-dev_1e-05-{scenario}.csv' \
                                for x in range(2) for y in range(2)]
            for file_ in file_list:
                data = np.genfromtxt(os.path.join(DATA_DIR, file_), delimiter=',').T
                if SORT:
                    data = sort_data(data)
                if TAKE_LOG:
                    data = [data[0], -np.log2(data[1])]
                data_list.append(data)

            ### Average
            if OPT == 'avg_in':
                data = np.average(data_list, axis=0)
                label = f'{class_name}'
            
            ### Maximize
            elif OPT == 'max_in':
                max_input = np.argmax(np.array(data_list)[:,1][:,0])
                data = data_list[max_input]
                class_name = f'class\ {cls}' if cls != 'CHSH' else 'CHSH'
                # class_name = class_name.replace("swap", "{swap}")
                class_name = class_name.replace("_swap", "\\textsubscript{swap}")
                label = r'{} $\displaystyle x^*y^*={:0>2b}$'.format(class_name, max_input)

            ### Minimize
            elif OPT == 'min_in':
                min_input = np.argmin(np.array(data_list)[:,1][:,0])
                data = data_list[min_input]
                label = f'{class_name} xy={min_input:0>2b}' # ({scenario})'

        color='gray'
        if '1' in cls:
            color = 'blue'
        elif '2' in cls:
            color = 'forestgreen'
        elif '3' in cls:
            color = 'firebrick'
        line = 'solid'
        if ALL_CLS:
            if re.match("[2-3]b", cls):
                line = 'dashed'
                if 'swap' in cls:
                    line = 'dotted'
            elif re.match("[2-3]c", cls):
                line = 'dashdot'

        marker = ''
        if cls == '2b':
            marker = '+'
        
        plt.plot(*data, label = label, color=color, linestyle=line, marker=marker, markersize=12)

    if SMALL:
        plt.locator_params(axis='y', nbins=5)
        plt.locator_params(axis='x', nbins=3)
    else:
        #lgd = plt.legend(ncol=2, bbox_to_anchor=(0.9, 1.2))
        lgd = plt.legend(bbox_to_anchor=(1, 1))
        # plt.legend(loc='best')
    
    if SMALL:
        plt.xlabel(r'$\displaystyle w_{exp}$', labelpad = 0)
    else:
        X_TITLE = r'$\displaystyle w_{exp}$'+' (winning probability)'
        if X_NORMALIZED:
            X_TITLE += ' (normalized with quantum bound)'
        plt.xlabel(X_TITLE, labelpad = 0)
        plt.ylabel(r"$\displaystyle H(A|BXYE')$")

    if not SMALL:
        SAVE_ARGS = {"bbox_extra_artists": (lgd,), "bbox_inches": 'tight'}
    else:
        SAVE_ARGS = {"transparent": True}

    if SAVE:
        COM = 'br-asymp'
        if ALL_CLS:
            COM += '-all_cls'
        else:
            COM += f'-{len(CLASSES)}cls'
        TAIL = 'test'
        FORMAT = 'png'
        OUT_NAME = f'{COM}-{OPT}-{scenario}'
        if TAIL:
            OUT_NAME += f'-{TAIL}'
        OUT_PATH = os.path.join(OUT_DIR, f'{OUT_NAME}.{FORMAT}')
        plt.savefig(OUT_PATH, **SAVE_ARGS)
    
    if SHOW:
        plt.show()        
