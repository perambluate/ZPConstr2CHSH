from matplotlib import pyplot as plt
import numpy as np
import os, sys
import re

### Add current directory to Python path
# (Note: this works when running the command in the dir. 'blindRandomness')
sys.path.append('.')
from common_func.plotting_helper import *

### Data path
TYPE = 'blind'                                            # Type of randomness (one/two/blind)
TOP_DIR = top_dir(TYPE)
if TYPE == 'blind':
    DATA_DIR = os.path.join(TOP_DIR, 'data/BFF21/asym_rate')
else:
    DATA_DIR = os.path.join(TOP_DIR, 'data/asym_rate')
OUT_DIR = os.path.join(TOP_DIR, './figures')

### General file setup
if TYPE == 'blind':
    COM = 'br'
elif TYPE =='one':
    COM = 'opr'
elif TYPE == 'two':
    COM = 'tpr'

CLASSES = ['chsh','1','2a','2b','2b_swap','2c','3a','3b']
if TYPE == 'two' and '2b_swap' in CLASSES:
    CLASSES.remove('2b_swap')

PUT_MARKER = False
SAVE = True     # To save file or not
SHOW = False    # To show figure or not

CLASS_INPUT_MAP = cls_inp_map(TYPE)
WTOL = 'wtol_2e-05'
ZTOL = 'ztol_1e-10'
QUAD = 'M_18'

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

plt.figure(figsize=FIG_SIZE, dpi=DPI)
plt.subplots_adjust(**SUBPLOT_PARAM)

for cls in CLASSES:
    if TYPE == 'one':
        INP = f'x_{CLASS_INPUT_MAP[cls]}'
    else:
        INP = f'xy_{CLASS_INPUT_MAP[cls]}'
    file = f'{COM}-{cls}-{INP}-{WTOL}-{ZTOL}-{QUAD}.csv'

    data = np.genfromtxt(os.path.join(DATA_DIR, file), delimiter=',', skip_header=3).T
    data = data[:2,:]
    label = 'CHSH'
    if cls != 'chsh':
        class_name = cls.replace("_swap", "\\textsubscript{swap}")
        label = r'{}'.format(class_name)
    
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

    marker = '' if not PUT_MARKER else 'x'
    
    plt.plot(*data, label = label, color=color, linestyle=line, marker=marker)

# lgd = plt.legend(bbox_to_anchor=(1.02, 1))
lgd_param = {'loc': 'best'}
if TYPE == 'blind':
    lgd_param['loc'] = 'upper left'
    plt.yticks(np.arange(0, 1.2, 0.2))
plt.legend(**lgd_param)
X_TITLE = r'$\displaystyle w_{exp}$' #+' (winning probability)'

plt.xlabel(X_TITLE)
if TYPE == 'blind':
    plt.ylabel(r"$\displaystyle H(A|XYBE')$")
elif TYPE =='one':
    plt.ylabel(r"$\displaystyle H(A|XYE')$")
elif TYPE == 'two':
    plt.ylabel(r"$\displaystyle H(AB|XYE')$")

plt.grid()
OUT_COM = f'{COM}-asym'
TAIL = '1'
FORMAT = 'png'
OUT_NAME = f'{OUT_COM}-{WTOL}-{ZTOL}-{QUAD}'
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
