from matplotlib import pyplot as plt
import numpy as np
import os, sys
import re

### Add current directory to Python path
sys.path.append('.')
from common_func.plotting_helper import *

### Data path
TYPE = 'blind'        # Type of randomness (one/two/blind)
DATA_DIR = './thesis/data/asym_rate/old/'
OUT_DIR = './thesis/figures'
SKIP_HEADER = 3

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
WTOL = 'wtol_1e-05'
ZTOL = 'ztol_1e-06'
PDG = 'pdgap_5e-05'
QUAD = 'M_12'
NP = 'N_20'
TAIL = 'wplb'

### Figure related
titlesize = 38
ticksize = 34
legendsize = 34
linewidth = 3

mplParams = plot_settings(title = titlesize, tick = ticksize,
                          legend = legendsize, linewidth = linewidth)
plt.rcParams.update(mplParams)

FIG_SIZE = (16, 9)
DPI = 200
SUBPLOT_PARAM = {'left': 0.11, 'right': 0.95, 'bottom': 0.12, 'top': 0.95}

plt.figure(figsize=FIG_SIZE, dpi=DPI)
plt.subplots_adjust(**SUBPLOT_PARAM)

for cls in CLASSES:
    if TYPE == 'one':
        INP = f'x_{CLASS_INPUT_MAP[cls]}'
    else:
        INP = f'xy_{CLASS_INPUT_MAP[cls]}'
    file = f'{COM}-{cls}-{INP}-{WTOL}-{ZTOL}-{PDG}-{QUAD}-{NP}-{TAIL}.csv'

    data = np.genfromtxt(os.path.join(DATA_DIR, file), delimiter=',', skip_header=SKIP_HEADER).T
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
    
    plt.plot(*data, label = label, color = color, linestyle = line, marker = marker)

lgd_param = {'loc': 'center right',
             'bbox_to_anchor': (1.22, 0.5),
             'handlelength': 1}

if TYPE == 'blind':
    plt.yticks(np.arange(0, 1.2, 0.2))

lgd = plt.legend(**lgd_param)
X_TITLE = r'$\displaystyle w_{exp}$'+' (winning probability)'

plt.xlabel(X_TITLE)
plt.ylabel(r"$\displaystyle h$")

plt.xticks(np.linspace(0.75, 0.85, 6, endpoint=True))
plt.grid()
OUT_COM = f'{COM}-asym'
FORMAT = 'png'
OUT_NAME = f'{OUT_COM}-{WTOL}-{ZTOL}-{QUAD}-{TAIL}'
OUT_FILE = f'{OUT_NAME}.{FORMAT}'
OUT_PATH = os.path.join(OUT_DIR, OUT_FILE)
SAVE_ARGS = {"bbox_extra_artists": (lgd,), "bbox_inches": 'tight'}

if SAVE:
    plt.savefig(OUT_PATH, **SAVE_ARGS)

if SHOW:
    plt.show()

