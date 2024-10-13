from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
import os, sys
import re
import itertools

### Add current directory to Python path
sys.path.append('.')
from ZPConstr2CHSH.DIRandomness import BEST_ZPC_CLASS, ZPC_BEST_INP_MAP
from common_func.plotting_helper import plot_settings

######### Plotting settings #########
FIG_SIZE = (16, 9)      # for aspect ratio 16:9
# FIG_SIZE = (12, 9)    # for aspect ratio 4:3
DPI = 200
SUBPLOT_PARAM = {'left': 0.12, 'right': 0.98, 'bottom': 0.11, 'top': 0.95, 'wspace': 0.28}

### Set font sizes and line width
titlesize = 30
ticksize = 28
legendsize = 26
linewidth = 3

### Set matplotlib plotting params
mplParams = plot_settings(title = titlesize, tick = ticksize,
                          legend = legendsize, linewidth = linewidth)
mplParams["xtick.major.pad"] = 10

#### Colors for different classes
plt.rcParams.update(mplParams)

#### Plot for max win prob
fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)


######### Protocol params #########
RAND_TYPE = 'loc'
EPSILON = 1e-8
BOUNDED_EPSCOM = True
EPSCOM_UB = 1e-6
WP_CHECK = 'lb'
N_NARROW = False

SETTINGs = {'chsh': [r'($\displaystyle w_{tol}=10^{-5}$)',
                     r'($\displaystyle w_{tol}=10^{-3}$)'],
            'zpclass': [r'($\displaystyle w_{tol}=10^{-5}, \eta_{z}=10^{-6}$)',
                        r'($\displaystyle w_{tol}=10^{-3}, \eta_{z}=10^{-3}$)']}

### Num of rounds
MAX_N = 1e15
N_SLICE = 200
if not N_NARROW:
    if RAND_TYPE == 'bli':
        MIN_N = 1e6
    elif RAND_TYPE in ('loc', 'glo'):
        MIN_N = 5e5
    N_POINT = re.sub('e\+0+|e\+', 'e', f'N_{MIN_N:.0e}_{MAX_N:.0e}_{N_SLICE}')

CLASSES = ['chsh', BEST_ZPC_CLASS[RAND_TYPE]]
LINES = itertools.cycle(('solid', 'dashed'))
MAX_R = 0

### Common setup in file names
QUAD = 'M_12'

for cls in CLASSES:
    ### Read param from file
    if RAND_TYPE == 'loc':
        HEAD = 'opr'
    elif RAND_TYPE =='glo':
        HEAD = 'tpr'
    else:
        HEAD = 'br'
    best_inp = ZPC_BEST_INP_MAP[RAND_TYPE][cls]
    if RAND_TYPE == 'loc':
        INP = f"x_{best_inp}"
    else:
        INP = f"xy_{best_inp[0]}{best_inp[1]}"
    PDG = 'pdgap_5e-05'
    NP = 'N_2'
    if WP_CHECK == 'lb':
        TAIL = 'wplb'
    else:
        TAIL = 'lpub'
    PARAM_FILE = f'{HEAD}-{cls}-{INP}-{PDG}-{QUAD}-{NP}-{TAIL}.csv'

    DATA_DIR = './thesis/data/best_setup/old'
    PARAM_FILE_PATH = os.path.join(DATA_DIR, PARAM_FILE)

    ### Load data
    data_mtf = np.genfromtxt(PARAM_FILE_PATH, delimiter=",", skip_header = 1)
    n_param = data_mtf.shape[0]

    for i in range(n_param):
        data_line = data_mtf[i]
        if cls != 'chsh':
            ztol = data_line[0]
            data_line = data_line[1:]
        else:
            ztol = 0

        wtol, _, _, wexp = data_line[:4]

        ### Num of rounds
        if N_NARROW:
            MIN_N = 1/2 * 100/(wtol**2)
            Ns = np.geomspace(MIN_N, MAX_N, num=N_SLICE, dtype=int)
            N_POINT = re.sub('e\+0+|e\+', 'e', f'N_{MIN_N:.0e}_{MAX_N:.0e}_{N_SLICE}')

        HEAD = f'fin-{RAND_TYPE}'
        WEXP = f'w_{wexp*10000:.0f}'.rstrip('0')
        ZTOL = f'ztol_{ztol:.0e}' if cls != 'chsh' else ''
        WTOL = f'wtol_{wtol:.0e}'
        EPS = f'eps_{EPSILON:.0e}'
        EPSCUB = f'epscub_{EPSCOM_UB:.0e}' if BOUNDED_EPSCOM else ''
        if N_NARROW:
            TAIL += '-narrow_n'
        file_name_elements = [HEAD, cls, WEXP, EPS, WTOL, ZTOL, QUAD, EPSCUB, N_POINT, TAIL]
        FILE_NAME = '-'.join([s for s in file_name_elements if s != ''])
        FINDATA_FILE = f'{FILE_NAME}.csv'
        FINDATA_DIR = './thesis/data/fin_rate'
        FINDATA_PATH = os.path.join(FINDATA_DIR, FINDATA_FILE)

        data = np.genfromtxt(FINDATA_PATH, delimiter=",", skip_header = 1)
        data = data.T
        Ns  = data[0]
        FRs = data[1]

        ### Colors for different zero tolerances
        if cls == 'chsh':
            color = 'grey'
            label = cls.upper()
            label += ' '+SETTINGs['chsh'][i]
        else:
            color = 'blue'
            label = cls
            label += ' '+SETTINGs['zpclass'][i]

        line = next(LINES)
        plt.plot(Ns, FRs, linestyle = line, color = color, label = label)
        MAX_R = max(MAX_R, np.max(FRs))

if round(MAX_R, 2) < round(MAX_R, 1):
    MAX_R = round(MAX_R, 1)
else:
    MAX_R = round(MAX_R, 1) + 0.1

if RAND_TYPE in ('bli', 'loc'):
    MAX_R = min(MAX_R, 1)

XLABEL = r'$\displaystyle n$'+' (number of rounds)'
YLABEL = r'$\displaystyle r$'+' (bit per round)'
plt.yticks(np.linspace(0, MAX_R, 5, endpoint=True))

plt.xscale("log")
plt.ylabel(YLABEL)
plt.xlabel(XLABEL)
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.LogLocator(numticks=99))
ax.xaxis.set_minor_locator(ticker.LogLocator(numticks=99, subs=(.2, .4, .6, .8)))

### Legends
lgd = plt.legend(prop={"weight":"bold"}, loc='best')

### Grids
plt.grid()

### Apply the graphic settings
plt.subplots_adjust(**SUBPLOT_PARAM)

### Save file
HEAD = f'fin-{RAND_TYPE}'
WEXP = f'qbound'
EPS = f'eps_{EPSILON:.0e}'
EPSCUB = f'epscub_{EPSCOM_UB:.0e}' if BOUNDED_EPSCOM else ''
TAIL = 'best_setup'
if WP_CHECK == 'lb':
    TAIL += '-wplb'
else:
    TAIL += '-lpub'
if N_NARROW:
    TAIL += '-narrow_n'
file_name_elements = [HEAD, WEXP, EPS, QUAD, EPSCUB, TAIL]
FILE_NAME = '-'.join([s for s in file_name_elements if s != ''])
FORMAT = 'png'
OUT_FILE = f'{FILE_NAME}.{FORMAT}'
OUT_DIR = './thesis/figures'
OUT_PATH = os.path.join(OUT_DIR, OUT_FILE)
SAVE_ARGS = {"format": FORMAT,
            "bbox_extra_artists": (lgd,),
            "bbox_inches": 'tight'}
plt.savefig(OUT_PATH, **SAVE_ARGS)

