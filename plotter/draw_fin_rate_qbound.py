"""
    This script generate a plot of finite rates of different types of randomness
    for each classes at quantum bound.
      - Type of randomness
          blind: blind randomness (asymptotic rate: H(A|BXYE))
          one: one-party randomness (asymptotic rate: H(A|XYE))
          two: two-party randomness (asymptotic rate: H(AB|XYE))
      - Notations (case insensitive)
          n     : number of rounds
          w_exp : expected winning probability
          wtol  : tolerance of winning probability
          ztol  : zero tolerance
          quad  : number of quadratures
          gamma : ratio of testing rounds to whole rounds
          class : class of specific correlation defined by zero-probability constraints
          beta, nu_prime : parameters to optimize for the correction terms
      - Directory and file names
          Note the directories where the data save maybe different,
            please change <TOP_DIR> and <DATA_DIR> according to the path you save.
          We assume the data files are named in the following formate:
            <HEAD>-<CLASS>-xy_<INP>-M_<N_QUAD>-wtol_<WTOL>-ztol_<ZTOL>.csv
          Choose the path <OUT_DIR> to where you want to output the figrues.
          The output figures are saved with the following naming formate:
            <COM>(-<CLASS>)(-<WEXP>)-<EPS>-<WTOL>-<GAM>-<QUAD>-<TAIL>.png
          (Note: The two parts in the parentheses are sometimes omited depending on
                the content of the figure.)
"""

from matplotlib import pyplot as plt, ticker as mticker
from functools import partial
from joblib import Parallel, delayed
import numpy as np
import math
import os, sys
import re

### Add current directory to Python path
sys.path.append('.')
from common_func.plotting_helper import *

PRINT_DATA = False  # To print values of data
SAVE = False         # To save figure or not
SHOW = True        # To show figure or not
SAVECSV = False     # To save data or not
DRAW_FROM_SAVED_DATA = False     # Plot the line with previous data if true
TYPE = 'blind'        # Type of randomness (one/two/blind)

### Change the GUI backend to get rid of the multi-threading runtime error with tkinter
if not SHOW:
    plt.switch_backend('agg')

### Data paths
TOP_DIR = top_dir(TYPE)
DATA_DIR = os.path.join(TOP_DIR, 'data')
OUTCSV_DIR = os.path.join(DATA_DIR, 'fin_rate')
if TYPE == 'blind':
    DATA_DIR = os.path.join(DATA_DIR, 'BFF21/w_max_77')
elif TYPE in ['one', 'two']:
    DATA_DIR = os.path.join(DATA_DIR, 'asymp_rate')
CLASS_INPUT_MAP = cls_inp_map(TYPE)
CLASS_MAX_WIN = cls_max_win_map()

### Parallel settings
N_JOB = 8

### Constant parameters
CHSH_W_Q = (2 + math.sqrt(2))/4    # CHSH win prob
EPSILON = 1e-12                     # Smoothness of smooth min entropy (related to secrecy)
WIN_TOL = 1e-4                      # Tolerant error for win prob
GAMMA = 1e-2                        # Testing ratio
INP_DIST = [1/4]*4

### General File Name Settings
OUT_DIR = os.path.join(TOP_DIR, 'figures/')
if TYPE == 'blind':
    HEAD = 'br'
elif TYPE =='one':
    HEAD = 'opr'
elif TYPE == 'two':
    HEAD = 'tpr'
EPS = f'eps_{EPSILON:.0e}'
WTOL = f'wtol_{WIN_TOL:.0e}'
QUAD = 'M_12'
# GAM = f'gam_{GAMMA:.0e}'
CSV_COM = 'opt_gamma'

######### Plotting settings #########
# FIG_SIZE = (16, 9)      # for aspect ratio 16:9
FIG_SIZE = (12, 9)    # for aspect ratio 4:3
DPI = 200
SUBPLOT_PARAM = {'left': 0.085, 'right': 0.98, 'bottom': 0.11, 'top': 0.95, 'wspace': 0.28}
if TYPE == 'two':
    SUBPLOT_PARAM['left'] = 0.1

### Set font sizes and line width
titlesize = 30
ticksize = 28
legendsize = 26
linewidth = 3

### Set matplotlib plotting params
mplParams = plot_settings(title = titlesize, tick = ticksize, legend = legendsize, linewidth = linewidth)
mplParams["xtick.major.pad"] = 10

#### Colors for different classes
plt.rcParams.update(mplParams)

### Num of rounds
if TYPE == 'blind':
    N_RANGE = (1e7, 1e14)
else:
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

### All classes to plot
# CLASSES = ['chsh','1','2a','2b','2b_swap','2c','3a','3b']
CLASSES = ['chsh','1','2c', '3b']
if TYPE == 'two':
    CLASSES.remove('2b_swap')
ZERO_TOLs = [1e-9]

################################ Iter for different parameters ################################
### Run over all classes in CLASSES
for cls in CLASSES:
    inp = CLASS_INPUT_MAP[cls]

    CLS = cls
    if TYPE == 'one':
        INP = f'x_{inp}'
    else:
        INP = f'xy_{inp}'

    ### Run over all zero tolerances in ZERO_TOLs
    for zero_tol in ZERO_TOLs:
        if DRAW_FROM_SAVED_DATA:
            WEXP = f'w_{CLASS_MAX_WIN[cls]*10000:.0f}'.rstrip('0')
            ZTOL = f'ztol_{zero_tol:.0e}'
            DATAFILE = f'{CSV_COM}-{CLS}-{WEXP}-{EPS}-{WTOL}-{ZTOL}-{QUAD}.csv'
            DATAPATH = os.path.join(OUTCSV_DIR, DATAFILE)
            data = np.genfromtxt(DATAPATH, delimiter=",", skip_header = 1).T
            Ns = data[0]
            FRs = data[1]
        else:
            ZTOL = f'ztol_{zero_tol:.0e}'
            DATAFILE = f'{HEAD}-{CLS}-{INP}-{QUAD}-{WTOL}-{ZTOL}.csv'
            DATAPATH = os.path.join(DATA_DIR, DATAFILE)

            ### Get the maximum winnin probability
            with open(DATAPATH) as file:
                max_p_win = float(file.readlines()[1])

            ### Load data
            data_mtf = np.genfromtxt(DATAPATH, delimiter=",", skip_header = 3)
            # print(data_mtf)

            ### Choose min-tradeoff function with expected winning prob
            if len(data_mtf.shape) == 1:
                data_mtf = np.array([data_mtf])

            #### For different w_exp (or other parameters) in the data
            win_prob, asym_rate, lambda_ = data_mtf[0][:3]
            c_lambda = data_mtf[0][-1]
            if cls != 'chsh':
                lambda_zeros = data_mtf[0][3:-1]
                c_lambda -= sum(lambda_zeros)*zero_tol

    ##################### Compute key rate with optimal parameters #####################
            ### Construct key rate function with fixed param (only leave n, beta tunable)
            fin_rate_func = partial(fin_rate_testing, asym_rate = asym_rate,
                                    lambda_ = lambda_, c_lambda = c_lambda,
                                    zero_tol = zero_tol, zero_class=cls, max_p_win = max_p_win)

            fr_gammas = Parallel(n_jobs=N_JOB, verbose = 0)(
                        delayed(opt_with_gamma)(N, beta_arr = BETAs, nup_arr = NU_PRIMEs, gam_arr = GAMMAs,
                                                inp_dist = INP_DIST, fin_rate_func = fin_rate_func) for N in Ns)
            fr_gammas = list(zip(*fr_gammas))
            FRs = np.array(fr_gammas[0])
            optGAMMAs = np.array(fr_gammas[1])

            if PRINT_DATA:
                print(np.column_stack((Ns, FRs)))

            if SAVECSV:
                data2save = np.column_stack((Ns, FRs, optGAMMAs))
                HEADER = 'rounds, rate, gamma'
                WEXP = f'w_{win_prob*10000:.0f}'.rstrip('0')
                ZTOL = f'ztol_{zero_tol:.0e}'
                N_POINT = f'N_{N_SLICE}'
                OUTCSV = f'{CSV_COM}-{CLS}-{WEXP}-{EPS}-{WTOL}-{ZTOL}-{QUAD}-{N_POINT}.csv'
                OUTCSV_PATH = os.path.join(OUTCSV_DIR, OUTCSV)
                np.savetxt(OUTCSV_PATH, data2save, fmt='%.5g', delimiter=',', header=HEADER)

    ##################################### Draw line ##################################### 
        class_name = f'class {cls}' if cls != 'chsh' else 'CHSH'
        class_name = class_name.replace("_swap", "\\textsubscript{swap}")
        if TYPE == 'one':
            label = r'{} $\displaystyle x^*={}$'.format(class_name, inp)
        else:
            label = r'{} $\displaystyle x^*y^*={}$'.format(class_name, inp)
    
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
        
        plt.plot(Ns, FRs, label=label, linestyle=line, color=color)


################################ Save figure ################################
YLABEL = r'$\displaystyle r$'+' (bit per round)'
XLABEL = r'$\displaystyle n$'+' (number of rounds)'    

plt.ylabel(YLABEL)
plt.xlabel(XLABEL)
plt.xscale("log")
ax = plt.gca()
ax.xaxis.set_major_locator(mticker.LogLocator(numticks=5))
ax.xaxis.set_minor_locator(mticker.LogLocator(numticks=4, subs="auto"))
plt.legend(prop={"weight":"bold"})

### Apply the graphic settings
plt.subplots_adjust(**SUBPLOT_PARAM)

### Save file
if SAVE:
    COM = f'fin_{HEAD}-inp_consump-qbound'
    TAIL = 'test'
    FORMAT = 'png'
    # OUT_NAME = f'{COM}-{EPS}-{WTOL}-{GAM}-{QUAD}'
    OUT_NAME = f'{COM}-{EPS}-{WTOL}-{QUAD}'
    if TAIL:
        OUT_NAME += f'-{TAIL}'
    OUT_PATH = os.path.join(OUT_DIR, f'{OUT_NAME}.{FORMAT}')
    plt.savefig(OUT_PATH, format = FORMAT)
if SHOW:
    plt.show()
