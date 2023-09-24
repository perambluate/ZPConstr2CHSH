"""
    This script generate a plot of finite rates for each classes at quantum bound.
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
            lr_bff21-<class>-xy_<inputs>-M_<num_quad>-wtol_<win_tol>-ztol_<zero_tol>.csv
        Choose the path <OUT_DIR> to where you want to output the figrues.
        The output figures are saved with the following naming formate:
            <COM>(-<CLASS>)(-<WIN_EXP>)-<EPS>-<WTOL>-<GAM>-<QUAD>-<TAIL>.png
        (Note: The two parts in the parentheses are sometimes omited depending on the content of the figure.)
"""

from matplotlib import pyplot as plt
from cycler import cycler
from functools import partial
from joblib import Parallel, delayed
import numpy as np
import math
import os, sys

### Add current directory to Python path
# (Note: this works when running the command in the dir. 'blindRandomness')
sys.path.append('..')
from common_func.plotting_helper import *

### An option to print values of data
PRINT_DATA = False
### An option to the data into a csv file
SAVECSV = False
### To save file or not
SAVE = True
### To show figure or not
SHOW = True

### Change the GUI backend to get rid of the multi-threading runtime error with tkinter
if not SHOW:
    plt.switch_backend('agg')

### Data paths
TOP_DIR = './'
DATA_DIR = os.path.join(TOP_DIR, 'data/BFF21/w_max_77')
OUTCSV_DIR = os.path.join(TOP_DIR, './data/BFF21/fin_rate')
OUT_DIR = os.path.join(TOP_DIR, 'figures/BFF21/fin_rate')

CLASS_INPUT_MAP = cls_inp_map('blind')

######### Plotting settings #########
# FIG_SIZE = (16, 9)      # for aspect ratio 16:9
FIG_SIZE = (12, 9)    # for aspect ratio 4:3
DPI = 200
SUBPLOT_PARAM = {'left': 0.085, 'right': 0.98, 'bottom': 0.11, 'top': 0.95, 'wspace': 0.28}

### Set font sizes and line width
titlesize = 30
ticksize = 28
legendsize = 28
linewidth = 3

### Set matplotlib plotting params
mplParams = plot_settings(title = titlesize, tick = ticksize, legend = legendsize, linewidth = linewidth)
mplParams["xtick.major.pad"] = 10

#### Colors for different classes
COLORS = ['blue','forestgreen','firebrick','gray']
# COLORS = ['blue','green','red','cyan','magenta', 'olive', 'gray', 'purple']
mplParams["axes.prop_cycle"] = cycler(color=COLORS)
plt.rcParams.update(mplParams)

### Parallel settings
N_JOB = 8

### Constant parameters
CHSH_W_Q = (2 + math.sqrt(2))/4    # CHSH win prob
EPSILON = 1e-12                     # Smoothness of smooth min entropy (related to secrecy)
WIN_TOL = 1e-4                      # Tolerant error for win prob
GAMMA = 1e-2                        # Testing ratio
INP_DIST = [1/4]*4

### General File Name Settings
EPS = f'eps_{EPSILON:.0e}'
WTOL = f'wtol_{WIN_TOL:.0e}'
# GAM = f'gam_{GAMMA:.0e}'
QUAD = 'M_12'

### Num of rounds
N_RANGE = (1e7, 1e14)
N_SLICE = 200
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
CLASSES = ['chsh','1','2c','3b']
ZERO_TOLs = [1e-9]

################################ Iter for different parameters ################################
### Run over all classes in CLASSES
for cls in CLASSES:
    input_ = CLASS_INPUT_MAP[cls]

    ### Specify the file record the data
    HEAD = 'br'
    CLS = cls
    INPUT = f'xy_{input_}'
    # WTOL = f'wtol_{WIN_TOL:.0e}'

    ### Run over all zero tolerances in ZERO_TOLs
    for zero_tol in ZERO_TOLs:
        ZTOL = f'ztol_{zero_tol:.0e}'
        DATA_FILE = f'{HEAD}-{CLS}-{INPUT}-{QUAD}-{WTOL}-{ZTOL}.csv'
        DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)

        ### Get the maximum winnin probability
        with open(DATA_PATH) as file:
            max_p_win = float(file.readlines()[1])

        ### Load data
        data_mtf = np.genfromtxt(DATA_PATH, delimiter=",", skip_header = 3)
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
        fr_func = partial(fin_rate_testing, asym_rate = asym_rate,
                            lambda_ = lambda_, c_lambda = c_lambda,
                            zero_tol = zero_tol, zero_class=cls, max_p_win = max_p_win)
        
        FRs = Parallel(n_jobs=N_JOB, verbose = 0)(
          delayed(opt_all)(n, beta_arr = BETAs, nup_arr = NU_PRIMEs, gam_arr = GAMMAs,
                           inp_dist = INP_DIST, fin_rate_func = fr_func) for n in Ns)
        FRs = np.array(FRs)
        
##################################### Draw line ##################################### 
        label = f'class {cls}' if cls != 'chsh' else 'CHSH'
        #### Single plot for max win prob
        label += r' $\displaystyle x^*y^*$'+f'={input_}' + \
                    r' $\displaystyle w_{exp}$'+f'={win_prob:.4f}'

        if PRINT_DATA:
            print(np.column_stack((Ns, FRs)))

        if SAVECSV:
            data2save = np.column_stack((Ns, FRs))
            HEADER = 'num_of_rounds_in_log, rate'
            WEXP = f'w_{win_prob*10000:.0f}'.rstrip('0')
            ZTOL = f'ztol_{zero_tol:.0e}'
            # OUTCSV = f'{CLS}-{WEXP}-{EPS}-{WTOL}-{ZTOL}-{GAM}-{QUAD}.csv'
            CSV_COM = 'fin_br'
            OUTCSV = f'{CSV_COM}-{CLS}-{WEXP}-{EPS}-{WTOL}-{ZTOL}-{QUAD}.csv'
            OUTCSV_PATH = os.join.path(OUTCSV_DIR, OUTCSV)
            np.savetxt(OUTCSV_PATH, data2save, fmt='%.5g', delimiter=',', header=HEADER)
    
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
if SAVE:
    COM = 'fin_br-inp_consump-qbound'
    TAIL = 'corrected_mtf'
    FORMAT = 'png'
    OUT_NAME = f'{COM}-{EPS}-{WTOL}-{QUAD}'
    if TAIL:
        OUT_NAME += f'-{TAIL}'
    OUT_PATH = os.path.join(OUT_DIR, f'{OUT_NAME}.{FORMAT}')
    plt.savefig(OUT_PATH, format = FORMAT)
if SHOW:
    plt.show()
