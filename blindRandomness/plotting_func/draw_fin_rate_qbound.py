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
            <COMMON>(-<CLASS>)(-<WIN_EXP>)-<EPS>-<WTOL>-<GAM>-<QUAD>-<TAIL>.png
        (Note: The two parts in the parentheses are sometimes omited depending on the content of the figure.)
"""

from matplotlib import pyplot as plt
from matplotlib import cycler
from functools import partial
from joblib import Parallel, delayed
import numpy as np
import math
import os, sys

### Add current directory to Python path
# (Note: this works when running the command in the dir. 'blindRandomness')
sys.path.append('.')
from common_func.plotting_helper import *

### An option to print values of data
PRINT_DATA = False
### An option to the data into a csv file
SAVE_CSV = False
### To save file or not
SAVE = True
### To show figure or not
SHOW = False

### Change the GUI backend to get rid of the multi-threading runtime error with tkinter
if not SHOW:
    plt.switch_backend('agg')

### Data paths
TOP_DIR = './'
DATA_DIR = os.path.join(TOP_DIR, 'data/bff21_zero/w_max_77')

CLASS_INPUT_MAP = {'CHSH': '00', '1': '01', '2a': '11',
                   '2b': '01', '2b_swap': '11', '2c': '10',
                   '3a': '11', '3b': '10'}

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
plt.rcParams.update(mplParams)

#### Colors for different classes
COLORS = ['blue','forestgreen','firebrick','gray']
# COLORS = ['blue','green','red','cyan','magenta', 'olive', 'gray', 'purple']
plt.rc('axes', prop_cycle=(cycler('color', COLORS)))

### Parallel settings
N_JOB = 8

### Constant parameters
CHSH_W_Q = (2 + math.sqrt(2))/4    # CHSH win prob
EPSILON = 1e-12                     # Smoothness of smooth min entropy (related to secrecy)
WIN_TOL = 1e-4                      # Tolerant error for win prob
GAMMA = 1e-2                        # Testing ratio

### General File Name Settings
OUT_DIR = os.path.join(TOP_DIR, 'figures/corrected_FER')
EPS = f'eps_{EPSILON:.0e}'
WTOL = f'wtol_{WIN_TOL:.0e}'
GAM = f'gam_{GAMMA:.0e}'

### Num of rounds
N_RANGE = (1e7, 1e14)
Ns = np.geomspace(*N_RANGE, num=200)

### Freely tunable params
#### Param related to alpha-Renyi entropy
BETA_SLICE = 150
BETAs = np.geomspace(1e-4, 1e-11, num=BETA_SLICE)
#### Another tunable param
NU_PRIME_SLICE = 50
NU_PRIMEs = np.linspace(1, 0.1, num=NU_PRIME_SLICE)

#### Plot for max win prob
fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)

### All classes to plot
CLASSES = ['CHSH','1','2c','3b']
ZERO_TOLs = [1e-9]

################################ Iter for different parameters ################################
### Run over all classes in CLASSES
for class_ in CLASSES:
    input_ = CLASS_INPUT_MAP[class_]

    ### Specify the file record the data
    HEAD = 'lr_bff21'
    CLS = f'class_{class_}' if class_ != 'CHSH' else 'CHSH'
    INPUT = f'xy_{input_}'
    QUAD = 'M_12'
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
        if class_ != 'CHSH':
            lambda_zeros = data_mtf[0][3:-1]
            c_lambda -= sum(lambda_zeros)*zero_tol

##################### Compute key rate with optimal parameters #####################
        ### Construct key rate function with fixed param (only leave n, beta tunable)
        kr_func = partial(fin_rate_testing, gamma = GAMMA, asym_rate = asym_rate,
                            lambda_ = lambda_, c_lambda = c_lambda,
                            zero_tol = zero_tol, zero_class=class_, max_p_win = max_p_win)
        
        def opt_all(n, beta_arr = BETAs, nu_prime_arr = NU_PRIMEs):
                return np.max(np.array([kr_func(n = n, beta = beta, nu_prime = nu_prime) \
                                        for nu_prime in nu_prime_arr for beta in beta_arr]))

        KRs = Parallel(n_jobs=N_JOB, verbose = 0)(delayed(opt_all)(N) for N in Ns)
        KRs = np.array(KRs)
        
##################################### Draw line ##################################### 
        label = f'class {class_}' if class_ != 'CHSH' else 'CHSH'
        #### Single plot for max win prob
        label += r' $\displaystyle x^*y^*$'+f'={input_}' + \
                    r' $\displaystyle w_{exp}$'+f'={win_prob:.4f}'

        if PRINT_DATA:
            print(np.column_stack((Ns, KRs)))

        if SAVE_CSV:
            data2save = np.column_stack((Ns, KRs))
            HEADER = 'num of rounds in log\trate'
            WEXP = f'w_{win_prob*10000:.0f}'.rstrip('0')
            ZTOL = f'ztol_{zero_tol:.0e}'
            OUTFILE = f'{CLS}-{WEXP}-{EPS}-{WTOL}-{ZTOL}-{GAM}-{QUAD}.csv'
            np.savetxt(OUTFILE, data2save, fmt='%.5g', delimiter=',', header=HEADER)
    
        plt.plot(Ns, KRs, label = label)


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
    COMMON = 'fin_blind_qbound'
    TAIL = 'test'
    FORMAT = 'png'
    OUT_NAME = f'{COMMON}-{EPS}-{WTOL}-{GAM}-{QUAD}'
    if TAIL:
        OUT_NAME += f'-{TAIL}'
    out_path = os.path.join(OUT_DIR, f'{OUT_NAME}.{FORMAT}')
    plt.savefig(out_path, format = FORMAT)
if SHOW:
    plt.show()
