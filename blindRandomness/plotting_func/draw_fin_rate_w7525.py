"""
    This script generate a plot of finite rates with a break axis for class 1, 2c, 3b, and CHSH
    at w_exp = 0.7525; an extra zero tolerance 1e-3 for class 3b to show the robustness.
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
            <COM>(-<CLS>)(-<WIN_EXP>)-<EPS>-<WTOL>-<GAM>-<QUAD>-<TAIL>.png
        (Note: The two parts in the parentheses are sometimes omited depending on the content of the figure.)
"""

from matplotlib import pyplot as plt
from functools import partial
from joblib import Parallel, delayed
import numpy as np
import itertools
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
SHOW = False

### Change the GUI backend to get rid of the multi-threading runtime error with tkinter
if not SHOW:
    plt.switch_backend('agg')

### Data paths
TOP_DIR = './'
DATA_DIR = os.path.join(TOP_DIR, 'data/BFF21/w7525')
OUTCSV_DIR = os.path.join(TOP_DIR, './data/BFF21/fin_rate')
OUT_DIR = os.path.join(TOP_DIR, 'figures/BFF21/fin_rate')

CLASS_INPUT_MAP = cls_inp_map('blind')

######### Plotting settings #########
FIG_SIZE = (16, 9)      # for aspect ratio 16:9
DPI = 200
SUBPLOT_PARAM = {'left': 0.125, 'right': 0.98, 'bottom': 0.12, 'top': 0.95, 'wspace': 0.28}


### Set font sizes and line width
titlesize = 30
ticksize = 28
legendsize = 26
linewidth = 3.5

### Set matplotlib plotting params
mplParams = plot_settings(title = titlesize, tick = ticksize, legend = legendsize, linewidth = linewidth)
mplParams["xtick.major.pad"] = 10
plt.rcParams.update(mplParams)

### Parallel settings
N_JOB = 8

### Constant parameters
CHSH_W_Q = (2 + math.sqrt(2))/4     # CHSH win prob
EPSILON = 1e-12                     # Smoothness of smooth min entropy (related to secrecy)
WIN_TOL = 1e-4                      # Tolerant error for win prob
# GAMMA = 1e-2                        # Testing ratio
INP_DIST = [1/4]*4

### General File Name Settings
EPS = f'eps_{EPSILON:.0e}'
WTOL = f'wtol_{WIN_TOL:.0e}'
# GAM = f'gam_{GAMMA:.0e}'

### Num of rounds
# N_RANGE = (1e8, 1e14)
N_RANGE = (1e10, 1e15)
N_SLICE = 100
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

fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
gs = fig.add_gridspec(2, hspace=0)
axs = gs.subplots(sharex=True)

CLASSES = ['chsh','1','2c','3b']
# COLORS = ('blue','firebrick','forestgreen','gray')
COLORS = ('blue','forestgreen','firebrick','gray')

color_iter = itertools.cycle(COLORS)

################################ Iter for different parameters ################################
### Run over all classes in CLASSES
for cls in CLASSES:
    input_ = CLASS_INPUT_MAP[cls]

    #### Line styles for different zero tolerances
    LINES = itertools.cycle(('solid', 'dashed', 'dashdot', 'dotted'))

    color = next(color_iter)

    ### Specify the file record the data
    HEAD = 'lr_bff21'
    CLS = cls
    INPUT = f'xy_{input_}'
    QUAD = 'M_12'

    ### Tolerant error for winning prob
    ZERO_TOLs = [1e-9]
    if cls == '3b':
        ZERO_TOLs.append(1e-3)

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
        
        line = next(LINES)

        ### Choose min-tradeoff function with expected winning prob
        if len(data_mtf.shape) == 1:
            data_mtf = np.array([data_mtf])

        win_prob, asym_rate, lambda_ = data_mtf[0][:3]
        c_lambda = data_mtf[0][-1]
        if cls != 'chsh':
            lambda_zeros = data_mtf[0][3:-1]
            c_lambda -= sum(lambda_zeros)*zero_tol

##################### Compute key rate with optimal parameters #####################
        ### Construct key rate function with fixed param (only leave n, beta tunable)
        kr_func = partial(fin_rate_testing, asym_rate = asym_rate,
                            lambda_ = lambda_, c_lambda = c_lambda,
                            zero_tol = zero_tol, zero_class=cls, max_p_win = max_p_win)
        
        def opt_all(n, beta_arr = BETAs, nup_arr = NU_PRIMEs, gam_arr = GAMMAs):
            # return np.max(np.array([kr_func(n = n, beta = beta, nu_prime = nu_prime) \
            #                         for nu_prime in nup_arr for beta in beta_arr]))
            gen_rand = np.array([[kr_func(n = n, beta = beta, nu_prime = nup, gamma = gamma) \
                            for nup in nup_arr for beta in beta_arr] for gamma in gam_arr])
            cost = np.array([inp_rand_consumption(gamma, INP_DIST) for gamma in gam_arr])
            net_rand = (gen_rand.T - cost).T
            return max(np.max(net_rand), 0)

        FRs = np.zeros(N_SLICE)
        if opt_all(N_RANGE[1]) > 0:
            FRs = Parallel(n_jobs=N_JOB, verbose = 0)(delayed(opt_all)(N) for N in Ns)
            FRs = np.array(FRs)
        
##################################### Draw line #####################################        
        ### Set labels of legends
        label = f'class {cls}' if cls != 'chsh' else 'CHSH'
        # label += r' $\displaystyle x^*y^*$'+f'={input_}' + \
        #             r' $\displaystyle \delta_{zero}$'+f'={zero_tol:.0e}'
        label += r' $\displaystyle \delta_{zero}$'+f'={zero_tol:.0e}'

        if PRINT_DATA:
            print(np.column_stack((Ns, FRs)))

        if SAVECSV:
            data2save = np.column_stack((Ns, FRs))
            HEADER = 'num of rounds in log\trate'
            WEXP = f'w_{win_prob*10000:.0f}'.rstrip('0')
            ZTOL = f'ztol_{zero_tol:.0e}'
            CSV_COM = 'fin_br'
            # OUTCSV = f'{CLS}-{WEXP}-{EPS}-{WTOL}-{ZTOL}-{GAM}-{QUAD}.csv'
            OUTCSV = f'{CSV_COM}-{CLS}-{WEXP}-{EPS}-{WTOL}-{ZTOL}-{QUAD}.csv'
            OUTCSV_PATH = os.join.path(OUTCSV_DIR, OUTCSV)
            np.savetxt(OUTCSV_PATH, data2save, fmt='%.5g', delimiter=',', header=HEADER)

################################ Plotting lines ################################        
        #### Plot for win 0.7525 with break axis
        axs[0].plot(Ns, FRs,
                linestyle = line, color = color, label = label)            
        axs[1].plot(Ns, FRs,
                linestyle = line, color = color, label = label)

################################ Save figure ################################
YLABEL = r'$\displaystyle r$'+' (bit per round)'
XLABEL = r'$\displaystyle n$'+' (number of rounds)'

### For break axis
axs[1].set_xlabel(xlabel=XLABEL)
axs[0].legend(loc='best')
axs[0].set_ylim(top=0.03, bottom=0.00085)
axs[1].set_ylim(top=0.0008, bottom=0)

for ax in axs:
    ax.set_xscale("log")

WIN_EXP = r'$\displaystyle {w}_{exp}=0.7525$'
fig.text(0.16, 0.85, WIN_EXP, va='center')
fig.supylabel(YLABEL)

### Apply the graphic settings
plt.subplots_adjust(**SUBPLOT_PARAM)

### Save file
if SAVE:
    COM = 'fin_blind-inp_consump-w_7525'
    TAIL = 'test'
    FORMAT = 'png'
    # OUT_NAME = f'{COM}-{EPS}-{WTOL}-{GAM}-{QUAD}'
    OUT_NAME = f'{COM}-{EPS}-{WTOL}-{QUAD}'
    if TAIL:
        OUT_NAME += f'-{TAIL}'
    out_path = os.path.join(OUT_DIR, f'{OUT_NAME}.{FORMAT}')
    plt.savefig(out_path, format = FORMAT)
if SHOW:
    plt.show()
