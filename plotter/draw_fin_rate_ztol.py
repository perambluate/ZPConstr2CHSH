"""
    This script generate several plots of finite rates for each classes at
    quantum bound and 0.77 with different zero tolerances (1e-9, 1e-5, 1e-3).
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

from matplotlib.ticker import ScalarFormatter, LogLocator
from matplotlib import pyplot as plt
from functools import partial
from joblib import Parallel, delayed
import numpy as np
import itertools
import os, sys
import re

### Add current directory to Python path
sys.path.append('.')
from common_func.plotting_helper import *

# Extend ScalarFormatter
class MyScalarFormatter(ScalarFormatter):
    # Override '_set_format' with your own
    def _set_format(self):
        self.format = '%.2f'  # Show 2 decimals without redundant zeros

PRINT_DATA = False              # To print values of data
SAVE = True                     # To save figure or not
SHOW = False                    # To show figure or not
SAVECSV = False                 # To save data or not
DRAW_FROM_SAVED_DATA = True     # Plot the line with previous data if true
TYPE = 'one'                    # Type of randomness (one/two/blind)

### Change the GUI backend to get rid of the multi-threading runtime error with tkinter
if not SHOW:
    plt.switch_backend('agg')

### Data paths
TOP_DIR = top_dir(TYPE)
DATA_DIR = os.path.join(TOP_DIR, 'data')
OUTCSV_DIR = os.path.join(DATA_DIR, 'fin_rate')
if TYPE == 'blind':
    # DATA_DIR = os.path.join(DATA_DIR, 'BFF21/w_max_77')
    DATA_DIR = os.path.join(DATA_DIR, 'BFF21')
elif TYPE in ['one', 'two']:
    DATA_DIR = os.path.join(DATA_DIR, 'asymp_rate')

CLASS_INPUT_MAP = cls_inp_map(TYPE)
# CLASS_MAX_WIN = cls_max_win_map()

### Parallel settings
N_JOB = 8

### Constant parameters
# CHSH_W_Q = (2 + math.sqrt(2))/4    # CHSH win prob
EPSILON = 1e-12                     # Smoothness of smooth min entropy (related to secrecy)
WIN_TOL = 1e-4                      # Tolerant error for win prob
# GAMMA = 1e-2                        # Testing ratio
INP_DIST = [1/4]*4

### General File Name Settings
OUT_DIR = os.path.join(TOP_DIR, 'figures/')
if TYPE == 'blind':
    HEAD = 'br'
elif TYPE =='one':
    HEAD = 'opr'
elif TYPE == 'two':
    HEAD = 'tpr'
# HEAD += '_ztoltest'
EPS = f'eps_{EPSILON:.0e}'
WTOL = f'wtol_{WIN_TOL:.0e}'
QUAD = 'M_12'
# GAM = f'gam_{GAMMA:.0e}'
CSVHEAD = ''

######### Plotting settings #########
# FIG_SIZE = (16, 9)      # for aspect ratio 16:9
FIG_SIZE = (12, 9)    # for aspect ratio 4:3
# FIG_SIZE = (7, 9)
DPI = 200
SUBPLOT_PARAM = {'left': 0., 'right': 0.98,'bottom': 0., 'top': 0.905, 'wspace': 0.}

### Set font sizes and line width
titlesize = 32
ticksize = 30
legendsize = 30
linewidth = 3

### Set matplotlib plotting params
mplParams = plot_settings(title = titlesize, tick = ticksize,
                          legend = legendsize, linewidth = linewidth)
mplParams["xtick.major.pad"] = 10
plt.rcParams.update(mplParams)

### Num of rounds
# N_RANGE = (1e6, 1e14)
if TYPE == 'two':
    N_RANGE = (5e5, 1e15)
elif TYPE == 'one':
    N_RANGE = (1e6, 1e14)
elif TYPE == 'blind':
    N_RANGE = (1e7, 1e15)
N_SLICE = 150
# N_POINT = f'N_{N_RANGE[0]}'
N_POINT = re.sub('e\+0+|e\+', 'e', f'N_{N_RANGE[0]:.0e}_{N_RANGE[1]:.0e}_{N_SLICE}')
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

### All classes to plot
CLASSES = ['chsh', '2a']
# CLASSES = ['chsh','1','2a','2b','2b_swap','2c','3a','3b']
# if TYPE == 'two':
#     CLASSES.remove('2b_swap')

ZERO_PROB = 1e-9

#### Colors for different winning probabilities
COLORS = ('darkcyan','darkred')

fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)

MAX_R = 0

################################ Iter for different parameters ################################
### Run over all classes in CLASSES
for cls in CLASSES:
    #### Two subplots for max win and 0.77
    # fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE, dpi=DPI)

    #### Line styles for different zero tolerances
    LINES = itertools.cycle(('solid', 'dashed', 'dashdot', 'dotted'))

    ### Specify the file record the data
    CLS = cls
    inp = CLASS_INPUT_MAP[cls]
    if TYPE == 'one':
        INP = f'x_{inp}'
    else:
        INP = f'xy_{inp}'

    #if cls == 'chsh':
    #    ZERO_TOLs = [ZERO_PROB]
    #else:
    #    ZERO_TOLs = [1e-9, 1e-3] #[1e-9, 1e-5, 1e-3]

    ### Run over all zero tolerances in ZERO_TOLs
    # for zero_tol in ZERO_TOLs:
    if cls == 'chsh':
        zero_tol = ZERO_PROB
        ZTOL = f'ztol_{ZERO_PROB:.0e}'
        MTF_DATA_FILE = f'{HEAD}-{CLS}-{INP}-{QUAD}-{WTOL}-{ZTOL}.csv'
    else:
        MTF_DATA_FILE = f'{HEAD}_ztoltest-{CLS}-{INP}-{WTOL}-{QUAD}.csv'
    MTF_DATA_PATH = os.path.join(DATA_DIR, MTF_DATA_FILE)

    ### Get the maximum winnin probability
    if cls == 'chsh':
        with open(MTF_DATA_PATH) as file:
            max_win = float(file.readlines()[1])

        ### Load data
        data_mtf = np.genfromtxt(MTF_DATA_PATH, delimiter=",", skip_header = 3)
    else:
        data_mtf = np.genfromtxt(MTF_DATA_PATH, delimiter=",", skip_header = 1)

    # print(data_mtf)
    
    # line = next(LINES)
    ### Colors for different winning probabilities
    # color_iter = itertools.cycle(COLORS)

    # titles = []

    ### Choose min-tradeoff function with expected winning prob
    if len(data_mtf.shape) == 1:
        data_mtf = np.array([data_mtf])
    # data_len = data_mtf.shape[0]
    if cls == 'chsh':
        _range = (0, 1)
    else:
        _range = (1, 3)

    #### For different w_exp (or other parameters) in the data
    for i in range(*_range):
        ### Read the data
        if cls == 'chsh':
            win_prob, asym_rate, lambda_ = data_mtf[i][:3]
            c_lambda = data_mtf[i][-1]
        else:
            zero_tol = data_mtf[i][0]
            max_win = data_mtf[i][1]
            win_prob, asym_rate, lambda_ = data_mtf[i][2:5]
            lambda_zeros = data_mtf[i][5:-1]
            c_lambda = data_mtf[i][-1] - sum(lambda_zeros)*zero_tol
        
        FILENOTFOUD = False
        if DRAW_FROM_SAVED_DATA:
            if cls == 'chsh':
                win_prob = data_mtf[i][0]
            else:
                win_prob = data_mtf[i][2]
            WEXP = f'w_{win_prob*10000:.0f}'.rstrip('0')
            ZTOL = f'ztol_{zero_tol:.0e}'
            FIN_DATA_FILE = f'{CLS}-{WEXP}-{EPS}-{WTOL}-{ZTOL}-{QUAD}-{N_POINT}.csv'
            FIN_DATA_PATH = os.path.join(OUTCSV_DIR, FIN_DATA_FILE)
            if os.path.exists(FIN_DATA_PATH):
                data = np.genfromtxt(FIN_DATA_PATH, delimiter=",", skip_header = 1).T
                Ns = data[0]
                FRs = data[1]
            else:
                print(f'File path: {FIN_DATA_PATH} does not exist, compute the data and save.')
                FILENOTFOUD = True
        
        if FILENOTFOUD or not DRAW_FROM_SAVED_DATA:
            # if cls == 'chsh':
            #     win_prob, asym_rate, lambda_ = data_mtf[i][:3]
            #     c_lambda = data_mtf[i][-1]
            # else:
            #     zero_tol = data_mtf[i][0]
            #     max_win = data_mtf[i][1]
            #     win_prob, asym_rate, lambda_ = data_mtf[i][2:5]
            #     lambda_zeros = data_mtf[i][5:-1]
            #     c_lambda = data_mtf[i][-1] - sum(lambda_zeros)*zero_tol

##################### Compute key rate with optimal parameters #####################
            ### Construct key rate function with fixed param (only leave n, beta tunable)
            fr_func = partial(fin_rate_testing, asym_rate = asym_rate,
                                    lambda_ = lambda_, c_lambda = c_lambda,
                                    zero_tol = zero_tol, zero_class=cls,
                                    max_win = max_win, min_win = 1 - max_win)

            FRs = Parallel(n_jobs=N_JOB, verbose = 0)(
                    delayed(opt_all)(n, beta_arr = BETAs, nup_arr = NU_PRIMEs, gam_arr = GAMMAs,
                                    inp_dist = INP_DIST, fin_rate_func = fr_func) for n in Ns)
            FRs = np.array(FRs)
            
            if PRINT_DATA:
                print(np.column_stack((Ns, FRs)))

            if SAVECSV or (DRAW_FROM_SAVED_DATA and FILENOTFOUD):
                data2save = np.column_stack((Ns, FRs))
                HEADER = 'rounds, rate'
                WEXP = f'w_{win_prob*10000:.0f}'.rstrip('0')
                ZTOL = f'ztol_{zero_tol:.0e}'
                # OUTCSV = f'{CLS}-{WEXP}-{EPS}-{WTOL}-{ZTOL}-{GAM}-{QUAD}.csv'
                OUTCSV = f'{CLS}-{WEXP}-{EPS}-{WTOL}-{ZTOL}-{QUAD}-{N_POINT}.csv'
                OUTCSV_PATH = os.path.join(OUTCSV_DIR, OUTCSV)
                np.savetxt(OUTCSV_PATH, data2save, fmt='%.5g', delimiter=',', header=HEADER)
        
##################################### Draw line ##################################### 
        ### Colors for different zero tolerances
        # color = next(color_iter)
        if cls == 'chsh':
            color = 'grey'
            label = 'CHSH'
        else:
            #if cls == '2a':
            color = 'blue'
            #else:
            #    color = 'darkred'
            label = CLS+r' $\displaystyle \eta_{z}$'+f'={zero_tol:.0e}'
        
        ### Set labels of legends
        # label = CLS+r' $\displaystyle \delta_{zero}$'+f'={zero_tol:.0e}'

################################ Plotting lines ################################
        line = next(LINES)
        # axs[i].plot(Ns, FRs, linestyle = line, color = color, label = label)
        plt.plot(Ns, FRs, linestyle = line, color = color, label = label)
        # titles.append(r'$\displaystyle w_{exp}$'+f'={win_prob:.4g}')
        MAX_R = max(MAX_R, np.max(FRs))

#### Draw WBC rate
DATAPATH = f'./WBC_inequality/data/wbc_{TYPE}-{WTOL}-{QUAD}.csv'
data_wbc = np.genfromtxt(DATAPATH, delimiter=",", skip_header = 1)
if len(data_wbc.shape) == 1:
    data_wbc = np.array([data_wbc])
data_len = data_wbc.shape[0]

# Labels = [r'$\mathrm{WBC}^{[16]}\ \displaystyle I_{\delta=0.5236}$',
#           r'$\mathrm{WBC}^{[16]}\ \displaystyle I_{\delta=0.08391}$']

for i in range(1):
    delta, qbound, win_prob, entropy, lambda_, c_lambda = data_wbc[i]
    ZTOL = f'ztol_{zero_tol:.0e}'
    FIN_DATA_FILE = f'wbc-fin-{TYPE}-{WEXP}-{EPS}-{WTOL}-{ZTOL}-{QUAD}-{N_POINT}.csv'
    FIN_DATA_PATH = os.path.join('./WBC_inequality/data', FIN_DATA_FILE)
    if os.path.exists(FIN_DATA_PATH):
        data = np.genfromtxt(FIN_DATA_PATH, delimiter=",", skip_header = 1).T
        Ns = data[0]
        FRs = data[1]
    else:
        print(f'File path: {FIN_DATA_PATH} does not exist, compute the data and save.')
        FILENOTFOUD = True
        fr_func = partial(fin_rate_testing, asym_rate = entropy, lambda_ = lambda_,
                        c_lambda = c_lambda, max_win = qbound, min_win = 1-qbound)
        
        FRs = Parallel(n_jobs = N_JOB, verbose = 0)(
            delayed(opt_all)(n, beta_arr = BETAs, nup_arr = NU_PRIMEs, gam_arr = GAMMAs,
                            inp_dist = INP_DIST, fin_rate_func = fr_func) for n in Ns)
        # print(FRs)

    label = r'$\mathrm{{WBC}}^{{[16]}}\ \displaystyle I_{{\delta={{{}}}}}$'.format(delta)
    plt.plot(Ns, FRs, color = 'darkred', label = label)
    MAX_R = max(MAX_R, np.max(FRs))
    if SAVECSV or (DRAW_FROM_SAVED_DATA and FILENOTFOUD):
        data2save = np.column_stack((Ns, FRs))
        HEADER = 'rounds, rate'
        np.savetxt(FIN_DATA_PATH, data2save, fmt='%.5g', delimiter=',', header=HEADER)

################################ Save figure ################################
YLABEL = r'$\displaystyle r$' #+' (bit per round)'
XLABEL = r'$\displaystyle n$' #+' (number of rounds)'
### For two subplots
# axs[0].label_outer()
# axs[0].set_ylabel(ylabel=YLABEL)

# for i in range(len(axs)):
#     axs[i].set_xlabel(xlabel=XLABEL)
#     axs[i].set_xscale("log")
#     axs[i].set_title(titles[i], y=1, pad=20)
#     axs[i].yaxis.set_major_formatter(MyScalarFormatter(useMathText=True))
#     axs[i].yaxis.major.formatter.set_powerlimits((-1,2))
#     axs[i].xaxis.set_major_locator(LogLocator(numticks=8))
#     if cls != 'chsh':
#         axs[i].legend(loc='lower right')

if round(MAX_R, 2) < round(MAX_R, 1):
    MAX_R = round(MAX_R, 1)
else:
    MAX_R = round(MAX_R, 1) + 0.1

plt.yticks(np.linspace(0, MAX_R, 5, endpoint=True))
plt.ylabel(ylabel=YLABEL)
plt.xlabel(xlabel=XLABEL)
plt.xscale("log")
ax = plt.gca()
# ax.yaxis.set_major_formatter(MyScalarFormatter(useMathText=True))
# ax.yaxis.major.formatter.set_powerlimits((-1,2))
ax.xaxis.set_major_locator(LogLocator(numticks=8))
ax.legend(loc='lower right')

### Apply the graphic settings
plt.subplots_adjust(**SUBPLOT_PARAM)
fig.tight_layout(pad=.2)

plt.grid()

if SAVE:
    # COM = f'fin-{TYPE}-chsh_{CLASSES[1]}-qbound'
    COM = f'fin-{TYPE}-compare_wbc_chsh_{CLASSES[1]}-qbound'
    TAIL = '0'
    FORMAT = 'png'
    # OUT_NAME = f'{COM}-{CLS}-{INP}-{EPS}-{WTOL}-{QUAD}'
    OUT_NAME = f'{COM}-{INP}-{EPS}-{WTOL}-{QUAD}'
    if TAIL:
        OUT_NAME += f'-{TAIL}'
    out_path = os.path.join(OUT_DIR, f'{OUT_NAME}.{FORMAT}')
    plt.savefig(out_path, format = FORMAT)
if SHOW:
    plt.show()
