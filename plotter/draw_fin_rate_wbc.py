"""
    Plot the finite rate of blind randomness for the delta-family Bell inequalities
    in (PRL. vol. 129, no. 15, 5, 150403)
"""

from matplotlib import pyplot as plt, ticker as mticker
from functools import partial
from joblib import Parallel, delayed
import numpy as np
import math
import os, sys
import re
import itertools

### Add current directory to Python path
# (Note: this works when running the command in the dir. 'blindRandomness')
sys.path.append('.')
from common_func.plotting_helper import *

def min_quantum_score(max_win_q, delta):
    max_win_ns = (1 + 2/math.sin(delta) + 1/math.cos(2*delta))/4
    return max_win_ns - max_win_q

PRINT_DATA = False  # To print values of data
SAVE = True        # To save figure or not
SHOW = False        # To show figure or not
SAVECSV = False      # To save data or not
DRAW_FROM_SAVED_DATA = True    # Plot the line with previous data if true
TYPE = 'blind'        # Type of randomness (one/two/blind)

### Parallel settings
N_JOB = 8

### Constant parameters
EPSILON = 1e-12                    # Smoothness of smooth min entropy (related to secrecy)
WIN_TOL = 1e-4                     # Tolerant error for win prob
GAMMA = 1e-2                       # Testing ratio
INP_DIST = [1/4]*4

######### Plotting settings #########
# FIG_SIZE = (16, 9)      # for aspect ratio 16:9
FIG_SIZE = (12, 9)    # for aspect ratio 4:3
DPI = 200
SUBPLOT_PARAM = {'left': 0.12, 'right': 0.98, 'bottom': 0.11, 'top': 0.95, 'wspace': 0.28}

### Set font sizes and line width
titlesize = 30
ticksize = 28
legendsize = 26
linewidth = 3

### Set matplotlib plotting params
mplParams = plot_settings(title = titlesize, tick = ticksize, legend = legendsize,
                          linewidth = linewidth)
mplParams["xtick.major.pad"] = 10

#### Colors for different classes
plt.rcParams.update(mplParams)

### Num of rounds
N_RANGE = (5e5, 1e14)
N_SLICE = 200
Ns = np.geomspace(*N_RANGE, num=N_SLICE)
N_POINT = re.sub('e\+0+|e\+', 'e', f'N_{N_RANGE[0]:.0e}_{N_RANGE[1]:.0e}_{N_SLICE}')

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

### Get data to compute finite rate
EPS = f'eps_{EPSILON:.0e}'
WTOL = f'wtol_{WIN_TOL:.0e}'
QUAD = 'M_12'

### Data paths
TOP_DIR = top_dir(TYPE)
DATA_DIR = os.path.join(TOP_DIR, 'data')
OUTCSV_DIR = os.path.join(DATA_DIR, 'fin_rate')
if TYPE == 'blind':
    DATA_DIR = os.path.join(DATA_DIR, 'BFF21/w_max_77')
elif TYPE in ['one', 'two']:
    DATA_DIR = os.path.join(DATA_DIR, 'asymp_rate')

OUT_DIR = os.path.join(TOP_DIR, 'figures/')
if TYPE == 'blind':
    HEAD = 'br'
elif TYPE =='one' or TYPE == 'two':
    HEAD = 'max_w77'

CLASS_INPUT_MAP = cls_inp_map(TYPE)
ZERO_PROB = 1e-9

### All classes to plot
CLASSES = ['3b', 'chsh']

MAX_R = 0

### Run over all classes in CLASSES
for cls in CLASSES:
    #### Two subplots for max win and 0.77
    # fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE, dpi=DPI)

    #### Line styles for different zero tolerances
    LINES = itertools.cycle(('solid', 'dashed'))

    ### Specify the file record the data
    CLS = cls
    inp = CLASS_INPUT_MAP[cls]
    if TYPE == 'one':
        INP = f'x_{inp}'
    else:
        INP = f'xy_{inp}'

    if cls == 'chsh':
        ZERO_TOLs = [ZERO_PROB]
    else:
        ZERO_TOLs = [1e-9, 1e-3] #[1e-9, 1e-5, 1e-3]

    ### Run over all zero tolerances in ZERO_TOLs
    for zero_tol in ZERO_TOLs:
        ZTOL = f'ztol_{zero_tol:.0e}'
        MTF_DATA_FILE = f'{HEAD}-{CLS}-{INP}-{QUAD}-{WTOL}-{ZTOL}.csv'
        MTF_DATA_PATH = os.path.join(DATA_DIR, MTF_DATA_FILE)

        ### Get the maximum winnin probability
        with open(MTF_DATA_PATH) as file:
            max_win = float(file.readlines()[1])

        ### Load data
        data_mtf = np.genfromtxt(MTF_DATA_PATH, delimiter=",", skip_header = 3)
        # print(data_mtf)
        
        line = next(LINES)
        titles = []

        ### Choose min-tradeoff function with expected winning prob
        if len(data_mtf.shape) == 1:
            data_mtf = np.array([data_mtf])
        data_len = data_mtf.shape[0]

        #### For different w_exp (or other parameters) in the data
        for i in range(1):
            FILENOTFOUD = False
            if DRAW_FROM_SAVED_DATA:
                win_prob = data_mtf[i][0]
                WEXP = f'w_{win_prob*10000:.0f}'.rstrip('0')
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
                win_prob, asym_rate, lambda_ = data_mtf[i][:3]
                c_lambda = data_mtf[i][-1]
                if cls != 'CHSH':
                    lambda_zeros = data_mtf[i][3:-1]
                    c_lambda -= sum(lambda_zeros)*zero_tol

    ##################### Compute key rate with optimal parameters #####################
                ### Construct key rate function with fixed param (only leave n, beta tunable)
                fr_func = partial(fin_rate_testing, asym_rate = asym_rate,
                                        lambda_ = lambda_, c_lambda = c_lambda,
                                        zero_tol = zero_tol, zero_class=cls,
                                        max_win = max_win, min_win = 1 - max_win)
                
                largest_FR = opt_all(Ns[-1], beta_arr = BETAs, nup_arr = NU_PRIMEs, gam_arr = GAMMAs,
                                       inp_dist = INP_DIST, fin_rate_func = fr_func)
                if largest_FR == 0:
                    FRs = np.zeros(N_SLICE)
                else:
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
                    OUTCSV = f'{CLS}-{WEXP}-{EPS}-{WTOL}-{ZTOL}-{QUAD}-{N_POINT}.csv'
                    OUTCSV_PATH = os.path.join(OUTCSV_DIR, OUTCSV)
                    np.savetxt(OUTCSV_PATH, data2save, fmt='%.5g', delimiter=',', header=HEADER)
            
##################################### Draw line ##################################### 
            ### Colors for different zero tolerances
            if cls == '3b':
                color = 'blue'
                # label = CLS+r' $\displaystyle \delta_{z}$'+f'={zero_tol:.0e}'
                label = r'Protocol 1 $\displaystyle \delta_{z}$'+f'={zero_tol:.0e}'
            if cls == 'chsh':
                color = 'grey'
                label = CLS.upper()

################################ Plotting lines ################################
            plt.plot(Ns, FRs, linestyle = line, color = color, label = label)
            MAX_R = max(MAX_R, np.max(FRs))

#### Draw WBC rate
DATAPATH = f'./WBC_inequality/data/wbc_{TYPE}-{WTOL}-{QUAD}.csv'
data_wbc = np.genfromtxt(DATAPATH, delimiter=",", skip_header = 1)
if len(data_wbc.shape) == 1:
    data_wbc = np.array([data_wbc])
data_len = data_wbc.shape[0]

Labels = [r'$\mathrm{WBC}^{[7]}\ \displaystyle I_{\delta=0.5236}$',
          r'$\mathrm{WBC}^{[7]}\ \displaystyle I_{\delta=0.08391}$']
Colors = ['darkred','darkgreen']

for i in range(1):
    delta, qbound, score, entropy, lambda_, c_lambda = data_wbc[i]
    fr_func = partial(fin_rate_testing, asym_rate = entropy, lambda_ = lambda_,
                      c_lambda = c_lambda, max_win = qbound, min_win = 1-qbound)
    
    FRs = Parallel(n_jobs=N_JOB, verbose = 0)(
          delayed(opt_all)(n, beta_arr = BETAs, nup_arr = NU_PRIMEs, gam_arr = GAMMAs,
                           inp_dist = INP_DIST, fin_rate_func = fr_func) for n in Ns)
    # print(FRs)

    label = Labels[i]
    plt.plot(Ns, FRs, color = Colors[i], label = label)
    MAX_R = max(MAX_R, np.max(FRs))

if round(MAX_R, 2) < round(MAX_R, 1):
    MAX_R = round(MAX_R, 1)
else:
    MAX_R = round(MAX_R, 1) + 0.1

################################ Save figure ################################
YLABEL = r'$\displaystyle r$'+' (bit per round)'
XLABEL = r'$\displaystyle n$'+' (number of rounds)'    

plt.xscale("log")
plt.ylabel(YLABEL)
plt.xlabel(XLABEL)
plt.legend(prop={"weight":"bold"})
plt.yticks(np.linspace(0, MAX_R, 5, endpoint=True))

### Apply the graphic settings
plt.subplots_adjust(**SUBPLOT_PARAM)

### Save file
if SAVE:
    COM = f'compare_wbc_chsh-{TYPE}-w_chsh_77'
    EPS = f'eps_{EPSILON:.0e}'
    WTOL = f'wtol_{WIN_TOL:.0e}'
    QUAD = 'M_12'
    TAIL = 'test'
    FORMAT = 'png'
    OUT_NAME = f'{COM}-{EPS}-{WTOL}-{QUAD}'
    if TAIL:
        OUT_NAME += f'-{TAIL}'
    OUT_FILE = f'{OUT_NAME}.{FORMAT}'
    OUT_DIR = './WBC_inequality/'
    OUT_PATH = os.path.join(OUT_DIR, OUT_FILE)
    plt.savefig(OUT_PATH, format = FORMAT)

if SHOW:
    plt.show()