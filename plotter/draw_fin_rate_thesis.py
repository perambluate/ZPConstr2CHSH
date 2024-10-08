"""
    Plot the finite rates of the randomness (including global, local, and blind)
    for the correlations with zero-probability constraints (Quantum 7, 1054 (2023))
    and the correlations from the delta-family Bell inequalities (PRL. vol. 129, no. 15, 5, 150403)
    by applying the generalized entropy accumulation (doi:10.1109/FOCS54457.2022.00085).
"""

from matplotlib import pyplot as plt
from matplotlib import ticker
from functools import partial
from joblib import Parallel, delayed
import numpy as np
import os, sys
import re
import itertools

### Add current directory to Python path
# (Note: this works when running the command in the repo dir.)
sys.path.append('.')
from common_func.plotting_helper import *

PRINT_DATA = False              # To print values of data
SAVE = True                     # To save figure or not
SHOW = False                    # To show figure or not
SAVECSV = False                  # To save data or not
DRAW_FROM_SAVED_DATA = True    # Plot the line with previous data if true
TYPE = 'two'                    # Type of randomness (one/two/blind)
PLOT_GAM = True                 # Plot gamma over number of rounds if true
GET_OPTPARAM = False             # Whether to get optimal params or not

### Parallel settings
N_JOB = 8

### Constant parameters
EPSILON = 1e-8                    # Smoothness of smooth min entropy (related to secrecy)
INP_DIST = [1/4]*4
d_K = 4 if TYPE == 'two' else 2

######### Plotting settings #########
FIG_SIZE = (16, 9)      # for aspect ratio 16:9
# FIG_SIZE = (12, 9)    # for aspect ratio 4:3
DPI = 200
SUBPLOT_PARAM = {'left': 0.12, 'right': 0.98, 'bottom': 0.11, 'top': 0.95, 'wspace': 0.28}

### Set font sizes and line width
titlesize = 38
ticksize = 34
legendsize = 34
linewidth = 3

### Set matplotlib plotting params
mplParams = plot_settings(title = titlesize, tick = ticksize,
                          legend = legendsize, linewidth = linewidth)
mplParams["xtick.major.pad"] = 10

#### Colors for different classes
plt.rcParams.update(mplParams)

### Num of rounds
if TYPE == 'blind':
    N_RANGE = (1e6, 1e15)
elif TYPE == 'one':
    N_RANGE = (5e5, 1e15)
elif TYPE == 'two':
    N_RANGE = (5e5, 1e15)
N_SLICE = 200
Ns = np.geomspace(*N_RANGE, num=N_SLICE)
N_POINT = re.sub('e\+0+|e\+', 'e', f'N_{N_RANGE[0]:.0e}_{N_RANGE[1]:.0e}_{N_SLICE}')

### Freely tunable params
#### Param related to alpha-Renyi entropy
BETA_SLICE = 50
BETAs = np.geomspace(1e-4, 1e-11, num=BETA_SLICE)

#### Testing ratio
GAMMA_SLICE = 100
GAMMAs = 1/np.geomspace(1, 100000, num=GAMMA_SLICE)

#### Another tunable param
NU_PRIME_SLICE = 50
NU_PRIMEs = np.linspace(1, 0.1, num=NU_PRIME_SLICE)

#### Plot for max win prob
fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)

### Get data to compute finite rate
EPS = f'eps_{EPSILON:.0e}'
PDG = 'pdgap_5e-05'
QUAD = 'M_12'

### Data paths
TOP_DIR = top_dir(TYPE)
DATA_DIR = os.path.join(TOP_DIR, 'data')
OUTCSV_DIR = os.path.join(DATA_DIR, 'fin_rate')
if TYPE == 'blind':
    DATA_DIR = os.path.join(DATA_DIR, 'BFF21/')

HEAD = 'best_setup-'
if TYPE == 'blind':
    HEAD += 'br'
    BEST_CLS = '3b'
elif TYPE =='one' or TYPE == 'two':
    HEAD += TYPE[0] + 'pr'
    BEST_CLS = '2a'

CLASS_INPUT_MAP = cls_inp_map(TYPE)

### Classes to plot
if PLOT_GAM:
    CLASSES = [BEST_CLS]
else:
    CLASSES = ['chsh', BEST_CLS]

MAX_R = 0

SETTINGs = {'chsh': [r'($\displaystyle w_{tol}=10^{-5}$)',
                     r'($\displaystyle w_{tol}=10^{-3}$)'],
            'zpclass': [r'($\displaystyle w_{tol}=10^{-5}, \eta_{z}=10^{-6}$)',
                        r'($\displaystyle w_{tol}=10^{-3}, \eta_{z}=10^{-3}$)']}

LINES = itertools.cycle(('solid', 'dashed'))

### Run over all classes in CLASSES
for cls in CLASSES:
    ### Specify the file record the data
    CLS = cls
    NP = 'N_2'
    inp = CLASS_INPUT_MAP[cls]
    if TYPE == 'one':
        INP = f'x_{inp}'
    else:
        INP = f'xy_{inp}'
    
    MTF_DATA_FILE = f'{HEAD}-{CLS}-{INP}-{PDG}-{QUAD}-{NP}-wplb.csv'
    MTF_DATA_PATH = os.path.join(DATA_DIR, MTF_DATA_FILE)

    ### Load data
    data_mtf = np.genfromtxt(MTF_DATA_PATH, delimiter=",", skip_header = 1)
    # print(data_mtf)
    
    ### Choose min-tradeoff function with expected winning prob
    if len(data_mtf.shape) == 1:
        data_mtf = np.array([data_mtf])

    #### For different w_exp (or other parameters) in the data
    for i in range(data_mtf.shape[0]):
        if cls == 'chsh':
            zero_tol = 0
            shift = 0
        else:
            zero_tol = data_mtf[i][0]
            shift = 1
        
        win_tol, max_win, min_win, win_prob, asym_rate, lambda_ = data_mtf[i][shift:shift+6]
        print(win_tol, max_win, min_win, win_prob, asym_rate, lambda_)

        ZTOL = f'ztol_{zero_tol:.0e}'
        WTOL = f'wtol_{win_tol:.0e}'
        FILENOTFOUD = False
        if DRAW_FROM_SAVED_DATA or SAVECSV:
            WEXP = f'w_{win_prob*10000:.0f}'.rstrip('0')
            TAIL = 'allparams'
            FIN_DATA_FILE = f'{CLS}-{WEXP}-{EPS}-{WTOL}-{ZTOL}-{QUAD}-{N_POINT}-{TAIL}.csv'
            FIN_DATA_PATH = os.path.join(OUTCSV_DIR, FIN_DATA_FILE)
            if DRAW_FROM_SAVED_DATA:
                if os.path.exists(FIN_DATA_PATH):
                    data = np.genfromtxt(FIN_DATA_PATH, delimiter=",", skip_header = 1).T
                    if PLOT_GAM:
                        if len(data) < 3:
                            print(f'File {FIN_DATA_PATH} does not include gammas.')
                            FILENOTFOUD = True
                        else:
                            Ns = data[0]
                            FRs = data[1]
                            optGAMMAs = data[2]
                    else:
                        Ns = data[0]
                        FRs = data[1]
                else:
                    print(f'File {FIN_DATA_PATH} does not exist, compute the data and save.')
                    FILENOTFOUD = True
        
        if FILENOTFOUD or not DRAW_FROM_SAVED_DATA:
##################### Compute key rate with optimal parameters #####################
            ### Construct key rate function with fixed param (only leave n, beta tunable)
            kr_func = partial(fin_rate_testing, asym_rate = asym_rate, d_K = d_K,
                                    lambda_ = lambda_, epsilon = EPSILON,
                                    zero_tol = zero_tol, zero_class=cls,
                                    max_win = max_win, min_win = min_win)
            
            largest_FR = opt_all(Ns[-1], beta_arr = BETAs, nup_arr = NU_PRIMEs, gam_arr = GAMMAs,
                                    inp_dist = INP_DIST, fin_rate_func = kr_func)
            if largest_FR == 0:
                FRs = np.zeros(N_SLICE)
            else:
                if GET_OPTPARAM:
                    fr_params = Parallel(n_jobs=N_JOB, verbose = 0)(
                                            delayed(opt_with_all)(N, beta_arr = BETAs,
                                                                  nup_arr = NU_PRIMEs,
                                                                  gam_arr = GAMMAs,
                                                                  inp_dist = INP_DIST,
                                                                  fin_rate_func = kr_func)
                                            for N in Ns)
                    fr_params = list(zip(*fr_params))
                    FRs = np.array(fr_params[0])
                    optGAMMAs = np.array(fr_params[1])
                    optBETAs = np.array(fr_params[2])
                    optNUPs = np.array(fr_params[3])
                elif PLOT_GAM:
                    fr_gammas = Parallel(n_jobs=N_JOB, verbose = 0)(
                                            delayed(opt_with_gamma)(N, beta_arr = BETAs,
                                                                    nup_arr = NU_PRIMEs,
                                                                    gam_arr = GAMMAs,
                                                                    inp_dist = INP_DIST,
                                                                    fin_rate_func = kr_func)
                                            for N in Ns)
                    fr_gammas = list(zip(*fr_gammas))
                    FRs = np.array(fr_gammas[0])
                    optGAMMAs = np.array(fr_gammas[1])
                else:
                    FRs = Parallel(n_jobs=N_JOB, verbose = 0)(
                                    delayed(opt_all)(n, beta_arr = BETAs,
                                                        nup_arr = NU_PRIMEs,
                                                        gam_arr = GAMMAs,
                                                        inp_dist = INP_DIST,
                                                        fin_rate_func = kr_func)
                                    for n in Ns)
                    FRs = np.array(FRs)
            
            if PRINT_DATA:
                print(np.column_stack((Ns, FRs)))

            if SAVECSV or (DRAW_FROM_SAVED_DATA and FILENOTFOUD):
                if GET_OPTPARAM:
                    HEADER = 'rounds, rate, gamma, beta, nuprime'
                    data2save = np.column_stack((Ns, FRs, optGAMMAs, optBETAs, optNUPs))
                elif PLOT_GAM:
                    HEADER = 'rounds, rate, gamma'
                    data2save = np.column_stack((Ns, FRs, optGAMMAs))
                else:
                    HEADER = 'rounds, rate'
                    data2save = np.column_stack((Ns, FRs))
                np.savetxt(FIN_DATA_PATH, data2save, fmt='%.5g', delimiter=',', header=HEADER)
        
##################################### Draw line ##################################### 
        ### Colors for different zero tolerances
        if cls == 'chsh':
            color = 'grey'
            label = CLS.upper()
            label += ' '+SETTINGs['chsh'][i]
        else:
            color = 'blue'
            label = CLS
            label += ' '+SETTINGs['zpclass'][i]
        

################################ Plotting lines ################################
        line = next(LINES)
        if PLOT_GAM:
            plt.plot(Ns[FRs > 0], optGAMMAs[FRs > 0],
                     linestyle = line, color = color, label = label)
        else:
            plt.plot(Ns, FRs, linestyle = line, color = color, label = label)
            MAX_R = max(MAX_R, np.max(FRs))


if round(MAX_R, 2) < round(MAX_R, 1):
    MAX_R = round(MAX_R, 1)
else:
    MAX_R = round(MAX_R, 1) + 0.1

if TYPE == 'blind' or TYPE == 'one':
    MAX_R = min(MAX_R, 1)

################################ Save figure ################################
XLABEL = r'$\displaystyle n$'+' (number of rounds)'
if PLOT_GAM:
    YLABEL = r'$\displaystyle \gamma$'
    plt.yscale("log")
else:
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
if PLOT_GAM:
    ax.grid(which='major', color='#DDDDDD', linestyle='-', linewidth=1.2)
    ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.8)
else:
    plt.grid()

### Apply the graphic settings
plt.subplots_adjust(**SUBPLOT_PARAM)


### Save file
if SAVE:
    if PLOT_GAM:
        HEAD = f'opt_gamma-{TYPE}-qbound'
    else:
        HEAD = f'fin-{TYPE}-qbound'
    TAIL = 'best_setup-wplb'
    FORMAT = 'png'
    OUT_FILE = f'{HEAD}-{EPS}-{PDG}-{QUAD}-{TAIL}.{FORMAT}'
    OUT_DIR = './'
    OUT_PATH = os.path.join(OUT_DIR, OUT_FILE)
    SAVE_ARGS = {"format": FORMAT,
                "bbox_extra_artists": (lgd,),
                "bbox_inches": 'tight'}
    plt.savefig(OUT_PATH, **SAVE_ARGS)

if SHOW:
    plt.show()
