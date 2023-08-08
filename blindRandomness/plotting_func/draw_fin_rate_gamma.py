from matplotlib import pyplot as plt
from cycler import cycler
import numpy as np
import itertools
from functools import partial
from joblib import Parallel, delayed
import math
import os, sys

### Add current directory to Python path
# (Note: this works when running the command in the dir. 'blindRandomness')
sys.path.append('.')
from blindRandomness.common_func.plotting_helper import *

######### Plotting settings #########
FIG_SIZE = (12, 9)    # for aspect ratio 4:3
DPI = 200
SUBPLOT_PARAM = {'left': 0.085, 'right': 0.98, 'bottom': 0.11, 'top': 0.95, 'wspace': 0.28}

titlesize = 30
ticksize = 28
legendsize = 28
linewidth = 3

### Set matplotlib plotting params
mplParams = plot_settings(title = titlesize, tick = ticksize, legend = legendsize, linewidth = linewidth)
mplParams["xtick.major.pad"] = 10

plt.rcParams.update(mplParams)
plt.rc('axes', prop_cycle=(cycler('color', ['gray']) *\
                           cycler('linestyle', 
                                  ['solid', 'dashed', 'dashdot', 'dotted', (0,(3, 1, 1, 1, 1, 1))])))

### Parallel settings
N_JOB = 8

### Constant parameters
INP_DIST = [1/4]*4
CHSH_W_Q = (2 + math.sqrt(2) )/4    # CHSH win prob
EPSILON = 1e-12                     # Smoothness of smooth min entropy (related to secrecy)
WIN_TOL = 1e-4                      # Tolerant error for win prob

### Data paths
TOP_DIR = './'
DATA_DIR = os.path.join(TOP_DIR, 'data/bff21_zero/w_max_77')
OUT_DIR = os.path.join(TOP_DIR, 'figures/corrected_FER')
CLASS_INPUT_MAP = {'CHSH': '00', '1': '01', '2a': '11',
                   '2b': '01', '2b_swap': '11', '2c': '10',
                   '3a': '11', '3b': '10'}

### To save file or not
SAVE = True
### To show figure or not
SHOW = True
### Common file names
QUAD = 'M_12'
EPS = f'eps_{EPSILON:.0e}'
WTOL = f'wtol_{WIN_TOL:.0e}'

### Varing params
#### Number of rounds
N_RANGE = (1e7, 1e14)
Ns = np.geomspace(*N_RANGE, num=200)

#### Param related to alpha-Renyi entropy
BETA_SLICE = 150
BETAs = np.geomspace(1e-4, 1e-11, num=BETA_SLICE)

#### Another tunable param
NU_PRIME_SLICE = 50
NU_PRIMEs = np.linspace(1, 0.1, num=NU_PRIME_SLICE)

### Plot with different gamma
GAMMAs = [1e-3, 1e-2, 5e-2, 0.1, 0.5]
CLASSES = ['CHSH','1','2c','3b']
ZERO_TOL = 1e-9

# color_iter = itertools.cycle(('blue','forestgreen','firebrick','gray'))

################################ Iter for different parameters ################################
### Run over all classes in CLASSES
for class_ in CLASSES:
    input_ = CLASS_INPUT_MAP[class_]

    HEAD = 'lr_bff21'
    CLS = f'class_{class_}' if class_ != 'CHSH' else 'CHSH'
    INP = f'xy_{input_}'
    ZTOL = f'ztol_{ZERO_TOL:.0e}'
    DATA_FILE = f'{HEAD}-{CLS}-{INP}-{QUAD}-{WTOL}-{ZTOL}.csv'
    DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)

    ### Get the maximum winnin probability
    with open(DATA_PATH) as file:
        max_p_win = float(file.readlines()[1])

    ### Load data
    data_mtf = np.genfromtxt(DATA_PATH, delimiter=",", skip_header = 3)
    # print(data_mtf)
    
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
    # LINES = itertools.cycle(('solid', 'dashed', 'dashdot', 'dotted', (0,(3, 1, 1, 1, 1, 1))))
    
    for gamma in GAMMAs:
        win_prob, asym_rate, lambda_ = data_mtf[0][:3]
        c_lambda = data_mtf[0][-1]
        if class_ != 'CHSH':
            lambda_zeros = data_mtf[0][3:-1]
            c_lambda -= sum(lambda_zeros)*ZERO_TOL

################################ Compute key rate ################################
        ### Construct key rate function with fixed param (only leave n, beta tunable)
        kr_func = partial(fin_rate_testing, gamma = gamma, asym_rate = asym_rate,
                            lambda_ = lambda_, c_lambda = c_lambda,
                            zero_tol = ZERO_TOL, zero_class=class_, max_p_win = max_p_win)
        
        def opt_all(n, beta_arr = BETAs, nu_prime_arr = NU_PRIMEs):
                return np.max(np.array([kr_func(n = n, beta = beta, nu_prime = nu_prime) \
                                        for nu_prime in nu_prime_arr for beta in beta_arr]))

        KRs = Parallel(n_jobs=N_JOB, verbose = 0)(delayed(opt_all)(N) for N in Ns)
        KRs = np.array(KRs)

        Cost = inp_rand_consumption(gamma, INP_DIST)
        netKRs = KRs - Cost
        netKRs[netKRs < 0] = 0

        label = r'$\displaystyle \gamma={:.1g}$'.format(gamma)
        plt.plot(Ns, netKRs, label = label)

    YLABEL = r'$\displaystyle r$'+' (bit per round)'
    XLABEL = r'$\displaystyle n$'+' (number of rounds)'
    plt.ylabel(YLABEL)
    plt.xlabel(XLABEL)
    plt.xscale("log")
    plt.legend(prop={"weight":"bold"})

    ### Apply the graphic settings
    plt.subplots_adjust(**SUBPLOT_PARAM)

    ### Save file
    if SAVE:
        ### General File Name Settings
        COM = 'diff_test_ratio'
        WEXP = 'QBOUND'
        TAIL = ''
        FORMAT = 'png'
        OUT_NAME = f'{COM}-{CLS}-{WEXP}-{EPS}-{WTOL}-{QUAD}'
        if TAIL:
            OUT_NAME += f'-{TAIL}'
        out_path = os.path.join(OUT_DIR, f'{OUT_NAME}.{FORMAT}')
        plt.savefig(out_path, format = FORMAT)
    if SHOW:
        plt.show()