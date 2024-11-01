"""
    This script generate a plot of finite rate of blind randomness
    with different and optimal gammas over number of rounds.
"""
from matplotlib import pyplot as plt
from cycler import cycler
import numpy as np
from functools import partial
from joblib import Parallel, delayed
import math
import os, sys

### Add directory to Python path
# (Note: this works when running the command in the dir. 'blindRandomness')
sys.path.append('..')
from common_func.plotting_helper import *

######### Plotting settings #########
FIG_SIZE = (12, 9)    # aspect ratio
DPI = 200
SUBPLOT_PARAM = {'left': 0.085, 'right': 0.98, 'bottom': 0.11, 'top': 0.95, 'wspace': 0.28}

titlesize = 30
ticksize = 28
legendsize = 26
linewidth = 3

### Set matplotlib plotting params
mplParams = plot_settings(title = titlesize, tick = ticksize, legend = legendsize, linewidth = linewidth)
mplParams["xtick.major.pad"] = 10
prop_cycler = cycler(color=['gray']) * cycler(linestyle=['solid', 'dashed', 'dashdot', 'dotted'])
mplParams["axes.prop_cycle"] = prop_cycler
plt.rcParams.update(mplParams)

### Parallel settings
N_JOB = 8

### Constant parameters
INP_DIST = [1/4]*4
CHSH_W_Q = (2 + math.sqrt(2) )/4    # CHSH win prob
EPSILON = 1e-12                     # Smoothness of smooth min entropy (related to secrecy)
WIN_TOL = 1e-4                      # Tolerant error for win prob

### Data paths
TOP_DIR = './'
DATA_DIR = os.path.join(TOP_DIR, 'data/BFF21/w_max_77')
OUT_DIR = os.path.join(TOP_DIR, 'figures/BFF21/fin_rate/gamma_test')
OUTCSV_DIR = os.path.join(TOP_DIR, './data/BFF21/fin_rate/opt_gamma')
CLASS_INPUT_MAP = cls_inp_map('blind')

SAVE = True         # To save figure or not
SHOW = False        # To show figure or not
SAVECSV = False     # To save data or not
### Common file names
QUAD = 'M_12'
EPS = f'eps_{EPSILON:.0e}'
WTOL = f'wtol_{WIN_TOL:.0e}'

### Varing params
#### Number of rounds
N_RANGE = (1e7, 1e14)
N_SLICE = 50
Ns = np.geomspace(*N_RANGE, num=N_SLICE)

#### Testing ratio
GAMMA_SLICE = 50
GAMMAs = 1/np.geomspace(10, 10000, num=GAMMA_SLICE)

#### Param related to alpha-Renyi entropy
BETA_SLICE = 50
BETAs = np.geomspace(1e-4, 1e-11, num=BETA_SLICE)

#### Another tunable param
NU_PRIME_SLICE = 50
NU_PRIMEs = np.linspace(1, 0.1, num=NU_PRIME_SLICE)

### Plot with different gamma
# GAMMAs = [1e-3, 1e-2, 5e-2, 0.1]
CLASSES = ['chsh', '1', '2c', '3b']
ZERO_TOL = 1e-9

################################ Iter for different parameters ################################
### Run over all classes in CLASSES
for cls in CLASSES:
    input_ = CLASS_INPUT_MAP[cls]

    HEAD = 'br'
    CLS = cls
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
    
    for gamma in [0.05, 0.01, 0.001]:
        win_prob, asym_rate, lambda_ = data_mtf[0][:3]
        c_lambda = data_mtf[0][-1]
        if cls != 'chsh':
            lambda_zeros = data_mtf[0][3:-1]
            c_lambda -= sum(lambda_zeros)*ZERO_TOL

################################ Compute key rate ################################
        ### Construct key rate function with fixed param (only leave n, beta tunable)
        def opt_all(n, beta_arr = BETAs, nu_prime_arr = NU_PRIMEs):
            fr_func = partial(fin_rate_testing, gamma = gamma, asym_rate = asym_rate,
                                lambda_ = lambda_, c_lambda = c_lambda,
                                zero_tol = ZERO_TOL, zero_class=cls, max_p_win = max_p_win)
            return np.max(np.array([fr_func(n = n, beta = beta, nu_prime = nup) \
                                    for nup in nu_prime_arr for beta in beta_arr]))

        FRs = Parallel(n_jobs=N_JOB, verbose = 0)(delayed(opt_all)(N) for N in Ns)
        FRs = np.array(FRs)

        Cost = inp_rand_consumption(gamma, INP_DIST)
        netFRs = FRs - Cost
        netFRs[netFRs < 0] = 0

        label = r'$\displaystyle \gamma={:.1g}$'.format(gamma)
        plt.plot(Ns, netFRs, label = label)

    def opt_with_gamma(n, beta_arr = BETAs, nup_arr = NU_PRIMEs, gam_arr = GAMMAs):
        fr_func = partial(fin_rate_testing, asym_rate = asym_rate,
                            lambda_ = lambda_, c_lambda = c_lambda,
                            zero_tol = ZERO_TOL, zero_class=cls, max_p_win = max_p_win)
        gen_rand = np.array([[fr_func(n = n, beta = beta, nu_prime = nup, gamma = gamma) \
                                for nup in nup_arr for beta in beta_arr] for gamma in gam_arr])
        cost = np.array([inp_rand_consumption(gamma, INP_DIST) for gamma in gam_arr])
        net_rand = (gen_rand.T - cost).T
        max_id =  np.argmax(net_rand) + 1
        max_gamma = gam_arr[math.ceil(max_id / (NU_PRIME_SLICE*BETA_SLICE))-1]
        opt_fin_rate = net_rand.flatten()[max_id - 1]
        return max(opt_fin_rate, 0), max_gamma
    
    opt_gamma = Parallel(n_jobs=N_JOB, verbose = 0)(delayed(opt_with_gamma)(N) for N in Ns)
    opt_gamma = list(zip(*opt_gamma))
    optFRs = np.array(opt_gamma[0])
    optGAMMAs = np.array(opt_gamma[1])

    plt.scatter(Ns, optFRs, label = r'optimal $\displaystyle \gamma$')
    nonzero = N_SLICE - len(optFRs[optFRs>=1e-6])
    for i in range(nonzero, N_SLICE):
        if (i-nonzero) % 10 == 0 or i == nonzero+4:
            txt = f'{optGAMMAs[i]:.1e}'
            plt.annotate(txt, (Ns[i], optFRs[i]+0.01), fontsize=24, horizontalalignment='right')

    ### Save optimal gamma to num of rounds to CSV file
    if SAVECSV:
        data2save = np.column_stack((Ns[nonzero:N_SLICE],
                                     optFRs[nonzero:N_SLICE],
                                     optGAMMAs[nonzero:N_SLICE]))
        HEADER = 'num_of_rounds\trate\tgamma'
        WEXP = f'w_{win_prob*10000:.0f}'.rstrip('0')
        OUTCSV = f'opt_gamma-{CLS}-{WEXP}-{EPS}-{WTOL}-{ZTOL}-{QUAD}.csv'
        OUTCSV_PATH = os.path.join(OUTCSV_DIR, OUTCSV)
        np.savetxt(OUTCSV_PATH, data2save, fmt='%.5g', delimiter=',', header=HEADER)

    ### Set x-/y- labels and other plotting stuff
    YLABEL = r'$\displaystyle r$'+' (bit per round)'
    XLABEL = r'$\displaystyle n$'+' (number of rounds)'
    plt.ylabel(YLABEL)
    plt.xlabel(XLABEL)
    plt.xscale("log")
    plt.legend(prop={"weight":"bold"}, loc='lower right')

    ### Apply the graphic settings
    plt.subplots_adjust(**SUBPLOT_PARAM)

    ### Save file
    if SAVE:
        ### General File Name Settings
        COM = 'opt_test_ratio'
        WEXP = 'QBOUND'
        TAIL = 'test'
        FORMAT = 'png'
        OUT_NAME = f'{COM}-{CLS}-{WEXP}-{EPS}-{WTOL}-{ZTOL}-{QUAD}'
        if TAIL:
            OUT_NAME += f'-{TAIL}'
        OUT_PATH = os.path.join(OUT_DIR, f'{OUT_NAME}.{FORMAT}')
        plt.savefig(OUT_PATH, format = FORMAT)
    if SHOW:
        plt.show()