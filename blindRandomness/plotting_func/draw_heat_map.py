"""
    This script provides heat map to find out the optimal parameters for min-tradeoff function.
    Specify the parameter to scan with variable 'SCAN'.
        (i)     beta (scanning) : Plot the finite rate as colored pixels with beta and number of rounds.
                                Optimize nu_prime for each pixel.
        (ii)    nup (scanning) : Plot the finite rate as colored pixels with nu_prime and number of rounds.
                                Optimize beta for each pixel.
        (iii)   both : Plot two heat map, one for beta scanning, and the other for nu_prime scanning.
"""
from matplotlib import pyplot as plt
from matplotlib import cm
from functools import partial
from joblib import Parallel, delayed
import numpy as np
import math
import os, sys

### Choose 'beta' or 'nup' to decide which parameter to scan;
###   use 'both' to produce two subplots for both params
SCAN = 'beta'

### To save file or not
SAVE = True
### To show figure or not
SHOW = False

### Change the GUI backend to get rid of the multi-threading runtime error with tkinter
if not SHOW:
    plt.switch_backend('agg')

### Parallel settings
N_JOB = 8

### Data paths
TOP_DIR = './'
DATA_DIR = os.path.join(TOP_DIR, 'data/bff21_zero/w_max_77')
CLASS_INPUT_MAP = {'CHSH': '00', '1': '01', '2a': '11',
                   '2b': '01', '2b_swap': '11', '2c': '10',
                   '3a': '11', '3b': '10'}

######### Plotting settings #########
DPI = 200
if SCAN == 'both':
    SUBPLOT_PARAM = {'left': 0.09, 'right': 0.87, 'bottom': 0.11, 'top': 0.925, 'wspace': 0.2}
    FIG_SIZE = (16, 9)
else:
    SUBPLOT_PARAM = {'left': 0.085, 'right': 0.98, 'bottom': 0.095, 'top': 0.905, 'wspace': 0.28}
    FIG_SIZE = (12, 9)

### Default font size
titlesize = 32
ticksize = 30

### Set matplotlib plotting params
mplParams = {
    #### Default font (family and size)
    "font.family": "serif",
    "font.sans-serif": "Latin Modern Roman",
    "font.size": ticksize,
    "figure.titlesize": titlesize,
    "axes.labelsize": titlesize,
    "xtick.labelsize": ticksize,
    "ytick.labelsize": ticksize,
    #### Xtick-axis padding
    "xtick.major.pad": 15,
    #### Use Latex
    "text.usetex": True,
    #### Make text and math font bold
    # "font.weight": "bold",
    # "text.latex.preamble": r"\boldmath"
}

plt.rcParams.update(mplParams)

### Constant parameters
CHSH_W_Q = (2 + math.sqrt(2) )/4    # CHSH win prob
EPSILON = 1e-12                     # Smoothness of smooth min entropy (related to secrecy)
WIN_TOL = 1e-4                      # Tolerant error for win prob
GAMMA = 1e-2                        # Testing ratio

### Num of rounds
N_RANGE = (1e7, 1e14)
Ns = np.geomspace(*N_RANGE, num=200)

### Free param related to alpha-Renyi entropy
BETA_SLICE = 100
BETAs = np.geomspace(1e-4, 1e-11, num=BETA_SLICE)

NU_PRIME_SLICE = 100
NU_PRIMEs = np.linspace(1, 0.1, num=NU_PRIME_SLICE)

# CLASSES = ['2a','2b','2b_swap','2c','3a','3b']
CLASSES = ['3b']

### Finite extractable rate for blind randomness extraction with testing
def key_rate_testing(n, beta, nu_prime, gamma, asym_rate, lambda_, c_lambda,
                        epsilon = EPSILON, win_tol = WIN_TOL, zero_tol = 0,
                        zero_class = 'CHSH', max_p_win = CHSH_W_Q):
    
    ln2 = math.log(2)
    gamma_0 = (1-gamma)/gamma
    nu_0 = 1/(2*gamma) - gamma_0*nu_prime
    
    D = c_lambda**2 + (lambda_/(2*gamma))**2 - gamma_0/gamma * lambda_**2 * (1-nu_prime)*nu_prime
    if nu_0 > max_p_win:
        var_f = D - (lambda_*(max_p_win - nu_prime))**2
    elif nu_0 < (1 - max_p_win):
        var_f = D - (lambda_*(1 - max_p_win - nu_prime))**2
    else:
        var_f = D

    max2min_f = lambda_*(gamma_0 * (1-nu_prime) + max_p_win)
    if math.sqrt(1 - epsilon**2) == 1:
        epsi_term = math.log2(1/2*epsilon**2)
    else:
        epsi_term = math.log2(1 - math.sqrt(1 - epsilon**2) )
    
    epsi_win = math.e ** (-2 * win_tol**2 * n)
    zero_tol = zero_tol/2
    epsi_zero = math.e ** (-2 * zero_tol**2 * n)
    try:
        log_prob = math.log2(1 - epsi_win)
        if zero_class != 'CHSH':
            log_prob += math.log2(1 - epsi_zero)
    except ValueError:
        log_prob = math.log2(sys.float_info.min)

    with np.errstate(over='raise'):
        try:
            lamb_term = 2 ** (2 + max2min_f)
            K_beta = 1/(6*ln2) * (beta**2)/((1-beta)**3) \
                      * (lamb_term ** beta) \
                      * (math.log(lamb_term + math.e**2)) **3

        except FloatingPointError:
            K_beta = 1/(6*ln2) * (beta**2)/((1-beta)**3) \
                      * (2 ** (beta * (2 + max2min_f) ) ) \
                      * ( (2 + max2min_f)/math.log2(math.e) ) **3

    key_rate =  asym_rate \
                - ln2/2 * beta * (math.log2(9) + math.sqrt(2 + var_f)) ** 2 \
                + 1/(beta * n) * ( (1+beta)*epsi_term + (1+2*beta)*log_prob) \
                - K_beta    
    return key_rate if key_rate > 0 else 0

### Make meshgrid for 3D plot/2D colormap
if SCAN == 'beta':
    X, Y = np.meshgrid(Ns, BETAs)
elif SCAN == 'nup':
    X, Y = np.meshgrid(Ns, NU_PRIMEs)


for class_ in CLASSES:
    input_ = CLASS_INPUT_MAP[class_]

    # ZERO_TOLs = [1e-9, 1e-5, 1e-3]    # Tolerant error for winning prob
    ZERO_TOLs = [1e-9]

    for zero_tol in ZERO_TOLs:
        ### Specify the file record the data
        HEAD = 'lr_bff21'
        CLASS = f'class_{class_}' if class_ != 'CHSH' else 'CHSH'
        INPUT = f'xy_{input_}'
        QUAD = 'M_12'
        if class_ =='CHSH':
            DATA_FILE = f'{HEAD}-{CLASS}-{INPUT}-{QUAD}-wtol_1e-04.csv'
        else:
            DATA_FILE = f'{HEAD}-{CLASS}-{INPUT}-{QUAD}-wtol_1e-04-ztol_{zero_tol:.0e}.csv'
        DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)

        ### Get the maximum winnin probability
        with open(DATA_PATH) as file:
            max_p_win = float(file.readlines()[1])

        ### Load data
        data_mtf = np.genfromtxt(DATA_PATH, delimiter=",", skip_header = 3)
        #print(data_mtf)

        ### Choose min-tradeoff function with expected winning prob
        for i in range(1):
        # for i in range(data_mtf.shape[0]):
            win_prob, asym_rate, lambda_ = data_mtf[i][:3]
            # win_prob, asym_rate, lambda_ = data_mtf[:3]
            c_lambda = data_mtf[i][-1]
            # c_lambda = data_mtf[-1]
            if class_ != 'CHSH':
                lambda_zeros = data_mtf[i][3:-1]
                # lambda_zeros = data_mtf[3:-1]
                c_lambda -= sum(lambda_zeros)*zero_tol

            ### Construct key rate function with fixed param (only leave n, beta tunable)
            kr_func = partial(key_rate_testing, gamma = GAMMA, asym_rate = asym_rate,
                                lambda_ = lambda_, c_lambda = c_lambda,
                                zero_tol = zero_tol, zero_class = class_, max_p_win = max_p_win)
            
            def opt_nup(n, beta, nu_prime_arr = NU_PRIMEs):
                return np.max(np.array([kr_func(n = n, beta = beta, nu_prime = nu_prime) \
                                        for nu_prime in nu_prime_arr]))
            
            def opt_beta(n, nu_prime, beta_arr = BETAs):
                return np.max(np.array([kr_func(n = n, nu_prime = nu_prime, beta = beta) \
                                        for beta in beta_arr]))           

            if SCAN == 'beta' or SCAN == 'nup':
                Z = []
                if SCAN == 'beta':
                    for x_arr, y_arr in zip(X, Y):
                        Z.append(Parallel(n_jobs=N_JOB, verbose = 0)(
                                        delayed(opt_nup)(x, y) for x,y in zip(x_arr, y_arr)))
                else:
                    for x_arr, y_arr in zip(X, Y):
                        Z.append(Parallel(n_jobs=N_JOB, verbose = 0)(
                                        delayed(opt_beta)(x, y) for x,y in zip(x_arr, y_arr)))
                Z = np.array(Z)
            
                ### 2D colormap
                fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
                if SCAN == 'beta':
                    plot = plt.pcolormesh(X, Y, Z, cmap=cm.coolwarm, shading='gouraud')
                    YLABEL = r'$\displaystyle \beta$'
                    plt.yscale("log")
                else:
                    plot = plt.pcolormesh(X, Y, Z, cmap=cm.coolwarm, shading='gouraud')
                    YLABEL = r"$\displaystyle \nu'$"

                cbar = plt.colorbar(plot)
                cbar.ax.tick_params(labelsize=22)
                # TITLE = r'$\displaystyle w_{exp}$='+f'{win_prob:.4f}'.rstrip('0') \
                #         + r'$\delta_{zero}$='+f'{zero_tol:.0e}'
                # plt.title(TITLE, pad=20, x=0.55)
                plt.ylabel(YLABEL)
                XLABEL = r'$\displaystyle n$'+' (number of rounds)'
                plt.xlabel(XLABEL)
                plt.xscale("log")

                ### Apply the graphic settings
                plt.subplots_adjust(**SUBPLOT_PARAM)
                plt.tight_layout(pad=2)

            elif SCAN == 'both':
                fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE, dpi=DPI)

                #### Scan beta with opt. nu_prime
                X, Y = np.meshgrid(Ns, BETAs)
                Z = []
                for x_arr, y_arr in zip(X, Y):
                        Z.append(Parallel(n_jobs=N_JOB, verbose = 0)(
                                        delayed(opt_nup)(x, y) for x,y in zip(x_arr, y_arr)))
                Z = np.array(Z)
                axs[0].pcolormesh(X, Y, Z, cmap=cm.coolwarm, shading='gouraud')
                axs[0].set_yscale("log")
                axs[0].set_ylabel(ylabel=r'$\displaystyle \beta$')
                SUBTITLE = "(a) Scanning " + r"$\displaystyle \nu'$" + \
                            " at optimal "+ r"$\displaystyle \beta$"
                axs[0].set_title(SUBTITLE, y=-0.22)

                #### Scan nu_prime with opt. beta
                X, Y = np.meshgrid(Ns, NU_PRIMEs)
                Z = []
                for x_arr, y_arr in zip(X, Y):
                        Z.append(Parallel(n_jobs=N_JOB, verbose = 0)(
                                        delayed(opt_beta)(x, y) for x,y in zip(x_arr, y_arr)))
                Z = np.array(Z)
                pcm = axs[1].pcolormesh(X, Y, Z, cmap=cm.coolwarm, shading='gouraud')
                axs[1].set_ylabel(ylabel=r"$\displaystyle \nu'$")
                SUBTITLE = "(a) Scanning " + r"$\displaystyle \beta$" + \
                            " at optimal "+ r"$\displaystyle \nu'$"
                axs[1].set_title(SUBTITLE, y=-0.22)
                
                XLABEL = r'$\displaystyle n$'+' (number of rounds)'
                for ax in axs:
                    ax.set_xlabel(xlabel=XLABEL)
                    ax.set_xscale("log")
                    # ax.tick_params(axis='x', which='major', pad=15)
                
                TITLE = 'Finite rate of blind randomness (' + \
                        r'$\displaystyle w_{exp}$='+f'{win_prob:.4f}'.rstrip('0') + ')'
                fig.suptitle(TITLE, fontsize=30)

                cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
                fig.colorbar(pcm, cax = cbar_ax, shrink=0.5)
                # fig.colorbar(pcm, ax = axs[:], shrink=0.8, location='right')
                fig.subplots_adjust(**SUBPLOT_PARAM)

            ### Save file
            if SAVE:
                OUT_DIR = os.path.join(TOP_DIR, 'figures/corrected_FER/h_map')
                if SCAN == 'beta':
                    COMMON = 'bscan'
                elif SCAN == 'nup':
                    COMMON = 'nupscan'
                elif SCAN == 'both':
                    COMMON = 'bothscan'
                WIN_EXP = f'w_{win_prob*10000:.0f}'.rstrip('0')
                EPS = f'eps_{EPSILON:.0e}'
                W_TOL = f'wtol_{WIN_TOL:.0e}'
                GAM = f'gam_{GAMMA:.0e}'
                Z_TOL = f'ztol_{zero_tol:.0e}'
                TAIL= '2'
                OUT_NAME = f'{COMMON}-{CLASS}-{WIN_EXP}-{INPUT}-{EPS}-{W_TOL}-{Z_TOL}-{GAM}-{QUAD}-{TAIL}'
                FORMAT = 'png'
                OUT_FILE = f'{OUT_NAME}.{FORMAT}'
                OUT_PATH = os.path.join(OUT_DIR, OUT_FILE)

                plt.savefig(OUT_PATH, format = FORMAT, bbox_inches='tight')
            
            if SHOW:
                plt.show()
