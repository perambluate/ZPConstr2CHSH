import numpy as np
import math
import sys

### Constant parameters
CHSH_W_Q = (2 + math.sqrt(2) )/4    # CHSH win prob
EPSILON = 1e-12                     # Smoothness of smooth min entropy (related to secrecy)
WIN_TOL = 1e-4                      # Tolerant error for win prob
DIGIT = 9                           # Accurate digit for the computation

### Finite extractable rate for blind randomness extraction with testing
def fin_rate_testing(n, beta, nu_prime, gamma, asym_rate, lambda_, c_lambda,
                        epsilon = EPSILON, win_tol = WIN_TOL, zero_tol = 0,
                        zero_class = 'chsh', max_p_win = CHSH_W_Q):
    
    ln2 = math.log(2)
    gamma_0 = (1-gamma)/gamma
    nu_0 = 1/(2*gamma) - gamma_0*nu_prime
    
    D = c_lambda**2 + (lambda_/(2*gamma))**2 - \
        gamma_0/gamma * lambda_**2 * (1-nu_prime)*nu_prime
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
        if zero_class != 'chsh':
            n_zero = int(zero_class[0])
            log_prob += n_zero * math.log2(1 - epsi_zero)
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

def bin_shannon_entropy(p: float):
    assert p >=0 and p <= 1, 'Input should be a valid probability!'
    if p ==0 or p == 1:
        return 0
    
    return - p*math.log2(p) - (1-p)*math.log2(1-p)

def shannon_entropy(Prob, digit=DIGIT):
    Prob = np.array(Prob)
    assert ((Prob >= 0).all() and (Prob <= 1).all() and \
            round(np.sum(Prob), digit) == 1),\
            'Input should be a valid probability distribution!'
    
    zero_indexes = np.argwhere(Prob == 0)
    for index in zero_indexes:
        Prob[index] = sys.float_info.min
    log_p = np.log2(Prob)
    entropy = -np.sum(Prob * log_p)
    
    return entropy

def inp_rand_consumption(gamma, p_xy):
    return bin_shannon_entropy(gamma) + gamma*shannon_entropy(p_xy)

def plot_settings(title: int, tick: int, legend: int, linewidth = 1,
                  family = "serif", font = "Latin Modern Roman"):
    plot_settings = {
        ## Default font (family and size)
        "font.family": family,
        "font.sans-serif": font,
        "font.size": legend,
        "figure.titlesize": title,
        "axes.labelsize": title,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "legend.fontsize": legend,
        ## Use Latex
        "text.usetex": True,
        ## line
        "lines.linewidth": linewidth
    }
    return plot_settings

RAND_TYPE = ['blind', 'one', 'two']

def top_dir(rand_type: str):
    assert rand_type in RAND_TYPE, "Wrong 'rand_type' when calling top_dir()!"
    if rand_type == 'blind':
        return './blindRandomness'
    elif rand_type == 'one':
        return './onePartyRandomness'
    elif rand_type == 'two':
        return './twoPartyRandomness'

def cls_inp_map(rand_type: str):
    assert rand_type in RAND_TYPE, "Wrong 'rand_type' when calling cls_inp_map()!"
    if rand_type == 'blind':
        return {'chsh':'00', '1':'01', '2a':'11', '2b':'01', '2b_swap':'11',
                '2c':'10', '3a':'11', '3b':'10'}
    elif rand_type == 'one':
        return {'chsh':'0', '1':'0', '2a':'1', '2b':'1', '2b_swap':'0',
                '2c':'0', '3a':'0', '3b':'1'}
    elif rand_type == 'two':
        return {'chsh':'00', '1':'01', '2a':'11', '2b':'01', '2c':'01',
                '3a':'00', '3b':'01'}
    
def cls_max_win_map():
    return {'chsh':0.8535, '1':0.8294, '2a':0.8125, '2b':0.8125, '2b_swap':0.8125,
            '2c':0.8039, '3a':0.7951, '3b':0.7837}