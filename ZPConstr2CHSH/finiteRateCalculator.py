"""Finite rate calculator for DIRE protocol with a primary predicate
    This script uses generalized entropy accumulation (arXiv:2203.04989) to compute finite rates
"""
import numpy as np
from scipy.stats import binom
from math import log, log2, sqrt, exp, ceil, floor
from functools import partial
from joblib import Parallel, delayed
import sys

DIGIT = 10       # Accurate digit for the computation

def bin_shannon_entropy(p):
    """Binary Shannon entropy
        - p: float, probability of an outcome of binary RV
    """
    if p < 0. or p > 1.:
        raise ValueError(f'Invalid input {p}. Expect a float in [0, 1].')
    if p == 0. or p == 1.:
        return 0.
    
    return - p*log2(p) - (1-p)*log2(1-p)

def shannon_entropy(prob, digit = DIGIT):
    """Shannon entropy of RV with arbitrary num of outcomes
        - prob: high dimensional list/np-array, probability distribution of the RV
        - digit: int, digit of accuracy to check normalization
    """
    prob = np.array(prob)
    if not ((prob >= 0).all() and (prob <= 1).all() and \
            round(np.sum(prob), digit) == 1):
        raise ValueError('Input should be a valid probability distribution!')

    zero_indexes = np.argwhere(prob == 0)
    for index in zero_indexes:
        prob[index] = sys.float_info.min
    log_p = np.log2(prob)
    entropy = -np.sum(prob * log_p)
    
    return entropy

def inp_rand_consumption(gamma, p_xy):
    """Compute randomness consmption to gen inputs in DIRE protocols
    """
    return bin_shannon_entropy(gamma) + gamma*shannon_entropy(p_xy)

def fin_rate(n, beta, nu_prime, gamma, asym_rate, d_K, lambda_, epsilon,
             max_win, min_win, inp_dist, WP_check_direction = 'lb'):
    """Finite extractable rate of DI randomness expansion
            (taking input randomness consumption into account)
        - n: int or float, num of rounds of the nonlocal game to perform
        - beta: float in interval (0, 1), related to alpha of alpha-Renyi entropy
        - nu_prime: float in the interval [0, 1], a free tunable constant in the crossover
                    min-tradeoff function
        - gamma: float in interval (0, 1], ratio of num of testing rounds to num of
                 total rounds
        - asym_rate: float, asymptotic rate of the DIRE protocol with specified settings
        - d_K: int, dimension of the system(s) to gen randomness
        - lambda_: list of float, vec of dual vars of the SDP to compute asymptotic rate
        - epsilon: float, smoothness of the smooth min-entropy, related to security param
        - max_win (min_win): float, maximum (minimum) winning probability of the nonlocal game
                             over the quantum set
        - inp_dist: high dimensional list/np-array, input probabilities
        - WP_check_direction: str ('lb', 'ub'), check WP satisfying the lower-bound ('lb') or
                              upper-bound ('ub') threshold
    """
    
    if not WP_check_direction in ('lb', 'ub'):
        raise ValueError(f'Invalid WP_check_direction. Should be "lb" or "ub" while got {WP_check_direction}')
    
    loge = log2(exp(1))
    ln2 = log(2)
    
    gam_inv = 1/gamma
    gamma_0 = (1-gamma)*gam_inv
    nu_0 = 1/(2*gamma) - gamma_0*nu_prime

    log_prob = log2(epsilon)
    log_dK = log2(d_K)
    
    if WP_check_direction == 'lb':
        max2min_f = lambda_ * (gam_inv - gamma_0*nu_prime - min_win)
    else:
        max2min_f = lambda_ * (gamma_0*nu_prime - max_win)
    
    with np.errstate(over='raise'):
        try:
            lamb_term = 2 ** (2*log_dK + max2min_f)
            K_beta = (lamb_term**beta) * (log(lamb_term + exp(2)))**3

        except FloatingPointError:
            try:
                K_beta = (2 ** (beta * (2*log_dK + max2min_f))) \
                            * ((2*log_dK + max2min_f)/loge)**3
            except FloatingPointError:
                return 0
    
    D = (lambda_**2) * gam_inv * (gam_inv/4 - gamma_0*(1-nu_prime)*nu_prime)
    if nu_0 > max_win:
        var_f = D - (lambda_*(max_win - nu_0))**2
    elif nu_0 < min_win:
        var_f = D - (lambda_*(min_win - nu_0))**2
    else:
        var_f = D

    if sqrt(1 - epsilon**2) == 1:
        epsi_term = log2(1/2*epsilon**2)
    else:
        epsi_term = log2(1 - sqrt(1 - epsilon**2) )

    Delta_fin = ln2/2 * beta * ( log2(2*(d_K**2) + 1) + sqrt(2+var_f) ) ** 2 \
                - 1/(beta*n) * ( (1+beta)*epsi_term + (1+2*beta)*log_prob ) \
                + 1/(6*ln2) * (beta**2)/((1-beta)**3) * K_beta
    
    Delta_inp = inp_rand_consumption(gamma, inp_dist)

    fin_rate =  (1-gamma) * asym_rate - Delta_fin - Delta_inp
    
    return fin_rate if fin_rate > 0 else 0

def completeness_wplb(n, gamma, wtol, wexp):
    """Completeness for protocols with a winning-probability lower-bound check
    """
    k = floor(n*gamma*(wexp-wtol))
    p= gamma*wexp
    return binom.cdf(k, floor(n), p)

def completeness_lpub(n, gamma, wtol, wexp):
    """Completeness for protocols with a lossing-probability upper-bound check
    """
    k = floor(n*gamma*(1-wexp+wtol))
    p= gamma*(1-wexp)
    return 1 - binom.cdf(k, floor(n), p)

class finiteRateCalculator():
    """Class to compute finite rates for DIRE protocols w/o ZP constraints
        - rand_type: str (loc, glo, bli), type of DI randomness to compute
        - finrate_params: dict with the following keys:
            - kappa: str ('1', '2a', '2b', '2c', '3a', '3b') or None, class of the ZP constraints
            - asym_rate: float, asymptotic rate of the DIRE protocol with specified settings
            - WP_check_direction: str ('lb', 'ub'), check WP satisfying the lower-bound ('lb') or
                upper-bound ('ub') threshold
            - inp_prob: high dimensional list/np-array, input probabilities
            - n: int or float, num of rounds of the nonlocal game to perform
            - gamma: float in interval (0, 1], ratio of num of testing rounds to num of total rounds
            - wexp: float, expected winning probability
            - wtol: float in interval (0, 1), tolerant level for WP constraint
            - ztol: float in interval (0, 1), tolerant level for ZP constraint
            - eps: float, smoothness of the smooth min-entropy, related to security param
            - lambda: list of float, vec of dual vars of the SDP to compute asymptotic rate
            - wp_Qbound: dict with the keys:
                - max (min): float, maximum (minimum) winning probability of the nonlocal game over the quantum set
        - nthreads: int, num of threads to perform the computation parallelly
    """
    def __init__(self, rand_type,
                 finrate_params = {'kappa': None, 'asym_rate': None,
                                   'WP_check_direction': 'lb',
                                   'inp_prob': [[1/4, 1/4],[1/4, 1/4]],
                                   'n': None, 'gamma': None, 'wexp': None,
                                   'wtol': None, 'ztol': 0, 'eps': None,
                                   'lambda': None},
                 wp_Qbound = {'max': None, 'min': None},
                 nthreads = 1):
        
        if rand_type not in ('loc', 'glo', 'bli'):
            raise ValueError(f'Invalid rand_type {rand_type}. Expect "loc", "glo", or "bli".')
        self.rand_type = rand_type

        if rand_type == 'glo':
            self.d_K = 4
        else:
            self.d_K = 2

        if finrate_params['WP_check_direction'] not in ('lb', 'ub'):
            raise ValueError(f'Invalid WP_check_direction {finrate_params["WP_check_direction"]}. Expect "lb" or "ub".')
        self.finrate_params = dict()
        self.finrate_params.update(finrate_params)

        if ('max' not in wp_Qbound) or ('min' not in wp_Qbound):
            raise ValueError(f'Invalid wp_Qbound {wp_Qbound}. wp_Qbound must contain "min" and "max" keys.')
        self.wp_Qbound = dict()
        self.wp_Qbound.update(wp_Qbound)

        self.nthreads = nthreads
        self.fin_rate_func = None
        self.comp_func = None

    def update_params(self, finrate_params = None, wp_Qbound = None):
        if (finrate_params is None) and (wp_Qbound is None):
            print('Null input, no update.')
            return
        
        if finrate_params:
            self.finrate_params.update(finrate_params)
        
        if wp_Qbound:
            self.wp_Qbound.update(wp_Qbound)

    def _init_fin_rate_func(self):
        """Init finite rate function according to the protocol params
        """
        if not self.finrate_params['asym_rate']:
            raise ValueError(f'No asym_rate. Plz give a valid asym_rate.')
        if not self.finrate_params['lambda']:
            raise ValueError(f'No lambda. Plz give a valid lambda.')
        if not self.finrate_params['eps']:
            raise ValueError(f'No eps. Plz give a valid eps.')
        if not self.wp_Qbound['max']:
            raise ValueError(f'No max wp_Qbound. Plz give a valid max wp_Qbound.')
        if not self.wp_Qbound['min']:
            raise ValueError(f'No min wp_Qbound. Plz give a valid min wp_Qbound.')

        self.fin_rate_func = partial(fin_rate,
                                     asym_rate = self.finrate_params['asym_rate'],
                                     d_K = self.d_K,
                                     lambda_ = self.finrate_params['lambda'],
                                     epsilon = self.finrate_params['eps'],
                                     max_win = self.wp_Qbound['max'],
                                     min_win = self.wp_Qbound['min'],
                                     inp_dist = self.finrate_params['inp_prob'],
                                     WP_check_direction = self.finrate_params['WP_check_direction'])

    def fin_rate(self, beta, nu_prime):
        """Compute finite rate according to the protocol params and given beta, nu_prime
        """
        if not self.finrate_params['n']:
            raise ValueError(f'No n. Plz give a valid n.')
        if not self.finrate_params['gamma']:
            raise ValueError(f'No gamma. Plz give a valid gamma.')

        if not self.fin_rate_func:
            self._init_fin_rate_func()
        
        return self.fin_rate_func(n = self.finrate_params['n'],
                                  beta = beta, nu_prime = nu_prime,
                                  gamma = self.finrate_params['gamma'])

    def fin_rate_opt_params(self, n, beta_list, nup_list, gam_list):
        """Given n, and lists of beta, nu_prime, and gamma, return the finite rate with optimal
            params (the ones in the list that give the best rate): beta, nu_prime, gamma
        """
        if not self.fin_rate_func:
            self._init_fin_rate_func()
        
        finrate_list = np.array([[self.fin_rate_func(n = n, beta = beta,
                                                     nu_prime = nup, gamma = gamma) \
                                for beta in beta_list for nup in nup_list] \
                                for gamma in gam_list])
        max_id =  np.argmax(finrate_list)
        opt_rate = finrate_list.flatten()[max_id]
        opt_gam = gam_list[ceil((max_id+1) / (len(nup_list)*len(beta_list)))-1]
        id_temp = max_id % (len(nup_list)*len(beta_list))
        opt_beta = beta_list[ceil((id_temp+1)/len(nup_list))-1]
        id_temp = id_temp % len(nup_list)
        opt_nup = nup_list[id_temp]

        return opt_rate, opt_gam, opt_beta, opt_nup
    
    def fin_rate_opt_epscom_bound(self, n, epscom_bound, beta_list, nup_list, gam_list):
        """Given n, and lists of beta, nu_prime, and gamma, compute the best finite rate over
            the params that satisfying the completeness upper bound; return the optimal rate
            and the associated params: gamma, beta, and nu_prime; if no params acheive a
            completeness does not exceed the threshold, then return an all-zero tuple
        """
        if not self.comp_func:
            self._init_comp_func()
        
        valid_gam_list = [gam for gam in gam_list if self.comp_func(n, gam) <= epscom_bound]

        if len(valid_gam_list):
            opt_rate, opt_gam, opt_beta, opt_nup = \
                self.fin_rate_opt_params(n, beta_list, nup_list, valid_gam_list)
            
            return opt_rate, opt_gam, opt_beta, opt_nup
        else:
            return 0,0,0,0

    def opt_fin_rate_for_ns(self, n_list, beta_list, nup_list, gam_list, epscom_bound = -1):
        """Given a list of n, and lists of beta, nu_prime, and gamma, return the finite rate with
            optimal params (beta, nu_prime, gamma) for each n in `n_list`; if epscom_bound is
            specified validily, then the optimization is conducted over the params satisfying the
            completeness upper bound
        """
        if not self.fin_rate_func:
            self._init_fin_rate_func()
        
        num_n = len(n_list)
        opt_fr_list = np.zeros(num_n)
        opt_gam_list = np.zeros(num_n)
        opt_beta_list = np.zeros(num_n)
        opt_nup_list = np.zeros(num_n)

        func_params = dict(beta_list = beta_list,
                           nup_list = nup_list,
                           gam_list = gam_list)
        
        if epscom_bound > 0:
            if not self.comp_func:
                self._init_comp_func()
            
            n_idx = num_n - sum([self.comp_func(n, 1.) <= epscom_bound for n in n_list])
            if num_n == n_idx:
                return opt_fr_list, opt_gam_list, opt_beta_list, opt_nup_list
            fr_func = self.fin_rate_opt_epscom_bound
            func_params['epscom_bound'] = epscom_bound
        else:
            n_idx = 0
            fr_func = self.fin_rate_opt_params

        fr_params = Parallel(n_jobs=self.nthreads, verbose = 0)(
                                    delayed(fr_func)(n = n, **func_params) for n in n_list[n_idx:num_n])
        
        fr_params = list(zip(*fr_params))
        opt_fr_list[n_idx:num_n] = np.array(fr_params[0])
        opt_gam_list[n_idx:num_n] = np.array(fr_params[1])
        opt_beta_list[n_idx:num_n] = np.array(fr_params[2])
        opt_nup_list[n_idx:num_n] = np.array(fr_params[3])

        return opt_fr_list, opt_gam_list, opt_beta_list, opt_nup_list
    
    def _init_comp_func(self):
        """Init completeness function according to the given protocol params
        """
        if not self.finrate_params['wtol']:
            raise ValueError(f'No wtol. Plz give a valid wtol.')
        if not self.finrate_params['wexp']:
            raise ValueError(f'No wexp. Plz give a valid wexp.')
        
        if self.finrate_params['WP_check_direction'] == 'lb':
            self.comp_func = partial(completeness_wplb,
                                     wtol = self.finrate_params['wtol'],
                                     wexp = self.finrate_params['wexp'])
        else:
            self.comp_func = partial(completeness_lpub,
                                     wtol = self.finrate_params['wtol'],
                                     wexp = self.finrate_params['wexp'])

    def completeness(self):
        """Compute the completeness according to the protocol params 
        """
        if not self.finrate_params['n']:
            raise ValueError(f'No n. Plz give a valid n.')
        if not self.finrate_params['gamma']:
            raise ValueError(f'No gamma. Plz give a valid gamma.')
        
        if not self.comp_func:
            self._init_comp_func()
        
        return self.comp_func(n = self.finrate_params['n'],
                              gamma = self.finrate_params['gamma'])
    
    def completeness_opt_param(self, n, beta_list, nup_list, gam_list, outrate = False,
                               outbeta = False, outnup = False, outgam = False):
        """Given n, and lists of beta, nu_prime, and gamma, compute the completeness with the
            protocol params that optimize the finite rate
            - outrate, outbeta, outnup, outgam: bool, control the info contained in output
        """
        if not self.comp_func:
            self._init_comp_func()
        
        opt_rate, opt_gam, opt_beta, opt_nup = \
            self.fin_rate_opt_params(n, beta_list, nup_list, gam_list)
        
        comp = self.comp_func(n = n, gamma = opt_gam)
        if opt_rate <= 0:
            comp = -1
        
        if any([outrate, outbeta, outnup, outgam]):
            param_dict = dict()
            if outrate:
                param_dict['rate'] = opt_rate
            if outbeta:
                param_dict['beta'] = opt_beta
            if outnup:
                param_dict['nup'] = opt_nup
            if outgam:
                param_dict['gam'] = opt_gam
            
            return (comp, param_dict)
        else:
            return comp

    def opt_completeness_for_ns(self, n_list, beta_list, nup_list, gam_list, outrate = False,
                                outbeta = False, outnup = False, outgam = False):
        """Given a list of n, and lists of beta, nu_prime, and gamma, compute the completeness
            with the protocol params that optimize the finite rate for each n in `n_list`
            - outrate, outbeta, outnup, outgam: bool, control the info contained in output
        """
        if not self.comp_func:
            self._init_comp_func()

        comp_param_list = Parallel(n_jobs=self.nthreads, verbose = 0)(
                                    delayed(self.completeness_opt_param)(
                                                n = n,
                                                beta_list = beta_list,
                                                nup_list = nup_list,
                                                gam_list = gam_list,
                                                outrate = outrate,
                                                outbeta = outbeta,
                                                outnup = outnup,
                                                outgam = outgam) \
                                            for n in n_list)
        
        if any([outrate, outbeta, outnup, outgam]):
            comp_param_list = list(zip(*comp_param_list))
            comp_list = comp_param_list[0]
            param_list = comp_param_list[1]

            return (comp_list, param_list)
        else:
            return comp_param_list

