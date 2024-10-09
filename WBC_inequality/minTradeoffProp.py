"""
    Functions to compute the properties of the min-tradeoff function,
        i.e., Max(f), Min_Sigma(f), and Var_Sigma(f).
    Note that we ignore the c_lambda term in both Max(f) and Min_Sigma(f)
        since the quantity we need in the end is Max(f) - Min_Sigma(f) where
        c_lambda will be canceled out in the substraction.
"""
import ncpol2sdpa as ncp
import numpy as np
import sys

sys.path.append('.')
from DIStrategyUtils.QuantumStrategy import *

class MinTradeoffFunction():
    """Class to compute properties of min-tradeoff function required for entropy accumulation
        - scenario: Bell scenario, e.g., [(2,2), (2,2,3)] repr a bipartite scenario where
                    first party has two meas to choose s.t. each get binary outcome;
                    second party has three meas s.t. the first two give binary outcome and
                    the third one is a three-outcome meas
        - lambda_vec: lambda vec, an vec of the dual vars corresponding to the score constraints
        - c_lambda: the constant term in the Lagrange dual function
        - num_score: num of the score constraints, equal to the length of the dual vars
        - inp_dist: a high dimentional np-array repr the input probabilities
        - score_coeffs: a list of np-arrays that record the coeffs of the scores
        - npa_level: level of NPA hierarchy
        - solver_config: SDP solver config
        - verbose: verbose level
    """
    def __init__(self, scenario, lambda_vec, c_lambda, score_coeffs = [],
                 npa_level = 2, solver_config = ['mosek'], verbose = 0):
        self.scenario = scenario
        self.lambda_vec = lambda_vec
        self.c_lambda = c_lambda
        self.num_score = len(lambda_vec)
        self.score_coeffs = score_coeffs
        self.npa_level = npa_level
        self.solver_config = solver_config
        self.verbose = verbose

    def __genScoreExpr(self, P, coeff):
        """Gen score expression in terms of POVM elements
        """
        out_combines = np.array(np.meshgrid(*self.scenario)).T
        inp_config = tuple(len(self.scenario[i]) for i in range(len(self.scenario)))
        inp_combines = genCombinations(inp_config)

        expr = 0
        for inps in inp_combines:
            out_comb = list(itertools.product(*[list(range(i)) for i in out_combines[inps]]))
            for outs in out_comb:
                expr += coeff[inps][outs] * P([*outs], [*inps])

        return expr
    
    def Max(self, nu_vec, gamma):
        """Return maximum of the min-tradeoff function
        """
        max_f = 1/gamma * np.max(self.lambda_vec) + (1-1/gamma) * np.dot(self.lambda_vec, nu_vec)
        if self.verbose:
            print(f'Max(f): {max_f}')
        return max_f

    def Min_Sigma(self):
        """Reutrn minimum of the min-tradeoff function over quantum realizable strategies
        """
        P = ncp.Probability(*self.scenario)

        obj = 0
        for s in range(self.num_score):
            obj += self.lambda_vec[s] * self.__genScoreExpr(P, self.score_coeffs[s])

        sdp = ncp.SdpRelaxation(P.get_all_operators(), verbose=max(self.verbose-3, 0))
        sdp.get_relaxation(level=self.npa_level, objective=obj, substitutions = P.substitutions)
        sdp.solve(*self.solver_config)
        min_sigma_f = sdp.primal

        if self.verbose:
            if self.verbose >= 2:
                print('Status\tPrimal\tDual')
                print(f'{sdp.status}\t{sdp.primal}\t{sdp.dual}')
            print(f'Min_Sigma(f): {min_sigma_f}')

        return min_sigma_f

    def Var_Sigma(self, nu_vec, gamma):
        """Return the variance of the min-tradeoff function over quantum realizable strategies
        """
        gam_inv = 1/gamma
        lambda_dot_nu = np.dot(self.lambda_vec, nu_vec)
        P = ncp.Probability(*self.scenario)

        lambda_dot_omega = 0
        lambda_square_dot_omega = 0
        for s in range(self.num_score):
            lambda_dot_omega += self.lambda_vec[s] * self.__genScoreExpr(P, self.score_coeffs[s])
            lambda_square_dot_omega += ((self.lambda_vec[s])**2) * \
                                        self.__genScoreExpr(P, self.score_coeffs[s])

        obj = -(lambda_dot_omega)**2 + gam_inv*lambda_square_dot_omega + \
                2*(1-gam_inv)*lambda_dot_nu*lambda_dot_omega
        sdp = ncp.SdpRelaxation(P.get_all_operators(), verbose=max(self.verbose-3, 0))
        sdp.get_relaxation(level=self.npa_level, objective=-obj, substitutions = P.substitutions)
        sdp.solve(*self.solver_config)
        var_sigma_f = (gam_inv-1)*(lambda_dot_nu)**2 - sdp.primal

        if self.verbose:
            if self.verbose >= 2:
                print('Status\tPrimal\tDual')
                print(f'{sdp.status}\t{sdp.primal}\t{sdp.dual}')
            print(f'Var_Sigma(f): {var_sigma_f}')

        return var_sigma_f
    

if __name__ == '__main__':
    LEVEL = 4                       # NPA relaxation level
    VERBOSE = 1                     # Relate to how detail of the info will be printed
    N_WORKER_SDP = 8                # Number of threads for solving a single SDP
    PRIMAL_DUAL_GAP = 1e-5          # Allowable gap between primal and dual
    SOLVER_CONFIG = ['mosek', {'dparam.intpnt_co_tol_rel_gap': PRIMAL_DUAL_GAP,
                               'iparam.num_threads': N_WORKER_SDP,
                               'iparam.infeas_report_auto': 1,
                               'iparam.infeas_report_level': 10}]
    configA = [2,2]
    configB = [2,2]
    SCENARIO = (configA, configB)
    INP_CONFIG = tuple(len(SCENARIO[i]) for i in range(len(SCENARIO)))
    INP_PROB = np.ones(INP_CONFIG)/np.prod(INP_CONFIG)
    ### Get data to compute finite rate
    MTFDATAPATH = './WBC_inequality/data/three_scores-wbc-one-wtol_1e-04-M_18.csv'
    data_mtf = np.genfromtxt(MTFDATAPATH, delimiter=",", skip_header = 1)
    NUM_SCORE = 3
    ScoreCoeffs = []
    ScoreCoeffs = []
    lambda_vec = data_mtf[-(NUM_SCORE+1):-1]
    c_lambda = data_mtf[-1]
    print(f'lambda_vec: {lambda_vec}')
    for s in range(NUM_SCORE):
        Coeff = np.zeros((2,2,2,2))
        if s == 0:      # for (x,y) = (0,0)
            Coeff[0,0] = [[0,1], [1,0]]
        elif s == 1:    # for (x,y) = (0,1) or (1,0)
            Coeff[0,1] = [[0,1], [1,0]]
            Coeff[1,0] = [[0,1], [1,0]]
        else:           # for (x,y) = (1,1)
            Coeff[1,1] = [[1,0], [0,1]]

        ScoreCoeffs += [Coeff * INP_PROB]

    nu_vec = [1/4, 1/2, 1/4]
    gamma = 1e-2

    MTF = MinTradeoffFunction(SCENARIO, lambda_vec, c_lambda,
                              score_coeffs = ScoreCoeffs, npa_level = LEVEL,
                              solver_config = SOLVER_CONFIG, verbose = VERBOSE)
    max_f = MTF.Max(nu_vec, gamma)
    min_sigma_f = MTF.Min_Sigma()
    var_sigma_f = MTF.Var_Sigma(nu_vec, gamma)

