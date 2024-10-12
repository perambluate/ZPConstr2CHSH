"""
    Device-independent (DI) randomness calculator in bipartite Bell scenarios
    This module uses the BFF21 method proposed by P. Brown, H. Fawzi, and O. Fawzi,
        Quantum 8, 1445 (2024) and built on top of the package ncpol2sdpa by Peter Wittek,
        ACM Trans. Math. Softw. 41, 1 (2015)
    Three types of randomness are considered: local (loc), global (glo), blind (bli)
    By default, the output of the first party is chosen as the source for local randomness
"""
from sympy.physics.quantum.dagger import Dagger
from joblib import Parallel, delayed
import ncpol2sdpa as ncp
from math import log
import numpy as np
import chaospy
import time
from argparse import ArgumentParser
import sys

LOG2 = log(2)

BEST_ZPC_CLASS = {'loc': '2a', 'glo': '2a', 'bli': '3b'}

ZP_POSITION_MAP = {'3a': [([0,0],[0,0]),([1,1],[0,1]),([1,1],[1,0])],
                   '3b': [([0,0],[0,0]),([1,1],[0,0]),([1,0],[1,1])],
                   '2a': [([0,0],[0,0]),([1,1],[0,0])],
                   '2b': [([0,0],[0,0]),([1,1],[1,0])],
                   '2b_swap': [([0,0],[0,0]),([1,1],[0,1])],
                   '2c': [([0,0],[0,0]),([1,0],[1,1])],
                   '1' : [([0,0],[0,0])]}

ZPC_BEST_INP_MAP = {'loc': {'chsh': 0, '1': 0, '2a': 1, '2b': 1,
                            '2b_swap': 0, '2c': 0, '3a': 0, '3b': 1},
                    'glo': {'chsh': (0,0), '1': (0,1), '2a': (1,1), '2b': (0,1),
                            '2c': (0,1), '3a': (0,0), '3b': (0,1)},
                    'bli': {'chsh': (0,0), '1': (0,1), '2a': (1,1), '2b': (0,1),
                            '2b_swap': (1,1), '2c': (1,0), '3a': (1,1), '3b': (1,0)}}
def truncate(number, digit):
    """
        Truncate the floating number up to the given digit after point
        It will ignore any number after the digit
    """
    truncated_number = round(number, digit)
    if round(number, digit+1) < truncated_number:
        truncated_number -= 10 ** (-digit)
    return truncated_number

def inner_quad_obj_bli(POVMs, inputs, Zops, t_i):
    """
        The inner-quadrature objective function to compute 
          blind randomness with BFF21 method
        - POVMs : Alice and Bob's POVMs in CG form
        - inputs : Alice and Bob's inputs for the computation
        - Zops : Eve's operators
        - t_i : Nodes in Gauss-Radau quadrature
    """
    x_star, y_star = inputs
    A, B = POVMs
    povmA = [*A[x_star], 1-sum(A[x_star])]
    povmB = [*B[y_star], 1-sum(B[y_star])]
    num_out = [len(povmA), len(povmB)]
    obj = 0
    for a in range(num_out[0]):
        for b in range(num_out[1]):
            ab = a*num_out[0]+b
            obj += povmA[a]*povmB[b]*(Zops[ab] + Dagger(Zops[ab]) \
                    + (1-t_i)*Dagger(Zops[ab])*Zops[ab])
            obj += t_i * povmB[b]*(Zops[ab]*Dagger(Zops[ab]))
    return obj

def inner_quad_obj_loc(POVMs, inputs, Zops, t_i):
    """
        The inner-quadrature objective function to compute 
          local randomness with BFF21 method
        - POVMs : Alice and Bob's POVMs in CG form
        - x_star : Alice for the key generation
        - Zops : Eve's operators
        - t_i : Nodes in Gauss-Radau quadrature
    """
    A, _ = POVMs
    x_star, _ = inputs
    povmA = [*A[x_star], 1-sum(A[x_star])]
    num_out = len(povmA)
    obj = 0
    for a in range(num_out):
        obj += povmA[a]*(Zops[a] + Dagger(Zops[a]) + (1-t_i)*Dagger(Zops[a])*Zops[a])
        obj += t_i *(Zops[a]*Dagger(Zops[a]))
    return obj

def inner_quad_obj_glo(POVMs, inputs, Zops, t_i):
    """
        The inner-quadrature objective function to compute 
          global randomness with BFF21 method
        - A, B : Alice and Bob's measurements in CG form
        - inputs : Alice and Bob's inputs for the computation
        - Zops : Eve's operators
        - t_i : Nodes in Gauss-Radau quadrature
    """
    x_star, y_star = inputs
    A, B = POVMs
    povmA = [*A[x_star], 1-sum(A[x_star])]
    povmB = [*B[y_star], 1-sum(B[y_star])]
    num_out = [len(povmA), len(povmB)]
    obj = 0
    for a in range(num_out[0]):
        for b in range(num_out[1]):
            ab = a*2+b
            obj += povmA[a]*povmB[b]*(Zops[ab] + Dagger(Zops[ab]) \
                    + (1-t_i)*Dagger(Zops[ab])*Zops[ab])
            obj += t_i *(Zops[ab]*Dagger(Zops[ab]))
    return obj

OBJ_FUNC_MAP = {'bli': inner_quad_obj_bli,
                'loc': inner_quad_obj_loc,
                'glo': inner_quad_obj_glo}

class DIRandCalculator():
    """Class to compute asymptotic rates for DIRE protocols
        - scenario: list of tuples, Bell scenario specified in the following form:
                    [(num_out_meas1_party1, num_out_meas2_party1, ...),
                     (num_out_meas1_party2, ...), ...]
                    (note: in this module, the scenarios are bipartite)
        - P: object of `Probability` class defined in `physics_utils.py` in module `ncpol2sdpa`
        - Aops (Bops): list of Alice's (Bob's) meas POVM elements in CG form
                       (POVM elements are objects of `sympy.physics.quantum.operator.Operator`
                        or `sympy.physics.quantum.operator.HermitianOperator`)
        - rand_type: str (loc, glo, bli), type of DI randomness to compute
        - inputs: two-integer tuple, target inputs to gen randomness
        - substs: dict of `sympy.core.exp.Expr`, record the equivalant operators as a map
        - Zops: same type as Aops/Bops, list of Z operators in BFF21 method
        - num_Z: int, num of Z operators
        - npa_level: int, level of NPA hierarchy to approx quantum set
        - extra_monos: list of `sympy.core.exp.Expr`, use to build a NPA hierarchy beyond
                       integer levels
        - constr_dict: dict[str, sympy.core.exp.Expr], record the constraint names and their
                       corresponding expressions
        - eq_constr (ineq_constr): list of `sympy.core.exp.Expr`, list of equality (inequality)
                                   constraints
        - nthread_sdp: int, num of threads to construct the NPA hierarchy and solve SDP 
        - nthread_quad: int, num of threads to compute the terms in the quadrature parallelly
        - radau_quad_params: dict of the form:
                             {'n_quad': `int`, 'endpoint': `float`, 'keep_endpoint': `bool`},
                             record info related to the computation for Gauss-Radau quadrature
                             - n_quad: int, num of terms in the quadrature, default is 12
                                       (odd int is equivalant to the closer larger even int)
                             - endpoint: float, val of the right endpoint, default is 1, and
                                         the interval is [0,1]; set to val slightly smaller
                                         than 1 to avoid instability of solving SDP
                             - keep_endpoint: bool, default is `True`, set to `False` to avoid
                                              instability of solving SDP at endpoint
        - solver_config: list of the form: [`solver_name`, `solver_options`],
                         - solver_name: str
                         - solver_options: dict, specifying option names and vals
        - verbose: int, level of verbose logging
    """
    def __init__(self, configA, configB, rand_type, target_inp = (),
                 npa_level = 2, nthread_sdp = 1,  nthread_quad = 1,
                 radau_quad_params = {'n_quad': 12, 'endpoint': 1., 'keep_endpoint': True},
                 solver_config = ['mosek'], verbose = 1):
        
        self.scenario = [configA, configB]
        self.P = ncp.Probability(configA, configB)
        self.Aops, self.Bops =  self.P.parties
        assert rand_type in ('loc', 'glo', 'bli'), \
            'Invalid rand_type. Plz set as one of the followings: "loc", "glo", or "bli".'
        self.rand_type = rand_type

        self.inputs = target_inp
        if not self.inputs:
            print('Plz set the input before call other methods.')

        self.substs = self.P.substitutions
        if target_inp:
            self._init_Zops()
            self._init_substitutions()
        else:
            self.num_Z = 0
            self.Zops = []

        self.npa_level = npa_level
        self.extra_monos = []
        if target_inp:
            self._init_extramonomials()

        self.constr_dict = dict()
        self.eq_constr = []
        self.ineq_constr = []
        self.nthread_sdp = nthread_sdp
        self.nthread_quad = nthread_quad
        self.radau_quad_params = radau_quad_params
        self.solver_config = solver_config
        self.verbose = verbose
    
    def _init_Zops(self):
        self.num_Z = self.scenario[0][self.inputs[0]]
        if self.rand_type in ('glo', 'bli'):
            self.num_Z *= self.scenario[1][self.inputs[1]]
        self.Zops = ncp.generate_operators('Z', self.num_Z, hermitian=False)

    def _init_substitutions(self):
        """Set substitution for projectors and commutting operators
        """
        for op in self.P.get_all_operators():
            for z in self.Zops:
                self.substs.update({z*op: op*z, Dagger(z)*op: op*Dagger(z)})
    
    def _init_extramonomials(self):
        """Init extra monomials according to the randomness type;
            call set_extramonos to overwrite them with customized monomials
        """
        for a in ncp.flatten(self.Aops):
            for b in ncp.flatten(self.Bops):
                for z in self.Zops:
                    if self.npa_level <= 2:
                        self.extra_monos += [a*b*z, a*b*Dagger(z)]
                        if self.rand_type == 'loc':
                            self.extra_monos += [a*Dagger(z)*z, a*z*Dagger(z)]
                        else:
                            self.extra_monos += [a*b*Dagger(z)*z, a*b*z*Dagger(z)]
                    elif self.npa_level == 3 and (self.rand_type in ('glo', 'bli')):
                        self.extra_monos += [a*b*Dagger(z)*z, a*b*z*Dagger(z)]
    
    def set_inputs(self, inputs):
        """(Re-)Set target inputs
        """
        self.inputs = inputs
        self._init_Zops()
        self._init_substitutions()
        self._init_extramonomials()
   
    def set_extramonos(self, extra_monos):
        """(Re-)Set extra monomials
        """
        self.extra_monos = extra_monos

    def _build_expression_from_probabilities(self, coeff, marginal = ''):
        """Return an object of class `sympy.core.exp.Expr` built based on the coefficients
            corresponding to the probabilities
            - coeff: a high dimensional list/np-array, the coefficients corresponding to
                     the probabilities to build an expression in terms of POVM elements
            - marginal: str ('A', 'B'), use when the coefficients are specified corresponding
                        to the marginal probabilities
        """
        coeff = np.array(coeff)
        if marginal:
            assert marginal in self.P.labels, \
                f'The specified marginal "{marginal}" is not in the party labels' + \
                    'when calling _build_expression_from_probabilities'
            
            monos = [coeff[x][a] * self.P([a],[x],[marginal])
                     for a in range(coeff.shape[1]) for x in range(coeff.shape[0])]
        else:
            monos = [coeff[x, y][a, b] * self.P([a,b],[x,y])
                        for b in range(coeff.shape[3]) for a in range(coeff.shape[2])
                        for y in range(coeff.shape[1]) for x in range(coeff.shape[0])]
            
        return sum(monos)

    def add_constraint_by_coeff(self, _type, coeff, val, name = '', marginal = '', direction = ''):
        """Add a new constraint specified by the coefficients corresponding to probabilities
            - _type: str ('eq', 'ineq'), equality or inequality constraint
            - coeff: high dimensional list/np-array, the coefficients corresponding to
                     the probabilities to build an expression
            - val: float, constant in the constraint
            - name: str, name of the constraint
            - marginal: str ('A', 'B'), marginal of the probabilities that the coefficients relate to
            - direction: str ('<', '>'), inequality direction
        """
        expr = self._build_expression_from_probabilities(coeff, marginal)
        self.add_constraint(_type, expr, val, name, marginal, direction)

    def add_constraint(self, _type, expr, val, name = '', marginal = '', direction = ''):
        """Add a new constraint by giving the full expression and value 
            - _type: str ('eq', 'ineq'), equality or inequality constraint
            - expr: obj of `sympy.core.exp.Expr`, expression of the constraint to add
            - val: float, constant in the constraint
            - name: str, name of the constraint
            - marginal: str ('A', 'B'), marginal of the probabilities that the coefficients relate to
            - direction: str ('<', '>'), inequality direction

        """
        assert _type in ('eq', 'ineq'), \
                    f'Invalid type for constraint. Plz set as "eq" or "ineq"!'

        if _type == 'eq':
            if not name:
                name = f'eq{len(self.eq_constr)+1}'
            constr = expr - val
            self.eq_constr.append(constr)
        else:
            assert direction in ('<', '>'), \
                'Invalid direction for constraint. Plz set as one of the followings: ">" or "<"'

            if not name:
                name = f'ineq{len(self.ineq_constr)+1}'
            if direction == '>':
                constr = expr - val
            else:
                constr = val - expr
            self.ineq_constr.append(constr)
        
        constraint_detail = {'constr': constr, 'expr': expr, 'val': val, 'type': _type}
        if _type == 'ineq':
            constraint_detail['direction'] = direction
        
        self.constr_dict.update({name: constraint_detail})

    def get_constraint_names(self):
        """Print the names of all constraints
        """
        return self.constr_dict.keys()

    def _init_npa_relaxation(self):
        """Init NPA relaxation
        """
        ops = self.P.get_all_operators()+self.Zops
        self.sdp = ncp.SdpRelaxation(ops, verbose=max(self.verbose-2, 0),
                                    parallel = bool(self.nthread_sdp >= 2),
                                    number_of_threads = self.nthread_sdp)
        
        self.sdp.get_relaxation(level = self.npa_level, objective = 0,
                                substitutions = self.substs,
                                momentequalities = self.eq_constr,
                                momentinequalities = self.ineq_constr,
                                extramonomials = self.extra_monos)
        
    def _printNorm(self, sdp, ops):
        for op in ops:
            norm = sdp[Dagger(op)*op]
            print(f'{op}\tNorm: {norm:.5g}')
    
    def _printProb(self, sdp):
        num_x = len(self.configA)
        num_y = len(self.configB)
        for y in range(num_y):
            for b in range(self.configB[y]):
                s = f'\t'.join([f'{sdp[self.P([a,b],[x,y])]:.4f}' \
                                for x in range(num_x) for a in range(self.configA[x])])
                print(s)
    
    def _inner_quad(self, sdp, quad_t, quad_w, obj_func):
        """Return the weighted portion of entropy corresponding to the node in the quadrature sum
            - sdp: obj of `ncpol2sdpa.SdpRelaxation`
            - quad_t: float, the node of the quadrature
            - quad_w: float, the weight corresponding to the qudrature node
            - obj_func: Callable, a function return the expression of objective corresponding to
                        the quadrature node and target inputs in terms of the POVM elements
                        and Z operators
        """
        objective = obj_func(self.P.parties, self.inputs, self.Zops, quad_t)
        sdp.set_objective(objective)
        sdp.solve(*self.solver_config)

        if self.verbose >= 2:
            print(f'Node of quadrature: {quad_t}')
            print(sdp.status, sdp.primal, sdp.dual)

        assert sdp.status == 'optimal' or \
            ('feasible' in sdp.status and 'infeasible' not in sdp.status), \
            f'Status {sdp.status}, not promised solved'
        
        if sdp.status != 'optimal' and self.verbose >= 1:
            print('Solution does not reach optimal!')

        coeff = quad_w/(quad_t * LOG2)
        ent_in_quad = coeff * (1 + sdp.dual)
        
        if self.verbose >= 2:
            print(f'Entropy in quadrature: {ent_in_quad}')

        if self.verbose >= 4:
            self._printNorm(sdp, self.Zops)
            if self.verbose >= 5:
                self._printProb(sdp, self.P)

        return ent_in_quad
    
    def _gen_Radau_quadrature(self):
        """Gen the nodes and weights of the Gauss-Radau quadrature according to
            `radau_quad_params['n_quad']` and `radau_quad_params['endpoint']`;
            if `radau_quad_params['keep_endpoint']` is `False`, the last term will be removed
        """
        T, W = chaospy.quadrature.radau(round(self.radau_quad_params['n_quad']/2),
                                        chaospy.Uniform(0, self.radau_quad_params['endpoint']),
                                        self.radau_quad_params['endpoint'])
        T = T[0]
        self.radau_quad_params['n_quad'] = len(T)
        
        if not self.radau_quad_params['keep_endpoint']:
            self.radau_quad_params['n_quad'] -= 1
        
        return T, W
        
    def asymptotic_rate(self):
        """Compute the asymptotic rate in terms of von Neumann entropy lower bound
            by BFF21 method
        """
        assert bool(self.inputs), 'No specified inputs! Plz set inputs first.'
        assert bool(self.eq_constr) or bool(self.ineq_constr), \
                'No constraints! Plz set at least a constraint first.'
        
        self._init_npa_relaxation()
        T, W = self._gen_Radau_quadrature()

        if self.verbose >= 2:
            print(f'Nodes of the Gauss-Radau quadrature:\n{T}')
            print(f'Weights of the Gauss-Radau quadrature:\n{W}')

        obj_func = OBJ_FUNC_MAP[self.rand_type]

        results = []
        if self.nthread_quad == 1:
            for i in reversed(range(self.radau_quad_params['n_quad'])):
                ent = self._inner_quad(self.sdp, T[i], W[i], obj_func)
                results += [ent]
        else:
            results = Parallel(n_jobs=self.nthread_quad, verbose=0)(
                                delayed(self._inner_quad)(self.sdp, T[i], W[i], obj_func) \
                                for i in reversed(range(self.radau_quad_params['n_quad'])))
        
        asymp_rate = sum(results)
        if self.verbose:
            print(f'Asymptotic rate: {asymp_rate}')
        
        return asymp_rate


class DIRandZPC_Calculator(DIRandCalculator):
    """Class to compute asymptotic rates for DIRE protocol based on ZP-constrained correlations
        - rand_type: str (loc, glo, bli), type of DI randomness to compute
        - npa_level: int, level of NPA hierarchy to approx quantum set
        - nthread_sdp: int, num of threads to construct the NPA hierarchy and solve SDP 
        - nthread_quad: int, num of threads to compute the terms in the quadrature parallelly
        - radau_quad_params: dict of the form:
                             {'n_quad': `int`, 'endpoint': `float`, 'keep_endpoint': `bool`},
                             record info related to the computation for Gauss-Radau quadrature
                             - n_quad: int, num of terms in the quadrature, default is 12
                                       (odd int is equivalant to the closer larger even int)
                             - endpoint: float, val of the right endpoint, default is 1, and
                                         the interval is [0,1]; set to val slightly smaller
                                         than 1 to avoid instability of solving SDP
                             - keep_endpoint: bool, default is `True`, set to `False` to avoid
                                              instability of solving SDP at endpoint
        - protocol_params: dict contains the following keys:
                           - kappa: str ('1', '2a', '2b', '2c', '3a', '3b') or None, class of the
                                    ZP constraints
                           - inputs: two-int tuple, target inputs
                           - WP_predicate: high dimensional list/np-array, winning-probability (WP)
                                           predicate in terms of the coefficients corresponding
                                           to the probabilities
                           - WP_check_direction: str ('lb', 'ub'), check WP satisfying the lower-
                                                 bound ('lb') or upper-bound ('ub') threshold
                           - inp_prob: high dimensional list/np-array, input probabilities
                           - wexp: float, expected winning probability
                           - wtol: float in interval (0, 1), tolerant level for WP constraint
                           - ztol: float in interval (0, 1), tolerant level for ZP constraint
        - solver_config: list of the form: [`solver_name`, `solver_options`],
                         - solver_name: str
                         - solver_options: dict, specifying option names and vals
        - verbose: int, level of verbose logging

    """
    def __init__(self, rand_type, npa_level = 2, nthread_sdp = 1,  nthread_quad = 1,
                 radau_quad_params = {'n_quad': 12, 'endpoint': 1., 'keep_endpoint': True},
                 protocol_params = {'kappa': None, 'inputs': (),
                                    'WP_predicate': None,
                                    'WP_check_direction': 'lb',
                                    'inp_prob': [[1/4, 1/4],[1/4, 1/4]],
                                    'wexp': 0., 'wtol': 0., 'ztol': 0.},
                 solver_config = ['mosek'], verbose = 1):
        
        super().__init__([2,2],[2,2], rand_type,
                         target_inp = protocol_params['inputs'],
                         npa_level = npa_level,
                         nthread_sdp = nthread_sdp,
                         nthread_quad = nthread_quad,
                         radau_quad_params = radau_quad_params,
                         solver_config = solver_config,
                         verbose = verbose)
        
        assert protocol_params['WP_check_direction'] in ('lb', 'ub'), \
            'Invalid WP_check_direction in protocol_params. \
                Plz set as one of the followings: "lb" or "ub".'
        
        self.protocol_params = dict()
        self.protocol_params.update(protocol_params)
        if self.protocol_params['WP_predicate'] is None:
            CHSH_predicate = [[[[0,1], [1,0]],
                               [[0,1], [1,0]]],
                              [[[0,1], [1,0]],
                               [[1,0], [0,1]]]]
            CHSH_predicate = np.array(CHSH_predicate)*np.array(self.protocol_params['inp_prob'])
            self.protocol_params['WP_predicate'] = CHSH_predicate

        self.WP_Q_BOUND = {'lb': None, 'ub': None}

    def update_protocol_params(self, params):
        self.protocol_params.update(params)
        self.constr_dict = dict()
        self.eq_constr = []
        self.ineq_constr = []

    def win_prob_Q_bound(self, direction = 'max', truncate_digit = 6):
        """Compute the maximum/minimum winnning probability over the quantum set
            - direction: str ('max', 'min'), decide to compute maximum or minimum
            - truncate_digit: int, the digit of the result up to truncate
        """
        assert direction in ('max', 'min'), \
            'Invalid direction for winning probability quantum bound. \
                Plz set as the followings: "max" or "min".'
        
        sdp_Q = ncp.SdpRelaxation(self.P.get_all_operators(), verbose = max(self.verbose-3, 0))
        objective = super()._build_expression_from_probabilities(self.protocol_params['WP_predicate'])
        if direction == 'max':
            objective = -objective

        zero_pos = ZP_POSITION_MAP.get(self.protocol_params['kappa'], [])
        zp_constr = []
        for pos in zero_pos:
            zp_constr += [self.protocol_params['ztol'] - self.P(*pos)]
        
        sdp_Q.get_relaxation(level = self.npa_level, objective = objective,
                             substitutions = self.P.substitutions,
                             momentinequalities = zp_constr)
        sdp_Q.solve(*self.solver_config)

        if self.verbose >= 2:
            print(f'Status: {sdp_Q.status}')
            print(f'Primal: {sdp_Q.primal}')
            print(f'Dual: {sdp_Q.dual}')
            if zero_pos:
                max_zero_pos = max([sdp_Q[self.P(*pos)] for pos in zero_pos])
                print(f'Max val in zero position: {max_zero_pos:.9g}')

        if sdp_Q.status != 'optimal' and sdp_Q.status != 'primal-dual feasible':
            print('Cannot compute quantum bound correctly!', file=sys.stderr)

        win_prob_Q_bound = truncate(abs(sdp_Q.primal), truncate_digit)
        flip = ((sdp_Q.primal < 0) and (direction == 'min')) or \
                ((sdp_Q.primal > 0) and (direction == 'max'))
        
        if flip:
            win_prob_Q_bound = -win_prob_Q_bound
        
        self.WP_Q_BOUND[direction] = win_prob_Q_bound

        return win_prob_Q_bound
    
    def _set_WP_constraint(self):
        if self.protocol_params['WP_check_direction'] == 'lb':
            direction = '>'
            wp_bound = self.protocol_params['wexp'] - self.protocol_params['wtol']
        else:
            direction = '<'
            wp_bound = self.protocol_params['wexp'] + self.protocol_params['wtol']
        super().add_constraint_by_coeff('ineq', self.protocol_params['WP_predicate'], wp_bound,
                                        name = 'WP_check', direction = direction)
    
    def _zeropos2str(self, pos):
        """Convert the position/index of the probability to string
        """
        pos = np.array(pos)
        flat_pos = pos.reshape(np.sum(pos.shape))
        pos_str = ''.join([str(i) for i in flat_pos])
        return pos_str

    def _set_ZP_constraint(self):
        zero_pos = ZP_POSITION_MAP.get(self.protocol_params['kappa'], [])
        for pos in zero_pos:
            super().add_constraint('ineq', self.P(*pos), self.protocol_params['ztol'],
                                   name = f'ZP_check_{self._zeropos2str(pos)}', direction = '<')
            
    def _inner_quad_with_dual_var(self, sdp, quad_t, quad_w, obj_func):
        """Solve the SDP corresponding to the quadrature node; after the SDP is solved,
            return the winning probability, the weighted portion of entropy, the vec of dual vars,
            and the probabilities that ZP constraints are emposed
            - sdp: obj of `ncpol2sdpa.SdpRelaxation`
            - quad_t: float, the node of the quadrature
            - quad_w: float, the weight corresponding to the qudrature node
            - obj_func: Callable, a function return the expression of objective corresponding to
                        the quadrature node and target inputs in terms of the POVM elements
                        and Z operators
        """
        ent_in_quad = super()._inner_quad(sdp, quad_t, quad_w, obj_func)
        
        p_win_in_quad = sdp[self.constr_dict['WP_check']['expr']]
        p_win_dual_vec = np.array(sdp.get_dual(self.constr_dict['WP_check']['constr']))

        if not p_win_dual_vec.shape:
            p_win_dual_var = p_win_dual_vec*1
        elif len(p_win_dual_vec) == 1:
            p_win_dual_var = p_win_dual_vec[0]
        else:
            # print(p_win_dual_vec)
            p_win_dual_var = p_win_dual_vec[0]-p_win_dual_vec[1]

        coeff = quad_w/(quad_t * LOG2)
        if self.protocol_params['kappa'] in ZP_POSITION_MAP.keys():
            n_zp_checks = len(ZP_POSITION_MAP[self.protocol_params['kappa']])
            lambda_in_quad =  np.zeros(n_zp_checks+1)
            p_zero_in_quad = np.zeros(n_zp_checks)
            lambda_in_quad[0] = coeff * p_win_dual_var
            for i in range(n_zp_checks):
                zero_pos = ZP_POSITION_MAP[self.protocol_params['kappa']][i]
                constr_name = f'ZP_check_{self._zeropos2str(zero_pos)}'
                zero_dual_var = np.squeeze(sdp.get_dual(self.constr_dict[constr_name]['constr']))
                lambda_in_quad[i+1] = coeff * zero_dual_var
                #print(zero_dual_var)
                p_zero = sdp[self.P(*zero_pos)]
                p_zero_in_quad[i] = p_zero
            if self.verbose >= 2:
                print(f'Largest value among ZPs: {max(p_zero_in_quad)}')
            return p_win_in_quad, ent_in_quad, lambda_in_quad, p_zero_in_quad
        else:
            lambda_in_quad = coeff * p_win_dual_var
            return p_win_in_quad, ent_in_quad, lambda_in_quad, np.empty(0)
    
    
    def asymptotic_rate_with_Lagrange_dual(self):
        """Compute the asymptotic rates for the protocol with given params and derive the
            vec of dual vars `lambda_` and const term `c_lambda` of Lagrange dual function
        """
        assert bool(self.inputs), 'No specified inputs! Plz set inputs first.'
        
        T, W = super()._gen_Radau_quadrature()

        if self.verbose >= 2:
            print(f'Nodes of the Gauss-Radau quadrature:\n{T}')
            print(f'Weights of the Gauss-Radau quadrature:\n{W}')

        obj_func = OBJ_FUNC_MAP[self.rand_type]
        self._set_WP_constraint()
        self._set_ZP_constraint()
        super()._init_npa_relaxation()

        if self.nthread_quad == 1:
            p_win_quad = []
            entropy_quad = []
            lambda_quad = []
            p_zero_quad = []
            for i in reversed(range(self.radau_quad_params['n_quad'])):
                p_win, ent, lamb, p_zero = self._inner_quad_with_dual_var(self.sdp, T[i], W[i], obj_func)
                p_win_quad += [p_win]
                entropy_quad += [ent]
                lambda_quad += [lamb]
                p_zero_quad += [p_zero]
        else:
            results = Parallel(n_jobs=self.nthread_quad, verbose=0)(
                                delayed(self._inner_quad_with_dual_var)(self.sdp, T[i], W[i], obj_func) \
                                for i in reversed(range(self.radau_quad_params['n_quad'])))
            p_win_quad, entropy_quad, lambda_quad, p_zero_quad = zip(*results)
        
        entropy = np.sum(np.array(entropy_quad))
        if self.verbose:
            print(f'Asymptotic rate: {entropy}')
        
        lambda_ = np.sum(np.array(lambda_quad), axis=0)
        p_zero_quad = np.array(p_zero_quad)

        c_lambda = entropy
        if np.any(p_zero_quad):
            min_p_zero = np.min(p_zero_quad, axis=0)
            c_lambda += np.sum(lambda_[1:] * (min_p_zero - self.protocol_params['ztol']))

        lambda_w = lambda_[0] if bool(np.shape(lambda_)) else lambda_
        if self.protocol_params['WP_check_direction'] == 'lb':
            opt_p_win = max(p_win_quad)
            c_lambda -= lambda_w * opt_p_win
        else:
            opt_p_win = min(p_win_quad)
            c_lambda += lambda_w * opt_p_win

        return entropy, lambda_, c_lambda


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-t", dest="test_class", required=True,
                        choices=['DIRand', 'DIRandZPC'],
                        help="choose the class to test")
    args = parser.parse_args()

    if args.test_class == 'DIRand':
        print('Test DIRandCalculator >>>>>>')
        configA = [2,2]
        configB = [2,2]
        target_inp = (0,0)
        rand_type = 'glo'
        radau_quad_params = {'n_quad': 12, 'endpoint': .999, 'keep_endpoint': True}
        npa_level = 2
        nthread_quad = 2
        nthread_sdp = 6
        solver_config = ['mosek', {'dparam.intpnt_co_tol_rel_gap': 5e-5,
                                'iparam.num_threads': nthread_sdp,
                                'iparam.infeas_report_level': 4}]
        verbose = 1
        
        Calculator = DIRandCalculator(configA, configB, rand_type, target_inp,
                                      npa_level = npa_level,
                                      nthread_sdp = nthread_sdp,
                                      nthread_quad = nthread_quad,
                                      radau_quad_params = radau_quad_params,
                                      solver_config = solver_config, verbose = verbose)
        
        CHSH_BellCoeff = [[[[1,0], [0,1]],
                           [[1,0], [0,1]]],
                          [[[1,0], [0,1]],
                           [[0,1], [1,0]]]]
        CHSH_BellCoeff = np.array(CHSH_BellCoeff)*1/4
        wexp = 0.85355
        
        Calculator.add_constraint_by_coeff('ineq', CHSH_BellCoeff, wexp,
                                           name = 'WP_check', direction = '>')
        asymp_rate = Calculator.asymptotic_rate()
    else:
        print('Test DIRandZPC_Calculator >>>>>>')
        rand_type = 'loc'
        WP_predicate = [[[[0,1], [1,0]],
                         [[0,1], [1,0]]],
                        [[[0,1], [1,0]],
                         [[1,0], [0,1]]]]
        WP_predicate = np.array(WP_predicate)*1/4
        protocol_params = {'kappa': '2a', 'inputs': (1,1),
                           'inp_prob': [[1/4, 1/4], [1/4, 1/4]],
                           'WP_predicate': WP_predicate,
                           'WP_check_direction': 'lb',
                           'wexp': 0.8125, 'wtol': 1e-5, 'ztol': 1e-6}
        radau_quad_params = {'n_quad': 12, 'endpoint': .999, 'keep_endpoint': True}
        npa_level = 2
        nthread_sdp = 6
        nthread_quad = 2
        solver_config = ['mosek', {'dparam.intpnt_co_tol_rel_gap': 5e-5,
                                   'iparam.num_threads': nthread_sdp,
                                   'iparam.infeas_report_level': 4}]
        verbose = 0
        
        Calculator = DIRandZPC_Calculator(rand_type, npa_level = npa_level,
                                          nthread_sdp = nthread_sdp,
                                          nthread_quad = nthread_quad,
                                          radau_quad_params = radau_quad_params,
                                          protocol_params = protocol_params,
                                          solver_config = solver_config,
                                          verbose = verbose)

        tic = time.time()
        entropy, lambda_, c_lambda = Calculator.asymptotic_rate_with_Lagrange_dual()
        toc = time.time()
        print(f'Asymptotic rate: {entropy}')
        print(f'Lambdas: {lambda_}')
        print(f'C_lambda: {c_lambda}')
        print(f'Elapsed time: {toc-tic}')

