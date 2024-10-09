import numpy as np
from math import log, pi, cos, sin
import functools
import itertools

id = np.eye(2)
sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])
sy = np.array([[0, -1j], [1j, 0]])

ket = lambda vec, scalar: (np.array(vec)*scalar).reshape((4,1))
proj = lambda vec: np.outer(vec, vec)
obs_on_xz = lambda theta: cos(theta)*sz + sin(theta)*sx
obs_on_xy = lambda theta: cos(theta)*sx + sin(theta)*sy

def tensor(op_list):
    return functools.reduce(np.kron, op_list)

def genCombinations(configs):
    """
        Gen all the combinations from a config list
        e.g. configs = [2,2] -> [(0,0), (0,1), (1,0), (1,1)]
             configs = [2,3,2] -> [(0,0,0), (0,0,1), (0,1,0), (0,1,1), (0,2,0), (0,2,1),
                                   (1,0,0), (1,0,1), (1,1,0), (1,1,1), (1,2,0), (1,2,1)]
    """
    comb = np.array(np.meshgrid(*[list(range(i)) for i in configs])).T.reshape(-1, len(configs))
    comb = [tuple(id_comb) for id_comb in comb]
    return comb

def addWhiteNoise(rho, p):
    """
        Add certain level of white noise to a quantum state
        - rho: the density matrix repr the state
        - p: visibility of the noisy state (larger p indicates less noisy)
    """
    assert p <= 1 and p >= 0, 'Invalid noise ratio.'
    return p*rho + (1-p)*np.eye(rho.shape[0])/rho.shape[0]

def binaryMeasNoclickBinning(measPOVM, eta = 1, bin_to = 1):
    """
        Convert a binary measurement to its ineficient version
            when binning the no-click events to the given outcomes
        - measPOVM: the POVM elements corresponding to the binary measurement
        - eta: detectoin efficiency (smaller eta indicates less efficient)
        - bin_to: the index of the outcome that the no-click event is binned to
    """
    E_plus = eta/2*(id+measPOVM)
    E_minus = eta/2*(id-measPOVM)
    if bin_to:
        E_minus = E_minus + (1-eta)*id
    else:
        E_plus = E_plus + (1-eta)*id

    return [E_plus, E_minus]

class QuantumStrategy():
    """
        Class to get correlation (conditional probabilities)
            for a fixed scenario and a given strategy
        - scenario: a list of tuples to repr the scenario for arbitrary parties
                    e.g., [(2,2), (2,3,3)] repr a bipartite scenario where first party can
                    choose two meas and each meas has two outcomes and second party has
                    three meas including a binay meas and two three-outcome meas
        - state: a two-dimensional array repr the density matrix
        - measurement: a three-dimensional lists of np-array recording the measment POVM element
                       for given party, input and outcome
        - correlations: a high dimensional np-array that recods the full set of conditional
                        probabilities of getting the outcomes given the inputs
    """
    def __init__(self, scenario):
        self.scenario = scenario
        self.inp_config = tuple(len(self.scenario[i]) for i in range(len(self.scenario)))
        self.n_parties = len(self.scenario)
        self.state = []
        self.measurements = []
        for p in range(self.n_parties):
            temp = []
            for inp in range(self.inp_config[p]):
                temp.append([[] for i in range(self.scenario[p][inp])])
            self.measurements.append(temp)

    def set_state(self, state):
        """Set the state for the strategy
        """
        self.state = state

    def set_measurement(self, party, inp, POVMs = []):
        """Set the measurement POVMS for given party and input
        """
        self.measurements[party][inp] = POVMs

    def probability(self, parties, inps, outs):
        """Return the conditional probability of given parties, inputs, and outcomes
        """
        meas_ops = [self.measurements[p][inp][out] for p, inp, out in zip(parties, inps, outs)]
        return np.trace(np.matmul(self.state, tensor(meas_ops)))

    def gen_correlations(self, parties = None):
        """Gen the full correlation for the given strategy
        """
        if parties is None:
            parties = list(range(self.n_parties))
        inp_config = [self.inp_config[p] for p in parties]
        max_out_config = [max(self.scenario[p]) for p in parties]

        self.correlations = np.zeros((*inp_config, *max_out_config))
        inp_combines = genCombinations(inp_config)
        out_combines = np.array(np.meshgrid(*[self.scenario[p] for p in parties])).T

        for inps in inp_combines:
            out_comb = list(itertools.product(*[list(range(i)) for i in out_combines[inps]]))
            for outs in out_comb:
                self.correlations[inps][outs] = self.probability(parties, inps, outs)

    def BellVal(self, BellCoeff, parties = None):
        """Return the Bell value of the Bell function specified by the coefficients weighting
            the conditional probabilities
        """
        if parties is None:
            parties = list(range(self.n_parties))
        inp_config = [self.inp_config[p] for p in parties]

        inp_combines = genCombinations(inp_config)
        out_combines = np.array(np.meshgrid(*[self.scenario[p] for p in parties])).T
        Bell_value = 0

        for inps in inp_combines:
            out_comb = list(itertools.product(*[list(range(i)) for i in out_combines[inps]]))
            for outs in out_comb:
                Bell_value += self.correlations[inps][outs] * BellCoeff[inps][outs]

        return Bell_value


class StrategiesInCHSHScenario(QuantumStrategy):
    """
        Quantum strategies for CHSH scenario (two party, two inputs, and two outputs)
        - angles: a dict records the angles to specify
                  `theta`: angle related to the state: cos(theta)|00> + sin(theta)|11>
                  `alpha`: angle related to Alice's first meas
                           in the form of cos(theta)*sz + sin(theta)*sx
                  `alpha1`: angle related to Alice's second meas
                  `beta`: angle related to Bob's first meas
                  `beta1`: angle related to Bob's second meas
        - noise_param: a dict records the params related to the noise
                       `visibility`: the weight of the original state mixed with white noise
                       `eta`: detection efficiency
    """
    def __init__(self, angles = {'theta': pi/4,
                                 'alpha': 0, 'alpha1': pi/2,
                                 'beta': pi/4, 'beta1': -pi/4},
                       noise_param = {'visibility': 1, 'eta': 1}):
        """Init the class and at the same time gen the correlation after initiation
        """
        super().__init__(([2,2], [2,2]))
        self.angles = angles
        self.noise_param = noise_param

        # init state
        state_ket = np.array([cos(self.angles['theta']), 0, 0, sin(self.angles['theta'])])
        rho = addWhiteNoise(proj(state_ket), self.noise_param['visibility'])
        super().set_state(rho)

        # init measurements
        self.observables = np.zeros((2,2,2,2))
        self.observables[0][0] = obs_on_xz(self.angles['alpha'])
        self.observables[0][1] = obs_on_xz(self.angles['alpha1'])
        self.observables[1][0] = obs_on_xz(self.angles['beta'])
        self.observables[1][1] = obs_on_xz(self.angles['beta1'])
        for party in range(2):
            for inp in range(2):
                POVM = binaryMeasNoclickBinning(self.observables[party][inp], self.noise_param['eta'])
                super().set_measurement(party, inp, POVM)
        
        super().gen_correlations()

    def set_visibility(self, visibility):
        """Set visibility after initiation
        """
        self.noise_param['visibility'] = visibility
        state_ket = np.array([cos(self.angles['theta']), 0, 0, sin(self.angles['theta'])])
        rho = addWhiteNoise(proj(state_ket), self.noise_param['visibility'])
        super().set_state(rho)

    def set_eta(self, eta):
        """Set detection efficiency after initiation
        """
        self.noise_param['eta'] = eta
        A0 = obs_on_xz(self.angles['alpha'])
        A1 = obs_on_xz(self.angles['alpha1'])
        B0 = obs_on_xz(self.angles['beta'])
        B1 = obs_on_xz(self.angles['beta1'])
        super().set_measurement(0,0, binaryMeasNoclickBinning(A0, self.noise_param['eta']))
        super().set_measurement(0,1, binaryMeasNoclickBinning(A1, self.noise_param['eta']))
        super().set_measurement(1,0, binaryMeasNoclickBinning(B0, self.noise_param['eta']))
        super().set_measurement(1,1, binaryMeasNoclickBinning(B1, self.noise_param['eta']))

