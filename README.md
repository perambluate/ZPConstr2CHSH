# ZPConstr2CHSH

A Python framework for analyzing device-independent randomness expansion protocols based on CHSH games with zero-probability constraints.

## Overview

This repository provides:
1. Implementation code for the research presented in [arXiv:2401.08452](https://arxiv.org/abs/2401.08452) and the associated [thesis](https://thesis.lib.ncku.edu.tw/thesis/detail/35155df76be1c65883ed908ef6ff1c0c/)
2. Constructing quantum strategies in Bell scenarios using `QuantumStrategy`, with specialized support for CHSH (Clauser-Horne-Shimony-Holt) scenarios via `StrategiesInCHSHScenario`.
3. Computing device-independent (DI) randomness generation rates from modified CHSH games using:
- `ZPConstr2CHSH`: an all-in-one version of module to compute both asymptotic performance and the one with finite-data effects, example for demonstration: `examples/DI_Randomness_ZPConstr2CHSH.py`
- `SDP_helper.py`: helper functions to generate or process SDPs, used in the scripts in `onePartyRandomness/`, `twoPartyRandomness/`, `blindRandomness/`, and also in `examples/DI_Randomness_common_func.py`
- `plotting_helper.py`: helper function to compute the finite rates with different optimized tunable params

The implementation is based on:
- *Brown-Fawzi-Fawzi* (or BFF21) method[^1] for asymptotic performance
- *generalized entropy accumulation*[^2] for finite-data analysis

[^1]: *P. Brown et al.*, Quantum 8, 1445 (2024)
[^2]: *T. Metger et al.*, 2022 IEEE 63rd Annual Symposium on Foundations of Computer Science (FOCS)

## Installation
### Prerequisites

- Python3.x
- An SDP solver (one of):
    - MOSEK (recommended)
    - SCS
    - CVXOPT
    - SDPA

### Core Dependencies

- [ncpol2sdpa](https://github.com/peterjbrown519/ncpol2sdpa) - Navascués-Pironio-Acín (NPA) hierarchy[^3] construction
- CVXPY - SDP solver interface
- numpy - High dimensional array operations
- scipy - Statistical functions (calculate properties of binomials distributions for completeness; generate Gamma distributions for parameter scanning)
- sympy - Symbolic mathematics
- chaospy - Gauss-Radau quadratures

### Additional Dependencies
- matplotlib - Plotting
- joblib - Parallelization
- functools - Function manipulation (make it easier to parallelize multi-parameter functions)
- argparse - CLI argument parsing (quick test for `ZPConstr2CHSH` module)
- itertools - Generate combinations of array elements and cyclable arrays

[^3]: See *M. Navascués et al.*, Phys. Rev. Lett. 98, 010401 (2007) and *M. Navascués et al.*, New J. Phys. 10, 073013 (2008)

## Folder structure
<pre>
├── DIStrategyUtils/     # Bell scenario correlation construction
├── ZPConstr2CHSH/       # Core implementation modules
├── common_func/         # Helper functions
├── examples/            # Usage examples
├── blindRandomness/     # Scripts for blind randomness computation
├── onePartyRandomness/  # Scripts for one-party randomness computation
├── twoPartyRandomness/  # Scripts for two-party randomness computation
├── WBC_inequality/      # Scripts for (blind, one-/two-party) randomness computation with WBC inequality constraint
└── plotter/            # Scripts for results visualization 
</pre>

## Usage
See the examples in `examples` directory:
- DI_Randomness_example.py - Standalone implementation
- DI_Randomness_ZPC2CHSH - Complete workflow using ZPConstr2CHSH

Use scripts in `blindRandomness`, `onePartyRandomness`, `twoPartyRandomness`, and `WBC_inequality` to reproduce data in [my thesis](https://thesis.lib.ncku.edu.tw/thesis/detail/35155df76be1c65883ed908ef6ff1c0c/) or [arXiv draft](https://arxiv.org/abs/2401.08452).

## Old README
Check `README.old.md` for more detail about the Python scripts

## Lisense
This project is licensed under the GNU General Public License v3.0.
