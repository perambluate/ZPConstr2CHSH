# ZPConstr2CHSH

This repo contains source code for [my thesis](https://thesis.lib.ncku.edu.tw/thesis/detail/35155df76be1c65883ed908ef6ff1c0c/) as well as [an draft on arXiv](https://arxiv.org/abs/2401.08452)[^1]. Since we are planing to submit the draft to a journal, to avoid copyright issue, data and figures are currently not available.

In this repo, you can \
(i) construct a strategy for arbitrary Bell scenarios with module `QuantumStrategy`. Especially, strategies in Clauser-Horne-Shimony-Holt (CHSH) scenario can be easily built with `class::StrategiesInCHSHScenario`. \
(ii) compute the amount of device-independent (DI) randomness generated from the outcomes of the CHSH game and its modification with zero-probability constraints by either the module `ZPConstr2CHSH` or the functions collected in `common_func/SDP_helper.py` and `common_func/plotting_helper.py`

The computation is performed according to the *Brown-Fawzi-Fawzi* (or BFF21) method[^2] and *generalized entropy accumulation*[^3], where the former is used to calculate asymptotic performance of the DI-RE/DI-RG protocols and the latter is employed to perform the finite-data analysis on them.

[^1]: *C.-Y. Chen et al.*, arXiv:2401.08452 (2024)

[^2]: *P. Brown et al.*, Quantum 8, 1445 (2024)

[^3]: *T. Metger et al.*, 2022 IEEE 63rd Annual Symposium on Foundations of Computer Science (FOCS)

## Software
Python3

### Required packages

Navascués-Pironio-Acín (NPA) hierarchy[^4] construction
- [ncpol2sdpa](https://github.com/peterjbrown519/ncpol2sdpa)

One the following SDP solvers
- SCS
- CVXOPT
- SDPA
- MOSEK
> Used in this package: Mosek

SDP solver interface
- CVXPY

Gauss-Radau quadrature construction
- chaospy

Abstract symbols, operators, and expression calculation
- sympy

High dimensional array operation
- numpy

Properties of binomial distribution
- scipy

Basic math calculation
- math

Constructing functions with less arguments from exsiting functions
- functools

Parallelization
- joblib

Plotting
- matplotlib

Timing
- time

Parsing arguments
- argparse

Regex string manipulation
- re

Getting sys info, buffer and Python path appending
- sys

Path editing and examing
- os

Optional packages
- itertools
> Lack of `itertools` will make `DIStrategyUtils.QuantumStrategy` and `minTradeoffProp` modules and some plotting scripts not usable.
- cycler
> Only influence the plotting scripts in `blindRandomness/plotting_func/`.

[^4]: See *M. Navascués et al.*, Phys. Rev. Lett. 98, 010401 (2007) and *M. Navascués et al.*, New J. Phys. 10, 073013 (2008)

## Folder structure
<pre>
├── bash_scripts/
├── blindRandomness/
├── common_func/
├── DIStrategyUtils/
├── onePartyRandomness/
├── examples/
├── plotter/
├── twoPartyRandomness/
├── WBC_inequality/ 
└── ZPConstr2CHSH/
</pre>

### Modules or function collection files

- `DIStrategyUtils/`: a module to construct the correlation (conditional probabilities) in arbitrary Bell scenarios with given strategy, especially for CHSH Bell scenario (bipartite, two-input, two-output).

- `ZPConstr2CHSH/`: a module contains two sub-modules \
    a. `DIRandomness`: Compute asymptotic rates for DI randomness expansion (DIRE) protocols with zero-probability (ZP) constraints by BFF21 method.\
    b. `finiteRateCalculator`: Compute (i) finite rates for DIRE protocols with ZP constraints by GEAT and (ii) completeness of the protocol with given protocol params.

- Auxiliary functions are included in `common_func/`. \
`SDP_helper.py` provides the common functions used in computation of asmptotic rates via solving semidefinite programming (SDP) problems. \
`plotting_helper.py` provides the function to compute finite rate and set up the common plotting settings.

### Main folders
- Folders: `blindRandomness/`, `onePartyRandomness/`, and `twoPartyRandomness/`, contains the scripts to compute the asymptotic rate or guessing probability (the latter one only for blind randomness)
> In `blindRandomness/README.md`, you can see its forder structure under the folder `blindRandomness/`. For `onePartyRandomness/` and `twoPartyRandomness/` the main script to compute the asymptotic rate is the same as the folder name with file extension`.py`. In default settings, the data are saved as CSV files in the subfolder `data/`, and the figure would be saved as PNG files in the subfolder `figures/`, please change the variable `OUT_DIR` in the scripts if you want to change the location to save.

- Folder `WBC_inequality/` includes scripts to compute asymptotic rates and finite rates for DIRE protocol based on Wooltorton-Brown-Colbeck (WBC) inequality[^5].
    a. `DIRandomness-wbc.py`: compute asymptotic rates with weighted winning probability according to WBC inequality.
    b. `DIRandomness-wbc-three_win_probs.py`: compute asymptotic rates with three scores based on the the correlators corresponding to the three different weights.
    c. `minTradeoffProp.py`: a module to compute the properties of min-tradeoff function, essential to finite rate computation by GEAT, for the three-score-based protocol.
    d. `fin_rate_computation.py`: compute finite rates for three-score-based protocols.

- Folder `plotter/` comprises scripts to plot the asymptotic rates `draw_asym_rate.py` and compute and plot the finite rates `draw_fin_rate_thesis.py`.

[^5]: *L. Wooltorton et al.*, Phys. Rev. Lett. 129, 150403 (2022)

### Others

- Folder `bash_scripts/` includes the bash script `run_py_bg.sh` to execute python scripts in background.

- Folder `examples/` contains scripts to compute either asymptotic rate by BFF21 method or finite rate via GEAT.
    a. `DI_Randomness_common_func.py`: a script use the functions in `common_func/SDP_helper.py` to compute the asymptotic rate. \
    b. `DI_Randomness_example.py`: a standalone script (without importing any other modules defined in this repo) to compute the asymptotic rate.
    > One may read this script for better understanding of the asymptotic rate computation by BFF21 method.
    >

    c. `DI_Randomness_ZPC2CHSH`: a script make use of the module `ZPConstr2CHSH` to calculate both the asymptotic rate and the finite rate; the protocol params and results are save into `protocol.json` and `finrate_params.json`.
    > By reading this script, one can see how to use the module `ZPConstr2CHSH` to derive both the asymptotic rate and the finite rate.
