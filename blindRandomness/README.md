# Blind-Randomness-from-Nonlocal-Game
In this project, we consider the scenario of generating blind (or local) randomness ([PRA 97, 032324 (2018)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.97.032324)) by running a two-party nonlocal game $n$ rounds. We consider two scenario, one in Fu and Miller's work ([PRA 97, 032324 (2018)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.97.032324)), and the other in Brown, Fawzi, and Fawzi's work ([arXiv:2106.13692v2](https://arxiv.org/abs/2106.13692)). The former method quantify the local randomness as *minus logrithm of guessing probability* (min-entropy); while the later one consider the *von Neumann entropy conditioned on the adversary's information*.

Since in Fu and Miller's work ([PRA 97, 032324 (2018)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.97.032324)), they consider the adversary performs a sequential of measurements which is equivalent to a set of projective measurements with more inputs/outputs ([Quantum 4, 344 (2020)](https://quantum-journal.org/papers/q-2020-10-19-344/)). We also consider the computation with this equivalent scenario.

Besides by computing the conditional von Neumann entropy with Brown-Fawzi-Fawzi method, we can further employ generalised entropy accumulation theorem ([arXiv:2203.04989v2](https://arxiv.org/abs/2203.04989)) to compute the finite-size rate of local randomness per round. We provide the finite rate function in the plotting function file.

## Folder structure
<pre>
├── bbs20/
|   └── blindRandomness-BBS20.py
├── bff21/
|   └── blindRandomness-BFF21.py
├── fm18/
|   └── blindRandomness-FM18.py
├── plotting_func/
│   ├── draw_asym_rate.py
│   ├── draw_fin_rate.py
│   ├── draw_heat_map.py
|   └── draw_min_tradeoff.py
├── data/
├── figures/
├── README.md 
└── .gitignore
</pre>

## Required packages
Numerical computation
- [ncpol2sdpa](https://github.com/peterjbrown519/ncpol2sdpa)
- chaospy
- sympy
- numpy

One needs to install at least one of the SDP solvers to run the computation.
- SCS
- CVXOPT
- SDPA
- MOSEK

SDP solver interface for some solvers
- CVXPY

In plotting function files
- functool
- joblib
> `joblib` provide multiprocessing to speed up the computation intensive for-loop;
> one can rewrite the code with normal for-loop to avoid installing it.
- matplotlib
- itertool
