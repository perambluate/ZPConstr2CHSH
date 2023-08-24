# ZPConstr2CHSH

A collection of python files to compute the randomness extracted from the Clauser-Horne-Shimony-Holt game and its modification with introducing zero-probability constraints.

In this repo, we consider three kinds of randomness: (a) **single-party (one-party) randomness** $H(A|XYE)$, `./onePartyrandomness` (b) **two-party randomness** $H(AB|XYE)$ `./twoPartyrandomness` and (c) **blind randomness** $H(A|XYBE)$ `./blindPartyrandomness`. For all kind of randomness, we first compute the asymptotic rate with the *Brown-Fawzi-Fawzi* method ([arXiv:2106.13692v2](https://arxiv.org/abs/2106.13692)), and then employ *generalized entropy accumulation* ([arXiv:2203.04989v2](https://arxiv.org/abs/2203.04989)) to derive the finite rate by subtracting the second order correction term.

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


## Folder structure
<pre>
├── blindRandomness/
├── common_func/
├── onePartyRandomness/
├── plotter/
├── twoPartyRandomness/
├── run_py_bg.sh
├── README.md 
└── .gitignore
</pre>

1. Main folder contains the scripts to compute the asymptotic rate or guessing probability (the latter one only for blind randomness) \
`blindRandomness/` \
`onePartyRandomness/` \
`twoPartyRandomness/`
> In `blindRandomness/`, you can see its forder structure in `README.md`. For `onePartyRandomness` and `twoPartyRandomness` the main script to compute the asymptotic rate would be same as the folder name with file extension`.py`. And you can use the file `draw_asym_rate.py` to plot the asymptotic rate. In default settings, the data are saved as CSV files in the subfolder `data/`, and the figure would be saved as PNG files in the subfolder `figures/`, please change the variable `OUT_DIR` in the scripts if you want to change the position.

2. Folder `plotter/` comprises the functions to compute the finite rates and plot them as figures.
> Currently, there are two files, `draw_fin_rate_qbound.py`, `draw_fin_rate_ztol.py`.

3. Auxiliary functions are included in `common_func/`. \
`SDP_helper.py` provides the common functions used in computation of asmptotic rate via solving an SDP. \
`plotting_helper.py` provides the function to compute finite rate and the common variables for plotting.

## Quick use

1. Modify the following variables in the scripts of which name contains *blindRandomness*, *onePartyRandomness*, or *twoPartyRandomness*. \
    (a) `ZERO_CLASS`: Classes of the correlations you want to run. \
    (b) `INPUTS`: Specific inputs to the nonlocal game. (For one-party randomness, it's `X_STARs` instead) \
    (c) `P_WINs`: Observed winning probabilities from the nonlocal game. \
    (d) `ZERO_TOLs`: Tolerable zero-probability deviations. (For those correlation without zero-probability constraints it's useless and set as default value `1e-9`.)
    (e) `OUT_DIR` and `OUT_FILE`: Directory and file where the data save.
    (f) Variables whose name contains `N_WORKER`: Related to the number of jobs (threads) running in parallel. \
    `N_WORKER_SDP`: Number of threads running to solve an SDP. \
    `N_WORKER_QUAD`: (BFF21 only) Number of threads to parallelly run the quadrature for-loop. \
    `N_WORKER_LOOP`: Number of threads to parallelly run the for-loop for other parameters. \
    > Toltal number of the threads would be the product of all the `N_WORKER` variables.

2. Employ the symptotic rate computation (solving SDPs) in background with `run_py_bg.sh`; one can also run the python script directly.
```
bash run_py_bg.sh -t TYPE
bash run_py_bg.sh -d DIR -f FILE
```

3. Use plotting function to plot the results.