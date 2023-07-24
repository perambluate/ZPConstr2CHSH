# ZPConstr2CHSH

A collection of python files to compute the randomness extracted from the Clauser-Horne-Shimony-Holt game and its modification with introducing zero-probability constraints.

In this repo, we consider three kinds of randomness: (i) single-party (one-party) randomness $H(A|XYE)$, `./onePartyrandomness` (ii) two-party randomness $H(AB|XYE)$ `./twoPartyrandomness` and (iii) blind randomness $H(A|XYBE)$ `./blindPartyrandomness`. For all kind of randomness, we first compute the asymptotic rate with the *Brown-Fawzi-Fawzi* method ([arXiv:2106.13692v2](https://arxiv.org/abs/2106.13692)), and then employ *generalized entropy accumulation* ([arXiv:2203.04989v2](https://arxiv.org/abs/2203.04989)) to derive the finite rate by subtracting the second order correction term.

## Folder structure
You can see the forder structure in `README.md` in `./blindRandomness`. For `onePartyRandomness` and `twoPartyRandomness` the main script to compute the asymptotic rate would be same as the folder name with file extension`.py`. And you can use the file `draw_asym_rate.py` to plot the asymptotic rate. In default settings, the data are saved as CSV files in the subfolder `data`, and the figure would be saved as PNG files in the subfolder `figures`, please change the variable `OUT_DIR` in the scripts if you want to change the position.