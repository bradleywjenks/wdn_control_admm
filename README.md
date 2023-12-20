# wdn_control_admm
This repository contains code implementation for the manuscript entitled "Distributed nonconvex optimization for control of water networks with time-coupling constraints." A preprint of the manuscript is available here: [10.48550/arXiv.2311.05180](https://doi.org/10.48550/arXiv.2311.05180).

Instantiate the `Project.toml` to replicate this research environment. The main scripts are `admm_two_level.jl` and `admm_standard.jl`.

Note that the `OpWater` package, used in the `make_problem_data` script, is private and not made available in this repository. Problem data is instead precompiled and made available in the data folder.