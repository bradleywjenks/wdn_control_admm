# wdn_control_admm
This repository contains code implementation for the manuscript entitled "A two-stage distributed algorithm for dynamic control of water networks with convergence gaurantees." The manuscript has been submitted to Optimization and Engineering, a preprint of which is available here: 'insert arXiv link'

Instantiate the `Project.toml` to replicate this research environment. The main script is `admm_distributed.jl`.

Note that the `OpWater` package, used in the `make_problem_data` script, is private and not made available in this repository. Problem data is instead precompiled in the data folder.