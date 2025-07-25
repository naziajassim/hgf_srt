This repository contains modelling scripts for fitting the **categorical state-transition Hierarchical Gaussian Filter (HGF)** to a four-choice probabilistic serial reaction time reversal learning task.  

Please refer to the preprint for details about this project:

> Neurochemical markers of uncertainty processing in humans
Nazia Jassim, Peter Thestrup Waade, Owen Parsons, Frederike H Petzschner, Caterina Rua, Christopher T Rodgers, Simon Baron-Cohen, John Suckling, Christoph Mathys, Rebecca P Lawson
bioRxiv 2025.02.19.639013; doi: https://doi.org/10.1101/2025.02.19.639013

Please refer the documentation for the Julia implementation of the generalised HGF and for details about installation and use of the [Hierarchical Gaussian Filtering package](https://ilabcode.github.io/HierarchicalGaussianFiltering.jl/)

Scripts in this repository:

**1. Custom action model**

_Script name: custom_action_model.jl_

_Purpose: Defines a custom action model to simulate and predict reaction times (RTs) based on belief states derived from a HGF tracking categorical state transitions. The model integrates surprise, expected and unexpected uncertainty, post-error, and post-reversal effects into a regression framework._

The workflow includes:

* Input unpacking and preparation: Takes observed transition categories (category_from, category_to), post-error and post-reversal trial flags, converting them for HGF input.
* HGF update: Updates the HGF state with the observed transition and computes posterior belief states related to transition probabilities.
* Belief state extraction: Extracts surprise, posterior means, precisions, and calculates expected and unexpected uncertainty using logistic transforms consistent with categorical beliefs.
* Regression for reaction time prediction: Combines the extracted belief states and trial flags with agent-specific regression parameters (intercept, betas for surprise, uncertainties, post-error, post-reversal) to compute predicted log RTs.
* Action distribution creation: Constructs a Normal distribution for the predicted reaction time with estimated noise (regression_noise), returning this as the model’s action.
* Agent creation helper: Provides a function to initialize an agent with the categorical HGF configured for 4 categories and regression parameters set to zero as defaults.

Key functions:
* reaction_time_action(agent, input): Takes an agent and trial input, updates the HGF, computes belief states, predicts log RT via regression, and returns a normal action distribution.
* create_agent(): Instantiates the HGF and agent with default regression parameters, ready for simulation or fitting.

Dependencies: LogExpFunctions.jl (for logistic and exponential computations)

Assumes availability of HGF utilities such as premade_hgf, update_hgf!, and init_agent from your modeling framework.

**2. Full model fitting**

_Script name: full_model_fitting.jl_

_Purpose: Performs hierarchical Bayesian model fitting of a custom reaction time action model, based on a categorical Hierarchical Gaussian Filter (HGF), to behavioral data from multiple participants using parallelized computation._

The workflow includes:

* Setup and package loading: Loads essential Julia packages for HGF modeling, action modeling, distributions, plotting, data I/O, and parallel processing.
* Parallelization initialization: Configures multiple worker processes (n_cores) for efficient parallel fitting of participant-specific models.
* Agent creation on all workers: Includes the custom_action_model.jl script and initializes the categorical HGF agent with regression parameters for RT prediction on every parallel worker.
* Data loading and preprocessing: Imports behavioural data from a CSV file. Optionally subsets participants for testing, or for running in smaller subsets if computational resources limited. 
* Prior specification: Defines informative priors for all model parameters including volatility, regression noise, intercept, and regression betas for surprise, expected/unexpected uncertainty, post-error, and post-reversal effects.
* Model fitting: Fits the hierarchical Bayesian model to each participant's data separately using the fit_model function. Inputs specify trial transition columns and outcome RT column. The process runs in parallel with specified iterations and chains.
* Cleanup and saving: Removes parallel workers after fitting and saves the fitting results to a .jld2 file for later analysis.

Key parameters and options:
* subset: Whether to subset the data for quick testing (default true here).
* n_cores: Number of CPU cores/workers for parallel fitting (default 4).
* n_iterations: Number of sampling iterations per chain (set low for testing, increase for full runs).
* n_chains: Number of MCMC chains per participant.
* independent_group_cols: Specifies participant ID column for independent fits.
* input_cols: Specifies the columns representing state transitions and trial flags (post-error, post-reversal).
* action_cols: Specifies the dependent variable column (log_RT) used for fitting.

Dependencies:
HierarchicalGaussianFiltering.jl, ActionModels.jl, Distributions.jl, LogExpFunctions.jl, StatsPlots.jl, CSV.jl, DataFrames.jl, JLD2.jl, Julia standard library Distributed for parallel computing

