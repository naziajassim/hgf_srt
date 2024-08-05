## SETUP ##

#Read packages
using HierarchicalGaussianFiltering #For the HGF
using ActionModels, Distributions, LogExpFunctions #For model specification and fitting
using StatsPlots #For plotting
using CSV, DataFrames, JLD2 #For loading and saving data
using Distributed #For parallelization

#Whether to subset
subset = true
#Number of cores for parallelization
n_cores = 4

#Create workers for parallelization
addprocs(n_cores, exeflags="--project=.")

######## CREATE AGENT ######
#On all workers
@everywhere begin
    using HierarchicalGaussianFiltering 
    using ActionModels, Distributions, LogExpFunctions

    #Create the action model
    include("custom_action_model.jl")

    #Initialize hgf
    config = Dict(
    "n_categories_from" => 4,
    "n_categories_to" => 4,
    "include_volatility_parent" => false,
    )
    hgf = premade_hgf("categorical_state_transitions", config)

    #Agent parameters are regression parameters
    agent_parameters = Dict(
    "regression_noise" => 0.1,
    "regression_intercept" => 0.5,
    "regression_beta_surprise" => 0.1,
    "regression_beta_expected_uncertainty" => 0.1,
    "regression_beta_unexpected_uncertainty" => 0.1,
    "regression_beta_post_error" => 0.1,
    "regression_beta_post_reversal" => 0.1
    )

    #Create agent
    agent = init_agent(reaction_time_action, substruct = hgf, parameters = agent_parameters);
end



######## READ DATA ######
#Read data
data = CSV.read("data/all_participants_data_for_hgf_clean.csv", DataFrame, missingstring="NA")

if subset
    #Subset the data for testing
    filter!(row -> row.SID in [1003, 1047], data);
    #Add a fake post_error column
    data[!,:post_error] = Int64.(ones(nrow(data)));
end

######## FIT TO DATA ######
priors = Dict(
    "xprob_volatility" => Normal(-3, 1),            # unchanged
    "regression_noise" => truncated(Normal(exp(-3), .5), lower = 0), # unchanged- throws out errors if noise is negative
    "regression_intercept" => Normal(log(500), 1.7),  # β0 
    "regression_beta_surprise" => Normal(0,2),            # β1
    "regression_beta_expected_uncertainty" => Normal(0,2), # β2
    "regression_beta_unexpected_uncertainty" => Normal(0,2), #β3
    "regression_beta_post_error" => Normal(0,1.5),           # β4
    "regression_beta_post_reversal" => Normal(0,1.5)           # β5
)

#Fit the model for each participant
results = fit_model(
        agent,
        priors,
        data;
        independent_group_cols = [:SID],
        input_cols = [Symbol("Stimt-1"), :Stimt, :post_error, :post_reversal],
        action_cols = [:log_RT],
        n_cores = n_cores,
        n_iterations = 100, #2000
        n_chains = 2, #4
    )


######## CLEANUP ######
# Remove workers
rmprocs(workers())

# Save data
save_object("results/fitting_results.jld2", results)

