
#Read packages
using HierarchicalGaussianFiltering #For the HGF
using ActionModels, Distributions, LogExpFunctions #For model specification and fitting
using StatsPlots #For plotting
using CSV, DataFrames, JLD2 #For loading and saving data
using Distributed #For parallelization


#Load the results 
results = load_object("results/fitting_results.jld2")


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





specific_results = results[1003]
plot(specific_results)
plot_parameter_distribution(specific_results, priors)

posteriors = get_posteriors(specific_results, type = "distribution")
posteriors = get_posteriors(specific_results, type = "median")

for (SID, specific_results) in results
    
    posteriors = get_posteriors(specific_results)

    #put in some dataframe

end
