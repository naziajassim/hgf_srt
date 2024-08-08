
#Read packages
using HierarchicalGaussianFiltering #For the HGF
using ActionModels, Distributions, LogExpFunctions #For model specification and fitting
using StatsPlots #For plotting
using CSV, DataFrames, JLD2 #For loading and saving data
using Distributed #For parallelization


#Load the results 
results = load_object("results/fitting_results.jld2")




# GET THINGS FROM BEFORE
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

#Create the action model
include("custom_action_model.jl")

agent = create_agent();


#Read data
data = CSV.read("data/all_participants_data_for_hgf_clean.csv", DataFrame, missingstring="NA");

#Filter for only subject 1043
data = filter(row -> row.SID == 1043, data);

#Extract inputs
inputs = Array(data[!, [Symbol("Stimt-1"), :Stimt, :post_error, :post_reversal]]);






#GET ESTIMATED PARAMETERS

# for (SID, specific_results) in results
    
#     posteriors = get_posteriors(specific_results)

#     #put in some dataframe

# end


specific_results = results[1003]

plot(specific_results)
plot_parameter_distribution(specific_results, priors)

posteriors_full = get_posteriors(specific_results, type = "distribution")
# and here this could be put into a dataframe for further analysis
# in a for loop
#Or get confidence intervals blabla


# SIMULATE BELIEF TRAJECTORIES WITH ESIMTATED PARAMETERS

#Get out posterior estimate medians
posteriors = get_posteriors(specific_results, type = "median")

#Set them in the agent
set_parameters!(agent, posteriors)
reset!(agent)

#Simulate forward with the agent
give_inputs!(agent, inputs)

#Plot belief trajectories over time
plot_trajectory(agent, "xcat_1")
plot_trajectory(agent, "xcat_2")
plot_trajectory(agent, "xcat_3")
plot_trajectory(agent, "xcat_4")

#Get out all belief trajectories (beliefs, prediction errors etc)
histories = get_history(agent)
