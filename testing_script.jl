### PETER:
# NEED TO HAVE
# Check get_surprise for categorical inputs

# NICE TO HAVE
# Fix Categorical HGF pretty print
# ActionModels -> give inputs as tuples
# Parallelization

### FUTURE: MODEL THE BREAK ###
#      - give a missing input with a time distane, make the time distance a parameter #

## READ PACKAGES ##
using HierarchicalGaussianFiltering
using ActionModels
using Plots, StatsPlots
using CSV, DataFrames, Distributions
using LogExpFunctions



######## CREATE AGENT ######
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

agent = create_agent();

give_inputs!(agent, inputs)

get_surprise(agent.substruct, "u1")




node = agent.substruct.all_nodes["u2"]
parent = node.edges.observation_parents[1]

parent.states.prediction

get_surprise(agent)


surprise = sum(-log.(exp.(log.(parent.states.prediction) .* parent.states.posterior)))


node.states.input_value

node.history.input_value


#Create agent
agent = init_agent(reaction_time_action, substruct = hgf, parameters = agent_parameters);

#See parameters in agent
get_parameters(agent)

#Set parameter values
parameters = Dict(
    "xprob_volatility" => -3,
    #"regression_noise" => 0.2,
)
set_parameters!(hgf, parameters)

#Reset as if it had not seen any inputs
reset!(agent)


### TEST SIMULATION ###
#Read dataframe
data = CSV.read("data/singlesub_sample_data_for_hgf_clean.csv", DataFrame, missingstring="NA")

#Set up inputs, one column per category from, the value is category_to
inputs = Array(data[!, [Symbol("Stimt-1"),:Stimt]]);

inputs = hcat(inputs, Int64.(zeros(size(inputs))))

#Give inputs and simulate actions
reset!(agent)
actions = give_inputs!(agent, inputs)

#Plot belief trajectories about all categories
plot_trajectory(agent, "xcat_1")



### SINGLE PARTICIPANT PARAMETER ESTIMATION ###
#Estimate a single parameter
priors = Dict(
    "xprob_volatility" => Normal(-3, 1),
    "regression_noise" => truncated(Normal(0, .5), lower = 0),
    "regression_intercept" => Normal(-1,1),
    "regression_beta_surprise" => Normal(0,1),
    "regression_beta_expected_uncertainty" => Normal(0,1),
    "regression_beta_unexpected_uncertainty" => Normal(0,1),
    "regression_beta_post_error" => Normal(0,1),
)
results = fit_model(agent, priors, inputs, actions)
#Plot the posterior
plot_parameter_distribution(results, priors)
plot(results)




### FULL PARAMETER ESTIMATION ###
#Read data
data = CSV.read("data/all_participants_data_for_hgf_clean.csv", DataFrame, missingstring="NA")

#Subset the data for testing
filter!(row -> row.SID in [1003, 1047], data)
#Add a fake post_error column
# data[!,:post_error] = Int64.(ones(nrow(data)))

#Fit the model for each participant
results = fit_model(
        agent,
        priors,
        data;
        independent_group_cols = [:SID],
        input_cols = [Symbol("Stimt-1"), :Stimt, :post_error],
        action_cols = [:log_RT],
        n_cores = 1,
        n_iterations = 200,
        n_chains = 1,
    )

plot(results[1047])
plot_parameter_distribution(results[1003], priors)
