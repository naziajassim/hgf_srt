#using Pkg; Pkg.activate("post_hoc_analysis"); #Pkg.instantiate()
using HierarchicalGaussianFiltering #For the HGF
using ActionModels, LogExpFunctions #For model specification and fitting
using StatsPlots #For plotting
using CSV, DataFrames, JLD2 #For loading and saving data
using Glob #For loading files
using ProgressMeter
#Create the action model
include("custom_action_model.jl")

#Make dictionary for storing outputs
output_data = Dict()

#Load original data
data = CSV.read("data/all_participants_data_for_hgf_clean.csv", DataFrame)

#Set parameter names
param_names = [
    :xprob_volatility,
    :regression_noise,
    :regression_intercept,
    :regression_beta_unexpected_uncertainty,
    :regression_beta_expected_uncertainty,
    :regression_beta_post_error,
    :regression_beta_surprise,
    :regression_beta_post_reversal
]

#Create HGF agent
agent = create_agent();

#Get filenames
filenames = glob("results/per_participant_posteriors/*.jld2")

#For each participant file
@showprogress for filename in filenames

    # Extract the ID
    ID = parse(
        Int,
        split(basename(filename), '_')[1])

    #Load the chains
    participant_chains = load_object(filename)[ID]

    #Extract shape of chains
    n_iter, _, n_chains = size(participant_chains.value)

    #Extract the inputs
    participant_inputs = Array(data[data.SID .== ID, ["Stimt-1", "Stimt", "post_error", "post_reversal"]])

    #Create container for simulated actions
    simulated_actions = zeros(size(participant_inputs, 1), n_iter*n_chains)

    #Go through each sample
    for chain in 1:n_chains
        for iter in 1:n_iter

                #Extract parameters
                param_dict = Dict(String(param) => participant_chains.value[iter,param,chain] for param in param_names)
                
                #Set the parameters in the agent
                set_parameters!(agent, param_dict)
                #Reset the agent
                reset!(agent)
    
                #Simulate actions and store them
                simulated_actions[:,(chain-1)*n_iter + iter] .= give_inputs!(agent, participant_inputs)
        end
    end

    #Store the simulated actions
    output_data[ID] = simulated_actions
end

#Save the output data
save_object("results/post_hoc/posterior_predictive.jld2", output_data)