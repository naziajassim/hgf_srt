
#Read packages
using HierarchicalGaussianFiltering #For the HGF
using ActionModels, Distributions, LogExpFunctions #For model specification and fitting
using StatsPlots #For plotting
using CSV, DataFrames, JLD2 #For loading and saving data
using Distributed #For parallelization


#Read posteriors
posteriors = load("results/SID1003_posteriors.jld2")
#Read inputs into a matrix
inputs = CSV.read("results/SID1003_inputs.csv", Tables.matrix)


#Create the action model
include("custom_action_model.jl")
#Create HGF agent
agent = create_agent();

#Set parameters in agent
set_parameters!(agent, posteriors)
reset!(agent)

#Extract surprises and evolve the agent
surprises = []

for input in Tuple.(eachrow(inputs))
    single_input!(agent, input)
    push!(surprises, get_surprise(agent))
end

#Get states
history_all_states = get_history(agent)


#history_all_states[("xcat_1", "value_prediction_error")]


#u -> categorical input node
#xcat -> category node for each column in the input
#xbin -> binary parent of the cateogrical node
#xprob -> probability node for each column in the input