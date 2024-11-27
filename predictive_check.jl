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
data = CSV.read("data/all_participants_data_for_hgf_clean.csv", DataFrame, missingstring="NA")

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

output_data = load_object("results/post_hoc/posterior_predictive.jld2")

# Initialize an empty DataFrame with the specified columns
results_df = DataFrame(ID = Int[], trial = Int[], log_RT = Float64[])

@showprogress for (ID, actions) in output_data
    medians = median(actions, dims=2)
    n_trials = size(medians, 1)
    for trial in 1:n_trials
        push!(results_df, (ID, trial, medians[trial]))
    end
end
# Write the results_df to a CSV file
CSV.write("results/post_hoc/predictive_check_medians.csv", results_df)





simulated_data = CSV.read("results/post_hoc/predictive_check_medians.csv", DataFrame)


# Plot the log_RTs of each participant in a single plot
@df simulated_data plot(:trial, :log_RT, group = :ID, 
    xlabel = "Trial", ylabel = "log_RT", 
    title = "log_RTs of each participant")


@df data plot(:Trial, :log_RT, group = :SID, 
    xlabel = "Trial", ylabel = "log_RT", 
    title = "log_RTs of each participant")

# Calculate the mean log_RT across participants for each trial
sim_mean_log_RT_across_trials = combine(groupby(dropmissing(simulated_data), :trial), 
    :log_RT => mean => :mean_log_RT,
    :log_RT => std => :std_log_RT)

real_mean_log_RT_across_trials = combine(groupby(dropmissing(data), [:Session, :Block, :Trial]), 
    :log_RT => mean => :mean_log_RT,
    :log_RT => std => :std_log_RT)


plot(sim_mean_log_RT_across_trials[:, :mean_log_RT])
plot(sim_mean_log_RT_across_trials[:, :mean_log_RT], ribbon = sim_mean_log_RT_across_trials[:, :std_log_RT])


plot(real_mean_log_RT_across_trials[:, :mean_log_RT])
plot(real_mean_log_RT_across_trials[:, :mean_log_RT], ribbon = real_mean_log_RT_across_trials[:, :std_log_RT])
