#Activate posthoc environmemt
using Pkg; Pkg.activate("post_hoc_analysis"); #Pkg.instantiate()

#Load packages
using ActionModels, HierarchicalGaussianFiltering
using CSV, DataFrames, JLD2
using Distributed
#Get script for creating agent
include("custom_action_model.jl")


###### FIND THE QUANTILES OF THE POSTERIORS ######
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


posteriors = CSV.read("results/hgf_posteriors_all_participants.csv", DataFrame)

# Create a DataFrame to store quantiles for each parameter
quantiles_df = DataFrame(param = String[], q0 = Float64[], q25 = Float64[], q50 = Float64[], q75 = Float64[], q100 = Float64[])

# Calculate quantiles for each parameter and add to the DataFrame
for param in param_names
    param_quantiles = quantile(posteriors[posteriors.parameters .== String(param), :mean], [0, 0.25, 0.5, 0.75, 1])
    push!(quantiles_df, (String(param), param_quantiles...))
end

# Display the DataFrame
quantiles_df


### PREP THE RECOVERY ###

#Set ID's to take input sequences from, and volatilities ot recover
participants_to_use = [1003, 1006, 1010, 1015, 1023, 1031, 1038, 1046]
volatilities_to_recover = collect(-9:1:-2)

#Load original data
data = CSV.read("data/all_participants_data_for_hgf_clean.csv", DataFrame, missingstring="NA")
#Create agent
agent = create_agent()
#Number of iterations
n_iter = 1000

### RUN THE RECOVERY ###
#Go through all combinations of participants and volatilities
for (ID, gen_volatility) in Iterators.product(participants_to_use, volatilities_to_recover)

    #Create filename
    filename = "results/recovery_results/inputs_$(ID)_genvol_$(gen_volatility)_iter_$n_iter.jld2"

    # Check if the file already exists
    if isfile(filename)
        println("File $filename already exists. Skipping...")
        continue
    end

    # Get the posterior means for the participant
    parameter_posteriors = Dict()
    for param in param_names
        parameter_posteriors[String(param)] = posteriors[posteriors.SID .== ID .&& posteriors.parameters .== String(param), :mean][1]
    end
    set_parameters!(agent, parameter_posteriors)

    # Set the generative parameters
    generative_params = Dict(
        "xprob_volatility" => gen_volatility,
    )
    set_parameters!(agent, generative_params)
    reset!(agent)

    #Get inputs and simulate actions
    inputs = Array(data[data.SID .== ID, ["Stimt-1", "Stimt", "post_error", "post_reversal"]])
    simulated_actions = give_inputs!(agent, inputs)

    #Get priors
    priors = Dict(
        "xprob_volatility" => Normal(-3, 1),  
        # "regression_noise" => truncated(Normal(exp(-3), .5), lower = 0), # σ
        # "regression_intercept" => Normal(log(500), 1.7),  # β0 
        # "regression_beta_surprise" => Normal(0,2),            # β1
        # "regression_beta_expected_uncertainty" => Normal(0,2), # β2
        # "regression_beta_unexpected_uncertainty" => Normal(0,2), #β3
        # "regression_beta_post_error" => Normal(0,1.5),           # β4
        # "regression_beta_post_reversal" => Normal(0,1.5)           # β5
    )

    #Fit the model
    model = create_model(
        agent,
        priors,
        inputs,
        simulated_actions,
    )
    fitted_model = fit_model(model, n_iterations = n_iter, n_chains = 1)

    #Save the results
    save_object(filename, fitted_model.chains)

end




# ##### RUN THE PARAM REC #####

# #Specify number of cores
# addprocs(4)

# @everywhere begin
#     using ActionModels, HierarchicalGaussianFiltering
#     using CSV, Tables
#     #using Turing
#     include("custom_action_model.jl")


#     # SET GENERARTIVE VALUES TO RECOVER LATER: LET THEM COVER THE SPACE OF ESTIMATES
#     #Parameters to be recovered
#     parameter_ranges = Dict(
#         "xprob_volatility" => collect(-9:1:-2),     
#         "regression_noise" => collect(0.1:0.1:0.3),
#         "regression_intercept" => collect(4.5:1:8.5), 
#         "regression_beta_surprise" => collect(-0.8:0.4:0.5), 
#         "regression_beta_expected_uncertainty" => collect(-2:0.5:1.5), 
#         "regression_beta_unexpected_uncertainty" => collect(-5.5:1:2.5), 
#         "regression_beta_post_error" => collect(-0.05:0.05:0.3),          
#         "regression_beta_post_reversal" => collect(-0.13:0.05:0.12), 
#     )



#     #Create HGF agent
#     agent = create_agent()

#     # SET FIXED VALUES FOR PARAMETRS THAT ARE NOT RECOVERED
#     fixed_parameters = Dict(
#         #"xprob_volatility" => -3,
#         # "regression_noise" => 0.5,
#         # "regression_intercept" => 0.5,
#         # "regression_beta_surprise" => 0.5,
#         # "regression_beta_expected_uncertainty" => 0.5,
#         "regression_beta_unexpected_uncertainty" => 0,
#         # "regression_beta_post_error" => 0.5,
#         # "regression_beta_post_reversal" => 0.5,
#     )
#     set_parameters!(agent, fixed_parameters)

#     #Input sequences to use #LOAD INPUTS HERE
#     input_for_one_participant = CSV.read("results/SID1003_inputs.csv", Tables.matrix) .|> Int64
#     input_sequences = [input_for_one_participant]

#     #Sets of priors to use
#     priors = Dict(
#             "xprob_volatility" => Normal(-3, 1),
#             "regression_noise" => truncated(Normal(exp(-3), .5), lower = 0),
#             "regression_intercept" => Normal(log(500), 1.7),
#             "regression_beta_surprise" => Normal(0,2),          
#             "regression_beta_expected_uncertainty" => Normal(0,2),
#             # "regression_beta_unexpected_uncertainty" => Normal(0,2),
#             "regression_beta_post_error" => Normal(0,1.5),          
#             "regression_beta_post_reversal" => Normal(0,1.5)          
#         )

#     #Times to repeat each simulation
#     n_simulations = 1

#     #Sampler settings
#     sampler_settings = (n_iterations = 1000, n_chains = 1)
# end

# #Run parameter recovery
# results_df = parameter_recovery(
#     agent,
#     parameter_ranges,
#     input_sequences,
#     priors,
#     n_simulations,
#     sampler_settings = sampler_settings,
#     parallel = true,
#     show_progress = true,
# )

# rmprocs(workers())

# #Save results as CSV
# CSV.write("results/parameter_recovery.csv", results_df)

# @show results_df









