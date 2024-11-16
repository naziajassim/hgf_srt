#Activate posthoc environmemt
using Pkg; Pkg.activate("post_hoc_analysis"); #Pkg.instantiate()

#Load packages
using ActionModels, HierarchicalGaussianFiltering
using CSV, DataFrames, JLD2
using Distributed
#Get script for creating agent
include("custom_action_model.jl")


#Specify number of cores
addprocs(4)

@everywhere begin
    using ActionModels, HierarchicalGaussianFiltering
    using CSV, Tables
    #using Turing
    include("custom_action_model.jl")

    #Create HGF agent
    agent = create_agent()

    # SET FIXED VALUES FOR PARAMETRS THAT ARE NOT RECOVERED
    fixed_parameters = Dict(
        #"xprob_volatility" => -3,
        # "regression_noise" => 0.5,
        # "regression_intercept" => 0.5,
        # "regression_beta_surprise" => 0.5,
        # "regression_beta_expected_uncertainty" => 0.5,
        "regression_beta_unexpected_uncertainty" => 0,
        # "regression_beta_post_error" => 0.5,
        # "regression_beta_post_reversal" => 0.5,
    )
    set_parameters!(agent, fixed_parameters)


    # SET GENERARTIVE VALUES TO RECOVER LATER
    #Parameters to be recovered
    parameter_ranges = Dict(
        "xprob_volatility" => collect(-7:1:-1),     
        "regression_noise" => collect(0.01:0.3:1),
        "regression_intercept" => collect(0.01:0.3:1), 
        "regression_beta_surprise" => collect(0.01:0.3:1), 
        "regression_beta_expected_uncertainty" => collect(0.01:0.3:1), 
        # "regression_beta_unexpected_uncertainty" => collect(0.01:0.3:1), 
        "regression_beta_post_error" => collect(0.01:0.3:1),          
        "regression_beta_post_reversal" => collect(0.01:0.3:1), 
    )

    #Input sequences to use #LOAD INPUTS HERE
    input_for_one_participant = CSV.read("results/SID1003_inputs.csv", Tables.matrix) .|> Int64
    input_sequences = [input_for_one_participant]

    #Sets of priors to use
    priors = Dict(
            "xprob_volatility" => Normal(-3, 1),
            "regression_noise" => truncated(Normal(exp(-3), .5), lower = 0),
            "regression_intercept" => Normal(log(500), 1.7),
            "regression_beta_surprise" => Normal(0,2),          
            "regression_beta_expected_uncertainty" => Normal(0,2),
            # "regression_beta_unexpected_uncertainty" => Normal(0,2),
            "regression_beta_post_error" => Normal(0,1.5),          
            "regression_beta_post_reversal" => Normal(0,1.5)          
        )

    #Times to repeat each simulation
    n_simulations = 1

    #Sampler settings
    sampler_settings = (n_iterations = 1000, n_chains = 1)
end

#Run parameter recovery
results_df = parameter_recovery(
    agent,
    parameter_ranges,
    input_sequences,
    priors,
    n_simulations,
    sampler_settings = sampler_settings,
    parallel = true,
    show_progress = true,
)

rmprocs(workers())

#Save results as CSV
CSV.write("results/parameter_recovery.csv", results_df)

@show results_df