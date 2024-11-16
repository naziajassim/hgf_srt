using Pkg; Pkg.activate("post_hoc_analysis"); #Pkg.instantiate()

using ActionModels, Turing
include("custom_action_model.jl")

using Distributed
addprocs(4)

@everywhere begin
    using ActionModels, HierarchicalGaussianFiltering
    using CSV, DataFrames
    include("custom_action_model.jl")

    agent = create_agent();

    data = CSV.read("data/all_participants_data_for_hgf_clean.csv", DataFrame, missingstring="NA");

    priors = Dict(
        "xprob_volatility" => Normal(-3, 1),            # unchanged
        "regression_noise" => truncated(Normal(exp(-3), .5), lower = 0), # σ
        "regression_intercept" => Normal(log(500), 1.7),  # β0 
        "regression_beta_surprise" => Normal(0,2),            # β1
        "regression_beta_expected_uncertainty" => Normal(0,2), # β2
        # "regression_beta_unexpected_uncertainty" => Normal(0,2), #β3  --- this is fixed!
        "regression_beta_post_error" => Normal(0,1.5),           # β4
        "regression_beta_post_reversal" => Normal(0,1.5)           # β5
    );

    model = create_model(
        agent, priors, data, 
        grouping_cols = "SID",
        input_cols = ["Stimt-1", "Stimt", "post_error", "post_reversal"],
        action_cols = "log_RT"
        );
end


sampler = NUTS(-1, 0.65; adtype = AutoReverseDiff(; compile = true))
n_iterations = 10
n_chains = 4

results = fit_model(
    model;
    parallelization = MCMCDistributed(),
    sampler = sampler,
    n_iterations = n_iterations,
    n_chains = n_chains,
)

rmprocs(workers())



using TuringBenchmarking
benchmark = benchmark_model(
           model;
           # Check correctness of computations
           check=true,
           # Automatic differentiation backends to check and benchmark
           adbackends=[:forwarddiff, :reversediff, :reversediff_compiled, :zygote],
       )