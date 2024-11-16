using LogExpFunctions

######## CREATE ACTION MODEL ########
function reaction_time_action(agent::Agent, input::Any)

    ### PREPARE INPUT AND UPDATE HGF ###

    #Unpack the input into the observed_category, whether it's a post-error trial, and whether it's post-reversal
    category_from, category_to, post_error_trial, post_reversal = input

    #Change to integers
    category_from, category_to = Int64(category_from), Int64(category_to)

    #Create a vector of missing
    hgf_input = Vector{Union{Missing, Real}}(missing, 4)

    #Set the input to the category we have just transitioned to
    hgf_input[category_from] = category_to

    #Extract the HGF
    hgf = agent.substruct

    #Update the hgf
    update_hgf!(hgf, hgf_input)

    ### GET REACTION TIME ###

    ## Extract relevant nodes ##
    #Extract the input node for the transition_from
    uᵢ = hgf.ordered_nodes.input_nodes[category_from]
    #Get the categorical node for that transition_from 
    cᵢ = uᵢ.edges.observation_parents[1]
    #Get the binary node tracking the observing for that specific transition
    bᵢⱼ = cᵢ.edges.category_parents[category_to]
    #Get the continuous HGF node tracking the probability of that transition
    xᵢⱼ = bᵢⱼ.edges.probability_parents[1]
    
    ## Get belief states relevant for the regression ##
    #Get the suprise for the node
    ℑ = get_surprise(uᵢ)

    #Get the posterior mean and precision for the node tracking the probaility of the observed transition
    μ₂ = get_states(xᵢⱼ, "posterior_mean")
    π₂ = get_states(xᵢⱼ, "posterior_precision")
    
    #Get the belief about the volatility of the probability 
    μ₃ = 0

    #Use names from the marshall paper 
    tendency_observed_transition = logistic(μ₂)
    expected_uncertainy_observed_transition = 1 / π₂

    #Combine to get overall expected uncertainty
    expected_uncertainty =  tendency_observed_transition *
                            (1 - tendency_observed_transition) *
                            expected_uncertainy_observed_transition

    ##Combine to get the volatility-dependent Unexpected uncertainty
    unexpected_uncertainty = tendency_observed_transition *
                             (1 - tendency_observed_transition) *
                             exp(μ₃)


    ### REGRESSION ###
    ## Extract regression parameters ##
    σ = agent.parameters["regression_noise"]
    α = agent.parameters["regression_intercept"]
    βℑ = agent.parameters["regression_beta_surprise"]
    β_expected_uncertainy = agent.parameters["regression_beta_expected_uncertainty"]
    β_unexpected_uncertainty = agent.parameters["regression_beta_unexpected_uncertainty"]
    β_post_error = agent.parameters["regression_beta_post_error"]
    β_post_reversal = agent.parameters["regression_beta_post_reversal"]

    ## Do the regression ##
    reaction_time_prediction =  α + 
                                βℑ * ℑ + 
                                β_expected_uncertainy * expected_uncertainty + 
                                β_unexpected_uncertainty * unexpected_uncertainty + 
                                β_post_error * post_error_trial + 
                                β_post_reversal * post_reversal

    ## Create final action distribution ##
    action_distribution = Normal(reaction_time_prediction, σ)

    #Actions should be log reaction times
    return action_distribution
end


function create_agent()
    
    #Initialize hgf
    config = Dict(
    "n_categories_from" => 4,
    "n_categories_to" => 4,
    "include_volatility_parent" => false,
    )

    hgf = premade_hgf("categorical_state_transitions", config, verbose = false)

    #Agent parameters are regression parameters
    agent_parameters = Dict(
    "regression_noise" => 0,
    "regression_intercept" => 0,
    "regression_beta_surprise" => 0,
    "regression_beta_expected_uncertainty" => 0,
    "regression_beta_unexpected_uncertainty" => 0,
    "regression_beta_post_error" => 0,
    "regression_beta_post_reversal" => 0
    )

    #Create agent
    agent = init_agent(reaction_time_action, substruct = hgf, parameters = agent_parameters);

    return agent
end