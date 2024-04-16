

### PETER: MAKE BETTER VERSION OF STATE_TRANSTIION ###
### PETER+NAZIA: MAKE ACTION MODEL ###



#### READ EXAMPLE DATA ####

#### CREATE AGENT ####

#### GIVE AGENTS INPUTS TO TEST ####

#### FIT SINGLE AGENTMMODEL TO DATA AS TEST ####

#### FIT ENTIRE DATASET WITH MULTIPLE PARTICIPANTS AND STATISTICAL MODEL ####



using HierarchicalGaussianFiltering
using ActioModels
using Plots, StatsPlots

#Initialize HGF
config = Dict(
    "n_categories_from" => 3,
    "n_categories_to" => 2,
    "include_volatility_parent" => false,
)
HGF = premade_hgf("categorical_state_transitions", config)

#Get all node names
keys(HGF_test.all_nodes)

#See all parameters in the model
get_parameters(HGF)

#Set parameter values
parameters = Dict(
    "xprob_volatility" => 3
)
set_parameters(HGF, parameters)

#Set up inputs, one column per category from, the value is category_to
test_inputs = [
    missing missing 2 missing
    missing 1 missing missing
    missing missing missing 3
    missing missing missing missing
    3 missing missing missing
]

#Give inputs
give_inputs!(HGF, test_inputs)

#Reset as if it had not seen any inputs
reset!(HGF)

#Plot beliefs about all categories
plot_trajectory(HGF, "xcat_4")
#Plot a predicted specific transition probability
plot_trajectory(HGF, "xbin_4_4")
#And its untransformed version
plot_trajectory(HGF, "xprob_4_4")