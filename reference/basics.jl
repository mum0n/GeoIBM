
# source:  https://juliadynamics.github.io/Agents.jl/stable/tutorial/


# 1. space
# GraphSpace: Agent positions are equivalent with nodes of a graph/network.
# GridSpace: Space is discretized into boxes, typical style for cellular automata.
# ContinuousSpace: Truthful representation of continuous space, regarding location, orientation, and identification of neighboring agents.
# OpenStreetMapSpace: A space based on Open Street Map, where agents are confined to move along streets of the map, using real-world meter values for the length of each street.


# 2. agent type
# subtypes of AbstractAgent. It is almost always the case that mutable Types make for a simpler modelling experience.
# must have the id field as first field
# pos field of appropriate type and a vel field of appropriate type.
# other additional fields relevant to your system, for example variable quantities like "status" or other "counters"
# eg: for a GraphSpace, use the @agent macro
mutable struct ExampleAgent <: AbstractAgent
  id::Int
  pos::Int
  weight::Float64
  happy::Bool
end

# afterwards: add_agent!, move_agent! or kill_agent!.



# 3. model
AgentBasedModel(AgentType [, space]; properties, kwargs...) → model
# or ABM(ExampleAgent) 
# space is a subtype of AbstractSpace
# Spaces are mutable objects and are not designed to be shared between models. Create a fresh instance of a space with the same properties if you need to do this.
# properties = nothing is additional model-level properties (typically a dictionary) that can be accessed as model.properties. If properties is a dictionary with key type Symbol, or of it is a struct, then the syntax model.name is short hand for model.properties[:name] (or model.properties.name for structs). This syntax can't be used for name being agents, space, scheduler, properties, rng, maxid, which are the fields of AgentBasedModel.

# scheduler = fastest decides the order with which agents are activated (see e.g. by_id and the scheduler API). scheduler is only meaningful if an agent-stepping function is defined for step! or run!, otherwise a user decides a scheduler in the model-stepping function, as illustrated in the Advanced stepping part of the tutorial.

# rng = Random.default_rng() provides random number generation to the model. Accepts any subtype of AbstractRNG and is accessed by model.rng.

# warn=true: Type tests for AgentType are done, and by default warnings are thrown when appropriate.


# 4. time evolvution
# Any ABM model should have at least one and at most two step functions.
step!(model, agent_step!, n::Int = 1)
# agent step function: is required by default. Such an agent step function defines what happens to an agent when it activates
#   two arguments: first, an agent object, and second, a model object.

step!(model, agent_step!, model_step!, n::Int = 1, agents_first::Bool = true)
# Update agents n steps according to the stepping function agent_step!. Agents will be activated as specified by the model.scheduler. model_step! is triggered after every scheduled agent has acted, unless the argument agents_first is false (which then first calls model_step! and then activates the agents).

step!(model, agent_step!, model_step!, n::Function, agents_first::Bool = true)
# In this version n is a function. Then step! runs the model until n(model, s) returns true, where s is the current amount of steps taken, starting from 0. For this method of step!, model_step! must be provided always (use dummystep if you have no model stepping dynamics).

dummystep(model)
dummystep(agent, model)
# model step function: Sometimes we also need a function that changes all agents at once, or changes a model property
#   one argument, that is the model object
#   dummystep as the agent step function

# this simplified approach offers the benefit of not having to write an explicit loop over existing agents inside the model_step!. 
# more advanced models require special handling for scheduling, or may need to schedule several times and act on different subsets of agents with different functions. In such a scenario, it is more sensible to provide only a model_step! function (and use dummystep as agent_step!), where all configuration is contained within. Notice that if you follow this road, the argument scheduler given to AgentBasedModel somewhat loses its meaning.
# see:  https://juliadynamics.github.io/Agents.jl/stable/api/#Schedulers
function complex_step!(model)
  for a in scheduler1(model)
      agent_step1!(a, model)
  end
  intermediate_model_action!(model)
  for a in scheduler2(model)
      agent_step2!(a, model)
  end
  final_model_action!(model)
end

step!(model, dummystep, complex_step!, n)


# collect/record data
adata::Vector means "agent data to collect" (aggregation)
#. If an entry is a Symbol, e.g. :weight, then the data for this entry is agent's field weight. If an entry is a Function, e.g. f, then the data for this entry is just f(a) for each agent a. The resulting dataframe columns are named with the input symbol (here :weight, :f).

adata::Vector{<:Tuple}
# if adata is a vector of tuples instead, data aggregation is done over the agent properties.
# For each 2-tuple, the first entry is the "key" (any entry like the ones mentioned above, e.g. :weight, f). The second entry is an aggregating function that aggregates the key, e.g. mean, maximum. So, continuing from the above example, we would have adata = [(:weight, mean), (f, maximum)].

# It's also possible to provide a 3-tuple, with the third entry being a conditional function (returning a Bool), which assesses if each agent should be included in the aggregate. For example: x_pos(a) = a.pos[1]>5 with (:weight, mean, x_pos) will result in the average weight of agents conditional on their x-position being greater than 5.

# The resulting data name columns use the function aggname, and create something like :mean_weight or :maximum_f_x_pos. This name doesn't play well with anonymous functions, but you can simply use DataFrames.rename! to change the returned dataframe's column names.

# Notice: Aggregating only works if there are agents to be aggregated over. If you remove agents during model run, you should modify the aggregating functions. E.g. instead of passing mean, pass mymean(a) = isempty(a) ? 0.0 : mean(a).

mdata::Vector means "model data to collect"  (no aggregation)
# and works exactly like adata. For the model, no aggregation is possible (nothing to aggregate over).

# in more complex situations: https://juliadynamics.github.io/Agents.jl/stable/api/#Data-collection
function assets(model)
  total_savings(model) = model.bank_balance + sum(model.assets)
  function stategy(model)
      if model.year == 0
          return model.initial_strategy
      else
          return get_strategy(model)
      end
  end
  return [:age, :details, total_savings, strategy]
end
run!(model, agent_step!, model_step!, 10; mdata = assets(model))

