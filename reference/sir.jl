






# # SIR model for the spread of COVID-19
# ![](covid_evolution.gif)
#
# This example illustrates how to use `GraphSpace` and how to model agents on an graph
# (network) where the transition probabilities between each node (position) is not constant.
# ## SIR model

# A SIR model tracks the ratio of Susceptible, Infected, and Recovered individuals within a population.
# Here we add one more category of individuals: those who are infected, but do not know it.
# Transmission rate for infected and diagnosed individuals is lower than infected and undetected.
# We also allow a fraction of recovered individuals to catch the disease again, meaning
# that recovering the disease does not bring full immunity.

# ## Model parameters
# Here are the model parameters, some of which have default values.
# * `Ns`: a vector of population sizes per city. The amount of cities is just `C=length(Ns)`.
# * `β_und`: a vector for transmission probabilities β of the infected but undetected per city.
#   Transmission probability is how many susceptible are infected per day by an infected individual.
#   If social distancing is practiced, this number increases.
# * `β_det`: an array for transmission probabilities β of the infected and detected per city.
#   If hospitals are full, this number increases.
# * `infection_period = 30`: how many days before a person dies or recovers.
# * `detection_time = 14`: how many days before an infected person is detected.
# * `death_rate = 0.02`: the probability that the individual will die after the `infection_period`.
# * `reinfection_probability = 0.05`: The probability that a recovered person can get infected again.
# * `migration_rates`: A matrix of migration probability per individual per day from one city to another.
# * `Is = [zeros(C-1)..., 1]`: An array for initial number of infected but undetected people per city.
#   This starts as only one infected individual in the last city.

# Notice that `Ns, β, Is` all need to have the same length, as they are numbers for each
# city. We've tried to add values to the infection parameters similar to the ones you would hear
# on the news about COVID-19.

# The good thing with Agent based models is that you could easily extend the model we
# implement here to also include age as an additional property of each agent.
# This makes ABMs flexible and suitable for research of virus spreading.

# ## Making the model in Agents.jl
# We start by defining the `PoorSoul` agent type and the ABM

using Agents, Random, DataFrames, LightGraphs
using Distributions: Poisson, DiscreteNonParametric
using DrWatson: @dict
using Plots
gr() # hide

mutable struct PoorSoul <: AbstractAgent
    id::Int
    pos::Int
    days_infected::Int  # number of days since is infected
    status::Symbol  # 1: S, 2: I, 3:R
end

function model_initiation(;
    Ns,
    migration_rates,
    β_und,
    β_det,
    infection_period = 30,
    reinfection_probability = 0.05,
    detection_time = 14,
    death_rate = 0.02,
    Is = [zeros(Int, length(Ns) - 1)..., 1],
    seed = 0,
)

    rng = MersenneTwister(seed)
    @assert length(Ns) ==
    length(Is) ==
    length(β_und) ==
    length(β_det) ==
    size(migration_rates, 1) "length of Ns, Is, and B, and number of rows/columns in migration_rates should be the same "
    @assert size(migration_rates, 1) == size(migration_rates, 2) "migration_rates rates should be a square matrix"

    C = length(Ns)
    ## normalize migration_rates
    migration_rates_sum = sum(migration_rates, dims = 2)
    for c in 1:C
        migration_rates[c, :] ./= migration_rates_sum[c]
    end

    properties = @dict(
        Ns,
        Is,
        β_und,
        β_det,
        β_det,
        migration_rates,
        infection_period,
        infection_period,
        reinfection_probability,
        detection_time,
        C,
        death_rate
    )
    space = GraphSpace(complete_digraph(C))
    model = ABM(PoorSoul, space; properties, rng)

    ## Add initial individuals
    for city in 1:C, n in 1:Ns[city]
        ind = add_agent!(city, model, 0, :S) # Susceptible
    end
    
    ## add infected individuals
    for city in 1:C
        inds = ids_in_position(city, model)
        for n in 1:Is[city]
            agent = model[inds[n]]
            agent.status = :I # Infected
            agent.days_infected = 1
        end
    end
    return model
end
nothing # hide


# We will make a function that starts a model with `C` number of cities,
# and creates the other parameters automatically by attributing some random
# values to them. You could directly use the above constructor and specify all
# `Ns, β`, etc. for a given set of cities.

# All cities are connected with each other, while it is more probable to travel from a city
# with small population into a city with large population.

using LinearAlgebra: diagind

function create_params(;
    C,
    max_travel_rate,
    infection_period = 30,
    reinfection_probability = 0.05,
    detection_time = 14,
    death_rate = 0.02,
    Is = [zeros(Int, C - 1)..., 1],
    seed = 19,
)

    Random.seed!(seed)
    Ns = rand(50:5000, C)
    β_und = rand(0.3:0.02:0.6, C)
    β_det = β_und ./ 10

    Random.seed!(seed)
    migration_rates = zeros(C, C)
    for c in 1:C
        for c2 in 1:C
            migration_rates[c, c2] = (Ns[c] + Ns[c2]) / Ns[c]
        end
    end
    maxM = maximum(migration_rates)
    migration_rates = (migration_rates .* max_travel_rate) ./ maxM
    migration_rates[diagind(migration_rates)] .= 1.0

    params = @dict(
        Ns,
        β_und,
        β_det,
        migration_rates,
        infection_period,
        reinfection_probability,
        detection_time,
        death_rate,
        Is
    )

    return params
end

params = create_params(C = 8, max_travel_rate = 0.01)
model = model_initiation(; params...)

# Alright, let's plot the cities as a graph to get an idea how the model "looks like",
# using the function [`abm_plot_on_graph`](@ref).

using Plots

# !!! info "AgentsPlots no longer supported"
#     [AgentsPlots.jl](https://github.com/JuliaDynamics/AgentsPlots.jl) is now deprecated
#     in favor of [InteractiveDynamics.jl](https://github.com/JuliaDynamics/InteractiveDynamics.jl).
#     This example has not yet been transferred to the new paradigm, but will do so in the
#     near future.

plotargs = (node_size = 0.2, method = :circular, linealpha = 0.4)

abm_plot_on_graph(model; plotargs...)

# The node size is proportional to the relative population of each city.
# In principle we could adjust the edge widths to be proportional with the
# migration rates, by doing:

g = model.space.graph
edgewidthsdict = Dict()
for node in 1:nv(g)
    nbs = neighbors(g, node)
    for nb in nbs
        edgewidthsdict[(node, nb)] = params[:migration_rates][node, nb]
    end
end

edgewidthsf(s, d, w) = edgewidthsdict[(s, d)] * 250

plotargs = merge(plotargs, (edgewidth = edgewidthsf,))

abm_plot_on_graph(model; plotargs...)

# In the following we will be coloring each node according to how large percentage of the
# population is infected. So we create a function to give to [`abm_plot_on_graph`](@ref) as a
# second argument

infected_fraction(x) = cgrad(:inferno)[count(a.status == :I for a in x) / length(x)]
abm_plot_on_graph(model; ac = infected_fraction, plotargs...)

# Here this shows all nodes as black, since we haven't run the model yet. Let's change that!

# ## SIR Stepping functions

# Now we define the functions for modelling the virus spread in time

function agent_step!(agent, model)
    migrate!(agent, model)
    transmit!(agent, model)
    update!(agent, model)
    recover_or_die!(agent, model)
end

function migrate!(agent, model)
    pid = agent.pos
    d = DiscreteNonParametric(1:(model.C), model.migration_rates[pid, :])
    m = rand(model.rng, d)
    if m ≠ pid
        move_agent!(agent, m, model)
    end
end

function transmit!(agent, model)
    agent.status == :S && return
    rate = if agent.days_infected < model.detection_time
        model.β_und[agent.pos]
    else
        model.β_det[agent.pos]
    end

    d = Poisson(rate)
    n = rand(model.rng, d)
    n == 0 && return

    for contactID in ids_in_position(agent, model)
        contact = model[contactID]
        if contact.status == :S ||
           (contact.status == :R && rand(model.rng) ≤ model.reinfection_probability)
            contact.status = :I
            n -= 1
            n == 0 && return
        end
    end
end

update!(agent, model) = agent.status == :I && (agent.days_infected += 1)

function recover_or_die!(agent, model)
    if agent.days_infected ≥ model.infection_period
        if rand(model.rng) ≤ model.death_rate
            kill_agent!(agent, model)
        else
            agent.status = :R
            agent.days_infected = 0
        end
    end
end
nothing # hide

# ## Example animation
model = model_initiation(; params...)

anim = @animate for i in 0:30
    i > 0 && step!(model, agent_step!, 1)
    p1 = abm_plot_on_graph(model; ac = infected_fraction, plotargs...)
    title!(p1, "Day $(i)")
end

gif(anim, "covid_evolution.gif", fps = 5)

# One can really see "explosive growth" in this animation. Things look quite calm for
# a while and then suddenly supermarkets have no toilet paper anymore!

# ## Exponential growth

# We now run the model and collect data. We define two useful functions for
# data collection:
infected(x) = count(i == :I for i in x)
recovered(x) = count(i == :R for i in x)
nothing # hide

# and then collect data
model = model_initiation(; params...)

to_collect = [(:status, f) for f in (infected, recovered, length)]
data, _ = run!(model, agent_step!, 100; adata = to_collect)
data[1:10, :]

# We now plot how quantities evolved in time to show
# the exponential growth of the virus

N = sum(model.Ns) # Total initial population
x = data.step
p = plot(
    x,
    log10.(data[:, aggname(:status, infected)]),
    label = "infected",
    xlabel = "steps",
    ylabel = "log10(count)",
)
plot!(p, x, log10.(data[:, aggname(:status, recovered)]), label = "recovered")
dead = log10.(N .- data[:, aggname(:status, length)])
plot!(p, x, dead, label = "dead")

# The exponential growth is clearly visible since the logarithm of the number of infected increases
# linearly, until everyone is infected.




# --------------------------------------

# SIR example with optimization
# https://juliadynamics.github.io/Agents.jl/stable/examples/optim/
# https://github.com/robertfeldt/BlackBoxOptim.jl


# use the evolutionary algorithms in BlackBoxOptim.jl with Agents.jl, 
# to optimize the parameters of an epidemiological model (SIR). 
# We explain this model in detail in SIR model for the spread of COVID-19. 

#or: include("siroptim.jl") # From the examples directory

using Agents, Random, DataFrames, LightGraphs
using Distributions: Poisson, DiscreteNonParametric
using DrWatson: @dict
using LinearAlgebra: diagind

mutable struct PoorSoul <: AbstractAgent
    id::Int
    pos::Int
    days_infected::Int  # number of days since is infected
    status::Symbol  # 1: S, 2: I, 3:R
end

function model_initiation(;
    Ns = [500, 500, 500],
    migration_rate = 0.2,
    death_rate = 0.1,
    β_det = 0.05,
    β_und = 0.1,
    infection_period = 10,
    reinfection_probability = 0.05,
    detection_time = 3,
    Is = ones(Int, length(Ns)),
    seed = nothing,
)

    if !isnothing(seed)
        Random.seed!(seed)
    end

    C = length(Ns)
    β_det = [β_det for i in 1:C]
    β_und = [β_und for i in 1:C]

    migration_rates = reshape([migration_rate for i in 1:(C * C)], C, C)
    # normalize migration_rates
    migration_rates_sum = sum(migration_rates, dims = 2)
    for c in 1:C
        migration_rates[c, :] ./= migration_rates_sum[c]
    end

    properties = @dict(
        Ns,
        β_und,
        β_det,
        migration_rates,
        infection_period,
        reinfection_probability,
        detection_time,
        death_rate,
        Is,
        C,
    )
    space = GraphSpace(complete_digraph(C))
    model = ABM(PoorSoul, space; properties)

    # Add initial individuals
    for city in 1:C, n in 1:Ns[city]
        add_agent!(city, model, 0, :S) # Susceptible
    end
    # add infected individuals
    for city in 1:C
        inds = ids_in_position(city, model)
        for n in 1:Is[city]
            agent = model[inds[n]]
            agent.status = :I # Infected
            agent.days_infected = 1
        end
    end
    return model
end

function agent_step!(agent, model)
    migrate!(agent, model)
    transmit!(agent, model)
    update_status!(agent, model)
    recover_or_die!(agent, model)
end

function migrate!(agent, model)
    pid = agent.pos
    d = DiscreteNonParametric(1:(model.C), model.migration_rates[pid, :])
    m = rand(model.rng, d)
    if m ≠ pid
        move_agent!(agent, m, model)
    end
end

function transmit!(agent, model)
    agent.status == :S && return
    rate = if agent.days_infected < model.detection_time
        model.β_und[agent.pos]
    else
        model.β_det[agent.pos]
    end

    d = Poisson(rate)
    n = rand(model.rng, d)
    n == 0 && return

    for contactID in ids_in_position(agent, model)
        contact = model[contactID]
        if contact.status == :S ||
           (contact.status == :R && rand(model.rng) ≤ model.reinfection_probability)
            contact.status = :I
            n -= 1
            n == 0 && return
        end
    end
end

update_status!(agent, model) = agent.status == :I && (agent.days_infected += 1)

function recover_or_die!(agent, model)
    if agent.days_infected ≥ model.infection_period
        if rand(model.rng) ≤ model.death_rate
            kill_agent!(agent, model)
        else
            agent.status = :R
            agent.days_infected = 0
        end
    end
end



# The cost function takes as arguments the model parameters that we want to tune; in a SIR model, that would be the migration rate, death rate, transmission rate, when an infected person has been detected (β_det), or when the remain undetected (β_und), infection period, reinfection probability, and time until the infection is detected.

using BlackBoxOptim, Random
using Statistics: mean

function cost(x)
    model = model_initiation(;
        Ns = [500, 500, 500],
        migration_rate = x[1],
        death_rate = x[2],
        β_det = x[3],
        β_und = x[4],
        infection_period = x[5],
        reinfection_probability = x[6],
        detection_time = x[7],
    )

    infected_fraction(model) =
        count(a.status == :I for a in allagents(model)) / nagents(model)

    _, data = run!(
        model,
        agent_step!,
        50;
        mdata = [infected_fraction],
        when_model = [50],
        replicates = 10,
    )

    return mean(data.infected_fraction)
end

Random.seed!(10)

x0 = [
    0.2,  # migration_rate
    0.1,  # death_rate
    0.05, # β_det
    0.3,  # β_und
    10,   # infection_period
    0.1,  # reinfection_probability
    5,    # detection_time
]
cost(x0)

result = bboptimize(
    cost,
    SearchRange = [
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (7.0, 13.0),
        (0.0, 1.0),
        (2.0, 6.0),
    ],
    NumDimensions = 7,
    MaxTime = 20,
)
best_fitness(result)

best_candidate(result)

x = best_candidate(result)

Random.seed!(0)

model = model_initiation(;
    Ns = [500, 500, 500],
    migration_rate = x[1],
    death_rate = x[2],
    β_det = x[3],
    β_und = x[4],
    infection_period = x[5],
    reinfection_probability = x[6],
    detection_time = x[7],
)

_, data =
    run!(model, agent_step!, 50; mdata = [nagents], when_model = [50], replicates = 10)

mean(data.nagents)


function cost_multi(x)
  model = model_initiation(;
      Ns = [500, 500, 500],
      migration_rate = x[1],
      death_rate = x[2],
      β_det = x[3],
      β_und = x[4],
      infection_period = x[5],
      reinfection_probability = x[6],
      detection_time = x[7],
  )

  initial_size = nagents(model)

  infected_fraction(model) =
      count(a.status == :I for a in allagents(model)) / nagents(model)
  n_fraction(model) = -1.0 * nagents(model) / initial_size

  _, data = run!(
      model,
      agent_step!,
      50;
      mdata = [infected_fraction, n_fraction],
      when_model = [50],
      replicates = 10,
  )

  return mean(data.infected_fraction), mean(data.n_fraction)
end


# Notice that our new objective n_fraction is negative. It would be simpler to state we'd like to 'maximise the living population', but the optimiser we're using here focuses on minimising objectives only, therefore we must 'minimise the number of agents dying'.

cost_multi(x0)

result = bboptimize(
    cost_multi,
    Method = :borg_moea,
    FitnessScheme = ParetoFitnessScheme{2}(is_minimizing = true),
    SearchRange = [
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (7.0, 13.0),
        (0.0, 1.0),
        (2.0, 6.0),
    ],
    NumDimensions = 7,
    MaxTime = 55,
)
best_fitness(result)

best_candidate(result)

# The algorithm managed to minimize the number of infected and deaths while still increasing death rate to 42%, reinfection probability to 53%, and migration rates to 33%. The most important change however, was decreasing the transmission rate when individuals are infected and undetected from 30% in our initial calculation, to 0.2%.

#  Over a longer period of time than 50 days, that high death rate will take its toll though. Let's reduce that rate and check the cost.

x = best_candidate(result)
x[2] = 0.02
cost_multi(x)

# The fraction of infected increases to 0.04%. This is an interesting result: since this virus model is not as deadly, the chances of re-infection increase. 



