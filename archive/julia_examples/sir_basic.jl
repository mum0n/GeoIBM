

# SIR ultra-simple version 
# based upon: https://juliadynamics.github.io/Agents.jl/stable/examples/sir.jl

# A SIR model tracks the ratio of Susceptible, Infected, and Recovered individuals within a population.
# * `Ns`: population size  
# * `β_und`:  transmission probabilities β of the infected but undetected 
#   Transmission probability is how many susceptible are infected per day by an infected individual.
#   If social distancing is practiced, this number increases.
# * `β_det`:   transmission probabilities β of the infected and detected .
#   If hospitals are full, this number increases.
# * `infection_period = 30`: how many days before a person dies or recovers.
# * `detection_time = 14`: how many days before an infected person is detected.
# * `death_rate = 0.02`: the probability that the individual will die after the `infection_period`.
# * `reinfection_probability = 0.05`: The probability that a recovered person can get infected again.
# * `Is = 1 initial number of infected but undetected people .



# # 

using Agents, Random, DataFrames
using Distributions: Poisson 


mutable struct PoorSoul <: AbstractAgent
    id::Int
    days_infected::Int  # number of days since is infected
    status::Symbol  # 1: S, 2: I, 3:R
end



function transmit!(agent, model)
    #agent=model[73]
    agent.status == :S && return
    rate = if agent.days_infected < model.detection_time
        model.β_und
    else
        model.β_det
    end

    d = Poisson(rate)
    n = rand(model.rng, d)
    n == 0 && return

    for contactID in 1:model.Ns
        contact = model[contactID]
        if contact.status == :S ||
           (contact.status == :R && rand(model.rng) ≤ model.reinfection_probability)
            contact.status = :I
            n -= 1
            n == 0 && return
        end
        
    end
end

function recover_or_die!(agent, model)
    if agent.days_infected ≥ model.infection_period
        if rand(model.rng) ≤ model.death_rate
            agent.status = :D
        else
            agent.status = :R
            agent.days_infected = 0
        end
    end
end

function agent_step!(agent, model)
    transmit!(agent, model)
    update!(agent, model)
    recover_or_die!(agent, model)
end

update!(agent, model) = agent.status == :I && (agent.days_infected += 1)

function sir_basic(
        Ns = 10000,
        Is = 1,
        β_und = rand(0.3:0.02:0.6),
        β_det = rand(0.3:0.02:0.6) / 10 ,
        infection_period = 30,
        reinfection_probability = 0.05,
        detection_time = 14,
        death_rate = 0.02,
        seed = 0 
    ) 

    Random.seed!(seed)
    rng = MersenneTwister(seed)

    properties = Dict( 
        :Ns => Ns ,
        :Is => Is ,
        :β_und => β_und ,
        :β_det => β_det ,
        :infection_period => infection_period ,
        :reinfection_probability => reinfection_probability ,
        :detection_time => detection_time ,
        :death_rate => death_rate 
    )

    model = ABM(PoorSoul; properties, rng)

    ## Add initial individuals
    for  n in 1:model.Ns 
        ind = add_agent!(model, 0, :S) # Susceptible ... n is the positional index but 
    end

    ## add infected individuals
    for n in 1:model.Is
        ind = add_agent!(model, 1, :I) # Susceptible ... n is the positional index but 
    end
    model
end

model = sir_basic()
model[100]


# tests to make sure it works and precompile
Agents.step!( model, transmit! )
Agents.step!( model, recover_or_die! )
Agents.step!( model, update! )
Agents.step!( model, agent_step! )


# We now run the model and collect data. We define two useful functions for
# data collection:
infected(x) = count(i == :I for i in x)
recovered(x) = count(i == :R for i in x)
deaths(x) = count(i == :D for i in x)

to_collect = [(:status, f) for f in (infected, recovered, deaths, length)]


if (0 < 0) 
   
    # to run from within R:

    library(JuliaCall)

    # install_julia()
    # julia_install_package_if_needed("Optim")
    # julia_installed_package("Optim")

    julia_source("/home/jae/models/abm/sir_basic.jl")  # load julia functions

    # run 1 iteration to test
    o = julia_eval( "data, _ = run!(model, agent_step!, 1; adata = to_collect) ")

    o = julia_eval( "data, _ = run!(model, agent_step!, 50; adata = to_collect) ")

    plot( infected_status ~ step, data=o[[1]], type="b" )
    lines( recovered_status ~ step, data=o[[1]], col="red" )
    lines( deaths_status ~ step, data=o[[1]], col="green" )

    
    ## --- below is a Julia call instead of R and plotting

    data, _ = run!(model, agent_step!, 100; adata = to_collect)
    data[1:10, :]

    # We now plot how quantities evolved in time to show
    # the exponential growth of the virus
    
    using Plots
    gr() # hide
    
    N = sum(model.Ns) # Total initial population
    x = data.step
    p = plot(
        x,
        (data[:, aggname(:status, infected)]),
        label = "infected",
        xlabel = "steps",
        ylabel = "log10(count)",
    )
    plot!(p, x, (data[:, aggname(:status, recovered)].+1), label = "recovered")
    plot!(p, x, (data[:, aggname(:status, deaths)].+1), label = "dead")
    
end