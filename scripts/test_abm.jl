
# test of Agents.jl
using Random, LinearAlgebra
using Agents

seed= 1

# Generate the RNG for the model
rng = MersenneTwister(seed)
rng = Xoshiro(seed)


@agent SizeGroup NoSpaceAgent begin
    size::Int
    n::Int
    maturity::Int
    sex::Int
    status::Int 
end
 
# params accessible as model.*
properties = (
  min_to_be_happy => min_to_be_happy
)  

model = ABM( SizeGroup; properties, rng ) # , scheduler = Schedulers.Randomly() )

nw = 100
for i in 1:nw
  add_agent!(model; 2, i, 1, 1, 1 ) 
end

# nagents(model)

function grow!(agent, model)
  
end

function migrate!(agent, model)
    pid = agent.pos
    m = sample(model.rng, 1:(model.C), Weights(model.migration_rates[pid, :]))
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

    n = rate * abs(randn(model.rng))
    n <= 0 && return

    for contactID in ids_in_position(agent, model)
        contact = model[contactID]
        if contact.status == :S ||
           (contact.status == :R && rand(model.rng) ≤ model.reinfection_probability)
            contact.status = :I
            n -= 1
            n <= 0 && return
        end
    end
end

update!(agent, model) = agent.status == :I && (agent.days_infected += 1)

function recover_or_die!(agent, model)
    if agent.days_infected ≥ model.infection_period
        if rand(model.rng) ≤ model.death_rate
            remove_agent!(agent, model)
        else
            agent.status = :R
            agent.days_infected = 0
        end
    end
end


function agent_step!(agent, model)
    migrate!(agent, model)
    transmit!(agent, model)
    update!(agent, model)
    recover_or_die!(agent, model)
end

step!(model, agent_step!)

infected(x) = count(i == :I for i in x)
recovered(x) = count(i == :R for i in x)

to_collect = [(:status, f) for f in (infected, recovered, length)]
to_collect = [:pos, :mood, :group]
to_collect = [(:mood, sum), (x, mean)]

data, _ = run!(model, agent_step!, 5; adata =to_collect)

data[1:10, :] # print only a few rows


# bio.snowcrab::size_distributions() # creates "/home/jae/bio.data/bio.snowcrab/output/size_distributions.rdz"







