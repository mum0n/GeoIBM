
#  1. Models.wright_fisher, https://juliadynamics.github.io/Agents.jl/stable/examples/wright-fisher/#Wright-Fisher-model-of-evolution

using Agents
using Statistics: mean

numagents = 100
mutable struct Haploid <: AbstractAgent
    id::Int
    trait::Float64
end

model = ABM(Haploid)  # no spatial structures
for i in 1:numagents
    add_agent!(model, rand(model.rng))
end

sample!(model, nagents(model))  # new generation

modelstep_neutral!(model::ABM) = sample!(model, nagents(model))

modelstep_selection!(model::ABM) = sample!(model, nagents(model), :trait)  # trait increases pr selection (weight)

data, _ = run!(model, dummystep, modelstep_neutral!, 20; adata = [(:trait, mean)])
data

data, _ = run!(model, dummystep, modelstep_selection!, 20; adata = [(:trait, mean)])
data


# 2. Models.hk, https://juliadynamics.github.io/Agents.jl/stable/examples/hk/
# This is an implementation of a simple version of the Hegselmann and Krause (2002) model.
# Synchronous update schedule changes made to an agent are not seen by other agents until the next step,
# A set of n Agents with opinions xᵢ in the range [0,1] as attribute
# A parameter ϵ called "bound" in (0, 0.3]
# The update rule: at each step every agent adopts the mean of the opinions which are within the confidence bound ( |xᵢ - xⱼ| ≤ ϵ).
using Agents
using Statistics: mean

mutable struct HKAgent <: AbstractAgent
    id::Int
    old_opinion::Float64
    new_opinion::Float64
    previous_opinon::Float64
end

function hk_model(; numagents = 100, ϵ = 0.2)
    model = ABM(HKAgent, scheduler = fastest, properties = Dict(:ϵ => ϵ))
    for i in 1:numagents
        o = rand(model.rng)
        add_agent!(model, o, o, -1)
    end
    return model
end

model = hk_model()

function boundfilter(agent, model)
    filter(
        j -> abs(agent.old_opinion - j) < model.ϵ,
        [a.old_opinion for a in allagents(model)],
    )
end

function agent_step!(agent, model)
    agent.previous_opinon = agent.old_opinion
    agent.new_opinion = mean(boundfilter(agent, model))
end

function model_step!(model)
    for a in allagents(model)
        a.old_opinion = a.new_opinion
    end
end

function terminate(model, s)
    if any(
        !isapprox(a.previous_opinon, a.new_opinion; rtol = 1e-12) for a in allagents(model)
    )
        return false
    else
        return true
    end
end

Agents.step!(model, agent_step!, model_step!, terminate)
model[1]

function model_run(; kwargs...)
    model = hk_model(; kwargs...)
    agent_data, _ = run!(model, agent_step!, model_step!, terminate; adata = [:new_opinion])
    return agent_data
end

data = model_run(numagents = 100)
data[(end-19):end, :]

model = hk_model()
agent_data, _ = run!(
    model,
    agent_step!,
    model_step!,
    terminate;
    adata = [:new_opinion],
    when = terminate,
)
agent_data

using DataFrames, CairoMakie

const cmap = cgrad(:lightrainbow)
plotsim(ax, data, ϵ) =
    map(groupby(data, :id)) do grp
        lines!(ax, grp.step, grp.new_opinion, color = cmap[grp.id[1]/100])
    end

eps = [0.05, 0.15, 0.3]
figure = Figure(resolution = (600, 450))
for (i, e) in enumerate(eps)
    ax = figure[i, 1] = Axis(figure; title = "epsilon = $e")
    e_data = model_run(ϵ = e)
    plotsim(ax, e_data, e)
end
figure




# 3. Flocking https://juliadynamics.github.io/Agents.jl/stable/examples/flock/

# each bird follows three simple rules:

# maintain a minimum distance from other birds to avoid collision
# fly towards the average position of neighbors
# fly in the average direction of neighbors

using Agents, LinearAlgebra

mutable struct Bird <: AbstractAgent
    id::Int
    pos::NTuple{2,Float64}
    vel::NTuple{2,Float64}
    speed::Float64
    cohere_factor::Float64
    separation::Float64
    separate_factor::Float64
    match_factor::Float64
    visual_distance::Float64
end

#The fields id and pos are required for every agent. The field vel is required for using move_agent! in ContinuousSpace. speed defines how far the bird travels in the direction defined by vel per step. seperation defines the minimum distance a bird must maintain from its neighbors. visual_distance refers to the distance a bird can see and defines a radius of neighboring birds. The contribution of each rule defined above recieves an importance weight: cohere_factor is the importance of maintaining the average position of neighbors, match_factor is the importance of matching the average trajectory of neighboring birds, and separate_factor is the importance of maining the minimum distance from neighboring birds.

function initialize_model(;
    n_birds = 100,
    speed = 1.0,
    cohere_factor = 0.25,
    separation = 4.0,
    separate_factor = 0.25,
    match_factor = 0.01,
    visual_distance = 5.0,
    extent = (100, 100),
    spacing = visual_distance / 1.5,
)
    space2d = ContinuousSpace(extent, spacing)
    model = ABM(Bird, space2d, scheduler = random_activation)
    for _ in 1:n_birds
        vel = Tuple(rand(model.rng, 2) * 2 .- 1)
        add_agent!(
            model,
            vel,
            speed,
            cohere_factor,
            separation,
            separate_factor,
            match_factor,
            visual_distance,
        )
    end
    return model
end

function agent_step!(bird, model)
    # Obtain the ids of neighbors within the bird's visual distance
    neighbor_ids = nearby_ids(bird, model, bird.visual_distance)
    N = 0
    match = separate = cohere = (0.0, 0.0)
    # Calculate behaviour properties based on neighbors
    for id in neighbor_ids
        N += 1
        neighbor = model[id].pos
        heading = neighbor .- bird.pos

        # `cohere` computes the average position of neighboring birds
        cohere = cohere .+ heading
        if edistance(bird.pos, neighbor, model) < bird.separation
            # `separate` repels the bird away from neighboring birds
            separate = separate .- heading
        end
        # `match` computes the average trajectory of neighboring birds
        match = match .+ model[id].vel
    end
    N = max(N, 1)
    # Normalise results based on model input and neighbor count
    cohere = cohere ./ N .* bird.cohere_factor
    separate = separate ./ N .* bird.separate_factor
    match = match ./ N .* bird.match_factor
    # Compute velocity based on rules defined above
    bird.vel = (bird.vel .+ cohere .+ separate .+ match) ./ 2
    bird.vel = bird.vel ./ norm(bird.vel)
    # Move bird according to new velocity and speed
    move_agent!(bird, model, bird.speed)
end

using InteractiveDynamics
import CairoMakie

const bird_polygon = Polygon(Point2f0[(-0.5, -0.5), (1, 0), (-0.5, 0.5)])
function bird_marker(b::Bird)
    φ = atan(b.vel[2], b.vel[1]) #+ π/2 + π
    scale(rotate2D(bird_polygon, φ), 2)
end

model = initialize_model()
figure, = abm_plot(model; am = bird_marker)
figure

abm_video(
    "flocking.mp4", model, agent_step!;
    am = bird_marker,
    framerate = 20, frames = 100,
    title = "Flocking"
)

using InteractiveDynamics
import CairoMakie

const bird_polygon = Polygon(Point2f0[(-0.5, -0.5), (1, 0), (-0.5, 0.5)])
function bird_marker(b::Bird)
    φ = atan(b.vel[2], b.vel[1]) #+ π/2 + π
    scale(rotate2D(bird_polygon, φ), 2)
end

model = initialize_model()
figure, = abm_plot(model; am = bird_marker)
figure






# x. source:  https://juliadynamics.github.io/Agents.jl/stable/examples/schelling/

using Agents

space = GridSpace((10, 10); periodic = false)

mutable struct SchellingAgent <: AbstractAgent
    id::Int             # The identifier number of the agent
    pos::NTuple{2, Int} # The x, y location of the agent on a 2D grid
    mood::Bool          # whether the agent is happy in its position. (true = happy)
    group::Int          # The group of the agent, determines mood as it interacts with neighbors
end

@agent SchellingAgent GridAgent{2} begin
    mood::Bool
    group::Int
end

properties = Dict(:min_to_be_happy => 3)

schelling = ABM(SchellingAgent, space; properties)

schelling2 = ABM(
    SchellingAgent,
    space;
    properties = properties,
    scheduler = property_activation(:group),
)

using Random # for reproducibility

function initialize(; numagents = 320, griddims = (20, 20), min_to_be_happy = 3, seed = 125)
    space = GridSpace(griddims, periodic = false)
    properties = Dict(:min_to_be_happy => min_to_be_happy)
    rng = Random.MersenneTwister(seed)
    model = ABM(
        SchellingAgent, space;
        properties, rng, scheduler = random_activation
    )

    # populate the model with agents, adding equal amount of the two types of agents
    # at random positions in the model
    for n in 1:numagents
        agent = SchellingAgent(n, (1, 1), false, n < numagents / 2 ? 1 : 2)
        add_agent_single!(agent, model)
    end
    return model
end

function agent_step!(agent, model)
  minhappy = model.min_to_be_happy
  count_neighbors_same_group = 0
  # For each neighbor, get group and compare to current agent's group
  # and increment count_neighbors_same_group as appropriately.
  # Here `nearby_agents` (with default arguments) will provide an iterator
  # over the nearby agents one grid point away, which are at most 8.
  for neighbor in nearby_agents(agent, model)
      if agent.group == neighbor.group
          count_neighbors_same_group += 1
      end
  end
  # After counting the neighbors, decide whether or not to move the agent.
  # If count_neighbors_same_group is at least the min_to_be_happy, set the
  # mood to true. Otherwise, move the agent to a random position.
  if count_neighbors_same_group ≥ minhappy
      agent.mood = true
  else
      move_agent_single!(agent, model)
  end
  return
end

model = initialize()

step!(model, agent_step!)  # one step forward
step!(model, agent_step!, 3)  # three steps forward

# run the model
adata = [:pos, :mood, :group]

model = initialize()
data, _ = run!(model, agent_step!, 5; adata)
data[1:10, :] # print only a few rows

# using x
x(agent) = agent.pos[1]
model = initialize()
adata = [x, :mood, :group]
data, _ = run!(model, agent_step!, 5; adata)
data[1:10, :]


using Statistics: mean
model = initialize();
adata = [(:mood, sum), (x, mean)]
data, _ = run!(model, agent_step!, 5; adata)
data

using InteractiveDynamics
import CairoMakie # choosing a plotting backend

groupcolor(a) = a.group == 1 ? :blue : :orange
groupmarker(a) = a.group == 1 ? :circle : :rect
figure, _ = abm_plot(model; ac = groupcolor, am = groupmarker, as = 10)
figure # returning the figure displays it

# animation .. saved  to home directory 
model = initialize();
abm_video(
    "schelling.mp4", model, agent_step!;
    ac = groupcolor, am = groupmarker, as = 10,
    framerate = 4, frames = 20,
    title = "Schelling's segregation model"
)



# interactive model
parange = Dict(:min_to_be_happy => 0:8)

adata = [(:mood, sum), (x, mean)]
alabels = ["happy", "avg. x"]

model = initialize(; numagents = 300) # fresh model, noone happy


figure, adf, mdf = abm_data_exploration(
    model, agent_step!, dummystep, parange;
    ac = groupcolor, am = groupmarker, as = 10,
    adata, alabels
)


# replication and parallel computing

model = initialize(numagents = 370, griddims = (20, 20), min_to_be_happy = 3)
data, _ = run!(model, agent_step!, 5; adata = adata, replicates = 3)
data[(end - 10):end, :]

using Distributed
addprocs(4)

@everywhere using Agents
@everywhere SchellingAgent <: AbstractAgent
  id::Int             # The identifier number of the agent
  pos::NTuple{2, Int} # The x, y location of the agent on a 2D grid
  mood::Bool          # whether the agent is happy in its position. (true = happy)
  group::Int          # The group of the agent, determines mood as it interacts with neighbors
end

@everywhere ... for all functions and data structures



data, _ = run!(model, agent_step!, 2, adata=adata,
replicates=5, parallel=true)



happyperc(moods) = count(x -> x == true, moods) / length(moods)
adata = [(:mood, happyperc)]

parameters = Dict(
    :min_to_be_happy => collect(2:5), # expanded
    :numagents => [200, 300],         # expanded
    :griddims => (20, 20),            # not Vector = not expanded
)

data, _ = paramscan(parameters, initialize; adata = adata, n = 3, agent_step! = agent_step!)
data

data, _ = paramscan(
  parameters,
  initialize;
  adata = adata,
  n = 3,
  agent_step! = agent_step!,
  replicates = 3,
)

data[(end - 10):end, :]

using DataFrames
gd = groupby(data, [:step, :min_to_be_happy, :numagents])
data_mean = combine(gd, [:happyperc_mood, :replicate] .=> mean)

out = select(data_mean, Not(:replicate_mean))
