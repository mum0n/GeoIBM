# An agent-based representation of snow crab life stages and dynamics 
 

```julia


project_directory = joinpath( homedir(), "projects", "model.abm", "julia" )  
 
fn_env = joinpath( project_directory, "abm_environment.jl" )  
include(  fn_env )

 
seed = 125
rng = Random.Xoshiro(seed)
scheduler = Schedulers.Randomly()

griddims = (100, 100)

total_agents_init = 100

space = GridSpace(griddims; periodic = false)

properties = Dict(
	:degreedays => 365.0 * 4.0,
	:time => 0,
	:time_max => 15*365  	
)

@agent SnowCrab GridAgent{2} begin  
    degdays::Float64
    size::Float64
    sex::Int  
    mature::Int
end

model = ABM( SnowCrab, space; properties, rng, scheduler=scheduler )

n = 0
while n != total_agents_init
    pos = Tuple(rand(model.rng, 1:100, 2) )
    if isempty(pos, model)
        degdays, size, sex, mat = sample_population( -1 )
        agent = SnowCrab(n, pos, degdays, size, sex, mat )
        add_agent_single!(agent, model)
        n += 1
    end
end

nagents(model)
allagents(model)
id=0; model[id]
abmproperties(model)

function model_step!(model)
    model.time += 1
    model.stock += dstock(model)
end


function agent_step!(agent, model)
    agent.time += 1
    agent.time >= model.time_max && remove_agent!(agent, model)
		agent.degdays += rand(Normal(4.0, 0.5))
		if agent.degdays > model.degreedays
			agent.size += growth_increment( size; sex=agent.sex, mat=0 )
		end 
end


function agent_step_complete!(agent, model)
    agent.time += 1
    agent.time >= model.time_max && remove_agent!(agent, model)
		if agent.nearby.cw > my.cw move_agent_single!(agent, model)
    check_degreedays()
    grow( ~ temp, sex, mat, cw, etc)
    mature( ~ temp, sex, cw)
    death(~ temp, sex, cw, density )
    min_dd = model.degreedays

    count_neighbors_same_group = 0
    # For each neighbor, get group and compare to current agent's group
    # and increment `count_neighbors_same_group` as appropriately.
    # Here `nearby_agents` (with default arguments) will provide an iterator
    # over the nearby agents one grid point away, which are at most 8.
    for neighbor in nearby_agents(agent, model)
            if agent.group == neighbor.group
                    count_neighbors_same_group += 1
            end
    end
    # After counting the neighbors, decide whether or not to move the agent.
    # If count_neighbors_same_group is at least the degreedays, set the
    # mood to true. Otherwise, move the agent to a random position, and set
    # mood to false.
    if count_neighbors_same_group ≥ min_dd
            agent.mood = true
    else
            agent.mood = false
            move_agent_single!(agent, model)
    end
    return

end
 
step!(model, agent_step!, 3)


# example sequence: 

adata = [:pos, :mood, :group]
adata = [(:mood, sum), (x, mean)]

data, _ = run!(model, agent_step!, 5; adata)
data[1:10, :] # print only a few rows


using CairoMakie # choosing a plotting backend

groupcolor(a) = a.group == 1 ? :blue : :orange
groupmarker(a) = a.group == 1 ? :circle : :rect
figure, _ = abmplot(model; ac = groupcolor, am = groupmarker, as = 10)
figure # returning the figure displays it

abmvideo(
    "schelling.mp4", model, agent_step!;
    ac = groupcolor, am = groupmarker, as = 10,
    framerate = 4, frames = 20,
    title = "Schelling's segregation model"
)
	
using Distributed
addprocs(4)

@everywhere using Agents
@everywhere @agent SchellingAgent ...
@everywhere agent_step!(...) = ...
adf, = ensemblerun!(models, agent_step!, dummystep, 5; adata, parallel = true)


	using Random, LinearAlgebra, MakieGL
	using Revise, RData, MKL, DataFrames, Distributions, Turing, Agents

	# DEFINE KEY DIRECTORIES:
	# these need to be defined
	
	project_directory = joinpath( homedir(), "projects", "model.abm", "julia" ) 
	bio_data_directory = joinpath( homedir(), "bio.data" )  
	outputs_directory = joinpath( project_directory, "fishery_model" ) 

    # RUN LEVEL OPTIONS: time, area model 
	year_assessment = 2022   # <<<<<<<<-- change
	yrs = 1999:year_assessment  
	
	# choose a region of interest"
	aulab ="cfanorth"   
	aulab ="cfasouth"   
	aulab ="cfa4x"     

	model_variation = "ssa_simple"   

	# helper functions to go into:
	# include( joinpath(project_directory, "agent_based-models_functions.jl" ) )   

```




```julia
	
	G = growth_schedule()


	plot( G[:, :cwm_med] )
	plot!( G[:, :cwf_med] )


	instars_m = 1:14
	instars_f = 1:11

	bins = (1:2:175)
	n_max = 10000
	
	# params to estimate: 
	s50 = 50
	s50_sigma  = (50 - 15)/6 # sub 10 mm CW not seen 12-15 seems to be the end point
	
	
	Gm = sqrt.( (llbm.(instars_m .+ 1) .* llbm.(instars_m )) )
	Gf = sqrt.( (llbf.(instars_f .+ 1) .* llbf.(instars_f )) )
	
	# males
	Wm, Om = growth_simulate(Gm; gcv = 0.4, n_max=n_max, s50=s50, s50_sigma=s50_sigma )  
	
	# females
	Wf, Of = growth_simulate(Gf; gcv = 0.4, n_max=n_max, s50=s50, s50_sigma=s50_sigma )  

	S = selectivity.(bins, s50=s50, s50_sigma=s50_sigma )

	pl = histogram( Wm[:,1], bins=bins )
	for i in 2:size(Wm)[2]
		pl = histogram!(pl, Wm[:,i], bins=bins )
	end
	pl 
	plot!(pl, bins, S*n_max*0.9) 

	pl = histogram( Om[:,1], bins=bins )
	for i in 2:size(Om)[2]
		pl = histogram!(pl, Om[:,i], bins=bins )
	end
	
	pl
	plot!(pl, bins, S*n_max*0.03)

	
```

 
