"""
    simulation.jl

Simulation engine for the organism Individual-Based Model (IBM).

Implements spatial environment initialization, daily agent stepping (energetics,
molting, natural and fishing mortality, dispersal, mating, and hatching), daily
environmental stepping (seasonal temperature cycling and food replenishment),
and simulation execution governed by granular process toggles.
"""

using Agents
using DataFrames
using Random
using Statistics
using Distributions
using TOML

"""
    default_ibm_configuration()

Return the default configuration dictionary for the organism IBM simulation,
encompassing spatial grid geometry, runtime duration, environmental forcings,
bioenergetics, life-history phenology, fishery dynamics, process toggles, and output handling.

# Returns
`Dict{Symbol, Any}` containing all baseline parameter keys and values.
"""
function default_ibm_configuration()
    return Dict{Symbol, Any}(
        # Process Toggles
        :use_energetics => true,
        :use_thermal_stress => true,
        :use_cannibalism => false,
        :use_senescence => true,
        :use_spermatheca => true,
        :use_movement => true,
        :use_directed_movement => true,
        :use_fishery => true,
        :use_degree_days => true,
        :use_molting => true,
        :use_reproduction => true,
        # Simulation
        :n_agents => 300,
        :days => 730,
        :grid => (100, 100),
        :dt => 1.0,
        :seed => 1234,
        # Environment
        :temp_base => 5.0,
        :temp_amplitude => 4.0,
        :temp_noise_sd => 0.50,
        :food_max => 10.0,
        :food_replenishment_rate => 0.10,
        :benthic_carrying_capacity => 300.0,
        :half_saturation_food => 2.0,
        :temp_base_degreedays => 0.0,
        # Organism Biology
        :max_size_male => 160.0,
        :max_size_female => 95.0,
        :max_stage_male => 16,
        :max_stage_female => 14,
        :maturity_size_mid => 60.0,
        :metabolic_rate_base => 0.01,
        :intake_rate_base => 0.05,
        :energy_molt_threshold => 50.0,
        :energy_molt_slope => 1.5,
        :molt_energy_retention => 0.10,
        # Phenology
        :molt_season_start => 15.0,
        :molt_season_end => 140.0,
        :mating_season_start => 30.0,
        :mating_season_end => 120.0,
        :mating_search_radius => 5,
        :brood_duration_base => 365.0,
        :egg_survival_base => 5.0e-5,
        # Fishery
        :legal_size_male => 95.0,
        :fishing_probability_daily => 0.0,
        :fishing_season_start => 90.0,
        :fishing_season_end => 150.0,
        # Output
        :save => true,
        :save_file => "simulation_results.csv",
        :snapshot_day => 274.0,
        :frequency_days => 365.0,
        :plot => false,
        :output_dir => "outputs",
        # Animation
        :animate => false,
        :animation_file => "organism_animation.mp4",
        :anim_frames => 120,
        :anim_dt => 2,
        :anim_fps => 15,
        :anim_field => "temperature"
    )
end

"""
    load_ibm_configuration(filepath::String; base_config=nothing)

Read a TOML configuration file and return a flat `Dict{Symbol, Any}` of settings,
optionally overlaying loaded values on top of a `base_config` dictionary.
"""
function load_ibm_configuration(
    filepath::String;
    base_config::Union{Nothing, AbstractDict} = nothing
)
    if !isfile(filepath)
        error("Configuration file not found: $(filepath)")
    end

    raw = TOML.parsefile(filepath)
    flat = base_config === nothing ? Dict{Symbol, Any}() : deepcopy(base_config)

    for (section, kvs) in raw
        if isa(kvs, AbstractDict)
            for (k, v) in kvs
                flat[Symbol(k)] = v
            end
        else
            flat[Symbol(section)] = kvs
        end
    end

    # Normalize grid dimensions if present as array
    if haskey(flat, :grid) && isa(flat[:grid], AbstractVector)
        flat[:grid] = (Int(flat[:grid][1]), Int(flat[:grid][2]))
    end

    return flat
end
load_ibm_configuration(filepath::String, base_config::AbstractDict) =
    load_ibm_configuration(filepath; base_config = base_config)


"""
    save_ibm_configuration(filepath::String, config::AbstractDict)

Serialize a simulation configuration dictionary to a structured TOML file.
"""
function save_ibm_configuration(filepath::String, config::AbstractDict)
    sections = Dict{String, Dict{String, Any}}(
        "toggles" => Dict{String, Any}(),
        "simulation" => Dict{String, Any}(),
        "environment" => Dict{String, Any}(),
        "organism" => Dict{String, Any}(),
        "phenology" => Dict{String, Any}(),
        "fishery" => Dict{String, Any}(),
        "output" => Dict{String, Any}(),
        "animation" => Dict{String, Any}()
    )

    section_map = Dict{Symbol, String}(
        :use_energetics => "toggles",
        :use_thermal_stress => "toggles",
        :use_cannibalism => "toggles",
        :use_senescence => "toggles",
        :use_spermatheca => "toggles",
        :use_movement => "toggles",
        :use_directed_movement => "toggles",
        :use_fishery => "toggles",
        :use_degree_days => "toggles",
        :use_molting => "toggles",
        :use_reproduction => "toggles",

        :n_agents => "simulation",
        :days => "simulation",
        :grid => "simulation",
        :dt => "simulation",
        :seed => "simulation",

        :temp_base => "environment",
        :temp_amplitude => "environment",
        :temp_noise_sd => "environment",
        :temp_base_degreedays => "environment",
        :food_max => "environment",
        :food_replenishment_rate => "environment",
        :benthic_carrying_capacity => "environment",
        :half_saturation_food => "environment",

        :max_size_male => "organism",
        :max_size_female => "organism",
        :max_stage_male => "organism",
        :max_stage_female => "organism",
        :maturity_size_mid => "organism",
        :metabolic_rate_base => "organism",
        :intake_rate_base => "organism",
        :energy_molt_threshold => "organism",
        :energy_molt_slope => "organism",
        :molt_energy_retention => "organism",

        :molt_season_start => "phenology",
        :molt_season_end => "phenology",
        :mating_season_start => "phenology",
        :mating_season_end => "phenology",
        :mating_search_radius => "phenology",
        :brood_duration_base => "phenology",
        :egg_survival_base => "phenology",

        :legal_size_male => "fishery",
        :fishing_probability_daily => "fishery",
        :fishing_season_start => "fishery",
        :fishing_season_end => "fishery",

        :save => "output",
        :save_file => "output",
        :snapshot_day => "output",
        :frequency_days => "output",
        :plot => "output",
        :output_dir => "output",

        :animate => "animation",
        :animation_file => "animation",
        :anim_frames => "animation",
        :anim_dt => "animation",
        :anim_fps => "animation",
        :anim_field => "animation"
    )

    for (k, v) in config
        sym_k = Symbol(k)
        sec = get(section_map, sym_k, "simulation")
        str_k = string(sym_k)
        if sym_k === :grid && (isa(v, Tuple) || isa(v, AbstractVector))
            sections[sec][str_k] = [Int(v[1]), Int(v[2])]
        else
            sections[sec][str_k] = v
        end
    end

    open(filepath, "w") do io
        TOML.print(io, sections)
    end

    return filepath
end

"""
    initialize_model(config_filepath::String; kwargs...)

Initialize an individual-based model loading base parameters from a TOML configuration
file, with optional keyword argument overrides.
"""
function initialize_model(config_filepath::String; kwargs...)
    config = load_ibm_configuration(config_filepath)
    return initialize_model(config; kwargs...)
end

"""
    initialize_model(config::AbstractDict; kwargs...)

Initialize an individual-based model from a configuration dictionary,
with optional keyword argument overrides.
"""
function initialize_model(config::AbstractDict; kwargs...)
    merged = deepcopy(config)
    for (k, v) in kwargs
        merged[Symbol(k)] = v
    end

    n_ag = get(merged, :N_agents, get(merged, :n_agents, 100))
    grid_dim = get(merged, :dims, get(merged, :grid, (50, 50)))
    if isa(grid_dim, AbstractVector)
        grid_dim = (Int(grid_dim[1]), Int(grid_dim[2]))
    end

    dt = Float64(get(merged, :dt_days, get(merged, :dt, 1.0)))
    seed = get(merged, :rng_seed, get(merged, :seed, 1234))

    # Construct toggles
    toggles = ProcessToggles(
        use_energetics = get(merged, :use_energetics, true),
        use_thermal_stress = get(merged, :use_thermal_stress, true),
        use_cannibalism = get(merged, :use_cannibalism, false),
        use_senescence = get(merged, :use_senescence, true),
        use_spermatheca = get(merged, :use_spermatheca, true),
        use_movement = get(merged, :use_movement, true),
        use_directed_movement = get(merged, :use_directed_movement, true),
        use_fishery = get(merged, :use_fishery, true),
        use_degree_days = get(merged, :use_degree_days, true),
        use_molting = get(merged, :use_molting, true),
        use_reproduction = get(merged, :use_reproduction, true)
    )

    # Construct environment
    env = EnvironmentParams(
        temp_base = Float64(get(merged, :temp_base, 5.0)),
        temp_amplitude = Float64(get(merged, :temp_amplitude, 4.0)),
        temp_noise_sd = Float64(get(merged, :temp_noise_sd, 0.50)),
        temp_base_degreedays = Float64(get(merged, :temp_base_degreedays, 0.0)),
        food_max = Float64(get(merged, :food_max, 10.0)),
        food_replenishment_rate = Float64(get(merged, :food_replenishment_rate, 0.10)),
        benthic_carrying_capacity = Float64(get(merged, :benthic_carrying_capacity, 300.0)),
        half_saturation_food = Float64(get(merged, :half_saturation_food, 2.0))
    )

    # Construct organism params
    org_params = OrganismParams(
        max_size_male = Float64(get(merged, :max_size_male, 160.0)),
        max_size_female = Float64(get(merged, :max_size_female, 95.0)),
        max_stage_male = Int(get(merged, :max_stage_male, 16)),
        max_stage_female = Int(get(merged, :max_stage_female, 14)),
        maturity_size_mid = Float64(get(merged, :maturity_size_mid, 60.0)),
        metabolic_rate_base = Float64(get(merged, :metabolic_rate_base, 0.01)),
        intake_rate_base = Float64(get(merged, :intake_rate_base, 0.05)),
        energy_molt_threshold = Float64(get(merged, :energy_molt_threshold, 50.0)),
        energy_molt_slope = Float64(get(merged, :energy_molt_slope, 1.5)),
        molt_energy_retention = Float64(get(merged, :molt_energy_retention, 0.10))
    )

    # Construct phenology
    phen = PhenologyParams(
        molt_season_start = Float64(get(merged, :molt_season_start, 15.0)),
        molt_season_end = Float64(get(merged, :molt_season_end, 140.0)),
        mating_season_start = Float64(get(merged, :mating_season_start, 30.0)),
        mating_season_end = Float64(get(merged, :mating_season_end, 120.0)),
        mating_search_radius = Int(get(merged, :mating_search_radius, 5)),
        brood_duration_base = Float64(get(merged, :brood_duration_base, 365.0)),
        egg_survival_base = Float64(get(merged, :egg_survival_base, 5.0e-5))
    )

    # Construct fishery
    fish = FisheryParams(
        legal_size_male = Float64(get(merged, :legal_size_male, 95.0)),
        fishing_probability_daily = Float64(get(merged, :fishing_probability_daily, 0.0)),
        fishing_season_start = Float64(get(merged, :fishing_season_start, 90.0)),
        fishing_season_end = Float64(get(merged, :fishing_season_end, 150.0))
    )

    params = IBMParameters(
        toggles = toggles,
        simulation = SimulationParams(dt = dt, seed = seed, n_agents = n_ag),
        environment = env,
        organism = org_params,
        phenology = phen,
        fishery = fish
    )

    return initialize_model(
        N_agents = n_ag,
        dims = grid_dim,
        dt_days = dt,
        params = params,
        rng_seed = seed
    )
end

"""
    initialize_model(; N_agents=100, dims=(50, 50), ...)

Construct and populate an `Agents.StandardABM` representing the organism ecosystem.

# Arguments
- `N_agents::Int`: Initial number of individual agents seeded into the environment (default: 100).
- `dims::Tuple{Int, Int}`: Grid spatial dimensions `(nx, ny)` (default: `(50, 50)`).
- `temp_map::Union{Nothing, Matrix{Float64}}`: Optional initial temperature field.
- `food_map::Union{Nothing, Matrix{Float64}}`: Optional initial food density field.
- `dt_days::Float64`: Time step in days (default: 1.0).
- `params::IBMParameters`: Parameters and process toggles.
- `rng_seed::Union{Nothing, Int}`: Optional RNG seed.

# Returns
Configured and populated `StandardABM` instance ready for simulation.
"""
function initialize_model(;
    N_agents::Int = 100,
    dims::Tuple{Int, Int} = (50, 50),
    temp_map::Union{Nothing, Matrix{Float64}} = nothing,
    food_map::Union{Nothing, Matrix{Float64}} = nothing,
    dt_days::Float64 = 1.0,
    params::IBMParameters = IBMParameters(),
    behavior::SpeciesBehavior = default_species_behavior(),
    rng_seed::Union{Nothing, Int} = nothing,
    kwargs...
)
    space = GridSpace(dims)

    initial_temp = if temp_map === nothing
        fill(params.environment.temp_base, dims)
    else
        copy(temp_map)
    end

    initial_food = if food_map === nothing
        fill(params.environment.food_max / 2.0, dims)
    else
        copy(food_map)
    end

    properties = Dict{Symbol, Any}(
        :current_time => 0.0,
        :temperature_field => initial_temp,
        :food_availability_field => initial_food,
        :dt => dt_days,
        :params => params,
        :total_eggs_spawned => 0,
        :annual_eggs_spawned => 0,
        :total_recruits_settled => 0,
        :total_catch_numbers => 0,
        :total_catch_biomass_kg => 0.0,
        :total_molts => 0,
        :deaths_predation => 0,
        :deaths_starvation => 0,
        :deaths_thermal => 0,
        :deaths_senescence => 0,
        :deaths_fishery => 0,
        :n_benthic_organisms => 0,
        :n_mature_males => 0,
        :n_mature_females => 0,
        :n_legal_males => 0,
        :clutch_recruitment_records => DataFrame(
            hatch_time = Float64[],
            mother_size = Float64[],
            mother_mass = Float64[],
            egg_count = Int[],
            recruits_settled = Int[],
            density_feedback = Float64[]
        ),
        :aggregate_populations => AggregatePopulations(),
        :species_behavior => behavior,
        :step_callback => get(kwargs, :step_callback, nothing)
    )

    rng = if rng_seed === nothing
        Random.default_rng()
    else
        Random.Xoshiro(rng_seed)
    end

    model = StandardABM(
        Organism,
        space;
        agent_step! = agent_step!,
        model_step! = model_step!,
        properties = properties,
        rng = rng
    )

    # --- Initialize Aggregate Populations (Egg stage 0 through stage 5) ---
    pop = AggregatePopulations()
    if N_agents > 0
        for stg in 0:5
            sz = (stg == 0) ? 0.60 : ((stg < 4) ? stage_size_lower_bound(behavior, Float64(stg), :male) : stage_size(behavior, stg, :male; type = :random, rng = rng))
            wt = (stg == 0) ? 0.00007 : calculate_mass(behavior, sz, :male; mat = 0, stochastic = true, rng = rng)
            grp = AggregateGroup(
                id = pop.next_group_id,
                stage = stg,
                abundance = (stg == 0) ? 2000.0 : 500.0,
                size = sz,
                mass = wt,
                age_days = stg * 90.0,
                degree_days = 0.0,
                last_growth_time = 0.0,
                pos = (rand(rng, 1:dims[1]), rand(rng, 1:dims[2]))
            )
            pop.next_group_id += 1
            push!(pop.groups, grp)
        end
    end
    properties[:aggregate_populations] = pop

    # --- Populate Initial Spatial Agents (Individuated: stage 6 onwards) ---
    benthic_cnt = 0
    mat_m = 0
    mat_f = 0
    leg_m = 0

    for _ in 1:N_agents
        stg = rand(rng, 6:9)
        sex = rand(rng, (male, female))
        sz = stage_size(behavior, stg, sex; type = :random, rng = rng)
        mat = (stg >= 10)
        wt = calculate_mass(behavior, sz, sex; mat = (mat ? 1 : 0), stochastic = true, rng = rng)
        age_days = (stg - 3.0) * 180.0 + rand(rng) * 90.0
        energy_reserve = params.organism.initial_energy
        last_molt_time = -age_days
        degree_days = rand(rng, Uniform(50.0, 300.0))
        is_mating_ready = mat
        egg_count = 0
        hatch_day = 0.0
        sperm_clutches = 0
        pos = (rand(rng, 1:dims[1]), rand(rng, 1:dims[2]))

        new_org = add_agent!(
            pos,
            model,
            stg,
            sex,
            age_days,
            sz,
            wt,
            energy_reserve,
            last_molt_time,
            degree_days,
            is_mating_ready,
            egg_count,
            hatch_day,
            sperm_clutches,
            mat;
            behavior = behavior
        )
        sample_individual_traits!(new_org, model, rng)
    end

    # Cache initial demographic aggregates
    for a in allagents(model)
        benthic_cnt += 1
        if is_adult_male(a)
            mat_m += 1
            if is_legal_male(a)
                leg_m += 1
            end
        elseif is_adult_female(a)
            mat_f += 1
        end
    end

    properties[:n_benthic_organisms] = benthic_cnt
    properties[:n_benthic_crabs] = benthic_cnt
    properties[:n_mature_males] = mat_m
    properties[:n_mature_females] = mat_f
    properties[:n_legal_males] = leg_m

    return model
end

"""
    sample_individual_traits!(org::Organism, model, rng)

Sample intrinsic empirical life-history traits for an individual `Organism`.
"""
function sample_individual_traits!(org::Organism, model, rng)
    org.trait_growth_quantile = rand(rng, Uniform(0.01, 0.99))
    org.trait_maturity_quantile = rand(rng, Uniform(0.01, 0.99))
    org.trait_fecundity_quantile = rand(rng, Uniform(0.01, 0.99))
    org.trait_mortality_quantile = rand(rng, Uniform(0.01, 0.99))
    return nothing
end

function sample_weighted(items, weights, rng)
    tot = sum(weights)
    if tot <= 0.0 || isempty(items)
        return items[rand(rng, 1:length(items))]
    end
    r = rand(rng) * tot
    cum = 0.0
    for i in eachindex(items)
        cum += weights[i]
        if r <= cum
            return items[i]
        end
    end
    return items[end]
end

"""
    organism_disperse!(org::Organism, model; rng=abmrng(model))

Execute spatial movement and habitat taxis for an `Organism` agent governed by
process toggles `use_movement` and `use_directed_movement`.
"""
function organism_disperse!(
    org::Organism,
    model;
    rng::AbstractRNG = abmrng(model)
)
    # Check movement toggle
    if !model.params.toggles.use_movement
        return org.pos
    end

    # Energy-depleted lethargy: 75% chance of remaining stationary when energetics active
    if model.params.toggles.use_energetics && org.energy_reserve <= 0.0 && rand(rng) < 0.75
        return org.pos
    end

    candidates = [org.pos; collect(nearby_positions(org.pos, model))]

    if !model.params.toggles.use_directed_movement
        # Stochastic random walk: discrete neighbor choice
        chosen_pos = candidates[rand(rng, 1:length(candidates))]
        if chosen_pos != org.pos
            move_agent!(org, chosen_pos, model)
        end
        return org.pos
    end

    # Benthic habitat taxis: evaluate local environmental suitability
    t_opt = 2.5
    t_sigma = 1.5
    half_sat = model.params.environment.half_saturation_food

    utilities = Float64[]
    for pos in candidates
        cx, cy = pos
        ctemp = model.temperature_field[cx, cy]
        cfood = model.food_availability_field[cx, cy]

        # Gaussian thermal suitability
        s_temp = exp(-((ctemp - t_opt)^2) / (2.0 * t_sigma^2))

        # Hyperbolic food suitability
        s_food = cfood / (cfood + half_sat)

        u = s_temp * (1.0 + 0.5 * s_food)
        push!(utilities, u)
    end

    # Softmax probabilities with temperature parameter tau = 0.5
    tau = 0.5
    max_u = maximum(utilities)
    weights = [exp((u - max_u) / tau) for u in utilities]

    chosen_pos = sample_weighted(candidates, weights, rng)
    if chosen_pos != org.pos
        move_agent!(org, chosen_pos, model)
    end

    return org.pos
end

"""
    agent_step!(org::Organism, model)

Execute one simulation time step for an individual `Organism`. Biological processes
are applied modularly based on the feature toggles in `model.params.toggles`.
"""
function agent_step!(org::Organism, model)
    org.age_days += model.dt
    day_of_year = mod(model.current_time, 365.0)

    # 1. Energetics (Holling Type II & Kleiber consumption)
    if model.params.toggles.use_energetics
        temp = model.temperature_field[org.pos...]
        food = model.food_availability_field[org.pos...]
        dE_dt, intake = calculate_energy_change(temp, food, org.size; params = model.params, mass = org.mass)
        org.energy_reserve += dE_dt * model.dt
        model.food_availability_field[org.pos...] = max(0.0, food - intake * model.dt)
    end

    # 2. Degree-day accumulation
    if model.params.toggles.use_degree_days
        temp = model.temperature_field[org.pos...]
        dd_inc = max(0.0, temp - model.params.environment.temp_base_degreedays) * model.dt
        org.degree_days += dd_inc
    end

    # 3. Development & Molting
    perform_development!(org.behavior, org, model)

    # 4. Mortality
    temp = model.temperature_field[org.pos...]
    local_density = Float64(count(_ -> true, nearby_agents(org, model, 1)))
    mort_prob = calculate_mortality_probability(org, temp, local_density, model.current_time; params = model.params)

    if rand(abmrng(model)) < mort_prob
        if model.params.toggles.use_senescence && org.is_mature
            model.deaths_senescence += 1
        else
            model.deaths_predation += 1
        end
        remove_agent!(org, model)
        return nothing
    end

    # 5. Fishery Harvest
    if model.params.toggles.use_fishery &&
       model.params.fishery.fishing_probability_daily > 0.0 &&
       (model.params.fishery.fishing_season_start <= day_of_year <= model.params.fishery.fishing_season_end) &&
       is_legal_male(org; legal_size = model.params.fishery.legal_size_male)

        harvest_prob = model.params.fishery.fishing_probability_daily *
                       selectivity(
                           org.size;
                           s50 = model.params.fishery.legal_size_male,
                           s50_sigma = org.behavior.params.selectivity_s50_sigma
                       )
        if rand(abmrng(model)) < harvest_prob
            model.total_catch_numbers += 1
            model.total_catch_biomass_kg += (org.mass / 1000.0)
            model.deaths_fishery += 1
            remove_agent!(org, model)
            return nothing
        end
    end

    # 6. Locomotion / Spatial Dispersal
    organism_disperse!(org, model)

    # 7. Reproduction
    perform_reproduction!(org.behavior, org, model)

    return nothing
end

function update_demographic_caches!(model)
    benthic_count = 0
    mat_males = 0
    mat_females = 0
    legal_males = 0
    for org in allagents(model)
        benthic_count += 1
        if is_adult_male(org)
            mat_males += 1
            if is_legal_male(org)
                legal_males += 1
            end
        elseif is_adult_female(org)
            mat_females += 1
        end
    end
    model.n_benthic_organisms = benthic_count
    model.n_benthic_crabs = benthic_count
    model.n_mature_males = mat_males
    model.n_mature_females = mat_females
    model.n_legal_males = legal_males
    return nothing
end

"""
    model_step!(model)

Advance environmental state, advance aggregate cohorts, and update demographic caches
for one simulation time step.
"""
function model_step!(model)
    dt = model.dt
    old_time = model.current_time
    model.current_time += dt

    # Year boundary detection for annual resets
    if floor(old_time / 365.0) < floor(model.current_time / 365.0)
        model.annual_eggs_spawned = 0.0
    end

    # 1. Environmental updates (Temperature & Food)
    day_of_year = mod(model.current_time, 365.0)
    sinusoidal_factor = sin(2.0 * π * (day_of_year - 80.0) / 365.0)

    t_base = model.params.environment.temp_base
    t_amp = model.params.environment.temp_amplitude
    t_sd = model.params.environment.temp_noise_sd

    noise = rand(abmrng(model), Normal(0.0, t_sd), size(model.temperature_field))
    @. model.temperature_field = t_base + t_amp * sinusoidal_factor + noise

    f_max = model.params.environment.food_max
    f_rate = model.params.environment.food_replenishment_rate
    @. model.food_availability_field += f_rate * (f_max - model.food_availability_field) * dt

    # 2. Advance pre-individuation aggregate cohort groups (stages 0 through 5)
    step_aggregate_groups!(model)

    # 3. Update demographic aggregate caches
    update_demographic_caches!(model)

    # 4. Optional step callback hook
    if haskey(abmproperties(model), :step_callback) && model.step_callback !== nothing
        model.step_callback(model)
    end

    return nothing
end

"""
    safe_mean(x)

Calculate sample mean of numeric collection `x`, returning 0.0 if empty.
"""
function safe_mean(x)
    return isempty(x) ? 0.0 : mean(x)
end

"""
    run_simulation(model, n_steps; adata=nothing, mdata=nothing, when=nothing, step_callback=nothing)

High-level wrapper executing `n_steps` of simulation using `Agents.run!`.
"""
function run_simulation(
    model,
    n_steps::Int;
    adata = nothing,
    mdata = nothing,
    when = nothing,
    step_callback = nothing
)
    if step_callback !== nothing
        model.step_callback = step_callback
    end

    default_adata = [
        (:stage, length),
        (:stage, safe_mean),
        (:size, safe_mean),
        (:mass, safe_mean),
        (:mass, sum),
        (is_adult, count),
        (is_adult_male, count),
        (is_adult_female, count),
        (is_legal_male, count),
        (is_ovigerous, count),
        (mass_if_adult_female, sum),
        (mass_if_adult_male, sum),
        (mass_if_legal_male, sum)
    ]

    default_mdata = [
        :total_eggs_spawned,
        :total_recruits_settled,
        :total_catch_numbers,
        :total_catch_biomass_kg,
        :total_molts,
        :deaths_predation,
        :deaths_starvation,
        :deaths_thermal,
        :deaths_senescence,
        :deaths_fishery,
        :n_benthic_organisms,
        :n_mature_males,
        :n_mature_females,
        :n_legal_males
    ]

    actual_adata = (adata === nothing) ? default_adata : adata
    actual_mdata = (mdata === nothing) ? default_mdata : mdata

    if when === nothing
        return run!(model, n_steps; adata = actual_adata, mdata = actual_mdata)
    else
        return run!(model, n_steps; adata = actual_adata, mdata = actual_mdata, when = when)
    end
end
