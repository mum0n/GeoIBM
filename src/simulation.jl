"""
Simulation engine for the Snow Crab Individual-Based Model (IBM).

Implements spatial environment initialization, daily agent stepping (energetics,
molting, natural and fishing mortality, dispersal, mating, and hatching), daily
environmental stepping (seasonal temperature cycling and food replenishment),
and simulation execution.
"""

using Agents
using DataFrames
using Random
using Statistics
using Distributions
using TOML

"""
    default_ibm_configuration()

Return the default configuration dictionary for the snow crab IBM simulation,
encompassing spatial grid geometry, runtime duration, environmental forcings,
bioenergetics, life-history phenology, fishery dynamics, and output handling.

# Returns
`Dict{Symbol, Any}` containing all baseline parameter keys and values.
"""
function default_ibm_configuration()
    return Dict{Symbol, Any}(
        # Simulation
        :n_agents => 300,
        :days => 730,
        :grid => (100, 100),
        :dt => 1.0,
        :seed => 1234,
        # Environment
        :temp_base => 5.0,
        :temp_amplitude => 4.0,
        :temp_amp => 4.0,
        :temp_noise_sd => 0.50,
        :food_max => 10.0,
        :food_replenishment_rate => 0.10,
        :benthic_carrying_capacity => 300.0,
        :carrying_capacity => 300.0,
        # Biology
        :max_size_male => 160.0,
        :max_size_female => 95.0,
        :maturity_size_mid => 60.0,
        :metabolic_rate_base => 0.01,
        :intake_rate_base => 0.05,
        :energy_molt_threshold => 50.0,
        :energy_molt_slope => 1.5,
        :molt_energy_retention => 0.10,
        :temp_base_degreedays => 0.0,
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
        :fishing_prob => 0.0,
        :fishing_season_start => 90.0,
        :fishing_season_end => 150.0,
        # Output
        :save => true,
        :save_file => "simulation_results.csv",
        :snapshot_day => 274.0,
        :frequency_days => 365.0,
        :output_frequency => 365.0,
        :plot => false,
        :output_dir => "outputs",
        # Animation
        :animate => false,
        :animation_file => "snowcrab_animation.mp4",
        :anim_frames => 120,
        :anim_dt => 2,
        :anim_fps => 15,
        :anim_field => "temperature"
    )
end

"""
    load_ibm_configuration(filepath::String;
                           base_config::Union{Nothing, Dict{Symbol, Any}}=nothing)

Parse a configuration file (in TOML format, such as `snowcrab_ibm.config`)
and merge its sections and parameters into a standardized configuration dictionary.
Any parameter explicitly defined in the file overrides `base_config` (or
`default_ibm_configuration()` if `base_config` is not supplied).

# Arguments
- `filepath::String`: Path to the configuration file.
- `base_config::Union{Nothing, Dict{Symbol, Any}}`: Optional baseline configuration
  dictionary to update.

# Returns
`Dict{Symbol, Any}` containing merged configuration settings.
"""
function load_ibm_configuration(
    filepath::String;
    base_config::Union{Nothing, Dict{Symbol, Any}} = nothing
)
    if !isfile(filepath)
        error("Configuration file not found: $(filepath)")
    end

    raw_toml = TOML.parsefile(filepath)
    config = (base_config === nothing) ? default_ibm_configuration() : copy(base_config)

    for (section_or_key, val) in raw_toml
        if isa(val, AbstractDict)
            # Sectioned entry, e.g. [simulation], [environment]
            for (k, v) in val
                sym_k = Symbol(k)
                if sym_k === :grid && isa(v, AbstractVector) && length(v) >= 2
                    config[:grid] = (Int(v[1]), Int(v[2]))
                else
                    config[sym_k] = v
                end
            end
        else
            # Flat entry
            sym_k = Symbol(section_or_key)
            if sym_k === :grid && isa(val, AbstractVector) && length(val) >= 2
                config[:grid] = (Int(val[1]), Int(val[2]))
            else
                config[sym_k] = val
            end
        end
    end

    # Reconcile common aliases
    if haskey(config, :temp_amplitude)
        config[:temp_amp] = config[:temp_amplitude]
    elseif haskey(config, :temp_amp)
        config[:temp_amplitude] = config[:temp_amp]
    end

    if haskey(config, :fishing_probability_daily)
        config[:fishing_prob] = config[:fishing_probability_daily]
    elseif haskey(config, :fishing_prob)
        config[:fishing_probability_daily] = config[:fishing_prob]
    end

    if haskey(config, :benthic_carrying_capacity)
        config[:carrying_capacity] = config[:benthic_carrying_capacity]
    elseif haskey(config, :carrying_capacity)
        config[:benthic_carrying_capacity] = config[:carrying_capacity]
    end

    if haskey(config, :frequency_days)
        config[:output_frequency] = config[:frequency_days]
        config[:frequency] = config[:frequency_days]
    elseif haskey(config, :output_frequency)
        config[:frequency_days] = config[:output_frequency]
        config[:frequency] = config[:output_frequency]
    elseif haskey(config, :frequency)
        config[:frequency_days] = config[:frequency]
        config[:output_frequency] = config[:frequency]
    end

    return config
end

"""
    save_ibm_configuration(filepath::String, config::AbstractDict)

Serialize a configuration dictionary into TOML format at `filepath`,
structuring keys into logical categories (`simulation`, `environment`,
`biology`, `phenology`, `fishery`, `output`, `animation`).

# Arguments
- `filepath::String`: Destination path for the `.config` or `.toml` file.
- `config::AbstractDict`: Configuration settings dictionary.

# Returns
`filepath` upon successful serialization.
"""
function save_ibm_configuration(filepath::String, config::AbstractDict)
    # Group settings into standardized sections
    sections = Dict{String, Dict{String, Any}}(
        "simulation" => Dict{String, Any}(),
        "environment" => Dict{String, Any}(),
        "biology" => Dict{String, Any}(),
        "phenology" => Dict{String, Any}(),
        "fishery" => Dict{String, Any}(),
        "output" => Dict{String, Any}(),
        "animation" => Dict{String, Any}()
    )

    section_map = Dict{Symbol, String}(
        :model => "simulation",
        :n_agents => "simulation",
        :days => "simulation",
        :grid => "simulation",
        :dt => "simulation",
        :seed => "simulation",

        :temp_base => "environment",
        :temp_amplitude => "environment",
        :temp_noise_sd => "environment",
        :food_max => "environment",
        :food_replenishment_rate => "environment",
        :benthic_carrying_capacity => "environment",

        :max_size_male => "biology",
        :max_size_female => "biology",
        :maturity_size_mid => "biology",
        :metabolic_rate_base => "biology",
        :intake_rate_base => "biology",
        :energy_molt_threshold => "biology",
        :energy_molt_slope => "biology",
        :molt_energy_retention => "biology",
        :temp_base_degreedays => "biology",

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

    # Aliases to skip in serialized output to prevent duplicate entries
    alias_keys = Set([:temp_amp, :fishing_prob, :carrying_capacity, :output_frequency, :frequency])

    for (k, v) in config
        sym_k = Symbol(k)
        if sym_k in alias_keys
            continue
        end
        sec = get(section_map, sym_k, "simulation")
        str_k = string(sym_k)
        if sym_k === :grid && (isa(v, Tuple) || isa(v, AbstractVector))
            sections[sec][str_k] = [Int(v[1]), Int(v[2])]
        else
            sections[sec][str_k] = v
        end
    end

    open(filepath, "w") do io
        TOML.print(io, sections; sorted=true)
    end

    return filepath
end

"""
    initialize_model(config_filepath::String; kwargs...)

Initialize a snow crab IBM simulation directly from a TOML configuration file on disk.
Optional keyword arguments override entries loaded from the configuration file.

# Arguments
- `config_filepath::String`: Path to `.config` or `.toml` configuration file.
- `kwargs...`: Optional keyword parameter overrides.

# Returns
`StandardABM` instance configured and populated with agents.
"""
function initialize_model(config_filepath::String; kwargs...)
    config = load_ibm_configuration(config_filepath)
    return initialize_model(config; kwargs...)
end

"""
    initialize_model(config::AbstractDict; kwargs...)

Convenience constructor to initialize the snow crab IBM directly from a configuration
dictionary loaded from `snowcrab_ibm.config` or generated programmatically.
Optional keyword arguments override entries in `config`.

# Arguments
- `config::AbstractDict`: Configuration settings dictionary.
- `kwargs...`: Additional keyword arguments to override dictionary entries.

# Returns
Configured and populated `StandardABM` instance ready for simulation.
"""
function initialize_model(config::AbstractDict; kwargs...)
    merged = Dict{Symbol, Any}()
    for (k, v) in config
        merged[Symbol(k)] = v
    end
    for (k, v) in kwargs
        merged[Symbol(k)] = v
    end


    n_agents = if haskey(merged, :N_agents)
        merged[:N_agents]
    else
        get(merged, :n_agents, 300)
    end

    grid_raw = if haskey(merged, :dims)
        merged[:dims]
    else
        get(merged, :grid, (100, 100))
    end
    dims = if isa(grid_raw, Tuple)
        (Int(grid_raw[1]), Int(grid_raw[2]))
    elseif isa(grid_raw, AbstractVector)
        (Int(grid_raw[1]), Int(grid_raw[2]))
    else
        (100, 100)
    end

    dt = Float64(haskey(merged, :dt_days) ? merged[:dt_days] : get(merged, :dt, 1.0))
    seed = haskey(merged, :rng_seed) ? merged[:rng_seed] : get(merged, :seed, 1234)
    params = IBMParameters(merged)

    return initialize_model(
        N_agents = Int(n_agents),
        dims = dims,
        dt_days = dt,
        params = params,
        rng_seed = (seed === nothing) ? nothing : Int(seed)
    )
end

"""
    initialize_model(; N_agents=100, dims=(50, 50),
                     temp_map=nothing, food_map=nothing, dt_days=1.0,
                     params=IBMParameters(), rng_seed=nothing)

Construct and populate an `Agents.StandardABM` representing the snow crab ecosystem.

# Arguments
- `N_agents::Int`: Initial number of crabs seeded into the environment (default: 100).
- `dims::Tuple{Int, Int}`: Grid spatial dimensions `(nx, ny)` (default: `(50, 50)`).
- `temp_map::Union{Nothing, Matrix{Float64}}`: Optional initial temperature field.
- `food_map::Union{Nothing, Matrix{Float64}}`: Optional initial food density field.
- `dt_days::Float64`: Time step in days (default: 1.0).
- `params::IBMParameters`: Bioenergetic and environmental parameters.
- `rng_seed::Union{Nothing, Int}`: Optional RNG seed for deterministic simulations.

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
    rng_seed::Union{Nothing, Int} = nothing
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
        :n_benthic_organisms => 0, # Changed from n_benthic_crabs to match instructions (more generic)
        :n_mature_males => 0,
        :n_mature_females => 0,
        :n_legal_males => 0,
        :clutch_recruitment_records => DataFrame(
            hatch_time = Float64[],
            mother_cw = Float64[],
            mother_weight_g = Float64[],
            eggs_count = Int[],
            recruits_settled = Int[],
            density_feedback = Float64[]
        ),
        :aggregate_populations => AggregatePopulations() # Added this!
    )

    rng = if rng_seed === nothing
        Random.default_rng()
    else
        Random.Xoshiro(rng_seed)
    end

    model = StandardABM(
        SnowCrab,
        space;
        agent_step! = agent_step!,
        model_step! = model_step!,
        properties = properties,
        rng = rng
    )

    # --- Initialize Aggregate Populations (Non-spatial stages) ---
    pop = AggregatePopulations()
    if params.binning.use_bins
        n = params.binning.n_age_bins
        edges = collect(range(0.0, stop=params.simulation.days * 2.0, length=n+1))

        # Initialize stage distributions with empty/zeroed bins based on configuration
        pop.eggs = StageBinDistribution(zeros(Float64, n), edges)
        pop.larvae = StageBinDistribution(zeros(Float64, n), edges)
        pop.megalopa = StageBinDistribution(zeros(Float64, n), edges)
        for _ in 1:5 # Instars 1 to 5 (pre-individuation stages)
            push!(pop.instars, StageBinDistribution(zeros(Float64, n), edges))
        end
    end
    properties[:aggregate_populations] = pop

    # --- Populate Initial Benthic Agents (Individuated: Instar >= 6 onwards only) ---
    benthic_cnt = 0
    mat_m = 0
    mat_f = 0
    leg_m = 0

    for _ in 1:N_agents
        # According to the hybrid model, spatial agents begin at Instar 6.
        instar = rand(rng, 6:7) # Start with some initial cohort of mature/late juveniles
        sex = rand(rng, (male, female))
        size = carapacewidth(instar, sex; type=:random, rng=rng)
        mass = calculate_mass(size, sex; mat=0)
        age_days = (instar - 3.0) * 180.0 + rand(rng)*90.0 # Approx age for these instars
        energy_reserve = params.organism.initial_energy

        last_molt_time = -age_days    # backdate so days_since_molt == age_days at t=0
        degree_days = rand(rng, Uniform(50.0, 300.0))
        is_mating_ready = false
        eggs_count = 0
        hatch_day = 0.0
        sperm_clutches = 0
        pos = (rand(rng, 1:dims[1]), rand(rng, 1:dims[2]))

        # Add agent with placeholder traits for the Organism struct fields in types.jl
        new_org = add_agent!(
            pos,
            model,
            juvenile, # Using juvenile as a generic LifeStage label if instar >=6 is still considered 'juvenile' or similar 
                      # Actually we should probably have more stages but for now let's keep it consistent.
            sex,
            instar,
            age_days,
            size,
            mass,
            energy_reserve,
            last_molt_time,
            degree_days,
            is_mating_ready,
            eggs_count,
            hatch_day,
            sperm_clutches,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0 # traits: l50, slope, f_int, f_slp, g_int, g_slp
        )

        # Sample intrinsic biological traits for this individual agent from population parameters/distributions
        sample_individual_traits!(new_org, model, rng)
    end

    # --- Cache initial demographic aggregates (for spatial agents only) ---
    for a in allagents(model)
        if is_adult(a) || a.stage === juvenile # If we consider instar 6+ as 'juvenile' or similar
            benthic_cnt += 1
        end
        if is_adult_male(a)
            mat_m += 1
            if is_legal_male(a)
                leg_m += 1
            end
        elseif is_adult_female(a)
            mat_f += 1
        end
    end

    # Update the properties dictionary with current counts for reporting/visualization
    properties[:n_benthic_organisms] = benthic_cnt
    properties[:n_mature_males] = mat_m
    properties[:n_mature_females] = mat_f
    properties[:n_legal_males] = leg_m

    return model
end

"""
    sample_weighted(items, weights, rng)

Sample an element from `items` with probability proportional to `weights`.
If total weight is non-positive or items are empty, falls back to uniform selection.

# Arguments
- `items`: Collection of selectable candidate elements.
- `weights`: Numerical vector of relative sampling weights.
- `rng`: Random number generator.

# Returns
Selected element from `items`.
"""
function sample_individual_traits!(org::Organism, model, rng)
    # Maturity traits (SnowCrab species specific)
    if org.sex == female
        org.trait_l50 = rand(rng, Normal(model.params.species_snowcrab.maturity_l50_female, model.params.species_snowcrab.maturity_l50_female_sd))
        org.trait_maturity_slope = max(1.0, rand(rng, Normal(model.params.species_snowcrab.maturity_slope_female, model.params.species_snowcrab.maturity_slope_female_sd)))
    else
        org.trait_l50 = rand(rng, Normal(model.params.species_snowcrab.maturity_l50_male, model.params.species_snowcrab.maturity_l50_male_sd))
        org.trait_maturity_slope = max(1.0, rand(rng, Normal(model.params.species_snowcrab.maturity_slope_male, model.params.species_snowcrab.maturity_slope_male_sd)))
    end

    # Fecundity traits (using primiparous as base identity)
    org.trait_fecundity_intercept = rand(rng, Normal(FECUNDITY_PRIMIPAROUS_INTERCEPT, FECUNDITY_PRIMIPAROUS_ERR))
    org.trait_fecundity_slope = rand(rng, Normal(FECUNDITY_PRIMIPAROUS_SLOPE, 0.05)) # Low CV for slope

    # Growth traits (bivariate normal - SnowCrab species specific)
    z1 = rand(rng, Normal(0.0, 1.0))
    z2 = rand(rng, Normal(0.0, 1.0))
    if org.sex == female
        org.trait_growth_intercept = model.params.species_snowcrab.growth_int_female + model.params.species_snowcrab.growth_beta1_female * z1
        org.trait_growth_slope = model.params.species_snowcrab.growth_slp_female + model.params.species_snowcrab.growth_beta2_female * z1 + model.params.species_snowcrab.growth_beta3_female * z2
    else
        org.trait_growth_intercept = model.params.species_snowcrab.growth_int_male + model.params.species_snowcrab.growth_beta1_male * z1
        org.trait_growth_slope = model.params.species_snowcrab.growth_slp_male + model.params.species_snowcrab.growth_beta2_male * z1 + model.params.species_snowcrab.growth_beta3_male * z2
    end
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
    crab_disperse!(crab::SnowCrab, model; rng=abmrng(model))

Execute directed spatial movement and habitat taxis for a `SnowCrab` agent:
- Planktonic larval stages (`zoea`, `megalopa`) undergo stochastic advection / random walk.
- Benthic juveniles and adults evaluate local environmental gradients (temperature and food)
  across adjacent grid cells and stochastically move toward preferred habitat:
  - Cold-stenothermic preference optimum at ~2.5°C with Gaussian decay (σ = 1.5°C).
  - Food foraging preference scaled via Holling Type II response.
  - Softmax probability weighting over candidate neighbor positions and current cell.
- Energetically exhausted crabs (`energy_reserve <= 0`) experience reduced mobility
  (75% probability of staying stationary to conserve somatic reserves).

# Arguments
- `crab::SnowCrab`: Agent executing movement.
- `model`: Simulation ABM instance.
- `rng`: Random number generator (defaults to `abmrng(model)`).

# Returns
`Tuple{Int, Int}`: Final grid coordinates of the agent.
"""
function organism_disperse!(
    org::Organism,
    model;
    rng::AbstractRNG = abmrng(model)
)
    # 1. Larval pelagic drift: simple stochastic dispersal
    if org.stage === zoea || org.stage === megalopa
        randomwalk!(org, model)
        return org.pos
    end

    # 2. Energy-depleted lethargy: 75% chance of remaining stationary
    if org.energy_reserve <= 0.0 && rand(rng) < 0.75
        return org.pos
    end

    # 3. Benthic habitat taxis: evaluate current position and Moore neighbors
    candidates = [org.pos; collect(nearby_positions(org.pos, model))]

    # Compute habitat utility for each candidate position
    # U(x, y) = S_temp * (1.0 + 0.5 * S_food)
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
    agent_step!(crab::SnowCrab, model)

Execute one simulation time step for an individual `crab`. Biological processes
(e.g., energetics, thermal stress, cannibalism, senescence) are applied
modularly based on the feature toggles in `model.params`.
"""
function agent_step!(org::Organism, model)
    # 1. Age progression
    org.age_days += model.dt
    day_of_year = mod(model.current_time, 365.0)

    # 2. Development (Modular via Behavior Interface)
    perform_development!(org.behavior, org, model)

    # 3. Natural mortality (Modular)
    size_excess = max(0.0, org.size - 5.0)
    pred_risk = (0.15 / 365.0) * exp(-0.03 * size_excess)
    base_mort = 0.20 / 365.0

    senescence_risk = 0.0
    if model.params.simulation.use_senescence && org.stage === adult
        days_since_molt = model.current_time - org.last_molt_time
        shell_age_years = days_since_molt / 365.0
        thresh = model.params.mortality.senescence_age_threshold
        if shell_age_years > thresh
            senescence_risk = (0.08 / 365.0) * exp(0.80 * (shell_age_years - thresh))
        end
    end

    mort_prob = base_mort + pred_risk + senescence_risk
    if mort_prob > 1.0
        error("Total mortality probability exceeded 1.0 at time $(model.current_time) (value: $mort_prob). Check model assumptions.")
    elseif mort_prob < 0.0
        mort_prob = 0.0
    end

    if rand(abmrng(model)) < mort_prob
        if model.params.simulation.use_senescence && org.stage === adult && senescence_risk > 0
             model.deaths_senescence += 1
        else
             model.deaths_predation += 1
        end
        remove_agent!(org, model)
        return nothing
    end

    # 4. Commercial fishery harvest (Modular)
    if model.params.fishery.fishing_probability_daily > 0.0 &&
       (model.params.fishery.fishing_season_start <= day_of_year && day_of_year <= model.params.fishery.fishing_season_end) &&
       org.stage === adult && org.sex === male &&
       org.size >= model.params.fishery.legal_size_male
        harvest_prob = model.params.fishery.fishing_probability_daily *
                       selectivity(
                           org.size;
                           s50 = model.params.fishery.legal_size_male,
                           s50_sigma = model.params.species_snowcrab.selectivity_s50_sigma
                       )
        if rand(abmrng(model)) < harvest_prob
            model.total_catch_numbers += 1
            model.total_catch_biomass_kg += (org.mass / 1000.0)
            model.deaths_fishery += 1
            remove_agent!(org, model)
            return nothing
        end
    end

    # 5. Locomotion: Habitat Taxis or Random Walk (Modular via Behavior/Config)
    organism_disperse!(org, model)

    # 6. Reproduction (Modular via Behavior Interface)
    perform_reproduction!(org.behavior, org, model)

    return nothing
end

function update_demographic_caches!(model)
    benthic_count = 0
    mat_males = 0
    mat_females = 0
    legal_males = 0
    for org in allagents(model)
        if org.stage === juvenile || org.stage === adult
            benthic_count += 1
        end
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
    model.n_mature_males = mat_males
    model.n_mature_females = mat_females
    model.n_legal_males = legal_males
    return nothing
end

"""
    model_step!(model)

Advance environmental state and update demographic caches for one simulation time step,
using feature toggles to determine which processes are active.
"""
function model_step!(model)
    dt = model.dt
    old_time = model.current_time
    model.current_time += dt

    # Year boundary detection for annual resets
    if floor(old_time/365.0) < floor(model.current_time/365.0)
        model.annual_eggs_spawned = 0.0
    end

    # 1. Environmental updates (Temperature & Food)
    if model.params.simulation.use_thermal_stress
        seasonal_temp = model.params.environment.temp_base +
                        model.params.environment.temp_amplitude *
                        sin(2.0 * π * (model.current_time - 169.0) / 365.0)

        noise_sd = model.params.environment.temp_noise_sd
        nx, ny = size(model.temperature_field)
        for y in 1:ny
            for x in 1:nx
                model.temperature_field[x, y] = seasonal_temp +
                                                randn(abmrng(model)) * noise_sd
            end
        end
    end

    if model.params.simulation.use_food_dynamics
        nx, ny = size(model.food_availability_field)
        for y in 1:ny
            for x in 1:nx
                current_food = model.food_availability_field[x, y]
                model.food_availability_field[x, y] = min(
                    model.params.environment.food_max,
                    current_food + model.params.environment.food_replenishment_rate
                )
            end
        end
    end

    # 2. Refresh demographic aggregate caches once per time step
    update_demographic_caches!(model)

    return nothing
end

"""
    safe_mean(itr)

Compute arithmetic mean of an iterator, returning 0.0 if empty.
"""
safe_mean(itr) = isempty(itr) ? 0.0 : mean(itr)

"""
    run_simulation(model, n_steps; adata=nothing, mdata=nothing, when=nothing)

Convenience wrapper to execute an IBM simulation for `n_steps` days, collecting
agent-level and model-level time-series metrics.

# Default Agent Metrics (`adata`)
- Total population count: `(:stage, length)`
- Average carapace width: `(:size_mm, safe_mean)`
- Total adults: `(is_adult, count)`
- Adult males: `(is_adult_male, count)`
- Adult females: `(is_adult_female, count)`
- Mean instar: `(:instar, safe_mean)`

# Returns
Tuple `(agent_df, model_df)`.
"""
function run_simulation(
    model,
    n_steps::Integer;
    adata = nothing,
    mdata = nothing,
    when = nothing,
    func_params::Union{Nothing, Vector{IBMParameters}} = nothing
)
    default_adata = [
        (:stage, length),
        (:size, safe_mean),
        (:mass, safe_mean),
        (is_adult, count),
        (is_adult_male, count),
        (is_adult_female, count),
        (is_legal_male, count),
        (is_ovigerous, count),
        (weight_if_adult_female, sum),
        (weight_if_adult_male, sum),
        (weight_if_legal_male, sum),
        (:instar, safe_mean)
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

    results = if when === nothing
        run!(model, n_steps; adata=actual_adata, mdata=actual_mdata)
    else
        run!(model, n_steps; adata=actual_adata, mdata=actual_mdata, when=when)
    end

    if !isnothing(func_params)
        using model_agent: plot_functional_relationships
        for type in [:maturity, :fecundity, :selectivity]
            p = plot_functional_relationships(func_params; type=type)
            savefig(p, "functional_space_$(type).png")
        end
    end

    return results
end
