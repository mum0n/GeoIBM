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
        :model => "simple",
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

    mode_raw = if haskey(merged, :model_mode)
        merged[:model_mode]
    elseif haskey(merged, :model)
        merged[:model]
    else
        :simple
    end
    model_mode = Symbol(lowercase(string(mode_raw)))
    if model_mode ∉ (:simple, :complex)
        model_mode = :simple
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
        model_mode = model_mode,
        N_agents = Int(n_agents),
        dims = dims,
        dt_days = dt,
        params = params,
        rng_seed = (seed === nothing) ? nothing : Int(seed)
    )
end

"""
    initialize_model(; model_mode=:simple, model=nothing, N_agents=100, dims=(50, 50),
                     temp_map=nothing, food_map=nothing, dt_days=1.0,
                     params=IBMParameters(), rng_seed=nothing)

Construct and populate an `Agents.StandardABM` representing the snow crab ecosystem.

# Arguments
- `model_mode::Union{Symbol, String}`: Model formulation: `:simple` (default baseline)
  or `:complex` (detailed bioenergetics & multi-driver mortality).
- `model::Union{Symbol, String, Nothing}`: Optional alias for `model_mode`.
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
    model_mode::Union{Symbol, String} = :simple,
    model::Union{Symbol, String, Nothing} = nothing,
    N_agents::Int = 100,
    dims::Tuple{Int, Int} = (50, 50),
    temp_map::Union{Nothing, Matrix{Float64}} = nothing,
    food_map::Union{Nothing, Matrix{Float64}} = nothing,
    dt_days::Float64 = 1.0,
    params::IBMParameters = IBMParameters(),
    rng_seed::Union{Nothing, Int} = nothing
)
    resolved_mode = if model !== nothing
        Symbol(lowercase(string(model)))
    else
        Symbol(lowercase(string(model_mode)))
    end
    if resolved_mode ∉ (:simple, :complex)
        resolved_mode = :simple
    end

    space = GridSpace(dims)

    initial_temp = if temp_map === nothing
        fill(params.temp_base, dims)
    else
        copy(temp_map)
    end

    initial_food = if food_map === nothing
        fill(params.food_max / 2.0, dims)
    else
        copy(food_map)
    end

    properties = Dict{Symbol, Any}(
        :model_mode => resolved_mode,
        :current_time => 0.0,
        :temperature_field => initial_temp,
        :food_availability_field => initial_food,
        :dt => dt_days,
        :params => params,
        :total_eggs_spawned => 0,
        :total_recruits_settled => 0,
        :total_catch_numbers => 0,
        :total_catch_biomass_kg => 0.0,
        :total_molts => 0,
        :deaths_predation => 0,
        :deaths_starvation => 0,
        :deaths_thermal => 0,
        :deaths_senescence => 0,
        :deaths_fishery => 0,
        :n_benthic_crabs => 0,
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
        )
    )

    rng = if rng_seed === nothing
        Random.default_rng()
    else
        Random.Xoshiro(rng_seed)
    end

    agent_step_fn = (resolved_mode === :simple) ? agent_step_simple! : agent_step_complex!
    model_step_fn = (resolved_mode === :simple) ? model_step_simple! : model_step_complex!

    model = StandardABM(
        SnowCrab,
        space;
        agent_step! = agent_step_fn,
        model_step! = model_step_fn,
        properties = properties,
        rng = rng
    )

    # Populate initial cohort across early benthic instars (instars 4 to 7)
    for _ in 1:N_agents
        instar = rand(rng, 4:7)
        sex = rand(rng, (male, female))
        size_mm = carapacewidth(instar, sex; type=:random, rng=rng)
        weight_g = crab_weight(size_mm, sex; mat=0)
        age_days = (instar - 3) * 180.0 + rand(rng) * 90.0
        energy_reserve = params.energy_molt_threshold * rand(rng, Uniform(0.3, 0.8))
        last_molt_time = 0.0
        degree_days = rand(rng, Uniform(50.0, 300.0))
        is_mating_ready = false
        eggs_count = 0
        hatch_day = 0.0
        sperm_clutches = 0
        pos = (rand(rng, 1:dims[1]), rand(rng, 1:dims[2]))

        add_agent!(
            pos,
            model,
            juvenile,
            sex,
            instar,
            age_days,
            size_mm,
            weight_g,
            energy_reserve,
            last_molt_time,
            degree_days,
            is_mating_ready,
            eggs_count,
            hatch_day,
            sperm_clutches
        )
    end

    # Cache initial demographic aggregates
    benthic_cnt = 0
    mat_m = 0
    mat_f = 0
    leg_m = 0
    for a in allagents(model)
        if a.stage === juvenile || a.stage === adult
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
    model.n_benthic_crabs = benthic_cnt
    model.n_mature_males = mat_m
    model.n_mature_females = mat_f
    model.n_legal_males = leg_m

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
function sample_weighted(items, weights, rng)
    tot = sum(weights)
    if tot <= 0.0 || isempty(items)
        return rand(rng, items)
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
function crab_disperse!(
    crab::SnowCrab,
    model;
    rng::AbstractRNG = abmrng(model)
)
    # 1. Larval pelagic drift: simple stochastic dispersal
    if crab.stage === zoea || crab.stage === megalopa
        randomwalk!(crab, model)
        return crab.pos
    end

    # 2. Energy-depleted lethargy: 75% chance of remaining stationary
    if crab.energy_reserve <= 0.0 && rand(rng) < 0.75
        return crab.pos
    end

    # 3. Benthic habitat taxis: evaluate current position and Moore neighbors
    neighbors = collect(nearby_positions(crab.pos, model))
    candidates = pushfirst!(neighbors, crab.pos)

    # Compute habitat utility for each candidate position
    # U(x, y) = S_temp * (1.0 + 0.5 * S_food)
    t_opt = 2.5
    t_sigma = 1.5
    half_sat = model.params.half_saturation_food

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
    if chosen_pos != crab.pos
        move_agent!(crab, chosen_pos, model)
    end

    return crab.pos
end

"""
    agent_step_simple!(crab::SnowCrab, model)

Execute one simulation time step for an individual `crab` under the simplified IBM formulation (`model = :simple`).

Distillation features:
- Ignores food limitation: somatic energy reserve is maintained at a healthy positive baseline.
- Ignores temperature interactions: ambient thermal gradients do not modulate molting, maturity,
  or mortality.
- Ignores degree-days: developmental progression is purely time/age-based.
- Ignores carapace condition: no soft-shell post-molt vulnerability or senescent shell degradation.
- Time/age-based molting: evaluated during seasonal molt window once intermolt days elapsed
  exceed `min_intermolt_days(stage, instar)`.
- Standard allometric maturation: evaluated via baseline logistic ogive without thermal anomaly.
- Ignores spermathecal storage: receptive adult females encountering any mature adult male
  extrude an egg clutch directly with a fixed 365-day brood incubation.
- Standard random-walk spatial dispersal.
- Natural and commercial fishery mortality.
"""
function agent_step_simple!(crab::SnowCrab, model)
    day_of_year = mod(model.current_time, 365.0)

    # 1. Age progression (without food limitation or degree-day accumulation)
    crab.age_days += model.dt
    crab.energy_reserve = model.params.energy_molt_threshold

    # 2. Time/age-based molting check and execution
    is_molt_season = (model.params.molt_season_start <= day_of_year <=
                      model.params.molt_season_end)

    if crab.stage === juvenile && is_molt_season
        days_since_molt = crab.age_days - crab.last_molt_time
        min_days = min_intermolt_days(crab.stage, crab.instar)
        if days_since_molt >= min_days
            # Daily molt hazard scaling with accumulated intermolt duration
            p_molt = clamp(
                (days_since_molt - min_days + 1.0) / 45.0,
                0.05,
                0.35
            )
            if rand(abmrng(model)) < p_molt
                # Execute ecdysis
                sex_code = (crab.sex === female) ? 1 : 0
                prob_mat = maturity(sex_code, log(crab.size_mm); temp = nothing, rng = abmrng(model))
                mat_status = (rand(abmrng(model)) < prob_mat) ? 1 : 0

                if mat_status == 1
                    crab.stage = adult
                    crab.is_mating_ready = true
                end

                crab.size_mm = predict_postmolt_cw(
                    crab.size_mm;
                    sex = sex_code,
                    mat = mat_status,
                    temp = nothing,
                    condition = 1.0,
                    rng = abmrng(model)
                )
                crab.instar += 1
                crab.last_molt_time = model.current_time
                crab.weight_g = crab_weight(crab; stochastic = false, rng = abmrng(model))

                if haskey(abmproperties(model), :total_molts)
                    model.total_molts += 1
                end
            end
        end
    end

    # 3. Baseline natural mortality (constant base + size-attenuated juvenile predation)
    cw_excess = max(0.0, crab.size_mm - 5.0)
    pred_risk = (0.15 / 365.0) * exp(-0.03 * cw_excess)
    base_mort = 0.20 / 365.0
    mort_prob = clamp(base_mort + pred_risk, 0.0, 1.0)

    if rand(abmrng(model)) < mort_prob
        if haskey(abmproperties(model), :deaths_predation)
            model.deaths_predation += 1
        end
        remove_agent!(crab, model)
        return nothing
    end

    # 4. Commercial fishery harvest (legal-sized adult males during season)
    is_fishing_season = (model.params.fishing_season_start <= day_of_year <=
                         model.params.fishing_season_end)

    if is_fishing_season && crab.stage === adult && crab.sex === male &&
       crab.size_mm >= model.params.legal_size_male
        harvest_prob = model.params.fishing_probability_daily *
                       selectivity(
                           crab.size_mm;
                           s50 = model.params.legal_size_male,
                           s50_sigma = 5.0
                       )
        if rand(abmrng(model)) < harvest_prob
            if haskey(abmproperties(model), :total_catch_numbers)
                model.total_catch_numbers += 1
                model.total_catch_biomass_kg += (crab.weight_g / 1000.0)
            end
            if haskey(abmproperties(model), :deaths_fishery)
                model.deaths_fishery += 1
            end
            remove_agent!(crab, model)
            return nothing
        end
    end

    # 5. Locomotion / Dispersal: standard random walk
    randomwalk!(crab, model)

    # 6. Reproduction: seasonal mating without spermathecal storage
    is_mating_season = (model.params.mating_season_start <= day_of_year <=
                        model.params.mating_season_end)

    if crab.stage === adult && crab.sex === female
        # Receptive adult female seeks any mature male within detection radius
        if is_mating_season && crab.is_mating_ready && crab.eggs_count == 0
            radius = model.params.mating_search_radius
            nearby_males = filter(
                m -> m.stage === adult && m.sex === male,
                collect(nearby_agents(crab, model, radius))
            )
            if !isempty(nearby_males)
                # Extrude brood clutch directly without multi-year spermathecal tracking
                egg_estimate = fecundity(
                    crab.size_mm;
                    method = "moncton.primiparous",
                    rng = abmrng(model)
                )
                crab.eggs_count = max(round(Int, egg_estimate), 1000)
                # Fixed brood incubation (~1 year) without temperature acceleration
                crab.hatch_day = model.current_time + model.params.brood_duration_base
                crab.is_mating_ready = false

                crab.weight_g = crab_weight(
                    crab;
                    include_eggs = true,
                    rng = abmrng(model)
                )

                if haskey(abmproperties(model), :total_eggs_spawned)
                    model.total_eggs_spawned += crab.eggs_count
                end
            end
        end
    end

    # 7. Larval hatching and benthic settlement
    if crab.eggs_count > 0 && model.current_time >= crab.hatch_day
        n_benthic = get(abmproperties(model), :n_benthic_crabs, 0)
        if n_benthic == 0
            n_benthic = count(
                c -> c.stage === juvenile || c.stage === adult,
                allagents(model)
            )
        end
        k_cap = model.params.benthic_carrying_capacity
        density_feedback = 1.0 + (n_benthic / k_cap)

        expected_recruits = (model.params.egg_survival_base * crab.eggs_count) /
                            density_feedback
        num_recruits = rand(abmrng(model), Poisson(expected_recruits))

        nx, ny = size(model.temperature_field)
        for _ in 1:num_recruits
            recruit_sex = rand(abmrng(model), (male, female))
            sex_code = (recruit_sex === female) ? 1 : 0
            recruit_cw = carapacewidth(
                4,
                sex_code;
                type = :random,
                rng = abmrng(model)
            )
            dx = rand(abmrng(model), -5:5)
            dy = rand(abmrng(model), -5:5)
            rx = clamp(crab.pos[1] + dx, 1, nx)
            ry = clamp(crab.pos[2] + dy, 1, ny)

            recruit_weight = crab_weight(recruit_cw, recruit_sex; mat = 0)

            add_agent!(
                (rx, ry),
                model,
                juvenile,
                recruit_sex,
                4,
                180.0,
                recruit_cw,
                recruit_weight,
                model.params.energy_molt_threshold,
                model.current_time,
                0.0,
                false,
                0,
                0.0,
                0
            )
        end

        if haskey(abmproperties(model), :clutch_recruitment_records)
            push!(abmproperties(model)[:clutch_recruitment_records], (
                hatch_time = model.current_time,
                mother_cw = crab.size_mm,
                mother_weight_g = crab.weight_g,
                eggs_count = crab.eggs_count,
                recruits_settled = num_recruits,
                density_feedback = density_feedback
            ))
        end

        if haskey(abmproperties(model), :total_recruits_settled)
            model.total_recruits_settled += num_recruits
        end
        crab.eggs_count = 0
        crab.is_mating_ready = true

        crab.weight_g = crab_weight(
            crab;
            include_eggs = false,
            rng = abmrng(model)
        )
    end

    return nothing
end

"""
    agent_step_complex!(crab::SnowCrab, model)

Execute one simulation time step for an individual `crab` under the realistic, multi-driver
IBM formulation (`model = :complex`):
1. Metabolize energy via mass-scaled bioenergetics (W^0.75) and consume food via Holling Type II.
2. Evaluate molting readiness via degree-day thresholds and thermal ecdysial inhibition.
3. Apply natural mortality (predation, thermal stress, starvation, senescence, cannibalism).
4. Apply seasonal commercial fishery harvest to legal-sized adult males.
5. Disperse via directed habitat taxis toward optimal thermal and trophic conditions.
6. Conduct size-assortative mating, sperm clutch replenishment, and clutch extrusion.
7. Hatch brooded embryos with regional drift dispersal.
"""
function agent_step_complex!(crab::SnowCrab, model)
    x, y = crab.pos
    current_temp = model.temperature_field[x, y]
    current_food = model.food_availability_field[x, y]

    # 1. Energetics and food depletion
    dE_dt, actual_intake = calculate_energy_change(
        crab,
        current_temp,
        current_food;
        params = model.params
    )
    model.food_availability_field[x, y] = max(0.0, current_food - actual_intake)
    crab.energy_reserve += dE_dt * model.dt
    crab.age_days += model.dt
    crab.degree_days += max(0.0, current_temp - model.params.temp_base_degreedays) * model.dt

    # 2. Molting check and execution
    if check_molt_conditions(crab, current_temp, model.current_time;
                             params = model.params, rng = abmrng(model))
        perform_molt!(
            crab,
            model.current_time,
            current_temp;
            params = model.params,
            rng = abmrng(model)
        )
        if haskey(abmproperties(model), :total_molts)
            model.total_molts += 1
        end
    end

    # 3. Natural mortality with density-dependent cannibalism coupling
    local_density = Float64(length(ids_in_position(crab.pos, model)))
    mort_prob = calculate_mortality_probability(
        crab,
        current_temp,
        current_food;
        params = model.params,
        local_density = local_density
    )
    if rand(abmrng(model)) < mort_prob
        # Tally dominant mortality driver
        if haskey(abmproperties(model), :deaths_starvation)
            days_since_molt = max(0.0, crab.age_days - crab.last_molt_time)
            thresh = model.params.senescence_age_threshold
            if crab.energy_reserve < 0.0
                model.deaths_starvation += 1
            elseif current_temp > 6.0
                model.deaths_thermal += 1
            elseif crab.stage === adult && (days_since_molt / 365.0) > thresh
                model.deaths_senescence += 1
            else
                model.deaths_predation += 1
            end
        end
        remove_agent!(crab, model)
        return nothing
    end

    # 4. Commercial fishery mortality (legal-sized adult males during season)
    day_of_year = mod(model.current_time, 365.0)
    is_season = (model.params.fishing_season_start <= day_of_year <=
                 model.params.fishing_season_end)

    if is_season && crab.stage === adult && crab.sex === male &&
       crab.size_mm >= model.params.legal_size_male
        harvest_prob = model.params.fishing_probability_daily *
                       selectivity(crab.size_mm; s50=model.params.legal_size_male,
                                   s50_sigma=5.0)
        if rand(abmrng(model)) < harvest_prob
            if haskey(abmproperties(model), :total_catch_numbers)
                model.total_catch_numbers += 1
                model.total_catch_biomass_kg += (crab.weight_g / 1000.0)
            end
            if haskey(abmproperties(model), :deaths_fishery)
                model.deaths_fishery += 1
            end
            remove_agent!(crab, model)
            return nothing
        end
    end

    # 5. Locomotion / Dispersal via directed habitat taxis
    crab_disperse!(crab, model; rng = abmrng(model))

    # 6. Reproduction: seasonal mating and egg clutch extrusion
    day_of_year = mod(model.current_time, 365.0)
    is_mating_season = (model.params.mating_season_start <= day_of_year <=
                        model.params.mating_season_end)

    if crab.stage === adult && crab.sex === female
        # Spring mating: receptive adult female seeks mature male within detection radius
        if is_mating_season && crab.is_mating_ready && crab.eggs_count == 0
            radius = model.params.mating_search_radius
            nearby_males = filter(
                m -> m.stage === adult && m.sex === male && m.energy_reserve > 5.0,
                collect(nearby_agents(crab, model, radius))
            )
            # If male found, copulate: replenish spermathecal storage with sperm
            # Larger males dominate competition (selection weighted by CW^2)
            if !isempty(nearby_males)
                male_weights = [m.size_mm^2 for m in nearby_males]
                selected_male = sample_weighted(nearby_males, male_weights, abmrng(model))

                # Ejaculate quantity scaled by male size
                crab.sperm_clutches = (selected_male.size_mm >= 95.0) ? 3 : 2
                selected_male.energy_reserve = max(
                    0.0,
                    selected_male.energy_reserve - 8.0
                )
            end
        end

        # Egg clutch extrusion: requires stored sperm, empty brood, and mating season
        if is_mating_season && crab.eggs_count == 0 && crab.sperm_clutches > 0 &&
           crab.energy_reserve > 5.0
            egg_estimate = fecundity(crab; rng = abmrng(model))
            crab.eggs_count = max(egg_estimate, 1000)
            crab.sperm_clutches -= 1

            # Brood incubation ~1 year, with modest temperature acceleration
            temp_dev = current_temp - model.params.temp_base
            incubation_days = model.params.brood_duration_base *
                              exp(-0.02 * temp_dev)
            crab.hatch_day = model.current_time + incubation_days

            # Metabolic and energetic allocation to clutch production
            crab.energy_reserve *= 0.60
            crab.is_mating_ready = false

            # Update female body weight to include extruded egg mass
            crab.weight_g = crab_weight(crab; include_eggs = true, rng = abmrng(model))

            if haskey(abmproperties(model), :total_eggs_spawned)
                model.total_eggs_spawned += crab.eggs_count
            end
        end
    end

    # 7. Larval hatching and benthic settlement (feeding the system)
    if crab.eggs_count > 0 && model.current_time >= crab.hatch_day
        # Benthic population count for density-dependent compensatory survival
        n_benthic = get(abmproperties(model), :n_benthic_crabs, 0)
        if n_benthic == 0
            n_benthic = count(
                c -> c.stage === juvenile || c.stage === adult,
                allagents(model)
            )
        end
        k_cap = model.params.benthic_carrying_capacity
        density_feedback = 1.0 + (n_benthic / k_cap)

        # Expected number of surviving recruits to benthic settlement (Instar 4)
        expected_recruits = (model.params.egg_survival_base * crab.eggs_count) /
                            density_feedback

        # Realized stochastic recruitment drawn from Poisson distribution
        num_recruits = rand(abmrng(model), Poisson(expected_recruits))

        nx, ny = size(model.temperature_field)
        for _ in 1:num_recruits
            recruit_sex = rand(abmrng(model), (male, female))
            sex_code = (recruit_sex === female) ? 1 : 0
            recruit_cw = carapacewidth(
                4,
                sex_code;
                type = :random,
                rng = abmrng(model)
            )

            # Disperse regionally reflecting pelagic larval drift (±8 cells)
            dx = rand(abmrng(model), -8:8)
            dy = rand(abmrng(model), -8:8)
            rx = clamp(crab.pos[1] + dx, 1, nx)
            ry = clamp(crab.pos[2] + dy, 1, ny)

            recruit_weight = crab_weight(recruit_cw, recruit_sex; mat = 0)

            add_agent!(
                (rx, ry),
                model,
                juvenile,
                recruit_sex,
                4,                    # Instar 4: primary post-larval benthic recruit
                180.0,                # Age in days (includes larval planktonic phase)
                recruit_cw,
                recruit_weight,
                10.0,                 # Modest starting energy reserve
                model.current_time,
                0.0,                  # Initial degree-days = 0
                false,
                0,
                0.0,
                0                     # Initial sperm clutches = 0
            )
        end

        if haskey(abmproperties(model), :clutch_recruitment_records)
            push!(abmproperties(model)[:clutch_recruitment_records], (
                hatch_time = model.current_time,
                mother_cw = crab.size_mm,
                mother_weight_g = crab.weight_g,
                eggs_count = crab.eggs_count,
                recruits_settled = num_recruits,
                density_feedback = density_feedback
            ))
        end

        if haskey(abmproperties(model), :total_recruits_settled)
            model.total_recruits_settled += num_recruits
        end
        crab.eggs_count = 0
        crab.is_mating_ready = true   # Ready for next spring reproductive season

        # Update female weight back to somatic mass without eggs
        crab.weight_g = crab_weight(crab; include_eggs = false, rng = abmrng(model))
    end

    return nothing
end

"""
    agent_step!(crab::SnowCrab, model)

Execute one simulation time step for an individual `crab`, automatically dispatching
to `agent_step_simple!` or `agent_step_complex!` depending on `model.model_mode`.
"""
function agent_step!(crab::SnowCrab, model)
    mode = get(abmproperties(model), :model_mode, :simple)
    if mode === :complex
        return agent_step_complex!(crab, model)
    else
        return agent_step_simple!(crab, model)
    end
end

"""
    model_step_simple!(model)

Advance simulation time and refresh demographic caches under `model = :simple`.
Does not compute sinusoidal temperature oscillations or food replenishment.
"""
function model_step_simple!(model)
    model.current_time += model.dt

    # Refresh demographic aggregate caches once per time step
    benthic_count = 0
    mat_males = 0
    mat_females = 0
    legal_males = 0
    for crab in allagents(model)
        if crab.stage === juvenile || crab.stage === adult
            benthic_count += 1
        end
        if is_adult_male(crab)
            mat_males += 1
            if is_legal_male(crab)
                legal_males += 1
            end
        elseif is_adult_female(crab)
            mat_females += 1
        end
    end
    if haskey(abmproperties(model), :n_benthic_crabs)
        model.n_benthic_crabs = benthic_count
    end
    if haskey(abmproperties(model), :n_mature_males)
        model.n_mature_males = mat_males
    end
    if haskey(abmproperties(model), :n_mature_females)
        model.n_mature_females = mat_females
    end
    if haskey(abmproperties(model), :n_legal_males)
        model.n_legal_males = legal_males
    end

    return nothing
end

"""
    model_step_complex!(model)

Advance environmental state and update demographic caches for one simulation time step under `model = :complex`:
1. Advance simulation time by `model.dt`.
2. Update sinusoidal seasonal bottom water temperature field with localized noise.
3. Replenish local food resources up to carrying capacity.
4. Update cached demographic counts (`:n_benthic_crabs`, `:n_mature_males`,
   `:n_mature_females`, `:n_legal_males`).
"""
function model_step_complex!(model)
    model.current_time += model.dt

    # Sinusoidal seasonal oscillation: T(t) = T_base + T_amp * sin(2π * t / 365)
    seasonal_temp = model.params.temp_base +
                    model.params.temp_amplitude *
                    sin(2.0 * π * model.current_time / 365.0)

    noise_sd = model.params.temp_noise_sd
    nx, ny = size(model.temperature_field)
    for y in 1:ny
        for x in 1:nx
            model.temperature_field[x, y] = seasonal_temp +
                                            randn(abmrng(model)) * noise_sd
            current_food = model.food_availability_field[x, y]
            model.food_availability_field[x, y] = min(
                model.params.food_max,
                current_food + model.params.food_replenishment_rate
            )
        end
    end

    # Update demographic aggregate caches once per time step
    benthic_count = 0
    mat_males = 0
    mat_females = 0
    legal_males = 0
    for crab in allagents(model)
        if crab.stage === juvenile || crab.stage === adult
            benthic_count += 1
        end
        if is_adult_male(crab)
            mat_males += 1
            if is_legal_male(crab)
                legal_males += 1
            end
        elseif is_adult_female(crab)
            mat_females += 1
        end
    end
    if haskey(abmproperties(model), :n_benthic_crabs)
        model.n_benthic_crabs = benthic_count
    end
    if haskey(abmproperties(model), :n_mature_males)
        model.n_mature_males = mat_males
    end
    if haskey(abmproperties(model), :n_mature_females)
        model.n_mature_females = mat_females
    end
    if haskey(abmproperties(model), :n_legal_males)
        model.n_legal_males = legal_males
    end

    return nothing
end

"""
    model_step!(model)

Advance environmental state and update demographic caches for one simulation time step,
automatically dispatching to `model_step_simple!` or `model_step_complex!` depending
on `model.model_mode`.
"""
function model_step!(model)
    mode = get(abmproperties(model), :model_mode, :simple)
    if mode === :complex
        return model_step_complex!(model)
    else
        return model_step_simple!(model)
    end
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
    when = nothing
)
    default_adata = [
        (:stage, length),
        (:size_mm, safe_mean),
        (:weight_g, safe_mean),
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
        :n_benthic_crabs,
        :n_mature_males,
        :n_mature_females,
        :n_legal_males
    ]

    actual_adata = (adata === nothing) ? default_adata : adata
    actual_mdata = (mdata === nothing) ? default_mdata : mdata

    if when === nothing
        return run!(model, n_steps; adata=actual_adata, mdata=actual_mdata)
    else
        return run!(model, n_steps; adata=actual_adata, mdata=actual_mdata, when=when)
    end
end
