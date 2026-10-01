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
    parameter_structs() → Tuple of (field, struct, section)

Single source of truth for the configuration surface. The field is the name of the
slot holding that struct inside `IBMParameters`, the struct is its type, and the
section is the TOML section that carries it.

`config_field_types`, `config_section_map`, `reconstruct_params` and
`ibm_configuration_sections` all derive from this, so a new parameter struct
cannot be added to one and forgotten in another — the omission mode that let the
GUI's Reset button drop a parameter, and that let a documented TOML key fail the
"every TOML key is a parameter" check.
"""
parameter_structs() = (
    (:toggles, ProcessToggles, "toggles"),
    (:simulation, SimulationParams, "simulation"),
    (:environment, EnvironmentParams, "environment"),
    (:organism, OrganismParams, "organism"),
    (:species, SpeciesParams, "species"),
    (:phenology, PhenologyParams, "phenology"),
    (:mortality, MortalityParams, "mortality"),
    (:fishery, FisheryParams, "fishery"),
    (:output, OutputParams, "output"),
    (:leslie, LeslieParams, "leslie"),
    (:criteria, CriteriaParams, "criteria"),
)

"""
    config_field_types()

Map every declared parameter field to its type, derived from the parameter
structs. Field names are unique across the structs, so this map is
unambiguous and stays correct as structs gain fields.
"""
function config_field_types()
    types = Dict{Symbol, DataType}()
    for (_, struct_type, _) in parameter_structs()
        for field in fieldnames(struct_type)
            types[field] = fieldtype(struct_type, field)
        end
    end
    return types
end

"""
    config_field_type(key::Symbol)

Declared type of parameter `key`, or `nothing` when the name is not a parameter.
Used to coerce command-line strings to the type the struct expects.
"""
config_field_type(key::Symbol) = get(config_field_types(), key, nothing)

"""
    parameter_struct_defaults()

Collect the declared default of every field across all parameter structs into a
flat `Dict{Symbol, Any}`. This is the single source of truth for default values
and is derived automatically, so a new struct field appears here with no edit.
"""
function parameter_struct_defaults()
    defaults = Dict{Symbol, Any}()
    for (_, struct_type, _) in parameter_structs()
        base = struct_type()
        for field in fieldnames(struct_type)
            defaults[field] = getfield(base, field)
        end
    end
    return defaults
end

"""
    default_ibm_configuration()

Return the default configuration dictionary for the organism IBM simulation,
encompassing spatial grid geometry, runtime duration, environmental forcings,
bioenergetics, life-history phenology, fishery dynamics, process toggles, and
output handling.

Values come from the parameter struct defaults, so `configs/default.toml` and
the structs can never silently disagree: the TOML is the documented,
user-editable record, and the structs are the programmatic fallback.

# Returns
`Dict{Symbol, Any}` containing all baseline parameter keys and values.
"""
function default_ibm_configuration()
    config = parameter_struct_defaults()

    # Grid is addressed as a pair in the flat configuration, so derive it from
    # the struct's `grid_nx`/`grid_ny` fields.
    config[:grid] = (config[:grid_nx], config[:grid_ny])

    # Path-valued keys are resolved relative to the project at write time.
    config[:output_dir] = "outputs"
    config[:save_file] = "simulation_results.csv"
    config[:animation_file] = "organism_animation.mp4"

    sync_grid_dimensions!(config)
    return config
end

"""
    sync_grid_dimensions!(config::AbstractDict; explicit=nothing)

Keep the two representations of the spatial grid in agreement.

`grid_nx` and `grid_ny` are the declared parameter fields, while `grid` is the
`(nx, ny)` pair that the model constructor consumes. Whichever spelling a
configuration layer supplies, the other is derived from it.

Configurations are assembled in layers (defaults, baseline file, species overlay,
command line, keyword overrides). Without the `explicit` set, a value inherited
from an earlier layer would masquerade as a local choice and overwrite what the
current layer actually set, so callers pass the keys this layer supplied.

# Arguments
- `config`: configuration dictionary, modified in place.
- `explicit`: keys set by the current layer, or `nothing` to accept whichever
  spelling is already present.
"""
function sync_grid_dimensions!(
    config::AbstractDict;
    explicit::Union{Nothing, Union{AbstractSet, AbstractVector}} = nothing
)
    grid_set = (explicit === nothing || :grid in explicit) && haskey(config, :grid)
    nx_set = (explicit === nothing || :grid_nx in explicit) && haskey(config, :grid_nx)
    ny_set = (explicit === nothing || :grid_ny in explicit) && haskey(config, :grid_ny)

    if grid_set
        grid_value = config[:grid]
        if isa(grid_value, AbstractVector)
            config[:grid] = (Int(grid_value[1]), Int(grid_value[2]))
        end
        if isa(config[:grid], Tuple)
            config[:grid_nx] = Int(config[:grid][1])
            config[:grid_ny] = Int(config[:grid][2])
        end
    elseif nx_set || ny_set
        nx = Int(get(config, :grid_nx, 100))
        ny = Int(get(config, :grid_ny, 100))
        config[:grid_nx] = nx
        config[:grid_ny] = ny
        config[:grid] = (nx, ny)
    end
    return config
end

"""
    runtime_config_keys() → Set{Symbol}

Configuration keys that are meaningful to a run but are not parameter struct
fields: paths, CLI-only switches, and the `grid` pair that the flat configuration
uses in place of `grid_nx`/`grid_ny`.

`load_ibm_configuration` treats a key that is neither a declared parameter nor one
of these as a mistake, because it is otherwise copied into the configuration and
then silently dropped when the parameter structs are built.
"""
runtime_config_keys() = Set{Symbol}([
    :grid,
    :output_dir,
    :save_file,
    :animation_file,
    :log_file,
    :csv_prefix,
    :save,
    :plot,
    :help,
    :gui,
    :config_file,
    :species_config_file,
    :fishery_on,
])

"""
    cli_switch_keys() → Set{Symbol}

Command-line switch names that are consumed while parsing the argument vector and
never appear as a key in a configuration file.

`--config` is the reason this is separate from [`runtime_config_keys`](@ref): the
pre-scan uses it to locate the configuration files before the main loop runs, and
it is recorded on the config dict as `config_file`, never as `config`. A switch
like that must not be reported as an unrecognized parameter when the generic
branch encounters it, and it must not be accepted as a TOML key either.
"""
cli_switch_keys() = Set{Symbol}([:config])

"""
    known_config_keys() → Set{Symbol}

Every key a configuration file may legitimately declare: all parameter struct
fields plus [`runtime_config_keys`](@ref).
"""
known_config_keys() = union(Set(keys(config_field_types())), runtime_config_keys())

"""
    unknown_config_keys(file_keys; known = known_config_keys()) → Vector{Symbol}

Subset of `file_keys` that names neither a declared parameter nor a runtime key,
sorted for a stable message.
"""
function unknown_config_keys(
    file_keys;
    known::Set{Symbol} = known_config_keys()
)
    return sort!(collect(setdiff(Set(file_keys), known)))
end

"""
    load_ibm_configuration(filepath::String; base_config=nothing)

Read a TOML configuration file and return a flat `Dict{Symbol, Any}` of settings,
optionally overlaying loaded values on top of a `base_config` dictionary.
Alias keys are normalized via `sync_grid_dimensions!`.

Keys that are neither declared parameters nor runtime keys are reported as a
warning. They used to be copied into the configuration and then dropped without a
word when the parameter structs were built, so a typo, an abbreviation, or a key
renamed in an overlay silently did nothing.
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

    # Track which keys this file actually set. Layers are loaded in order
    # (defaults, baseline, species overlay), so a value inherited from an
    # earlier layer must never be allowed to overwrite an explicit value from
    # the file being loaded.
    file_keys = Set{Symbol}()
    for (section, kvs) in raw
        if isa(kvs, AbstractDict)
            for (k, v) in kvs
                key = Symbol(k)
                flat[key] = v
                push!(file_keys, key)
            end
        else
            key = Symbol(section)
            flat[key] = kvs
            push!(file_keys, key)
        end
    end

    # Grid dimensions have two spellings: a `grid` pair, which is the flat-config
    # form consumed by `initialize_model`, and the `grid_nx`/`grid_ny` struct
    # fields. Derive whichever spelling this file did not supply, so a species
    # overlay that sets only `grid_nx`/`grid_ny` still takes effect.
    if :grid in file_keys
        grid_value = flat[:grid]
        if isa(grid_value, AbstractVector)
            flat[:grid] = (Int(grid_value[1]), Int(grid_value[2]))
        end
        if isa(flat[:grid], Tuple)
            flat[:grid_nx] = Int(flat[:grid][1])
            flat[:grid_ny] = Int(flat[:grid][2])
        end
    elseif (:grid_nx in file_keys) || (:grid_ny in file_keys)
        nx = Int(get(flat, :grid_nx, 100))
        ny = Int(get(flat, :grid_ny, 100))
        flat[:grid_nx] = nx
        flat[:grid_ny] = ny
        flat[:grid] = (nx, ny)
    end

    sync_grid_dimensions!(flat; explicit = file_keys)

    # Report keys that will have no effect. Silently dropping them is how a
    # misspelled or renamed key survived in a shipped configuration file: the run
    # completes, the setting simply never applies, and nothing says so.
    unknown = unknown_config_keys(file_keys)
    if !isempty(unknown)
        @warn "configuration file declares keys that are not parameters and will be ignored" file =
            filepath keys = unknown
    end

    return flat
end
load_ibm_configuration(filepath::String, base_config::AbstractDict) =
    load_ibm_configuration(filepath; base_config = base_config)


"""
    ibm_configuration_sections()

Ordered list of TOML section names used by `save_ibm_configuration`.
"""
ibm_configuration_sections() = [section for (_, _, section) in parameter_structs()]

"""
    config_section_map()

Derive the configuration key to TOML section mapping directly from the parameter
struct definitions, so that adding a field to any parameter struct automatically
routes it to the correct section. This removes the hand-maintained duplicate
mapping that previously had to be edited in lockstep with the structs.
"""
function config_section_map()
    section_map = Dict{Symbol, String}()
    for (_, struct_type, section) in parameter_structs()
        for field in fieldnames(struct_type)
            section_map[field] = section
        end
    end
    return section_map
end

"""
    save_ibm_configuration(filepath::String, config::AbstractDict)

Serialize a simulation configuration dictionary to a structured TOML file.
Keys are routed to sections by `config_section_map()`.
"""
function save_ibm_configuration(filepath::String, config::AbstractDict)
    sections = Dict{String, Dict{String, Any}}(
        name => Dict{String, Any}() for name in ibm_configuration_sections()
    )
    section_map = config_section_map()

    # `grid` is a derived pair of `grid_nx`/`grid_ny`; persisting it as well
    # would duplicate those fields in the emitted TOML.
    write_grid = !(haskey(config, :grid_nx) && haskey(config, :grid_ny))

    for (k, v) in config
        sym_k = Symbol(k)
        (sym_k === :grid && !write_grid) && continue
        sec = get(section_map, sym_k, "simulation")
        # TOML has no tuple type, so render tuples as arrays.
        value = isa(v, Tuple) ? [Int(x) for x in v] : v
        sections[sec][string(sym_k)] = value
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
    struct_from_config(::Type{T}, config::AbstractDict) where {T}

Build parameter struct `T` from its own declared defaults, overriding any field
whose name appears in `config` and coercing the value to the field's declared
type.

Constructing the structs field-by-field at each call site duplicated the field
list, so a parameter added to a struct silently kept its default whenever the
duplicated list was not updated in step. Deriving the values from `T`'s own
definition makes that class of drift impossible.
"""
function struct_from_config(::Type{T}, config::AbstractDict) where {T}
    values = Any[]
    for field in fieldnames(T)
        declared = fieldtype(T, field)
        if haskey(config, field)
            push!(values, coerce_to_declared(declared, config[field], field))
        else
            push!(values, getfield(T(), field))
        end
    end
    return T(values...)
end

"""
    coerce_to_declared(declared::Type, value, field::Symbol)

Convert a configuration value to a parameter field's declared type, converting
numeric widths (a TOML float landing on an `Int` field, or vice versa) and
leaving values that already match alone.

Strings are parsed rather than rejected: `Float64("3.5")` is a `MethodError` in
Julia, so a numeric setting that arrived as text (a command-line argument, an
environment variable, a hand-edited overlay that quoted its numbers) could not be
coerced at all, which is the one case this function exists to handle.

Raises with a message naming the field and the offending value if the conversion
is impossible. Use [`try_coerce_to_declared`](@ref) for a non-throwing form.
"""
function coerce_to_declared(declared::Type, value, field::Symbol)
    declared === Any && return value
    try
        if declared <: Bool
            return value isa Bool ? value :
                   !(lowercase(strip(string(value))) in ("false", "0", "no", "off", ""))
        elseif declared <: Integer
            value isa Integer && return value
            f = to_number(value, field)
            return round(Int, f)
        elseif declared <: AbstractFloat
            value isa Real && return Float64(value)
            return to_number(value, field)
        elseif declared <: AbstractString
            return string(value)
        elseif declared <: AbstractVector
            values = value isa AbstractVector ? value : [value]
            parsed = [to_number(v, field) for v in values]
            return declared <: AbstractVector{<:Integer} ?
                   declared.(Int.(round.(Int, parsed))) : declared(Float64.(parsed))
        end
        return declared === typeof(value) ? value : convert(declared, value)
    catch err
        err isa CoercionError && rethrow()
        error(
            "Configuration value for `$(field)` could not be converted to " *
            "$(declared): got $(repr(value)) ($(sprint(showerror, err)))"
        )
    end
end

"""
    CoercionError

Raised internally when a configuration value cannot be converted to the declared
type, carrying the field name so the message can name it.
"""
struct CoercionError <: Exception
    field::Symbol
    declared::Type
    value::Any
end

function Base.showerror(io::IO, e::CoercionError)
    print(io, "configuration value for `", e.field, "` cannot be converted to ",
        e.declared, ": got ", repr(e.value))
    return nothing
end

"""
    to_number(value, field::Symbol) → Float64

Numeric value of `value`, parsing it if it is text. Used by
[`coerce_to_declared`](@ref) so that a quoted number from a configuration source
is accepted rather than rejected with a `MethodError`.
"""
function to_number(value, field::Symbol)
    value isa Real && return Float64(value)
    if value isa AbstractString
        parsed = tryparse(Float64, strip(value))
        parsed === nothing || return parsed
    elseif value isa AbstractChar || value isa Symbol
        parsed = tryparse(Float64, strip(string(value)))
        parsed === nothing || return parsed
    end
    throw(CoercionError(field, Float64, value))
end

"""
    try_coerce_to_declared(declared::Type, value, field::Symbol) → Union{Some, Nothing}

Non-throwing [`coerce_to_declared`](@ref). Returns `Some(converted)` on success
and `nothing` when the value cannot be converted, so a caller that wants to
distinguish "bad setting" from "good setting" need not catch.
"""
function try_coerce_to_declared(declared::Type, value, field::Symbol)
    try
        return Some(coerce_to_declared(declared, value, field))
    catch
        return nothing
    end
end

"""
    initialize_model(config::AbstractDict; kwargs...)

Initialize an individual-based model from a configuration dictionary,
with optional keyword argument overrides.
"""
function initialize_model(config::AbstractDict; kwargs...)
    merged = deepcopy(config)
    # Keyword overrides are the highest-precedence layer, so they must be applied
    # to the grid pair before the grid is read back out of the merged dictionary.
    override_keys = Set(Symbol.(keys(kwargs)))
    for (k, v) in kwargs
        merged[Symbol(k)] = v
    end
    sync_grid_dimensions!(merged; explicit = override_keys)

    n_ag = get(merged, :n_agents, 100)
    grid_dim = get(merged, :grid, (50, 50))
    if isa(grid_dim, AbstractVector)
        grid_dim = (Int(grid_dim[1]), Int(grid_dim[2]))
    end

    dt = Float64(get(merged, :dt, 1.0))
    seed = get(merged, :seed, 1234)

    # Parameter structs are built generically from the merged configuration. Each
    # struct is instantiated from its own declared defaults and any field present in
    # the configuration is overridden, coerced to the field's declared type. Listing
    # fields by hand here meant a new parameter silently fell back to its default
    # whenever this function was not updated alongside the struct.
    toggles = struct_from_config(ProcessToggles, merged)
    env = struct_from_config(EnvironmentParams, merged)
    org_params = struct_from_config(OrganismParams, merged)
    phen = struct_from_config(PhenologyParams, merged)
    fish = struct_from_config(FisheryParams, merged)

    params = IBMParameters(
        toggles = toggles,
        simulation = SimulationParams(
            dt = dt, seed = seed, n_agents = n_ag,
            # Keep the recorded grid consistent with the space that is actually
            # built, so anything reading `params.grid_nx` sees the real extent.
            grid_nx = Int(grid_dim[1]), grid_ny = Int(grid_dim[2])
        ),
        environment = env,
        organism = org_params,
        phenology = phen,
        fishery = fish
    )

    return initialize_model(
        n_agents = n_ag,
        grid = grid_dim,
        dt = dt,
        params = params,
        seed = seed
    )
end

"""
    initialize_model(; n_agents=100, grid=(50, 50), ...)

Construct and populate an `Agents.StandardABM` representing the organism ecosystem.

# Arguments
- `n_agents::Int`: Initial number of individual agents seeded into the environment (default: 100).
- `grid::Tuple{Int, Int}`: Grid spatial dimensions `(nx, ny)` (default: `(50, 50)`).
- `temp_map::Union{Nothing, Matrix{Float64}}`: Optional initial temperature field.
- `food_map::Union{Nothing, Matrix{Float64}}`: Optional initial food density field.
- `dt::Float64`: Time step in days (default: 1.0).
- `params::IBMParameters`: Parameters and process toggles.
- `behavior::Union{Nothing, SpeciesBehavior}`: Optional custom species behavior. If nothing, created from params.species.
- `seed::Union{Nothing, Int}`: Optional RNG seed. A fixed seed makes a run
  repeatable only up to the activation order; see `scheduler`.
- `scheduler`: Agent activation order. Defaults to
  `Agents.Schedulers.Randomly()`, which draws its order from the model's RNG, so
  a fixed seed fixes the activation order by construction while still using every
  thread. The Agents.jl default, `Schedulers.fastest`, takes the agent
  container's own id order, which is not derived from the RNG and is not
  guaranteed stable. That matters here because organisms interact: mortality,
  mate search and cannibalism all read neighbours, so the activation order is
  part of the state transition.

# Returns
Configured and populated `StandardABM` instance ready for simulation.
"""
function initialize_model(;
    n_agents::Int = 100,
    grid::Tuple{Int, Int} = (50, 50),
    temp_map::Union{Nothing, Matrix{Float64}} = nothing,
    food_map::Union{Nothing, Matrix{Float64}} = nothing,
    habitat_map::Union{Nothing, Matrix{Float64}} = nothing,
    dt::Float64 = 1.0,
    params::IBMParameters = IBMParameters(),
    behavior::Union{Nothing, SpeciesBehavior} = nothing,
    seed::Union{Nothing, Int} = nothing,
    scheduler = Agents.Schedulers.Randomly(),
    kwargs...
)
    # Create species behavior from params if not provided
    if behavior === nothing
        behavior = SpeciesBehaviorImpl(params.species)
    end
    
    space = GridSpace(grid)

    rng = if seed === nothing
        Random.default_rng()
    else
        Random.Xoshiro(seed)
    end

    initial_temp = if temp_map === nothing
        fill(params.environment.temp_base, grid)
    else
        copy(temp_map)
    end

    initial_food = if food_map === nothing
        fill(params.environment.food_max / 2.0, grid)
    else
        copy(food_map)
    end

    initial_habitat = if habitat_map === nothing
        μ = params.environment.habitat_quality_mean
        σ = params.environment.habitat_quality_sd
        clamp.(randn(rng, grid) .* σ .+ μ, 0.0, 1.0)
    else
        copy(habitat_map)
    end

    properties = Dict{Symbol, Any}(
        :current_time => 0.0,
        :temperature_field => initial_temp,
        :food_availability_field => initial_food,
        :habitat_quality_field => initial_habitat,
        :dt => dt,
        :params => params,
        :total_eggs_spawned => 0,
        :annual_eggs_spawned => 0,
        :total_recruits_settled => 0,
        :total_catch_numbers => 0,
        :total_catch_biomass_kg => 0.0,
        :total_molts => 0,
        :deaths_predation => 0,
        :deaths_baseline => 0,
        :deaths_soft_shell => 0,
        :deaths_starvation => 0,
        :deaths_thermal => 0,
        :deaths_senescence => 0,
        :deaths_cannibalism => 0,
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
        # Published by `model_step!` and read by `agent_step!` and the accessors, so
        # neither recomputes it per organism. Seeded to the value for time zero.
        :day_of_year => 0.0,
        :step_callback => get(kwargs, :step_callback, nothing)
    )

    model = StandardABM(
        Organism,
        space;
        agent_step! = agent_step!,
        model_step! = model_step!,
        properties = properties,
        rng = rng,
        scheduler = scheduler
    )
    properties[:scheduler] = scheduler
    # `Schedulers.fastest` returns the agent container's own id order, which for a
    # `Dict` container is whatever the hash table happens to hold and is not
    # derived from the RNG. `Randomly` draws its order from the model's RNG, so a
    # fixed seed fixes the activation order by construction. That matters here
    # because organisms interact: mortality, mate search and cannibalism all read
    # neighbours, so the activation order is part of the model's state transition.
    properties[:seeded_activation_order] =
        scheduler !== Agents.Schedulers.fastest
    # Seeded so the accessor below never has to fall back to recomputing on the
    # first step. `model_step!` refreshes it once per step.
    properties[:density_feedback] = benthic_density_feedback(model)

    # --- Initialize Aggregate Populations (stage 0 through juvenile_max_stage - 1) ---
    #
    # The range comes from the configuration, not a literal. It previously ran
    # `0:5`, which happened to match the default `juvenile_max_stage = 6` but
    # would silently mis-seed any other value: individuation happens at stage 6, so
    # a group per stage below that is what feeds the benthic population.
    pop = AggregatePopulations()
    sim_params = params.simulation
    org_params = params.organism
    juvenile_max = org_params.juvenile_max_stage
    if n_agents > 0 && juvenile_max > 0
        for stg in 0:(juvenile_max - 1)
            sz = (stg == 0) ? org_params.egg_size_mm :
                 ((stg < Int(params.species.benthic_anchor_stage)) ?
                  stage_size_lower_bound(behavior, Float64(stg), :male) :
                  stage_size(behavior, stg, :male; type = :random, rng = rng, params = params))
            wt = (stg == 0) ? org_params.egg_mass_g :
                 calculate_mass(behavior, sz, :male; mat = 0, stochastic = true, rng = rng)
            grp = AggregateGroup(
                id = pop.next_group_id,
                stage = stg,
                abundance = (stg == 0) ? sim_params.initial_egg_cohort_abundance :
                           sim_params.initial_stage_cohort_abundance,
                size = sz,
                mass = wt,
                age_days = stg * sim_params.initial_cohort_days_per_stage,
                degree_days = 0.0,
                last_growth_time = 0.0,
                pos = (rand(rng, 1:grid[1]), rand(rng, 1:grid[2]))
            )
            pop.next_group_id += 1
            push!(pop.groups, grp)
        end
    end
    properties[:aggregate_populations] = pop

    # --- Populate Initial Spatial Agents (Individuated: benthic stages onwards) ---
    #
    # Two properties the previous seeding did not have.
    #
    # It drew stages from `6:9` only, so stages 10 upwards were never seeded. Since
    # it also set `mat = (stg >= 10)`, that meant *no individual was ever mature at
    # initialization*, and so no legal male either — the run began with an empty
    # spawning and fishable stock and could not produce a brood for a full year.
    # That is visible in a GUI run as `mean_weight_legal_kg == 0` for the first two
    # annual snapshots.
    #
    # It now covers the whole benthic range for each sex, guarantees at least one
    # individual in every (sex, stage) cell so no stage is empty, and sets maturity
    # from the maturity ogive at the individual's own size and trait quantile — the
    # same test `perform_development!` applies during a run — rather than from a
    # stage threshold.
    benthic_cnt = 0
    mat_m = 0
    mat_f = 0
    leg_m = 0

    # First individuated stage. Recruits are individuated *into* `juvenile_max_stage`
    # by the aggregate-to-agent transition, so the benthic range starts there, not
    # one stage above it — otherwise the seeded population can never contain a
    # stage-6 recruit, which is the main inflow into the benthic stock.
    first_benthic = max(juvenile_max, 1)
    max_stage_of(sex) = (sex === female) ? params.organism.max_stage_female :
                        params.organism.max_stage_male

    # Stage-abundance weighting: geometrically decreasing with stage, so the initial
    # cohort is juvenile-skewed as a real population is, while every benthic stage is
    # still represented. The decay is a calibration choice; that a stage must be
    # reachable at all is not.
    stage_decay = sim_params.initial_stage_decay

    function seed_one(stg::Int, sex, position)
        sz = stage_size(behavior, stg, sex; type = :random, rng = rng, params = params)
        age_days = (stg - sim_params.initial_age_first_stage) *
                   sim_params.initial_age_days_per_stage +
                   rand(rng) * sim_params.initial_age_jitter_days
        energy_reserve = params.organism.initial_energy
        last_molt_time = -age_days
        degree_days = rand(
            rng,
            Uniform(
                sim_params.initial_degree_days_min,
                sim_params.initial_degree_days_max
            )
        )
        max_stage = max_stage_of(sex)

        new_org = add_agent!(
            position, model, stg, sex, age_days, sz, 0.0, energy_reserve,
            last_molt_time, degree_days, false, 0, 0.0, 0, false;
            behavior = behavior
        )
        sample_individual_traits!(new_org, model, rng)

        # Maturity, by the same test a run applies. An individual at its sex's
        # terminal stage is mature by definition, exactly as the stage ceiling in
        # `perform_development!` forces.
        sex_code = (sex === female) ? 1 : 0
        prob_mat = maturity(
            behavior, sex_code, log(new_org.size);
            maturity_quantile = new_org.trait_maturity_quantile
        )
        new_org.is_mature = (stg >= max_stage) || (rand(rng) < prob_mat)
        new_org.is_mating_ready = new_org.is_mature
        new_org.mass = calculate_mass(
            behavior, new_org.size, sex;
            mat = (new_org.is_mature ? 1 : 0), stochastic = true, rng = rng
        )
        return new_org
    end

    if n_agents > 0
        # Every (sex, stage) cell, ordered by sex then stage, as the coverage pool.
        cells = Tuple{Int, Any}[]
        weights = Float64[]
        for sex in (male, female)
            for stg in first_benthic:max_stage_of(sex)
                push!(cells, (stg, sex))
                # Juvenile-skewed, as a real population is, so the remainder lands
                # mostly on the young stages while every cell is still represented.
                push!(weights, stage_decay^(stg - first_benthic))
            end
        end

        # `n_agents` is honoured exactly. When the budget covers every cell, seed
        # them all and distribute the remainder by weight. When it does not, spread
        # the budget across the ordered cells so the stage range is still spanned
        # end to end, rather than filling only the youngest stages.
        if n_agents <= length(cells)
            # `range(1, k, length = 1)` is invalid — a one-element range cannot span
            # differing endpoints — so the single-agent case is handled directly.
            picks = (n_agents == 1) ? [1] :
                    unique(round.(Int, range(1, length(cells), length = n_agents)))
            for i in picks
                stg, sex = cells[i]
                seed_one(stg, sex, (rand(rng, 1:grid[1]), rand(rng, 1:grid[2])))
            end
        else
            for (stg, sex) in cells
                seed_one(stg, sex, (rand(rng, 1:grid[1]), rand(rng, 1:grid[2])))
            end
            for _ in 1:(n_agents - length(cells))
                stg, sex = sample_weighted(cells, weights, rng)
                seed_one(stg, sex, (rand(rng, 1:grid[1]), rand(rng, 1:grid[2])))
            end
        end
    end

    # Cache initial demographic aggregates.
    # Maturity and legal size are counted independently: `is_legal_male` is a
    # size-based regulation test, so a large male that never matured is still
    # legal. Gating the legal count on maturity would make `n_legal_males`
    # disagree with the `count_is_legal_male` column and the GUI's legal series.
    for a in allagents(model)
        benthic_cnt += 1
        if is_mature_male(a)
            mat_m += 1
        elseif is_mature_female(a)
            mat_f += 1
        end
        if is_legal_male(a, params.fishery.legal_size_male,
                         params.organism.adult_size_threshold)
            leg_m += 1
        end
    end

    properties[:n_benthic_organisms] = benthic_cnt
    properties[:n_mature_males] = mat_m
    properties[:n_mature_females] = mat_f

    # The density feedback multiplies every mortality hazard by
    # `1 + (N / K)^exponent`, so a stock that starts above K is not a slow
    # calibration problem: at K = 300 against the default seed of 4800 the
    # multiplier is 257 and every hazard is a certainty on day one. That looks
    # like an extinction the model never modelled, so it is reported here rather
    # than discovered in the results.
    k_cap = params.environment.benthic_carrying_capacity
    seeded_stock = total_benthic_abundance(model)
    if k_cap > 0.0 && seeded_stock > k_cap
        multiplier = 1.0 +
                     (seeded_stock / k_cap)^params.environment.benthic_carrying_capacity_exponent
        @warn "seeded benthic stock already exceeds the carrying capacity" stock =
            round(seeded_stock; digits = 1) capacity = k_cap ratio = round(
            seeded_stock / k_cap;
            digits = 2,
        ) mortality_multiplier = round(multiplier; digits = 1) initial_egg_cohort_abundance =
            sim_params.initial_egg_cohort_abundance initial_stage_cohort_abundance =
            sim_params.initial_stage_cohort_abundance n_agents = n_agents
    end
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

    # Energy-depleted lethargy: the individual may stay put when energetics are active.
    if model.params.toggles.use_energetics && org.energy_reserve <= 0.0 &&
       rand(rng) < model.params.organism.energy_deficit_removal_probability
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

    # Benthic habitat taxis: evaluate local environmental suitability. The thermal
    # preference governs direction of travel and is distinct from the metabolic
    # optimum in [organism], which sets the energetic cost curve instead.
    env = model.params.environment
    t_opt = env.habitat_temperature_optimum
    t_sigma = env.habitat_temperature_sigma
    food_weight = env.habitat_food_weight
    half_sat = env.half_saturation_food

    utilities = Float64[]
    for pos in candidates
        cx, cy = pos
        ctemp = model.temperature_field[cx, cy]
        cfood = model.food_availability_field[cx, cy]

        # Gaussian thermal suitability
        s_temp = exp(-((ctemp - t_opt)^2) / (2.0 * t_sigma^2))

        # Hyperbolic food suitability
        s_food = cfood / (cfood + half_sat)

        # Substrate/Refuge quality
        chabitat = model.habitat_quality_field[cx, cy]

        u = chabitat * s_temp * (1.0 + food_weight * s_food)
        push!(utilities, u)
    end

    # Softmax over suitability. A larger temperature spreads the heading
    # distribution over more directions; a smaller one concentrates it.
    tau = env.movement_softmax_temperature
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
    # `model.params` is a symbol lookup into the model's properties dictionary on
    # every access, and this function reads it a dozen times per organism per day.
    # At a few thousand organisms over a two-year run that is tens of millions of
    # dictionary lookups. Hoist it once.
    params = model.params
    # The day of year is the same for every organism on a given day. `model_step!`
    # has already computed it for the seasonal cycle, so read that rather than
    # recomputing the modulo per organism.
    day_of_year = current_day_of_year(model)
    dt = model.dt

    # Constants for this step, resolved once. Each of these is a nested parameter
    # access, and none of them varies between organisms.
    toggles = params.toggles
    use_energetics = toggles.use_energetics
    use_degree_days = toggles.use_degree_days
    use_cannibalism = toggles.use_cannibalism
    use_fishery = toggles.use_fishery
    # Whether the fishery is open, and the maturity threshold for a legal animal,
    # are properties of the day, not of the organism.
    in_fishing_season = use_fishery &&
                        params.fishery.fishing_probability_daily > 0.0 &&
                        params.fishery.fishing_season_start <= day_of_year <=
                        params.fishery.fishing_season_end
    # The organism does not move until step 6 and the environment does not change
    # within a step, so its own cell's temperature is fixed for steps 1 to 5. It was
    # read four times: energetics, degree-day accrual, the molt decision inside
    # `perform_development!`, and mortality.
    temp = model.temperature_field[org.pos...]

    # 1. Energetics (Holling Type II & Kleiber consumption)
    if use_energetics
        food = model.food_availability_field[org.pos...]
        dE_dt, intake = calculate_energy_change(temp, food, org.size; params = params, mass = org.mass)
        org.energy_reserve += dE_dt * dt
        model.food_availability_field[org.pos...] = max(0.0, food - intake * dt)
    end

    # 2. Degree-day accumulation
    if use_degree_days
        org.degree_days += max(0.0, temp - params.environment.temp_base_degreedays) * dt
    end

    # 3. Development & Molting
    perform_development!(org.behavior, org, model; temp = temp)

    # 4. Mortality
    # `temp` is still the organism's own cell: nothing before this step moves it.
    # The neighbourhood query that feeds the cannibalism hazard is only worth its
    # cost when cannibalism is switched on. It was run for every organism on every
    # day regardless, and it is a grid lookup over the surrounding cells.
    local_density = use_cannibalism ?
                    Float64(count(_ -> true, nearby_agents(org, model, 1))) : 0.0
    causes = mortality_components(org, temp, local_density, model.current_time;
        params = params,
        density_feedback = density_feedback(model))

    if rand(abmrng(model)) < causes.total
        # Attribute the death to the hazard that produced it. The previous rule —
        # `is_mature → senescence, else → predation` — had no relationship to the
        # summed probability, so juveniles dying of background mortality were
        # counted as predation deaths and mature individuals dying of anything were
        # counted as senescence deaths. `deaths_starvation` and `deaths_thermal`
        # were never incremented at all.
        cause = dominant_mortality_cause(causes)
        cause === :baseline && (model.deaths_baseline += 1)
        cause === :predation && (model.deaths_predation += 1)
        cause === :soft_shell && (model.deaths_soft_shell += 1)
        cause === :starvation && (model.deaths_starvation += 1)
        cause === :thermal && (model.deaths_thermal += 1)
        cause === :senescence && (model.deaths_senescence += 1)
        cause === :cannibalism && (model.deaths_cannibalism += 1)
        remove_agent!(org, model)
        return nothing
    end

    # 5. Fishery Harvest
    if in_fishing_season
        legal_size = params.fishery.legal_size_male
        if is_legal_male(org, legal_size, params.organism.adult_size_threshold)
            # Legality is checked before the draw, not after. It used to consume one
            # random number per organism in season — selectivity already zeroes the
            # probability for a sublegal animal, so those draws were spent on
            # nothing and only shifted the stream for everything after them.
            fishing_probability_daily = params.fishery.fishing_probability_daily *
                                       selectivity(
                                           org.size;
                                           s50 = legal_size,
                                           s50_sigma =
                                               org.behavior.params.selectivity_s50_sigma
                                       )
            if rand(abmrng(model)) < fishing_probability_daily
                model.total_catch_numbers += 1
                model.total_catch_biomass_kg += (org.mass / 1000.0)
                model.deaths_fishery += 1
                remove_agent!(org, model)
                return nothing
            end
        end
    end

    # 6. Locomotion / Spatial Dispersal
    organism_disperse!(org, model)

    # 7. Reproduction
    perform_reproduction!(org.behavior, org, model)

    return nothing
end

function update_demographic_caches!(model)
    # Both thresholds are read once per organism. The compiler cannot hoist them
    # out of the loop because the loop body is opaque, so leaving them in place
    # meant resolving two nested parameter accesses per agent per day.
    legal_size = model.params.fishery.legal_size_male
    adult_size = model.params.organism.adult_size_threshold
    benthic_count = 0
    mat_males = 0
    mat_females = 0
    legal_males = 0
    for org in allagents(model)
        benthic_count += 1
        if is_mature_male(org)
            mat_males += 1
        elseif is_mature_female(org)
            mat_females += 1
        end
        if is_legal_male(org, legal_size, adult_size)
            legal_males += 1
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

Advance environmental state, advance aggregate cohorts, and update demographic caches
for one simulation time step.
"""
function model_step!(model)
    dt = model.dt
    sim = model.params.simulation
    old_time = model.current_time
    model.current_time += dt

    # Year boundary detection for annual resets
    if floor(old_time / sim.days_per_year) < floor(model.current_time / sim.days_per_year)
        model.annual_eggs_spawned = 0.0
    end

    # 1. Environmental updates (Temperature & Food)
    #
    # These touch every cell, so at a 1000 x 1000 grid they are 1e6 elements per
    # field per day. Both updates are written to avoid allocating: a
    # `rand(rng, Normal(0, sd), dims)` here would allocate a second full-size
    # matrix every step, which is 8 MB per day and 5.8 GB over a two-year run, and
    # would also make every other allocation in the step wait on the collector.
    day_of_year = mod(model.current_time, sim.days_per_year)
    sinusoidal_factor = sin(2.0 * π * (day_of_year - sim.temp_seasonal_phase_day) /
                           sim.days_per_year)

    env = model.params.environment
    t_base = env.temp_base
    t_amp = env.temp_amplitude
    t_sd = env.temp_noise_sd

    # The seasonal target is spatially uniform; the field's only structure is the
    # white noise. Drawing the standard normals straight into the field and scaling
    # in place gives the same values from the same RNG stream with one pass and no
    # allocation.
    seasonal_target = t_base + t_amp * sinusoidal_factor
    temperature = model.temperature_field
    if t_sd > 0.0
        randn!(abmrng(model), temperature)
        @. temperature = seasonal_target + t_sd * temperature
    else
        fill!(temperature, seasonal_target)
    end

    # Published for the agent phase, which needs the same day of year for every
    # organism and used to recompute the modulo once per agent per day.
    model.day_of_year = day_of_year

    f_max = env.food_max
    f_rate = env.food_replenishment_rate
    # `F += r * (f_max - F) * dt` written as the affine map it is: one multiply
    # and one add per cell rather than a subtract, two multiplies and an add.
    food_decay = 1.0 - f_rate * dt
    food_gain = f_rate * f_max * dt
    @. model.food_availability_field = food_decay * model.food_availability_field +
                                       food_gain

    # 2. Advance pre-individuation aggregate cohort groups (stages 0 through 5)
    step_aggregate_groups!(model)

    # 3. Refresh the density multiplier once for this step.
    #
    # It depends only on the population, so evaluating it per organism recomputed
    # the same value once per agent per day, walking the aggregate cohorts each
    # time. It is refreshed once per step instead, and *after* the cohorts have
    # advanced: individuation adds recruits to the population the agents are about
    # to face, so reading it before would apply last step's density to this step's
    # animals.
    model.density_feedback = benthic_density_feedback(model)

    # 4. Update demographic aggregate caches
    update_demographic_caches!(model)

    # 5. Optional step callback hook
    if haskey(abmproperties(model), :step_callback) && model.step_callback !== nothing
        model.step_callback(model)
    end

    return nothing
end

"""
    safe_mean(x)

Mean of `x`, returning `0.0` when it is empty.

`Agents` hands aggregators a one-shot generator, not a collection, so
`mean(x)` throws on an empty one and probing with `isempty(x)` first consumes an
element. The obvious fix is `collect(x)` then `mean`, which is correct but
allocates a `Vector` per column per step: with about ten mean columns collected
every day, that is ten population-sized allocations per simulated day, and it is
why observation collection costs so much more than the dynamics themselves.

This accumulates in one pass with no allocation. The population counts change
between steps and can reach zero, so it has to handle an empty input anyway.
"""
function safe_mean(x)
    total = 0.0
    n = 0
    for v in x
        total += v
        n += 1
    end
    return n == 0 ? 0.0 : total / n
end

"""
    safe_sum(x)

Sum of `x`, returning `0.0` when it is empty.

`Agents._add_col_data!` calls `agg(generator)` with no `init`, so a bare `sum`
throws `reducing over an empty collection is not allowed` the moment the
population reaches zero. That is precisely the run this model produces: the
population declines, and rather than recording the final empty state the run
crashed inside the data-collection step. `length` and `count` are already safe on
an empty input; `sum` and `mean` are the two that are not.
"""
safe_sum(x) = sum(x; init = 0.0)

"""
    total_sum

Total function with `sum`'s name, so a column aggregated with it is still called
`sum_*`. See [`NamedAggregate`](@ref) for why the name matters.
"""
const total_sum = NamedAggregate(:sum, safe_sum)

"""
    default_agent_observations(params::IBMParameters)

Standard per-agent observations recorded by `run_simulation`.

The biological predicates take their thresholds from `params` — legal size and
the adult size boundary are species settings — so the observation list is built
from the model's own parameters rather than from hardcoded defaults.
"""
function default_agent_observations(params::IBMParameters)
    adult_size = params.organism.adult_size_threshold
    legal_size = params.fishery.legal_size_male
    return [
        (:stage, length),
        (:stage, safe_mean),
        (:size, safe_mean),
        (:mass, safe_mean),
        (:mass, total_sum),
        (named_predicate(is_adult_male, :is_adult_male), count),
        (is_mature_male, count),
        (is_mature_female, count),
        (named_predicate(is_legal_male, :is_legal_male, legal_size, adult_size), count),
        (is_ovigerous, count),
        (mass_if_mature_female, total_sum),
        (mass_if_mature_male, total_sum),
        (
            named_predicate(
                mass_if_legal_male,
                :mass_if_legal_male,
                legal_size,
                adult_size
            ),
            total_sum
        )
    ]
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

    default_adata = default_agent_observations(model.params)

    default_mdata = [
        :total_eggs_spawned,
        :total_recruits_settled,
        :total_catch_numbers,
        :total_catch_biomass_kg,
        :total_molts,
        :deaths_baseline,
        :deaths_predation,
        :deaths_soft_shell,
        :deaths_starvation,
        :deaths_thermal,
        :deaths_senescence,
        :deaths_cannibalism,
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
        return run!(model, n_steps; adata = actual_adata, mdata = actual_mdata,
                    when = when)
    end
end
