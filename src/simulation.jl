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

    # Default n_agents is the sum of the initial sex cohort vectors.
    config[:n_agents] = round(
        Int,
        sum(config[:initial_female_cohort]) + sum(config[:initial_male_cohort])
    )

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
    :n_agents,
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
    :gui_simple,
    :gui_compare,
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

    # If n_agents was not explicitly set in the loaded file, update or initialize it
    # from the resolved initial cohort vectors.
    if !(:n_agents in file_keys) && haskey(flat, :initial_female_cohort) &&
       haskey(flat, :initial_male_cohort)
        flat[:n_agents] = round(
            Int,
            sum(flat[:initial_female_cohort]) + sum(flat[:initial_male_cohort])
        )
    end

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
            # A vector parameter may be declared with a non-numeric element type.
            # `stage_labels` is a list of plot names, and coercing its entries with
            # `to_number` failed on the first one, so every configuration load that
            # carried a name list raised rather than reading it.
            if eltype(declared) <: AbstractString
                # Broadcast over the *element* type. `declared.(x)` would call
                # `Vector{String}(x)`, which is not a constructor.
                return eltype(declared).(string.(values))
            end
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
    fishery_open(params::IBMParameters, day_of_year) -> Bool

Whether the fishery is open to harvest on a given day of the year.

The window is `fishery.fishing_season_start` to `fishing_season_end`, given as
**day of year** (1 to `days_per_year`), and it may wrap the year end: a start
greater than the end is a season running from the start through 31 December and
into January. A plain `start <= day <= end` test reduces that to *never open*, and
the failure is invisible, because an always-closed season looks exactly like the
fishery having been switched off.

The model carries day-of-year 0-based internally (`mod(time, days_per_year)`), so
the shift is made once here rather than at every call site.
"""
function fishery_open(params::IBMParameters, day_of_year::Real)
    fishery_active(params) || return false
    f = params.fishery
    dpy = params.simulation.days_per_year
    lo, hi = f.fishing_season_start, f.fishing_season_end
    d = mod(day_of_year, dpy) + 1.0
    for (name, v) in (("fishing_season_start", lo), ("fishing_season_end", hi))
        if !(0.0 <= v <= dpy)
            @warn "Fishing season bound lies outside the calendar; the fishery " *
                  "will never open." parameter = name value = v days_per_year = dpy
            return false
        end
    end
    lo <= hi && return lo <= d <= hi
    return d >= lo || d <= hi
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
    mort = struct_from_config(MortalityParams, merged)
    spec = struct_from_config(SpeciesParams, merged)
    leslie_params = struct_from_config(LeslieParams, merged)
    crit = struct_from_config(CriteriaParams, merged)
    outp = struct_from_config(OutputParams, merged)

    params = IBMParameters(
        toggles = toggles,
        # Built from the configuration, not field by field. It used to be assembled
        # from `dt`, `seed` and the grid alone, so everything else the file declares
        # — including the three initial-abundance vectors — was silently dropped on
        # this path, and a run configured through a file seeded nothing at all.
        # `dt`, `seed` and the grid are already in `merged`: `sync_grid_dimensions!`
        # wrote the pair from `:grid`, so they are read here rather than restated.
        simulation = struct_from_config(SimulationParams, merged),
        environment = env,
        organism = org_params,
        phenology = phen,
        # These four were never built from `merged` and so were silently dropped on
        # the configuration path: `[mortality]`, `[species]`, `[leslie]` and
        # `[criteria]` had no effect on a run started from a file, and every value in
        # them fell back to the struct default. `[mortality]` was the expensive one —
        # it made every "which mortality term is doing this" experiment return the
        # same answer, because the rates under test were never the ones being used.
        mortality = mort,
        fishery = fish,
        output = outp,
        species = spec,
        leslie = leslie_params,
        criteria = crit
    )

    return initialize_model(
        grid = grid_dim,
        dt = dt,
        params = params,
        seed = seed,
        # `n_agents` is the total the seeding vectors sum to, not a parameter of
        # its own, so it is a keyword here rather than a key in the file. It is
        # forwarded explicitly: merging it into `merged` alone left it unread, and
        # every configuration-path caller asking for a population got none.
        n_agents = get(merged, :n_agents, nothing)
    )
end

"""
    build_habitat_quality_field(params, grid, rng)

Generate the initial habitat suitability index (HSI) field on `grid`.

The field is a clipped Gaussian noise floor about `habitat_quality_mean` carrying
`habitat_peaks` additive Gaussian bumps at uniformly random positions. The noise
alone has no spatial correlation, so it describes a landscape with nowhere better
than anywhere else; the peaks supply coherent high-quality habitat for the taxis
in `organism_disperse!` to aggregate in and disperse between.

Peak widths are `habitat_peak_width_fraction` of the domain rather than a cell
count, so the structure looks the same on a 10x10 test grid and a 1000x1000
operating grid. Peak centres come from `rng`, so the layout is reproducible under
a fixed seed.

`habitat_peaks = 0` reproduces the previous pure-noise field exactly.
"""
function build_habitat_quality_field(params::EnvironmentParams,
    grid::Tuple{Int, Int}, rng::AbstractRNG)
    μ = params.habitat_quality_mean
    σ = params.habitat_quality_sd
    field = clamp.(randn(rng, grid) .* σ .+ μ, 0.0, 1.0)

    n_peaks = params.habitat_peaks
    n_peaks > 0 || return field
    width_frac = params.habitat_peak_width_fraction
    width_frac > 0.0 || error(
        "habitat_peak_width_fraction must be positive, got $(width_frac)")
    height = params.habitat_peak_height

    nx, ny = grid
    # Fractional width to a per-axis Gaussian sigma, in cell units. `max(..., eps)`
    # keeps a degenerate width from dividing by zero and blanking the field.
    sigma_x = max(width_frac * nx, 1e-6)
    sigma_y = max(width_frac * ny, 1e-6)

    # Coordinate vectors, so each peak is a rank-1 outer product instead of a
    # fresh allocation over the whole grid.
    xs = collect(1:nx)
    ys = collect(1:ny)

    for _ in 1:n_peaks
        cx = rand(rng, 1:nx)
        cy = rand(rng, 1:ny)
        gx = exp.(-0.5 .* ((xs .- cx) ./ sigma_x) .^ 2)
        gy = exp.(-0.5 .* ((ys .- cy) ./ sigma_y) .^ 2)
        field .+= Float64(height) .* (gx * transpose(gy))
    end

    return clamp.(field, 0.0, 1.0)
end

"""
    build_geodata_environmental_fields(
        env_params::EnvironmentParams,
        grid::Tuple{Int, Int};
        depth_map::Union{Nothing, Matrix{Float64}} = nothing,
        verbose::Bool = false
    )

Construct bathymetry, bottom temperature, and habitat suitability index (HSI)
fields using real physical datasets accessed through `GeoData`.

# Mathematical and Analytical Workflow
1. Spatial coordinate mesh:
   Discretizes the spatial domain `(domain_lon, domain_lat)` into regular `nx` by `ny`
   geographic coordinates.
2. Seabed Bathymetry:
   If `depth_map` is not provided, queries ERDDAP/ETOPO via
   `GeoDataSources.fetch_erddap_bathymetry` and builds continuous interpolator
   `GeoData.get_bathymetry_interpolator`. Depth is extracted as `max(0.0, -elevation)`.
3. Seabed Bottom Temperature:
   Queries climatological hydrography via `GeoDataSources.fetch_boundary_hydrography_geodata`
   (e.g., WOA23 or GLORYS12v1) and samples temperature at seabed depth `z`.
4. Habitat Suitability Index (HSI):
   Evaluates `calculate_habitat_suitability(temperature, depth)` across the grid.

# Arguments
- `env_params`: `EnvironmentParams` specifying domain bounds and data providers.
- `grid`: Spatial grid dimensions `(nx, ny)`.
- `depth_map`: Optional pre-computed depth matrix.
- `verbose`: Diagnostic reporting toggle.

# Returns
NamedTuple `(depth = depth_field, temperature = temp_field, hsi = hsi_field)`.
"""
function build_geodata_environmental_fields(
    env_params::EnvironmentParams,
    grid::Tuple{Int, Int};
    depth_map::Union{Nothing, Matrix{Float64}} = nothing,
    verbose::Bool = false
)
    nx, ny = grid
    lon_bounds = env_params.domain_lon
    lat_bounds = env_params.domain_lat
    length(lon_bounds) >= 2 || error("domain_lon requires [min_lon, max_lon]")
    length(lat_bounds) >= 2 || error("domain_lat requires [min_lat, max_lat]")

    lon_range = (Float64(lon_bounds[1]), Float64(lon_bounds[2]))
    lat_range = (Float64(lat_bounds[1]), Float64(lat_bounds[2]))

    lons = range(lon_range[1], lon_range[2], length = nx)
    lats = range(lat_range[1], lat_range[2], length = ny)

    # 1. Bathymetric depth field
    depth_field = if depth_map !== nothing
        size(depth_map) == grid || error(
            "depth_map dimensions $(size(depth_map)) do not match grid $(grid)")
        abs.(Float64.(copy(depth_map)))
    else
        bathy_ds = GeoDataSources.fetch_erddap_bathymetry(
            lon_range = lon_range,
            lat_range = lat_range,
            dataset_id = env_params.bathymetry_source == "etopo2022" ?
                "ETOPO_2022_v1_15s" : env_params.bathymetry_source,
            # The store, not a run-local cache. `geodata_input_dir` stays honoured for a
            # configuration written before the store existed; when `geodata_store_dir` is set
            # it wins, and assimilation and ingestion then agree by construction.
            output_path = joinpath(GeoIBM.environmental_store_dir(env_params),
                                   "bathymetry.zarr"),
            verbose = verbose
        )
        bathy_itp = GeoData.get_bathymetry_interpolator(bathy_ds)
        d_mat = zeros(Float64, nx, ny)
        for j in 1:ny, i in 1:nx
            d_mat[i, j] = max(0.0, -bathy_itp(lons[i], lats[j]))
        end
        d_mat
    end

    # 2. Bottom and upper water column (pelagic) temperature from hydrography
    temp_src = Symbol(env_params.temperature_source)
    pelagic_z = -abs(Float64(env_params.pelagic_depth))

    # Fetch initial/specified month hydrography
    hydro = GeoDataSources.fetch_boundary_hydrography_geodata(
        temp_src;
        lon_range = lon_range,
        lat_range = lat_range,
        input_dir = GeoIBM.environmental_store_dir(env_params),
        month = env_params.temperature_month,
        verbose = verbose
    )

    temp_field = zeros(Float64, nx, ny)
    surface_temp_field = zeros(Float64, nx, ny)
    hsi_field = zeros(Float64, nx, ny)
    for j in 1:ny, i in 1:nx
        z = depth_field[i, j]
        t_bottom = hydro.T(lons[i], lats[j], -z, 0.0)
        t_surf = hydro.T(lons[i], lats[j], pelagic_z, 0.0)
        temp_field[i, j] = t_bottom
        surface_temp_field[i, j] = t_surf
        hsi_field[i, j] = calculate_habitat_suitability(t_bottom, z)
    end

    # Build 12-month seasonal hydrography sequence if dynamic climatology requested
    monthly_bottom = Vector{Matrix{Float64}}(undef, 12)
    monthly_surface = Vector{Matrix{Float64}}(undef, 12)
    for m in 1:12
        m_hydro = GeoDataSources.fetch_boundary_hydrography_geodata(
            temp_src;
            lon_range = lon_range,
            lat_range = lat_range,
            input_dir = GeoIBM.environmental_store_dir(env_params),
            month = m,
            verbose = false
        )
        b_mat = zeros(Float64, nx, ny)
        s_mat = zeros(Float64, nx, ny)
        for j in 1:ny, i in 1:nx
            z = depth_field[i, j]
            b_mat[i, j] = m_hydro.T(lons[i], lats[j], -z, 0.0)
            s_mat[i, j] = m_hydro.T(lons[i], lats[j], pelagic_z, 0.0)
        end
        monthly_bottom[m] = b_mat
        monthly_surface[m] = s_mat
    end

    return (
        depth = depth_field,
        temperature = temp_field,
        surface_temperature = surface_temp_field,
        hsi = hsi_field,
        monthly_bottom = monthly_bottom,
        monthly_surface = monthly_surface
    )
end

"""
    initialize_model(; grid=(50, 50), ...)

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
    n_agents::Union{Int, Nothing} = nothing,
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

    initial_depth = if haskey(kwargs, :depth_map) && kwargs[:depth_map] !== nothing
        copy(kwargs[:depth_map])
    else
        nothing
    end

    initial_temp = if temp_map !== nothing
        copy(temp_map)
    else
        nothing
    end

    initial_habitat = if habitat_map !== nothing
        copy(habitat_map)
    else
        nothing
    end

    if params.environment.use_geodata
        geo_env = build_geodata_environmental_fields(
            params.environment,
            grid;
            depth_map = initial_depth
        )
        if initial_depth === nothing
            initial_depth = geo_env.depth
        end
        if initial_temp === nothing
            initial_temp = geo_env.temperature
        end
        if initial_habitat === nothing
            initial_habitat = geo_env.hsi
        end
    end

    if initial_depth === nothing
        initial_depth = fill(150.0, grid)
    end
    if initial_temp === nothing
        initial_temp = fill(params.environment.temp_base, grid)
    end
    if initial_habitat === nothing
        initial_habitat = build_habitat_quality_field(params.environment, grid, rng)
    end

    initial_food = if food_map === nothing
        fill(params.environment.food_max / 2.0, grid)
    else
        copy(food_map)
    end

    initial_surface_temp = if params.environment.use_geodata && @isdefined(geo_env)
        geo_env.surface_temperature
    else
        copy(initial_temp)
    end

    properties = Dict{Symbol, Any}(
        :current_time => 0.0,
        :temperature_field => initial_temp,
        :surface_temperature_field => initial_surface_temp,
        :depth_field => initial_depth,
        :baseline_temperature_field => copy(initial_temp),
        :baseline_surface_temperature_field => copy(initial_surface_temp),
        :food_availability_field => initial_food,
        :habitat_quality_field => initial_habitat,
        :monthly_bottom => (params.environment.use_geodata && @isdefined(geo_env)) ?
            geo_env.monthly_bottom : nothing,
        :monthly_surface => (params.environment.use_geodata && @isdefined(geo_env)) ?
            geo_env.monthly_surface : nothing,
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

    # Valid marine habitat cells for benthic organisms (strictly non-land: depth >= 20.0 and hsi > 0)
    depth_min_bound = 20.0
    marine_cells = Tuple{Int, Int}[]
    marine_weights = Float64[]
    for j in 1:grid[2], i in 1:grid[1]
        z = initial_depth[i, j]
        h = initial_habitat[i, j]
        if z >= depth_min_bound && h > 0.0
            push!(marine_cells, (i, j))
            push!(marine_weights, h)
        end
    end
    if isempty(marine_cells)
        for j in 1:grid[2], i in 1:grid[1]
            if initial_depth[i, j] >= depth_min_bound
                push!(marine_cells, (i, j))
                push!(marine_weights, 1.0)
            end
        end
    end
    if isempty(marine_cells)
        marine_cells = [(i, j) for i in 1:grid[1], j in 1:grid[2]]
        marine_weights = fill(1.0, length(marine_cells))
    end
    sample_marine_pos() = sample_weighted(marine_cells, marine_weights, rng)

    # --- Initialize Aggregate Populations (stage 0 through juvenile_max_stage - 1) ---
    pop = AggregatePopulations()
    sim_params = params.simulation
    org_params = params.organism
    juvenile_max = org_params.juvenile_max_stage
    if juvenile_max > FIRST_COHORT_STAGE
        cohort_abundance = sim_params.initial_aggregate_cohort
        for stg in FIRST_COHORT_STAGE:(juvenile_max - 1)
            abundance = stg + 1 <= length(cohort_abundance) ? cohort_abundance[stg+1] : 0.0
            abundance > 0.0 || continue
            is_first = stg == FIRST_COHORT_STAGE
            sz = (is_first) ? org_params.egg_size_mm :
                 ((stg < Int(params.species.benthic_anchor_stage)) ?
                  stage_size_lower_bound(behavior, Float64(stg), :male) :
                  stage_size(behavior, stg, :male; type = :random, rng = rng, params = params))
            wt = (is_first) ? org_params.egg_mass_g :
                 calculate_mass(behavior, sz, :male; mat = 0, stochastic = true, rng = rng)
            grp = AggregateGroup(
                id = pop.next_group_id,
                stage = stg,
                abundance = abundance,
                size = sz,
                mass = wt,
                age_days = stg * sim_params.initial_cohort_days_per_stage,
                degree_days = 0.0,
                last_growth_time = 0.0,
                pos = sample_marine_pos()
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

    # Stage-abundance distribution, read from `initial_female_cohort` and
    # `initial_male_cohort`, element `i` being stage `i - 1`.
    #
    # This replaces a geometric stage decay that existed only to spread a *total*
    # over the stage range. The total is gone too: the vectors state the numbers,
    # so there is no separate count to keep consistent with them.
    function abundance_at(vec, stg::Int)
        return (stg + 1 <= length(vec)) ? vec[stg+1] : 0.0
    end

    # What the configuration asks for, and what the caller overrides it with.
    # `n_agents` is the sum of the two sex vectors unless given.
    seed_sum = sum(sim_params.initial_female_cohort) + sum(sim_params.initial_male_cohort)

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

    # Seed individuals from the two sex vectors.
    #
    # `n_agents` is the sum of the vectors when the caller does not state it, and
    # an override when they do. Two cases, because they are different requests:
    #
    #  * The vectors state a distribution. That is the point of them, so their
    #    *shape* is honoured; an override only rescales it, so the caller can ask
    #    for a different total without restating the shape.
    #  * The vectors are empty and `n_agents` is given. There is no distribution to
    #    honour, so the count is spread evenly across the stage range of each sex.
    #    Uniform rather than juvenile-skewed: a skew is a shape choice and shapes
    #    now live in the vectors, so inventing one here would reintroduce exactly
    #    the hidden constant the vectors were added to remove.
    seed_target = something(n_agents, seed_sum)
    if seed_target > 0
        # The (stage, sex) cells, so the two cases below share one enumeration.
        cells = [(stg, sex) for sex in (female, male)
                 for stg in first_benthic:max_stage_of(sex)]
        if seed_sum > 0
            # The vectors state a distribution; honour the shape and let `n_agents`
            # only rescale it.
            for (k, (stg, sex)) in enumerate(cells)
                vec = (sex === female) ? sim_params.initial_female_cohort :
                      sim_params.initial_male_cohort
                n_here = round(Int, abundance_at(vec, stg) * (seed_target / seed_sum))
                for _ in 1:n_here
                    seed_one(stg, sex, sample_marine_pos())
                end
            end
        else
            # No distribution stated: cycle the total evenly over the cells.
            for k in 1:seed_target
                stg, sex = cells[mod1(k, length(cells))]
                seed_one(stg, sex, sample_marine_pos())
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
        ) mortality_multiplier = round(multiplier; digits = 1) aggregate_cohorts =
            round(sum(sim_params.initial_aggregate_cohort); digits = 1) seeded_individuals =
            round(seed_sum; digits = 1)
    end
    properties[:n_legal_males] = leg_m

    return model
end

"""
    sample_individual_traits!(org::Organism, model, rng)

Sample intrinsic empirical life-history traits for an individual `Organism`.

Each trait is an individual's position on its own distribution — a quantile, not a
value — and the four quantiles share one range, `OrganismParams.trait_quantile_min`
and `trait_quantile_max`. They were four separate `Uniform(0.01, 0.99)` draws, so
the range was written down four times and changing it meant finding all four.
"""
function sample_individual_traits!(org::Organism, model, rng)
    op = model.params.organism
    q = Uniform(op.trait_quantile_min, op.trait_quantile_max)
    org.trait_growth_quantile = rand(rng, q)
    org.trait_maturity_quantile = rand(rng, q)
    org.trait_fecundity_quantile = rand(rng, q)
    org.trait_mortality_quantile = rand(rng, q)
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

        # Substrate/Refuge quality and depth check (avoiding land)
        chabitat = model.habitat_quality_field[cx, cy]
        cdepth = model.depth_field[cx, cy]

        if cdepth < 20.0 || chabitat <= 0.0 || ctemp >= 10.0
            u = 0.0
        else
            u = chabitat * s_temp * (1.0 + food_weight * s_food)
        end
        push!(utilities, u)
    end

    # Softmax over suitability. Zero utility (land) gets zero probability weight.
    tau = env.movement_softmax_temperature
    max_u = maximum(utilities)
    weights = [u > 0.0 ? exp((u - max_u) / tau) : 0.0 for u in utilities]

    if sum(weights) > 0.0
        chosen_pos = sample_weighted(candidates, weights, rng)
        if chosen_pos != org.pos
            move_agent!(org, chosen_pos, model)
        end
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
    # Whether the fishery is open, and the maturity threshold for a legal animal,
    # are properties of the day, not of the organism. Whether there is a fishery at
    # all is `fishery.fishing_probability_daily`: a probability of zero means no
    # harvest, so that single number is the switch.
    in_fishing_season = fishery_open(params, day_of_year)
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

    # 3b. Maturity, evaluated every step rather than only at ecdysis.
    #
    # It used to be drawn inside `perform_molt!`, which ecdysis only reaches inside
    # the molt-season window, so an animal that failed its draw in one spring waited
    # a full year: maturation was an annual step function and legal-male biomass
    # pulsed several-fold year to year.
    #
    # The ogive probability is a probability of being mature *at this size*, so on a
    # daily trial it has to be a daily **hazard**. Divided by `days_per_year` the
    # cumulative probability over a year comes back to the ogive value — the same
    # conversion [mortality] applies to its rates, and the reason it is resolution
    # independent. Drawing the raw probability daily instead compounds it: 0.027 a
    # day is effectively certain within a year, which matured every male above the
    # 35 mm size gate and emptied the fishery.
    if !org.is_mature && params.toggles.use_molting
        prob_mat = maturity(
            org.behavior,
            (org.sex === female) ? 1 : 0,
            log(org.size);
            temp = temp,
            maturity_quantile = org.trait_maturity_quantile,
            rng = abmrng(model)
        )
        if prob_mat >= 0.9999
            org.is_mature = true
            org.is_mating_ready = true
        elseif prob_mat > 0.0
            # Continuous daily hazard rate lambda such that 1 - exp(-lambda * days) = prob_mat:
            # lambda = -log(1 - prob_mat) / days_per_year
            daily_hazard = -log(1.0 - prob_mat) / params.simulation.days_per_year
            if rand(abmrng(model)) < (1.0 - exp(-daily_hazard * dt))
                org.is_mature = true
                org.is_mating_ready = true
            end
        end
    end

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

                # Handling a caught and returned crab kills some of them. Charged on
                # capture, only to an individual's *first* capture — the damage
                # happens at the event, so a crab already handled before is not
                # charged again — and only inside the configured window, after
                # which the term is zero.
                #
                # Which individuals have been handled is per-run state rather than a
                # trait, so it lives on the model instead of on `Organism`: adding a
                # field there would mean touching its four positional constructors
                # and every call site.
                if model.current_time <= params.mortality.handling_mortality_days
                    handled = get!(Set{Int}, abmproperties(model), :handled_ids)
                    if !(org.id in handled)
                        push!(handled, org.id)
                        if rand(abmrng(model)) < params.mortality.rate_handling
                            model.deaths_fishery += 1
                            remove_agent!(org, model)
                            return nothing
                        end
                    end
                end
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
    mprops = abmproperties(model)
    has_monthly = env.use_geodata &&
                  haskey(mprops, :monthly_bottom) &&
                  model.monthly_bottom !== nothing

    if has_monthly
        # Monthly dynamic hydrographic sequence with linear interpolation between month midpoints
        # Midpoint of month m is (m - 0.5) * 365 / 12
        month_frac = (day_of_year / sim.days_per_year) * 12.0
        m1 = mod(floor(Int, month_frac - 0.5), 12) + 1
        m2 = mod(m1, 12) + 1
        frac = (month_frac - 0.5) - floor(month_frac - 0.5)

        b1 = model.monthly_bottom[m1]
        b2 = model.monthly_bottom[m2]
        s1 = model.monthly_surface[m1]
        s2 = model.monthly_surface[m2]

        surface_temp = model.surface_temperature_field
        if t_sd > 0.0
            randn!(abmrng(model), temperature)
            @. surface_temp = (1.0 - frac) * s1 + frac * s2 + t_sd * temperature
            @. temperature = (1.0 - frac) * b1 + frac * b2 + t_sd * temperature
        else
            @. surface_temp = (1.0 - frac) * s1 + frac * s2
            @. temperature = (1.0 - frac) * b1 + frac * b2
        end
        if haskey(mprops, :depth_field)
            @. model.habitat_quality_field = calculate_habitat_suitability(
                temperature, model.depth_field)
        end
    elseif env.use_geodata && haskey(mprops, :baseline_temperature_field)
        base_field = model.baseline_temperature_field
        base_surf = get(mprops, :baseline_surface_temperature_field, base_field)
        surface_temp = get(mprops, :surface_temperature_field, temperature)
        if t_sd > 0.0
            randn!(abmrng(model), temperature)
            @. surface_temp = base_surf + t_amp * sinusoidal_factor + t_sd * temperature
            @. temperature = base_field + t_amp * sinusoidal_factor + t_sd * temperature
        else
            @. surface_temp = base_surf + t_amp * sinusoidal_factor
            @. temperature = base_field + t_amp * sinusoidal_factor
        end
        if haskey(mprops, :depth_field)
            @. model.habitat_quality_field = calculate_habitat_suitability(
                temperature, model.depth_field)
        end
    else
        surface_temp = get(mprops, :surface_temperature_field, temperature)
        if t_sd > 0.0
            randn!(abmrng(model), temperature)
            @. surface_temp = seasonal_target + t_sd * temperature
            @. temperature = seasonal_target + t_sd * temperature
        else
            fill!(surface_temp, seasonal_target)
            fill!(temperature, seasonal_target)
        end
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
