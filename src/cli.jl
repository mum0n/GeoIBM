"""
    src/cli.jl

Command-line interface (CLI) execution engine and entry points for the
organism individual-based model (IBM).

Provides parameter parsing, configuration hierarchy resolution, terminal
reporting, live snapshot telemetry, batch figure export, interactive GUI
launching, and native C-callable binary entry points (`julia_main`).
"""

using Printf
using Random
using Statistics
using DataFrames
using CairoMakie
import GLMakie

"""
    resolve_configuration_files(; explicit_path="")

Locate the baseline and overlay TOML configuration files.

Configurations are always applied in two layers, lowest priority first:

1. the baseline, `default.toml`, which records every declared parameter, and
2. an overlay holding the values that differ for a particular run.

`explicit_path` names the overlay. If it *is* the baseline, the conventional
`snowcrab.toml` sibling is used as the overlay instead, so `--config` always
selects the highest-priority layer. Without `explicit_path` both files are
located under `configs/` relative to the working directory and the package root.

The previous behaviour treated the explicit path as the baseline and overlaid
whatever other `*.toml` sat beside it. Passing a species file therefore loaded
`default.toml` *after* it, so the baseline silently overwrote the very file the
user had asked for.

# Returns
`(baseline, overlay)` tuple of paths; either entry is `""` when absent. The
baseline is always loaded first and the overlay second.
"""
function resolve_configuration_files(; explicit_path::AbstractString = "")
    dir = if isempty(explicit_path)
        ""
    else
        isdir(explicit_path) ? explicit_path : dirname(explicit_path)
    end

    if !isempty(dir) && isdir(dir)
        baseline = joinpath(dir, "default.toml")
        isfile(baseline) || (baseline = "")

        overlay = if isempty(explicit_path)
            species = joinpath(dir, "snowcrab.toml")
            isfile(species) ? species : ""
        elseif isfile(explicit_path) && abspath(explicit_path) != abspath(baseline)
            String(explicit_path)
        else
            # The explicit path is the baseline, so the species sibling overlays it.
            species = joinpath(dir, "snowcrab.toml")
            isfile(species) ? species : ""
        end

        return (String(baseline), String(overlay))
    end

    for root in (pwd(), joinpath(@__DIR__, ".."))
        candidate = joinpath(root, "configs", "default.toml")
        isfile(candidate) || continue
        species = joinpath(root, "configs", "snowcrab.toml")
        overlay = isfile(species) ? species : ""
        if !isempty(explicit_path) && isfile(explicit_path) &&
           abspath(explicit_path) != abspath(candidate)
            return (String(candidate), String(explicit_path))
        end
        return (String(candidate), String(overlay))
    end
    return ("", isfile(explicit_path) ? String(explicit_path) : "")
end

"""
    coerce_config_value(::Type{T}, raw::AbstractString)

Convert a command-line string into the declared parameter type `T`.

The CLI accepts every parameter that the structs declare, so values must be
coerced to the same type the struct field expects. Without this, a numeric
parameter arriving as `String` would either fail deep inside a constructor or be
silently stored as text.
"""
function coerce_config_value(::Type{T}, raw::AbstractString) where {T}
    text = strip(String(raw))
    if T === Bool
        lowered = lowercase(text)
        return !(lowered in ("false", "0", "no", "off", ""))
    elseif T === Int
        return parse(Int, text)
    elseif T === Float64
        return parse(Float64, text)
    elseif T === String
        return text
    elseif T <: Tuple
        parts = parse.(Int, [strip(p) for p in split(replace(text, r"[\(\)\[\]]" => ""), ',')])
        return T(parts)
    elseif T <: AbstractVector
        if occursin(',', text) || occursin(';', text)
            tokens = [strip(p) for p in split(replace(text, ',' => ';'), ';') if !isempty(strip(p))]
        else
            tokens = split(text)
        end
        if eltype(T) === Float64
            return T(parse.(Float64, tokens))
        elseif eltype(T) === Int
            return T(parse.(Int, tokens))
        end
        return T(tokens)
    end
    return convert(T, text)
end

"""
    parse_cli_args(args::Vector{String})

Parse command-line argument vector into a configuration dictionary.

Follows a strict three-tier precedence hierarchy:
1. Internal defaults (`default_ibm_configuration()`)
2. Configuration file overrides (`--config=<path>` or auto-detected default)
3. Explicit command-line arguments (highest priority, supporting `--key=value`)

# Arguments
- `args::Vector{String}`: Command line arguments from `ARGS`.

# Returns
`Dict{Symbol, Any}` containing simulation configuration.
"""
function parse_cli_args(args::AbstractVector)
    # Accept any vector of strings, including the `[]` literal, so callers do not
    # have to annotate the element type.
    argv = String[String(a) for a in args]
    config = default_ibm_configuration()
    config[:help] = false
    config[:config_file] = ""
    config[:output_dir] = "outputs"
    config[:gui] = false
    config[:gui_simple] = false

    function extract_key_val(idx::Int)
        tok = argv[idx]
        if contains(tok, '=')
            parts = split(tok, '=', limit = 2)
            k_str = lstrip(parts[1], '-')
            v_str = String(parts[2])
            return (k_str, v_str, idx)
        elseif startswith(tok, '-')
            k_str = lstrip(tok, '-')
            if idx < length(argv) && !startswith(argv[idx + 1], '-')
                return (k_str, argv[idx + 1], idx + 1)
            else
                return (k_str, "true", idx)
            end
        else
            return ("", "", idx)
        end
    end

    # Pre-scan for `--config` so the configuration files are located before the
    # main loop runs, since a file value has lower precedence than a command-line
    # value and must be in place first. `break` on the first match is deliberate
    # only in the sense that a single `--config` is the documented form; taking the
    # last match instead makes a repeated flag behave like every other one, where
    # the later occurrence wins.
    custom_config_path = ""
    idx = 1
    while idx <= length(args)
        k_str, v_str, next_idx = extract_key_val(idx)
        if k_str == "config"
            custom_config_path = v_str
        end
        idx = (next_idx > idx) ? (next_idx + 1) : (idx + 1)
    end

    default_config, species_config = resolve_configuration_files(
        explicit_path = custom_config_path
    )

    for path in (default_config, species_config)
        isempty(path) && continue
        isfile(path) || continue
        file_cfg = load_ibm_configuration(path; base_config = config)
        for (k, v) in file_cfg
            config[k] = v
        end
    end
    if !isempty(default_config)
        config[:config_file] = default_config
    end
    if !isempty(species_config)
        config[:species_config_file] = species_config
    end

    sync_grid_dimensions!(config)

    # Command-line arguments form the highest-precedence layer. Snapshot the
    # reconciled state so the keys this layer actually changes can be identified
    # afterwards, and pass them to the final reconcile so an explicit alias on the
    # command line is not overwritten by a canonical name inherited from a file.
    before_cli = copy(config)

    i = 1
    while i <= length(args)
        k_str, v_str, next_i = extract_key_val(i)
        i = (next_i > i) ? (next_i + 1) : (i + 1)

        if isempty(k_str)
            continue
        end

        # Only the flags that are not declared parameters are handled by name.
        # Every declared parameter reaches the generic branch below, which looks
        # the name up in the parameter structs and coerces the value to the field
        # type. That keeps the command line automatically in step with the
        # structs, so a new parameter is settable without editing this parser and
        # an unknown name is reported rather than silently dropped.
        if k_str == "help"
            config[:help] = true
        elseif k_str in ("gui", "gui_simple", "gui-simple")
            config[:gui] = (lowercase(v_str) != "false")
            config[:gui_simple] = config[:gui]
        elseif k_str == "config"
            # Already consumed by the pre-scan that located the configuration
            # files; it is recorded on the config dict rather than set here.
        elseif k_str == "grid"
            # `grid` is the `(nx, ny)` pair form of the `grid_nx`/`grid_ny`
            # parameters; `sync_grid_dimensions!` propagates it to both.
            parts = parse.(Int, [strip(p) for p in split(replace(v_str, r"[\(\)\[\]]" => ""), ',')])
            config[:grid] = (parts[1], parts[2])
        elseif k_str == "n_agents"
            config[:n_agents] = parse(Int, strip(v_str))
        else
            sym_k = Symbol(k_str)
            declared = config_field_type(sym_k)
            if declared === nothing
                # A CLI switch that reaches the generic branch has already been
                # consumed (like `--config`, which the pre-scan above uses to
                # locate the configuration files). Reporting it as an unrecognized
                # parameter would tell the user to fix a spelling that is correct,
                # and the message names a different line from the one that made the
                # decision. `cli_switch_keys()` is the authority on which names are
                # command-line switches that never appear in a config file.
                if sym_k in cli_switch_keys()
                    @debug "configuration switch already handled" key = k_str
                else
                    @warn "Ignoring unrecognized configuration key; expected an exact parameter name" key =
                        k_str
                end
            else
                config[sym_k] = coerce_config_value(declared, v_str)
            end
        end
    end

    cli_keys = Set{Symbol}()
    for (key, value) in config
        if !haskey(before_cli, key) || !isequal(before_cli[key], value)
            push!(cli_keys, key)
        end
    end
    sync_grid_dimensions!(config; explicit = cli_keys)

    return config
end

"""
    display_cli_help()

Print detailed command-line interface usage instructions.

Takes an `io` so the text can be captured, matching `summarize_empirical_model`.
"""
function display_cli_help(io::IO = stdout)
    println(io, """
================================================================================
  ORGANISM INDIVIDUAL-BASED MODEL (IBM) - COMMAND LINE INTERFACE
================================================================================
Usage:
  GeoIBM [OPTIONS]
  julia scripts/organism_ibm.jl [OPTIONS]

Every parameter accepts exactly one name, matching the field names declared in
`configs/default.toml` and the parameter structs. There are no abbreviations or
alternative spellings: an unrecognized name is reported and ignored.

Flags:
  --config=<Path>            Configuration file (TOML format)
  --gui=<Bool>               Launch the interactive real-time GLMakie dashboard
  --help                     Show this help message

Common parameters (see configs/default.toml for the full list):
  --days=<Int>                             Total simulation days (default: 730)
   --grid=<Nx,Ny>                           Spatial grid dimensions (default: 100,100)
  --grid_nx=<Int>                          Spatial grid width (default: 100)
  --grid_ny=<Int>                          Spatial grid height (default: 100)
  --dt=<Float>                             Simulation time step in days (default: 1.0)
  --seed=<Int>                             Random seed (default: 1234)
  --temp_offset=<Float>                    Baseline bottom temperature offset (default: 0.0)
  --temp_multiplier=<Float>                Seasonal temperature multiplier (default: 1.0)
  --temp_base=<Float>                      Legacy alias for --temp_offset
  --temp_amplitude=<Float>                 Legacy alias for --temp_multiplier
  --benthic_carrying_capacity=<Float>      Total benthic capacity for the whole model,
                                           not per cell (default: 12000.0)
  --frequency_days=<Float>                 Output sampling interval (default: 365.0)
  --snapshot_day=<Float>                   Annual survey snapshot day of year (default: 274.0)
  --save=<Bool>                            Write annual indicators to CSV (default: true)
  --save_file=<Path>                       Output CSV filename
  --plot=<Bool>                            Export diagnostic figures (default: false)
  --animate=<Bool>                         Record an animation (default: false)
  --animation_file=<Path>                  Animation output filename
  --output_dir=<Path>                      Output directory (default: outputs)

Any parameter declared in a TOML section is settable, for example:
  --maturity_l50_male=<Float>              [species] maturity size-at-50 for males
  --legal_size_male=<Float>                [fishery] minimum legal size
  --rate_predation=<Float>                 [mortality] annual predation rate
  --survival_stage1=<Float>                [leslie] stage-4 survival
  --use_energetics=<Bool>                  [toggles] enable bioenergetics
  --intermolt_days_schedule=60,75,90       [species] stage-indexed schedule
================================================================================
""")
end

"""
    print_simulation_report(summary, init_agents, days, elapsed_time; io = stdout)

Print formatted terminal report of simulation outcomes and population structure.

Takes an `io` so the text can be captured, matching `summarize_empirical_model`.
"""
function print_simulation_report(
    summary,
    init_agents::Int,
    days::Int,
    elapsed_time::Float64;
    io::IO = stdout
)
    println(io)
    println(io, "="^80)
    println(io, "                  ORGANISM IBM SIMULATION RESULTS                     ")
    println(io, "="^80)
    @printf(io, " Simulation Duration   : %d days (%.2f years)\n", days, days / 365.0)
    @printf(io, " Computation Time      : %.2f seconds\n", elapsed_time)
    @printf(io, " Initial Population    : %d organisms\n", init_agents)
    total_orgs = hasproperty(summary, :total_organisms) ? summary.total_organisms :
                 (hasproperty(summary, :total_organisms) ? summary.total_organisms : 0)
    @printf(io, " Final Living Organisms : %d organisms\n", total_orgs)

    survival_pct = (init_agents > 0) ? (total_orgs / init_agents * 100.0) : 0.0
    @printf(io, " Net Population Change : %+.1f%%\n", (survival_pct - 100.0))
    println(io, "-"^80)

    println(io, " [Key Output Indicators]")
    @printf(io, "   - Total Number of Mature Females   : %4d\n", summary.total_mature_females)
    @printf(io, "   - Total Number of Mature Males     : %4d\n", summary.total_mature_males)
    @printf(io, "   - Total Mass of Fishable Males     : %8.3f kg (%.4f tonnes)\n",
            summary.total_fishable_male_mass_kg, summary.total_fishable_male_mass_kg / 1000.0)
    @printf(io, "   - Total Living Population Biomass  : %8.3f kg (%.4f tonnes)\n",
            summary.total_biomass_kg, summary.total_biomass_kg / 1000.0)
    c_num = hasproperty(summary, :total_catch_numbers) ? summary.total_catch_numbers : 0
    c_bio = hasproperty(summary, :total_catch_biomass_kg) ? summary.total_catch_biomass_kg : 0.0
    @printf(io, "   - Cumulative Harvest Removals      : %4d organisms (%.3f kg)\n",
            c_num, c_bio)
    println(io, "-"^80)

    if hasproperty(summary, :count_by_stage_male)
        println(io, "   - Demographic Stage & Sex Structure:")
        # `mass_by_stage` is the mean individual mass at that stage, in grams:
        # `summarize_population` divides the stage's summed mass by its summed
        # count. The column previously read "Total Mass (kg)", which was wrong
        # twice over, so a 95 mm female was reported as 290.70 kg when her mass
        # was 0.29 kg. The biomass totals above are the kilogram quantities.
        println(io, "       Stage | Total Count |   Males | Females | Mean M (mm) | Mean F (mm) | Mean Mass (g)")
        println(io, "       ------+-------------+---------+---------+-------------+-------------+----------------")
        for stg in sort(collect(keys(summary.size_by_stage)))
            count_stg = summary.count_by_stage[stg]
            c_m = get(summary.count_by_stage_male, stg, 0)
            c_f = get(summary.count_by_stage_female, stg, 0)
            sz_m = get(summary.size_by_stage_male, stg, 0.0)
            sz_f = get(summary.size_by_stage_female, stg, 0.0)
            m_mass = hasproperty(summary, :mass_by_stage) ?
                get(summary.mass_by_stage, stg, 0.0) : 0.0
            @printf(io, "         %2d  | %11d | %7d | %7d |      %6.2f |      %6.2f |      %10.2f\n",
                    stg, count_stg, c_m, c_f, sz_m, sz_f, m_mass)
        end
    end
    println(io, "="^80)
    println()
end

"""
    print_save_step_outputs(ann_df::DataFrame; io = stdout)

Print formatted table of demographic indicators across all save steps.

Takes an `io` so the text can be captured. An empty frame prints nothing at all,
rather than a header with no rows.
"""
function print_save_step_outputs(ann_df::DataFrame; io::IO = stdout)
    if isempty(ann_df)
        return
    end
    println(io)
    println(io, "="^108)
    println(io, "                                PERIODIC SAVE STEP POPULATION INDICATORS")
    println(io, "="^108)
    println(io, " Idx|  Day  | Total N | Adult F | Adult M | Legal N | Total(kg) | SSB F(kg) | MMB(kg) | Legal(kg) | Eggs(ann) | Catch")
    println(io, "----+--------+---------+---------+---------+---------+-----------+-----------+---------+-----------+-----------+------")
    for row in eachrow(ann_df)
        @printf(
            io,
            " %2d | %6.0f | %7d | %7d | %7d | %7d | %9.2f | %9.2f | %7.2f | %9.2f | %9.2e | %5d\n",
            row.sample_index,
            row.day,
            round(Int, row.total_numbers),
            round(Int, row.mature_females),
            round(Int, row.mature_males),
            round(Int, row.legal_numbers),
            row.total_biomass_kg,
            row.mature_female_biomass_kg,
            row.mature_male_biomass_kg,
            row.fishable_male_biomass_kg,
            row.annual_egg_production,
            round(Int, row.catch_total)
        )
    end
    println(io, "="^108)
    println(io)
end

"""
    generate_summary_plots(model, agent_df, output_dir; model_df=nothing)

Generate and save all diagnostic figures into `output_dir`.
"""
function generate_summary_plots(
    model,
    agent_df::DataFrame,
    output_dir::String;
    model_df = nothing
)
    files = export_all_plots(
        model,
        agent_df,
        output_dir;
        model_df = model_df
    )
    for f in files
        println("  Saved: $(f)")
    end
    return files
end

"""
    run_ibm_cli(args=ARGS)::Int

Execute the full organism IBM workflow from command line argument vector.

# Arguments
- `args::Vector{String}`: Command line arguments.

# Returns
`Int`: Exit status code (0 for success).
"""
function run_ibm_cli(args::AbstractVector = ARGS)::Int
    config = parse_cli_args(args)

    if config[:help]
        display_cli_help()
        return 0
    end

    if get(config, :gui, false) || get(config, :gui_simple, false)
        println("="^80)
        println(" Launching Organism IBM GUI Dashboard (GLMakie)...")
        println("="^80)
        fig, state = launch_organism_gui(
            config;
            display_window = true
        )
        println(" GUI launched. Close the window to terminate.")
        try
            screen = GLMakie.display(fig)
            wait(screen)
        catch
        end
        return 0
    end

    println("="^80)
    println(" Initializing Organism IBM Simulation...")
    println(" Configuration:")
    cfg_src = isempty(config[:config_file]) ? "Internal Defaults" : config[:config_file]
    println("   - Config Source   : $(cfg_src)")
    println(" Parameters:")
    n_init = get(config, :n_agents, sum(config[:initial_female_cohort]) + sum(config[:initial_male_cohort]))
    @printf("   - Agents          : %d\n", n_init)
    @printf("   - Days            : %d (%.2f years)\n", config[:days], config[:days] / 365.0)
    @printf("   - Grid Size       : %d x %d\n", config[:grid][1], config[:grid][2])
    t_offset = get(config, :temp_offset, get(config, :temp_base, 0.0))
    t_mult = get(config, :temp_multiplier, get(config, :temp_amplitude, 1.0))
    f_prob = get(config, :fishing_probability_daily, 0.0)
    carrying_cap = get(config, :benthic_carrying_capacity, 300.0)
    @printf("   - Temperature     : offset %+.1f °C, multiplier %.2f×\n", t_offset, t_mult)
    @printf("   - Harvest Prob    : %.4f daily\n", f_prob)
    @printf("   - Carrying Cap    : %.1f organisms (model total)\n", carrying_cap)
    @printf("   - Random Seed     : %d\n", config[:seed])
    @printf("   - Save Results    : %s\n", config[:save] ? "Yes" : "No")
    @printf("   - Plot Figures    : %s\n", config[:plot] ? "Yes" : "No")
    println("="^80)

    params = IBMParameters(config)
    t_start = time()
    model = initialize_model(
        n_agents = n_init,
        grid = config[:grid],
        dt = config[:dt],
        params = params,
        seed = config[:seed]
    )

    adata = default_agent_observations(params)
    mdata = [
        :total_eggs_spawned,
        :total_recruits_settled,
        :total_catch_numbers,
        :total_catch_biomass_kg
    ]

    agent_df, model_df = run_simulation(
        model,
        config[:days];
        adata = adata,
        mdata = mdata
    )
    elapsed_time = time() - t_start

    summary = summarize_population(model)
    print_simulation_report(summary, n_init, config[:days], elapsed_time)

    # A short run legitimately has no annual samples, but a failure here is not
    # the same as that and must not be reported as a clean run.
    ann_df = try
        df = extract_annual_production(
            agent_df,
            model_df;
            frequency_days = Float64(config[:frequency_days]),
            snapshot_day = Float64(config[:snapshot_day])
        )
        print_save_step_outputs(df)
        df
    catch err
        @warn "annual production extraction failed; no annual output written" exception =
            err
        DataFrame()
    end

    if config[:save] && !isempty(ann_df)
        mkpath(config[:output_dir])
        save_path = isabspath(config[:save_file]) ?
            config[:save_file] : joinpath(config[:output_dir], config[:save_file])
        println(" Saving annual simulation results to: $(save_path)")
        write_dataframe_csv(save_path, ann_df)
        println(" Simulation results saved successfully.")
    end

    if config[:plot]
        println(" Generating summary plots in: $(config[:output_dir])")
        generated = generate_summary_plots(
            model,
            agent_df,
            config[:output_dir];
            model_df = model_df
        )
        println(" Exported $(length(generated)) figure(s).")
    end

    if get(config, :animate, false)
        mkpath(config[:output_dir])
        anim_path = isabspath(config[:animation_file]) ?
            config[:animation_file] : joinpath(config[:output_dir], config[:animation_file])
        println(" Recording Agents.jl simulation animation to: $(anim_path)")
        t_anim = time()
        anim_model = initialize_model(
            n_agents = n_init,
            grid = config[:grid],
            dt = config[:dt],
            params = params,
            seed = config[:seed]
        )
        animate_organism_simulation(
            anim_model;
            filepath = anim_path,
            frames = Int(config[:anim_frames]),
            dt = Int(config[:anim_dt]),
            framerate = Int(config[:anim_fps]),
            heat_field = Symbol(config[:anim_field])
        )
        @printf(" Animation video recorded in %.2f seconds.\n", time() - t_anim)
    end

    return 0
end

"""
    julia_main()::Cint

Standard C-callable application entry point for PackageCompiler binaries.

# Returns
`Cint`: 0 on successful termination, 1 on caught unhandled exception.
"""
Base.@ccallable function julia_main()::Cint
    try
        return Cint(run_ibm_cli(ARGS))
    catch
        Base.invokelatest(Base.display_error, Base.catch_stack())
        return Cint(1)
    end
end
