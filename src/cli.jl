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
function parse_cli_args(args::Vector{String})
    config = default_ibm_configuration()
    config[:help] = false
    config[:config_file] = ""
    config[:output_dir] = "outputs"
    config[:gui] = false

    function extract_key_val(idx::Int)
        tok = args[idx]
        if contains(tok, '=')
            parts = split(tok, '=', limit = 2)
            k_str = lstrip(parts[1], '-')
            v_str = String(parts[2])
            return (k_str, v_str, idx)
        elseif startswith(tok, '-')
            k_str = lstrip(tok, '-')
            if idx < length(args) && !startswith(args[idx + 1], '-')
                return (k_str, args[idx + 1], idx + 1)
            else
                return (k_str, "true", idx)
            end
        else
            return ("", "", idx)
        end
    end

    custom_config_path = ""
    idx = 1
    while idx <= length(args)
        k_str, v_str, next_idx = extract_key_val(idx)
        if k_str in ("config", "c")
            custom_config_path = v_str
            break
        end
        idx = (next_idx > idx) ? (next_idx + 1) : (idx + 1)
    end

    config_file_to_load = if !isempty(custom_config_path)
        custom_config_path
    else
        candidates = [
            joinpath(pwd(), "organism_ibm.config"),
            joinpath(pwd(), "scripts", "organism_ibm.config"),
            joinpath(pwd(), "scripts", "snowcrab_ibm.config"),
            joinpath(@__DIR__, "..", "scripts", "organism_ibm.config"),
            joinpath(@__DIR__, "..", "scripts", "snowcrab_ibm.config")
        ]
        found = ""
        for c in candidates
            if isfile(c)
                found = c
                break
            end
        end
        found
    end

    if !isempty(config_file_to_load) && isfile(config_file_to_load)
        config[:config_file] = config_file_to_load
        file_cfg = load_ibm_configuration(config_file_to_load)
        for (k, v) in file_cfg
            config[k] = v
        end
    end

    if haskey(config, :fishing_probability_daily) && !haskey(config, :fishing_prob)
        config[:fishing_prob] = config[:fishing_probability_daily]
    end
    if haskey(config, :temp_amplitude) && !haskey(config, :temp_amp)
        config[:temp_amp] = config[:temp_amplitude]
    end
    if haskey(config, :benthic_carrying_capacity) && !haskey(config, :carrying_capacity)
        config[:carrying_capacity] = config[:benthic_carrying_capacity]
    end

    i = 1
    while i <= length(args)
        k_str, v_str, next_i = extract_key_val(i)
        i = (next_i > i) ? (next_i + 1) : (i + 1)

        if isempty(k_str)
            continue
        end

        if k_str in ("help", "h")
            config[:help] = true
        elseif k_str in ("gui", "interactive")
            config[:gui] = (lowercase(v_str) != "false")
        elseif k_str == "days"
            config[:days] = parse(Int, v_str)
        elseif k_str in ("n_agents", "agents", "n")
            config[:n_agents] = parse(Int, v_str)
        elseif k_str == "grid"
            coords = parse.(Int, split(replace(v_str, r"[\(\)\[\]]" => ""), ','))
            config[:grid] = (coords[1], coords[2])
        elseif k_str == "temp_base"
            config[:temp_base] = parse(Float64, v_str)
        elseif k_str in ("temp_amp", "temp_amplitude")
            config[:temp_amp] = parse(Float64, v_str)
            config[:temp_amplitude] = parse(Float64, v_str)
        elseif k_str in ("fishing_prob", "harvest_prob")
            config[:fishing_prob] = parse(Float64, v_str)
            config[:fishing_probability_daily] = parse(Float64, v_str)
        elseif k_str in ("carrying_capacity", "k_cap")
            config[:carrying_capacity] = parse(Float64, v_str)
            config[:benthic_carrying_capacity] = parse(Float64, v_str)
        elseif k_str == "dt"
            config[:dt] = parse(Float64, v_str)
        elseif k_str in ("seed", "rng_seed")
            config[:seed] = parse(Int, v_str)
        elseif k_str == "save"
            config[:save] = (lowercase(v_str) != "false")
        elseif k_str == "save_file"
            config[:save_file] = v_str
        elseif k_str in ("frequency_days", "save_freq")
            config[:frequency_days] = parse(Float64, v_str)
        elseif k_str == "snapshot_day"
            config[:snapshot_day] = parse(Float64, v_str)
        elseif k_str == "plot"
            config[:plot] = (lowercase(v_str) != "false")
        elseif k_str in ("animate", "video")
            config[:animate] = (lowercase(v_str) != "false")
        elseif k_str in ("animation_file", "video_file")
            config[:animation_file] = v_str
        elseif k_str == "anim_frames"
            config[:anim_frames] = parse(Int, v_str)
        elseif k_str == "anim_dt"
            config[:anim_dt] = parse(Int, v_str)
        elseif k_str == "anim_fps"
            config[:anim_fps] = parse(Int, v_str)
        elseif k_str == "anim_field"
            config[:anim_field] = v_str
        elseif k_str == "output_dir"
            config[:output_dir] = v_str
        elseif k_str in ("use_energetics", "energetics")
            config[:use_energetics] = (lowercase(v_str) != "false")
        elseif k_str in ("use_thermal_stress", "thermal_stress")
            config[:use_thermal_stress] = (lowercase(v_str) != "false")
        elseif k_str in ("use_food_dynamics", "food_dynamics")
            config[:use_food_dynamics] = (lowercase(v_str) != "false")
        elseif k_str in ("use_habitat_taxis", "habitat_taxis")
            config[:use_habitat_taxis] = (lowercase(v_str) != "false")
        elseif k_str in ("use_cannibalism", "cannibalism")
            config[:use_cannibalism] = (lowercase(v_str) != "false")
        elseif k_str in ("use_senescence", "senescence")
            config[:use_senescence] = (lowercase(v_str) != "false")
        end
    end

    return config
end

"""
    display_cli_help()

Print detailed command-line interface usage instructions.
"""
function display_cli_help()
    println("""
================================================================================
  ORGANISM INDIVIDUAL-BASED MODEL (IBM) - COMMAND LINE INTERFACE
================================================================================
Usage:
  model_agent [OPTIONS]
  julia scripts/organism_ibm.jl [OPTIONS]

Options:
  --config=<Path>, -c <Path> Configuration file (TOML format)
  --gui, --interactive       Launch real-time interactive multi-dashboard GUI
  --days=<Int>               Total simulation days (default: 730)
  --n_agents=<Int>, -n <Int> Initial population size (default: 300)
  --grid=<Nx,Ny>             Spatial grid dimensions (default: 100,100)
  --temp_base=<Float>        Mean bottom water temperature in °C (default: 5.0)
  --temp_amp=<Float>         Annual temperature amplitude in °C (default: 4.0)
  --fishing_prob=<Float>     Daily commercial harvest probability (default: 0.0005)
  --carrying_capacity=<Float> Benthic carrying capacity per cell (default: 1000.0)
  --dt=<Float>               Simulation time step in days (default: 1.0)
  --seed=<Int>               Random seed for reproducibility (default: 42)
  --save                     Save annual population indicators to CSV (default: true)
  --save_file=<Path>         Output filename for simulation results CSV
  --plot                     Export diagnostic figures and dashboard (default: true)
  --animate, --video         Record simulation video animation (.mp4)
  --output_dir=<Path>        Output directory for figures and data (default: outputs)
  --help, -h                 Show this help message
================================================================================
""")
end

"""
    print_simulation_report(summary, init_agents, days, elapsed_time)

Print formatted terminal report of simulation outcomes and population structure.
"""
function print_simulation_report(
    summary,
    init_agents::Int,
    days::Int,
    elapsed_time::Float64
)
    println()
    println("="^80)
    println("                  ORGANISM IBM SIMULATION RESULTS                     ")
    println("="^80)
    @printf(" Simulation Duration   : %d days (%.2f years)\n", days, days / 365.0)
    @printf(" Computation Time      : %.2f seconds\n", elapsed_time)
    @printf(" Initial Population    : %d organisms\n", init_agents)
    total_orgs = hasproperty(summary, :total_organisms) ? summary.total_organisms :
                 (hasproperty(summary, :total_crabs) ? summary.total_crabs : 0)
    @printf(" Final Living Organisms : %d organisms\n", total_orgs)

    survival_pct = (init_agents > 0) ? (total_orgs / init_agents * 100.0) : 0.0
    @printf(" Net Population Change : %+.1f%%\n", (survival_pct - 100.0))
    println("-"^80)

    println(" [Key Output Indicators]")
    @printf("   - Total Number of Mature Females   : %4d\n", summary.total_mature_females)
    @printf("   - Total Number of Mature Males     : %4d\n", summary.total_mature_males)
    @printf("   - Total Mass of Fishable Males     : %8.3f kg (%.4f tonnes)\n",
            summary.total_fishable_male_mass_kg, summary.total_fishable_male_mass_kg / 1000.0)
    @printf("   - Total Living Population Biomass  : %8.3f kg (%.4f tonnes)\n",
            summary.total_biomass_kg, summary.total_biomass_kg / 1000.0)
    c_num = hasproperty(summary, :total_catch_numbers) ? summary.total_catch_numbers : 0
    c_bio = hasproperty(summary, :total_catch_biomass_kg) ? summary.total_catch_biomass_kg : 0.0
    @printf("   - Cumulative Harvest Removals      : %4d organisms (%.3f kg)\n",
            c_num, c_bio)
    println("-"^80)

    if hasproperty(summary, :count_by_stage_male)
        println("   - Demographic Stage & Sex Structure:")
        println("       Stage | Total Count |   Males | Females | Mean M (mm) | Mean F (mm) | Total Mass (kg)")
        println("       ------+-------------+---------+---------+-------------+-------------+----------------")
        for stg in sort(collect(keys(summary.size_by_stage)))
            count_stg = summary.count_by_stage[stg]
            c_m = get(summary.count_by_stage_male, stg, 0)
            c_f = get(summary.count_by_stage_female, stg, 0)
            sz_m = get(summary.size_by_stage_male, stg, 0.0)
            sz_f = get(summary.size_by_stage_female, stg, 0.0)
            m_mass = hasproperty(summary, :mass_by_stage) ?
                get(summary.mass_by_stage, stg, 0.0) : 0.0
            @printf("         %2d  | %11d | %7d | %7d |      %6.2f |      %6.2f |      %10.2f\n",
                    stg, count_stg, c_m, c_f, sz_m, sz_f, m_mass)
        end
    end
    println("="^80)
    println()
end

"""
    print_save_step_outputs(ann_df::DataFrame)

Print formatted table of demographic indicators across all save steps.
"""
function print_save_step_outputs(ann_df::DataFrame)
    if isempty(ann_df)
        return
    end
    println()
    println("="^108)
    println("                                PERIODIC SAVE STEP POPULATION INDICATORS")
    println("="^108)
    println(" Yr |   Day  | Total N | Adult F | Adult M | Legal N | Total(kg) | SSB F(kg) | MMB(kg) | Legal(kg) | Eggs(ann) | Catch")
    println("----+--------+---------+---------+---------+---------+-----------+-----------+---------+-----------+-----------+------")
    for row in eachrow(ann_df)
        @printf(
            " %2d | %6.0f | %7d | %7d | %7d | %7d | %9.2f | %9.2f | %7.2f | %9.2f | %9.2e | %5d\n",
            row.year,
            row.time,
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
    println("="^108)
    println()
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
function run_ibm_cli(args::Vector{String} = ARGS)::Int
    config = parse_cli_args(args)

    if config[:help]
        display_cli_help()
        return 0
    end

    if config[:gui]
        println("="^80)
        println(" Launching Organism IBM Interactive Real-Time Dashboard...")
        println("="^80)
        fig, state = launch_organism_gui(
            config;
            display_window = false
        )
        println(" GUI launched. Close the window to terminate.")
        try
            screen = display(fig)
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
    @printf("   - Agents          : %d\n", config[:n_agents])
    @printf("   - Days            : %d (%.2f years)\n", config[:days], config[:days] / 365.0)
    @printf("   - Grid Size       : %d x %d\n", config[:grid][1], config[:grid][2])
    t_base = get(config, :temp_base, 5.0)
    t_amp = get(config, :temp_amp, get(config, :temp_amplitude, 4.0))
    f_prob = get(config, :fishing_prob, get(config, :fishing_probability_daily, 0.0005))
    k_cap = get(config, :carrying_capacity, get(config, :benthic_carrying_capacity, 1000.0))
    @printf("   - Temperature     : %.1f ± %.1f °C\n", t_base, t_amp)
    @printf("   - Harvest Prob    : %.4f daily\n", f_prob)
    @printf("   - Carrying Cap    : %.1f organisms\n", k_cap)
    @printf("   - Random Seed     : %d\n", config[:seed])
    @printf("   - Save Results    : %s\n", config[:save] ? "Yes" : "No")
    @printf("   - Plot Figures    : %s\n", config[:plot] ? "Yes" : "No")
    println("="^80)

    params = IBMParameters(config)
    t_start = time()
    model = initialize_model(
        N_agents = config[:n_agents],
        dims = config[:grid],
        dt_days = config[:dt],
        params = params,
        rng_seed = config[:seed]
    )

    adata = [
        (:stage, length),
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
        (mass_if_legal_male, sum),
        (:stage, safe_mean)
    ]
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
    print_simulation_report(summary, config[:n_agents], config[:days], elapsed_time)

    ann_df = DataFrame()
    try
        ann_df = extract_annual_production(agent_df, model_df)
        print_save_step_outputs(ann_df)
    catch
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
        generate_summary_plots(
            model,
            agent_df,
            config[:output_dir];
            model_df = model_df
        )
        println(" All figures successfully exported.")
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
