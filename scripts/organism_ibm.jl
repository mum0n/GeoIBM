"""
Command-Line Interface (CLI) for the Organism Individual-Based Model (IBM).

Usage:
    julia --project=. scripts/organism_ibm.jl [options]

Options:
    --n_agents <Int>    Initial population size (default: 300)
    --days <Int>        Simulation run duration in days (default: 730)
    --grid <nx,ny>      Grid spatial dimensions (default: 50,50)
    --temp_base <Float> Mean bottom temperature in °C (default: 5.0)
    --temp_amp <Float>  Seasonal temperature amplitude in °C (default: 4.0)
    --dt <Float>        Simulation time step in days (default: 1.0)
    --seed <Int>        Random number generator seed (default: 1234)
    --plot              Generate and save visualization plots
    --animate, --video  Generate and record Agents.jl spatial animation video
    --gui, --interactive Launch real-time interactive GUI dashboard
    --output_dir <Path> Directory to save generated figures (default: outputs)
    --help, -h          Print this help message
"""

using Printf
using Random
using Statistics
using DataFrames
using CairoMakie

# Activate environment and load model_agent
using DrWatson
@quickactivate "model_agent"
using model_agent

"""
    parse_cli_args(args)

Parse command-line argument vector into a configuration dictionary.

Follows a strict three-tier precedence hierarchy:
1. Internal defaults (`default_ibm_configuration()`)
2. Configuration file overrides (default `configs/default.toml` + `configs/snowcrab.toml` or `--config=<path>`)
3. Explicit command-line arguments (highest priority, supporting `--key=value`)

# Arguments
- `args::Vector{String}`: Command line arguments from `ARGS`.

# Returns
`Dict{Symbol, Any}` containing simulation configuration.
"""
function parse_cli_args(args::Vector{String})
    # 1. Baseline internal defaults
    config = default_ibm_configuration()
    config[:help] = false
    config[:config_file] = ""
    config[:output_dir] = projectdir("outputs")
    config[:gui] = false

    # Helper function to extract (key, val, next_idx) from token or token pair
    function extract_key_val(idx::Int)
        tok = args[idx]
        if contains(tok, '=')
            parts = split(tok, '=', limit=2)
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

    # 2. First pass: Locate explicit --config=<path> or --config <path>
    custom_config_path = ""
    idx = 1
    while idx <= length(args)
        k_str, v_str, next_idx = extract_key_val(idx)
        if k_str == "config" || k_str == "c"
            custom_config_path = v_str
            break
        end
        idx = (next_idx > idx) ? (next_idx + 1) : (idx + 1)
    end

    # 3. Determine config file path (explicit path or auto-detected default)
    config_file_to_load = if !isempty(custom_config_path)
        custom_config_path
    else
        candidates = [
            joinpath(pwd(), "configs", "default.toml"),
            projectdir("configs", "default.toml"),
            joinpath(@__DIR__, "..", "configs", "default.toml"),
        ]
        found = ""
        for cand in candidates
            if isfile(cand)
                found = cand
                break
            end
        end
        # Also try to load species-specific config
        species_config = if !isempty(found)
            replace(found, "default.toml" => "snowcrab.toml")
        else
            ""
        end
        (found, species_config)
    end

    # 4. Load configuration file if located, overriding internal defaults
    default_config, species_config = config_file_to_load
    if !isempty(default_config) && isfile(default_config)
        config = load_ibm_configuration(default_config; base_config = config)
        config[:config_file] = default_config
    end
    # Load species-specific config if available
    if !isempty(species_config) && isfile(species_config)
        config = load_ibm_configuration(species_config; base_config = config)
        config[:species_config_file] = species_config
    end

    # 5. Second pass: Parse command-line argument overrides (highest priority)
    idx = 1
    while idx <= length(args)
        arg = args[idx]
        if arg == "--help" || arg == "-h"
            config[:help] = true
            idx += 1
            continue
        elseif arg == "--plot"
            config[:plot] = true
            idx += 1
            continue
        elseif arg == "--no-plot"
            config[:plot] = false
            idx += 1
            continue
        elseif arg == "--save"
            config[:save] = true
            idx += 1
            continue
        elseif arg == "--no-save"
            config[:save] = false
            
            idx += 1
            continue
        elseif arg == "--animate" || arg == "--video"
            config[:animate] = true
            idx += 1
            continue
        elseif arg == "--no-animate" || arg == "--no-video"
            config[:animate] = false
            idx += 1
            continue
        elseif arg == "--gui" || arg == "--interactive"
            config[:gui] = true
            idx += 1
            continue
        end

        k_str, v_str, next_idx = extract_key_val(idx)
        idx = (next_idx > idx) ? (next_idx + 1) : (idx + 1)

        if isempty(k_str)
            continue
        end

        if k_str == "config" || k_str == "c"
            # Handled in first pass
            continue
        elseif k_str == "help" || k_str == "h"
            config[:help] = true
        elseif k_str in ("model", "mode", "model_mode")
            config[:model] = lowercase(v_str)
        elseif k_str == "grid"
            parts = split(v_str, ',')
            if length(parts) == 2
                nx = tryparse(Int, strip(parts[1]))
                ny = tryparse(Int, strip(parts[2]))
                if nx !== nothing && ny !== nothing
                    config[:grid] = (nx, ny)
                end
            end
        elseif k_str == "save_file" || k_str == "save_results"
            config[:save_file] = v_str
            config[:save] = true
        elseif k_str == "output_dir"
            config[:output_dir] = v_str
        elseif k_str == "plot"
            config[:plot] = (lowercase(v_str) in ("true", "1", "yes"))
        elseif k_str == "save"
            config[:save] = (lowercase(v_str) in ("true", "1", "yes"))
        elseif k_str in ("animate", "video")
            config[:animate] = (lowercase(v_str) in ("true", "1", "yes"))
        elseif k_str in ("gui", "interactive")
            config[:gui] = (lowercase(v_str) in ("true", "1", "yes"))
        elseif k_str in ("animation_file", "anim_file")
            config[:animation_file] = v_str
        elseif k_str in ("anim_frames", "frames")
            parsed_frames = tryparse(Int, v_str)
            if parsed_frames !== nothing
                config[:anim_frames] = parsed_frames
            end
        elseif k_str == "anim_dt"
            parsed_dt = tryparse(Int, v_str)
            if parsed_dt !== nothing
                config[:anim_dt] = parsed_dt
            end
        elseif k_str in ("anim_fps", "fps")
            parsed_fps = tryparse(Int, v_str)
            if parsed_fps !== nothing
                config[:anim_fps] = parsed_fps
            end
        elseif k_str == "anim_field"
            config[:anim_field] = v_str
        elseif k_str == "param" || k_str == "set"
            pair = split(v_str, '=', limit=2)
            if length(pair) == 2
                pkey = Symbol(strip(pair[1]))
                pval_str = strip(pair[2])
                pval = tryparse(Float64, pval_str)
                config[pkey] = (pval === nothing) ? pval_str : pval
            end
        else
            sym_k = Symbol(k_str)
            int_val = tryparse(Int, v_str)
            if int_val !== nothing && !contains(v_str, '.')
                config[sym_k] = int_val
            else
                float_val = tryparse(Float64, v_str)
                if float_val !== nothing
                    config[sym_k] = float_val
                else
                    config[sym_k] = v_str
                end
            end

            # Reconcile parameter aliases
            if sym_k === :temp_base
                config[:temp] = config[sym_k]
            elseif sym_k === :temp
                config[:temp_base] = config[sym_k]
            elseif sym_k === :temp_amp || sym_k === :temp_amplitude
                config[:temp_amp] = config[sym_k]
                config[:temp_amplitude] = config[sym_k]
            elseif sym_k === :fishing_prob || sym_k === :fishing_probability_daily
                config[:fishing_prob] = config[sym_k]
                config[:fishing_probability_daily] = config[sym_k]
            elseif sym_k === :carrying_capacity || sym_k === :benthic_carrying_capacity
                config[:carrying_capacity] = config[sym_k]
                config[:benthic_carrying_capacity] = config[sym_k]
            elseif sym_k === :snapshot_day || sym_k === :survey_day || sym_k === :annual_snapshot_day
                config[:snapshot_day] = Float64(config[sym_k])
            elseif sym_k in (:frequency, :frequency_days, :output_frequency)
                config[:frequency_days] = Float64(config[sym_k])
                config[:output_frequency] = Float64(config[sym_k])
            end
        end
    end

    # Ensure canonical and alias keys are mutually populated
    config[:temp_base] = Float64(get(config, :temp_base, get(config, :temp, 5.0)))
    config[:temp] = config[:temp_base]
    config[:temp_amplitude] = Float64(get(config, :temp_amplitude, get(config, :temp_amp, 4.0)))
    config[:temp_amp] = config[:temp_amplitude]
    config[:fishing_probability_daily] = Float64(get(config, :fishing_probability_daily, get(config, :fishing_prob, 0.0)))
    config[:fishing_prob] = config[:fishing_probability_daily]
    config[:benthic_carrying_capacity] = Float64(get(config, :benthic_carrying_capacity, get(config, :carrying_capacity, 300.0)))
    config[:carrying_capacity] = config[:benthic_carrying_capacity]
    config[:frequency_days] = Float64(get(config, :frequency_days, get(config, :frequency, 365.0)))
    config[:frequency] = config[:frequency_days]
    config[:snapshot_day] = Float64(get(config, :snapshot_day, 274.0))

    return config
end

"""
    display_cli_help()

Print command-line usage instructions, configuration loading rules, and supported options.
"""
function display_cli_help()
    println("""
================================================================================
Organism Individual-Based Model (IBM) CLI Tool
================================================================================
Usage:
    julia --project=. scripts/organism_ibm.jl [OPTIONS]

Parameter Syntax:
    Parameters are specified using '=' (e.g. --key=value).

Configuration File Override:
    --config=<Path>, -c=<Path> Path to TOML configuration file (overrides internal
                               defaults; e.g. --config=snowcrab_ibm.config)
    If not specified, the tool automatically checks for 'snowcrab_ibm.config'
    in the current directory or project root as a baseline override.

Precedence Hierarchy:
    Internal Defaults -> Configuration File -> Command-Line Arguments

Options:
    --model=<simple|complex>  Model formulation: 'simple' (default) or 'complex'
    --n_agents=<Int>          Number of initial crabs (default: 300)
    --days=<Int>              Simulation duration in days (default: 730)
    --grid=<nx,ny>            Grid size in cells, comma-separated (default: 100,100)
    --temp_base=<Float>       Annual mean bottom temperature in °C (default: 5.0)
    --temp_amp=<Float>        Seasonal oscillation amplitude in °C (default: 4.0)
    --dt=<Float>              Time step in days (default: 1.0)
    --fishing_prob=<Float>    Daily harvest probability on legal males (default: 0.0)
    --carrying_capacity=<Flt> Target benthic carrying capacity (default: 300.0)
    --molt_season_start=<Flt> Day of year molt window begins (default: 15.0)
    --molt_season_end=<Flt>   Day of year molt window ends (default: 140.0)
    --seed=<Int>              RNG seed for deterministic runs (default: 1234)
    --param=<key>=<value>     Override any parameter (e.g. --param=food_max=12.0)
    --save                    Save annual production results to CSV (default: true)
    --no-save                 Disable saving annual production results to CSV
    --save_file=<Filename>    Filename/path for saved results (default: simulation_results.csv)
    --snapshot_day=<Float>    Day of calendar year for annual output snapshot (default: 274.0)
    --frequency=<Float>       Output frequency in days (default: 365.0 = 1 year)
    --plot                    Flag to generate and export PNG summary figures
    --no-plot                 Disable figure generation
    --animate, --video        Record spatial ABM animation video/gif using Agents.abmvideo
    --no-animate              Disable animation recording
    --animation_file=<Name>   Animation output filename (default: organism_animation.mp4)
    --anim_frames=<Int>       Total animation frames to record (default: 120)
    --anim_dt=<Int>           Days advanced per animation frame (default: 2)
    --anim_fps=<Int>          Playback framerate in fps (default: 15)
    --anim_field=<String>     Environmental background field (temperature or food)
    --output_dir=<Path>       Directory for saved figures and data (default: outputs)
    --help, -h                Show this help text
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
    total_orgs = hasproperty(summary, :total_organisms) ? summary.total_organisms : 0
    @printf(" Final Living Organisms : %d organisms\n", total_orgs)

    survival_pct = (init_agents > 0) ? (total_orgs / init_agents * 100.0) : 0.0
    @printf(" Net Population Change : %+.1f%%\n", (survival_pct - 100.0))
    println("-"^80)

    println(" [Key Output Indicators]")
    @printf("   - Total Number of Mature Females   : %4d\n", summary.total_mature_females)
    @printf("   - Total Number of Mature Males     : %4d\n", summary.total_mature_males)
    @printf("   - Total Mass of Fishable Males     : %8.3f kg (%.4f tonnes)\n",
            summary.total_fishable_male_mass_kg, summary.total_fishable_male_mass_kg / 1000.0)
    @printf("   - Total Mass of Mature Males       : %8.3f kg (%.4f tonnes)\n",
            summary.total_mature_male_mass_kg, summary.total_mature_male_mass_kg / 1000.0)
    @printf("   - Total Mass of Mature Females     : %8.3f kg (%.4f tonnes)\n",
            summary.total_mature_female_mass_kg, summary.total_mature_female_mass_kg / 1000.0)
    @printf("   - Total Egg Production             : %d eggs\n", summary.total_egg_production)
    println("-"^80)

    println(" [Stage Distribution]")
    has_agg = hasproperty(summary, :n_stage_0)
    if has_agg
        @printf("   - Stage 0 (Aggregate Eggs/Larvae)    : %7d\n", summary.n_stage_0)
        @printf("   - Stages 1 to 5 (Aggregate Cohorts)  : %7d\n", summary.n_stages_1_to_5)
        if hasproperty(summary, :count_by_aggregate_stage)
            for stg in 1:5
                cnt = get(summary.count_by_aggregate_stage, stg, 0)
                @printf("       * Stage %d                         : %7d\n", stg, cnt)
            end
        end
    end
    @printf("   - Planktonic Larvae (Zoea & Megalopa) : %7d\n", summary.n_larvae)
    @printf("   - Benthic Juveniles (Stage 4+)       : %7d\n", summary.n_juveniles)
    @printf("   - Mature Adults (Terminal Molt)       : %7d\n", summary.n_adults)
    @printf("       * Adult Males                     : %7d\n", summary.n_adult_males)
    @printf("       * Adult Females                   : %7d\n", summary.n_adult_females)
    println("-"^80)

    println(" [Sex Composition]")
    @printf("   - Males   : %7d\n", summary.n_males)
    @printf("   - Females : %7d\n", summary.n_females)
    sex_ratio = (summary.n_females > 0) ? (summary.n_males / summary.n_females) : 0.0
    @printf("   - Sex Ratio (M:F) : %.2f\n", sex_ratio)
    println("-"^80)

    println(" [Reproduction & Recruitment]")
    @printf("   - Ovigerous Females Brooding Eggs  : %7d", summary.n_ovigerous_females)
    if summary.n_adult_females > 0
        pct_ovig = summary.n_ovigerous_females / summary.n_adult_females * 100.0
        @printf(" (%.1f%% of adult females)\n", pct_ovig)
    else
        println(" (0.0%)\n")
    end
    @printf("   - Total Eggs Currently Brooded     : %d eggs\n", summary.total_eggs_brooded)
    @printf("   - Cumulative Eggs Spawned          : %d eggs\n", summary.total_eggs_spawned)
    @printf("   - Cumulative Benthic Recruits      : %d recruits (Stage 4)\n", summary.total_recruits_settled)
    println("-"^80)

    println(" [Biomass & Stock Assessment Metrics]")
    @printf("   - Total Living Biomass             : %8.3f kg (%.4f tonnes)\n",
            summary.total_biomass_kg, summary.total_biomass_kg / 1000.0)
    @printf("   - Female Spawning Stock Biomass    : %8.3f kg (%.4f tonnes)\n",
            summary.female_ssb_kg, summary.female_ssb_kg / 1000.0)
    @printf("   - Mature Male Biomass (MMB)        : %8.3f kg (%.4f tonnes)\n",
            summary.male_mature_biomass_kg, summary.male_mature_biomass_kg / 1000.0)
    @printf("   - Total Spawning Stock Biomass(SSB): %8.3f kg (%.4f tonnes)\n",
            summary.total_ssb_kg, summary.total_ssb_kg / 1000.0)
    @printf("   - Legal Commercial Male Biomass    : %8.3f kg (%.4f tonnes)\n",
            summary.legal_biomass_kg, summary.legal_biomass_kg / 1000.0)
    @printf("   - Mean Individual Body Mass        : %8.2f grams\n",
            summary.mean_mass)
    println("-"^80)

    println(" [Size and Stage Structure]")
    @printf("   - Mean Carapace Size (All Organisms): %.2f mm\n", summary.mean_size)
    has_sex_stage = hasproperty(summary, :count_by_stage_male)
    if has_sex_stage
        println("   - Stage Breakdown (Counts, Mean Size, and Mean Mass by Sex):")
        println("       Stage |   Total |   Males | Females | Male Size | Female Size | Mean Mass(g)")
        println("       ------+---------+---------+---------+-----------+-------------+-------------")
        for stg in sort(collect(keys(summary.size_by_stage)))
            mean_size = summary.size_by_stage[stg]
            count_stg = summary.count_by_stage[stg]
            c_m = get(summary.count_by_stage_male, stg, 0)
            c_f = get(summary.count_by_stage_female, stg, 0)
            sz_m = get(summary.size_by_stage_male, stg, 0.0)
            sz_f = get(summary.size_by_stage_female, stg, 0.0)
            m_mass = hasproperty(summary, :mass_by_stage) ?
                get(summary.mass_by_stage, stg, 0.0) : 0.0
            @printf("         %2d  | %7d | %7d | %7d |    %6.2f |      %6.2f |      %7.2f\n",
                    stg, count_stg, c_m, c_f, sz_m, sz_f, m_mass)
        end
    else
        println("   - Stage Mean Carapace Size Breakdown:")
        println("       Stage | Living Count | Mean Carapace Size (mm)")
        println("       -------+--------------+-------------------------")
        for stg in sort(collect(keys(summary.size_by_stage)))
            mean_size = summary.size_by_stage[stg]
            count_stg = summary.count_by_stage[stg]
            @printf("         %2d   |     %4d     |        %6.2f\n",
                    stg, count_stg, mean_size)
        end
    end
    println("="^80)
    println()
end

"""
    print_save_step_outputs(ann_df::DataFrame)

Print formatted table of demographic, abundance, and biomass indicators across
all annual or periodic save steps.
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

Generate and save all diagnostic figures and the executive simulation
dashboard into `output_dir`.
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
    main()

Entry point for the organism IBM CLI simulation execution.
"""
function main()
    config = parse_cli_args(ARGS)

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
    cfg_display = isempty(config[:config_file]) ? "Internal Defaults" : config[:config_file]
    println("   - Config Source   : $(cfg_display)")
    features = [
        (:use_energetics, "Energetics"),
        (:use_thermal_stress, "Thermal Stress"),
        (:use_food_dynamics, "Food Dynamics"),
        (:use_habitat_taxis, "Habitat Taxis"),
        (:use_cannibalism, "Cannibalism"),
        (:use_senescence, "Senescence")
    ]
    for (f, name) in features
        status = get(config, f, false) ? "[ON]" : "[OFF]"
        println("     * $(lpad(name, 15)) : $status")
    end
    println(" Parameters:")
    @printf("   - Agents          : %d\n", config[:n_agents])
    @printf("   - Days            : %d (%.2f years)\n", config[:days], config[:days] / 365.0)
    @printf("   - Grid Size       : %d x %d\n", config[:grid][1], config[:grid][2])
    @printf("   - Temperature     : %.1f ± %.1f °C\n", config[:temp_base], config[:temp_amp])
    @printf("   - Harvest Prob    : %.4f daily\n", config[:fishing_prob])
    @printf("   - Carrying Cap    : %.1f organisms\n", config[:carrying_capacity])
    @printf("   - Random Seed     : %d\n", config[:seed])
    @printf("   - Save Results    : %s (%s)\n", config[:save] ? "Yes" : "No", config[:save_file])
    @printf("   - Output Frequency: %.1f days (%.2f years)\n",
            Float64(config[:frequency_days]), Float64(config[:frequency_days]) / 365.0)
    @printf("   - Snapshot Day    : Day %.1f (~1 September)\n", Float64(config[:snapshot_day]))
    @printf("   - Plot Figures    : %s\n", config[:plot] ? "Yes" : "No")
    @printf("   - Animate Video   : %s (%s)\n",
            config[:animate] ? "Yes" : "No", config[:animation_file])
    println("="^80)

    # Construct complete IBMParameters from full configuration dictionary
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

    # Configure live snapshot day progress reporting
    snapshot_day = Float64(config[:snapshot_day])
    frequency_days = Float64(config[:frequency_days])
    first_snap = (snapshot_day == 274.0 && frequency_days < 274.0) ?
        frequency_days : snapshot_day
    snap_days = Float64[]
    curr_t = first_snap
    while curr_t <= Float64(config[:days])
        push!(snap_days, curr_t)
        curr_t += frequency_days
    end

    snap_count = Ref(0)
    snap_callback = function(m)
        t = m.current_time
        if any(d -> abs(t - d) < (m.dt / 2.0), snap_days)
            snap_count[] += 1
            if snap_count[] == 1
                println()
                println("="^116)
                println("                                LIVE SNAPSHOT DAY POPULATION INDICATORS")
                println("="^116)
                println(
                    " Yr |   Day  | Total N | Adult F | Adult M | Legal N | " *
                    "Total(kg) | SSB F(kg) | MMB(kg) | Legal(kg) | Stage 0 | Stages 1-5"
                )
                println(
                    "----+--------+---------+---------+---------+---------+-" *
                    "----------+-----------+---------+-----------+---------+-----------"
                )
            end
            s = summarize_population(m)
            @printf(
                " %2d | %6.0f | %7d | %7d | %7d | %7d | %9.2f | %9.2f | %7.2f | %9.2f | %7d | %10d\n",
                snap_count[],
                t,
                s.total_organisms,
                s.n_adult_females,
                s.n_adult_males,
                s.legal_numbers,
                s.total_biomass_kg,
                s.female_ssb_kg,
                s.male_mature_biomass_kg,
                s.legal_biomass_kg,
                s.n_stage_0,
                s.n_stages_1_to_5
            )
            flush(stdout)
        end
    end

    println(" Executing simulation across $(config[:days]) time steps...")
    agent_df, model_df = run_simulation(
        model,
        config[:days];
        adata = adata,
        mdata = mdata,
        step_callback = snap_callback
    )
    if snap_count[] > 0
        println("="^116)
        println()
    end
    elapsed = time() - t_start

    summary = summarize_population(model)
    print_simulation_report(summary, config[:n_agents], config[:days], elapsed)

    # Extract and display periodic save-step metrics
    ann_df = extract_annual_production(
        agent_df,
        model_df;
        frequency_days = Float64(config[:frequency_days]),
        snapshot_day = Float64(config[:snapshot_day])
    )
    print_save_step_outputs(ann_df)

    if config[:save]
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

    anim_file_path = ""
    if config[:animate]
        mkpath(config[:output_dir])
        anim_file_path = isabspath(config[:animation_file]) ?
            config[:animation_file] : joinpath(config[:output_dir], config[:animation_file])
        println(" Recording Agents.jl simulation animation to: $(anim_file_path)")
        t_anim = time()
        anim_model = initialize_model(
            N_agents = config[:n_agents],
            dims = config[:grid],
            dt_days = config[:dt],
            params = params,
            rng_seed = config[:seed]
        )
        animate_organism_simulation(
            anim_model;
            filepath = anim_file_path,
            frames = Int(config[:anim_frames]),
            dt = Int(config[:anim_dt]),
            framerate = Int(config[:anim_fps]),
            heat_field = Symbol(config[:anim_field])
        )
        @printf(" Animation video successfully recorded in %.2f seconds.\n",
                time() - t_anim)
    end

    return (
        model = model,
        agent_df = agent_df,
        model_df = model_df,
        summary = summary,
        config = config,
        animation_path = anim_file_path
    )
end

# If invoked as a standalone script from terminal, execute main()
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
