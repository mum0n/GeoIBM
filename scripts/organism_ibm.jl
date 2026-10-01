"""
Command-Line Interface (CLI) for the Organism Individual-Based Model (IBM).

Usage:
    julia --project=. scripts/organism_ibm.jl --key=value [options]

Parameter names are matched exactly against the fields declared in
`configs/default.toml` and the parameter structs. There are no abbreviations.
Run with `--help` for the full list of flags and common parameters.

Common options:
    --config <Path>     Configuration file (TOML format)
    --n_agents <Int>    Initial population size (default: 300)
    --days <Int>        Simulation run duration in days (default: 730)
    --grid <nx,ny>      Grid spatial dimensions (default: 100,100)
    --temp_base <Float> Mean bottom temperature in Â°C (default: 5.0)
    --temp_amplitude <Float>  Seasonal temperature amplitude in Â°C (default: 4.0)
    --dt <Float>        Simulation time step in days (default: 1.0)
    --seed <Int>        Random number generator seed (default: 1234)
    --plot <Bool>       Generate and save visualization plots
    --animate <Bool>    Generate and record a spatial animation
    --gui <Bool>        Launch the real-time interactive GUI dashboard
    --output_dir <Path> Directory to save generated figures (default: outputs)
    --help              Print this help message
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

# Argument parsing is provided by `model_agent.parse_cli_args`, which owns the
# canonical alias table and configuration discovery. It is re-exported here so
# `scripts/organism_ibm.jl` and the package stay on a single implementation.
const parse_cli_args = model_agent.parse_cli_args
const display_cli_help = model_agent.display_cli_help

# The report printers and the plot wrapper live in `model_agent`. They were
# duplicated here as full copies, which is how the script's copy of
# `print_save_step_outputs` kept printing the old `Yr`/`time` columns after the
# package's had been corrected to `sample_index`/`day`. Alias them, exactly as
# `parse_cli_args` and `display_cli_help` are aliased above.
const print_simulation_report = model_agent.print_simulation_report
const print_save_step_outputs = model_agent.print_save_step_outputs
const generate_summary_plots = model_agent.generate_summary_plots

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
    @printf("   - Temperature     : %.1f Â± %.1f Â°C\n", config[:temp_base], config[:temp_amplitude])
    @printf("   - Harvest Prob    : %.4f daily\n", config[:fishing_probability_daily])
    @printf("   - Carrying Cap    : %.1f organisms\n", config[:benthic_carrying_capacity])
    @printf("   - Random Seed     : %d\n", config[:seed])
    @printf("   - Save Results    : %s (%s)\n", config[:save] ? "Yes" : "No", config[:save_file])
    @printf("   - Output Frequency: %.1f days (%.2f years)\n",
            Float64(config[:frequency_days]),
            Float64(config[:frequency_days]) / Float64(config[:days_per_year]))
    @printf("   - Snapshot Day    : Day %.1f (~1 September)\n", Float64(config[:snapshot_day]))
    @printf("   - Plot Figures    : %s\n", config[:plot] ? "Yes" : "No")
    @printf("   - Animate Video   : %s (%s)\n",
            config[:animate] ? "Yes" : "No", config[:animation_file])
    println("="^80)

    # Construct complete IBMParameters from full configuration dictionary
    params = IBMParameters(config)

    t_start = time()
    model = initialize_model(
        n_agents = config[:n_agents],
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

    # Configure live snapshot day progress reporting. Uses the same target
    # computation as extract_annual_production, so the progress lines land on the
    # instants the data rows actually sample.
    snap_days = annual_production_targets(
        Float64(config[:days]);
        dt_year = Float64(config[:days_per_year]),
        frequency_days = Float64(config[:frequency_days]),
        snapshot_day = Float64(config[:snapshot_day])
    )

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
        dt_year = Float64(config[:days_per_year]),
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
        generated = generate_summary_plots(
            model,
            agent_df,
            config[:output_dir];
            model_df = model_df
        )
        println(" Exported $(length(generated)) figure(s).")
    end

    anim_file_path = ""
    if config[:animate]
        mkpath(config[:output_dir])
        anim_file_path = isabspath(config[:animation_file]) ?
            config[:animation_file] : joinpath(config[:output_dir], config[:animation_file])
        println(" Recording Agents.jl simulation animation to: $(anim_file_path)")
        t_anim = time()
        anim_model = initialize_model(
            n_agents = config[:n_agents],
            grid = config[:grid],
            dt = config[:dt],
            params = params,
            seed = config[:seed]
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
