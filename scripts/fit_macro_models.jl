"""
Post-hoc macro-ecological population dynamic model fitting CLI.

Fits discrete-time Schaefer logistic surplus production models and
Deriso-Schnute delay-difference models to simulation results (from a specified file),
evaluating both the whole population and the fishable component (legal adult males).

Usage:
    julia --project=. scripts/fit_macro_models.jl [options]

Options:
    --file <path>        Path to simulation results (.csv or .jls) [REQUIRED]
    --lag <Int>          Delay-difference recruitment lag in years (default: 1)
    --retro              Run retrospective pattern analysis and compute Mohn's rho
    --peels <Int>        Number of retrospective peels for reanalysis (default: 5)
    --output_dir <path>  Output directory to save plots (default: outputs)
    --plot               Generate and save diagnostic figures (default: true)
    --help, -h           Show this help message
"""

using DataFrames
using CairoMakie
using Printf
using Random
using Statistics

using TethysBio

function parse_macro_args(args::Vector{String})
    config = Dict{Symbol, Any}(
        :file => "",
        :lag => 1,
        :retro => false,
        :peels => 5,
        :leslie => false,
        :output_dir => projectdir("outputs"),
        :plot => true,
        :help => false
    )

    idx = 1
    while idx <= length(args)
        arg = args[idx]
        if arg == "--help" || arg == "-h"
            config[:help] = true
            idx += 1
            continue
        elseif arg == "--retro"
            config[:retro] = true
            idx += 1
            continue
        elseif arg == "--leslie"
            config[:leslie] = true
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
        end

        k_str = ""
        v_str = ""
        if contains(arg, '=')
            parts = split(arg, '=', limit=2)
            k_str = lstrip(parts[1], '-')
            v_str = String(parts[2])
            idx += 1
        elseif startswith(arg, '-')
            k_str = lstrip(arg, '-')
            if idx < length(args) && !startswith(args[idx + 1], '-')
                idx += 1
                v_str = args[idx]
            else
                v_str = "true"
            end
            idx += 1
        else
            idx += 1
            continue
        end

        if k_str == "file"
            config[:file] = v_str
        elseif k_str == "lag"
            config[:lag] = parse(Int, v_str)
        elseif k_str == "peels"
            config[:peels] = parse(Int, v_str)
        elseif k_str == "output_dir"
            config[:output_dir] = v_str
        end
    end
    return config
end


function display_help()
    println("="^75)
    println("Organism Macro-Ecological Dynamic Model Fitting & Retrospective CLI")
    println("="^75)
    println("Usage:")
    println("    julia --project=. scripts/fit_macro_models.jl --file=<path> [OPTIONS]")
    println()
    println("Options:")
    println("    --file=<path>        Path to simulation output (.csv or .jls) [REQUIRED]")
    println("    --lag=<Int>          Delay-difference recruitment lag in years (default: 1)")
    println("    --retro              Perform retrospective analysis & Mohn's rho")
    println("    --peels=<Int>        Number of retrospective peels (default: 5)")
    println("    --leslie             Fit generalized Leslie stage-transition model & compare")
    println("    --output_dir=<path>  Output directory for figures (default: outputs)")
    println("    --plot / --no-plot   Flag to control figure export (default: true)")
    println("    --help, -h           Display this help text")
    println("="^75)
end

function main()
    config = parse_macro_args(ARGS)

    if config[:help] || isempty(config[:file])
        display_help()
        if isempty(config[:file]) && !config[:help]
            println("\n[Error] Missing required parameter: --file <path_to_results>")
            return 1
        end
        return 0
    end

    input_file = config[:file]
    if !isfile(input_file)
        println("\n[Error] Specified simulation results file not found: $(input_file)")
        return 1
    end

    println("="^75)
    println(" Loading simulation results from: $(input_file)")
    println(" Fitting Schaefer Logistic and Delay-Difference models (lag = $(config[:lag]) yr)...")
    println("="^75)

    macro_fits = fit_population_dynamics(input_file; lag = config[:lag])
    report = summarize_population_dynamics(macro_fits)
    println(report)

    if config[:plot]
        mkpath(config[:output_dir])
        fig_path = joinpath(config[:output_dir], "macro_population_dynamics.png")
        p = plot_population_dynamics_fits(macro_fits)
        savefig(p, fig_path)
        println(" Diagnostic 4-panel figure saved to: $(fig_path)")
    end

    # Retrospective analysis
    if config[:retro]
        println("\n" * "="^75)
        println(" Conducting retrospective reanalysis (peels = $(config[:peels]))...")
        println("="^75)
        try
            retro_res = retrospective_analysis(
                input_file;
                n_peels = config[:peels],
                lag = config[:lag]
            )
            retro_report = summarize_retrospective(retro_res)
            println(retro_report)

            if config[:plot]
                retro_fig_path = joinpath(config[:output_dir], "macro_retrospective.png")
                p_retro = plot_retrospective_analysis(retro_res)
                savefig(p_retro, retro_fig_path)
                println(" Retrospective 4-panel diagnostic figure saved to: $(retro_fig_path)")
            end
        catch err
            println(" [Error during retrospective analysis]: ", err)
        end
    end

    # Generalized Leslie Model & 3-Way Macro Comparison
    if config[:leslie]
        println("\n" * "="^75)
        println(" Fitting Generalized Leslie Stage-Structured Transition Matrix Model...")
        println("="^75)
        try
            stage_mat, catch_mat, yrs = extract_annual_stage_matrix(input_file)
            leslie_fit = fit_generalized_leslie_model(stage_mat, catch_mat; years = yrs)
            leslie_report = summarize_leslie_model(leslie_fit)
            println(leslie_report)

            if config[:plot]
                mkpath(config[:output_dir])
                stages_fig_path = joinpath(config[:output_dir], "stage_specific_dynamics.png")
                p_stages = plot_stage_specific_dynamics(leslie_fit)
                savefig(p_stages, stages_fig_path)
                println(" Stage-specific 4-panel diagnostic figure saved to: $(stages_fig_path)")
            end

            println("\n" * "="^75)
            println(" Conducting Three-Way Macro Model Comparison (Fishable Stock)...")
            println("="^75)
            macro_comp = compare_macro_models(input_file, leslie_fit; lag = config[:lag])
            println(summarize_macro_comparison(macro_comp))

            if config[:plot]
                comp_fig_path = joinpath(config[:output_dir], "fishable_macro_comparison.png")
                p_comp = plot_macro_model_comparison(macro_comp)
                savefig(p_comp, comp_fig_path)
                println(" Comparative 4-panel diagnostic figure saved to: $(comp_fig_path)")
            end
        catch err
            println(" [Error during generalized Leslie / comparison analysis]: ", err)
        end
    end

    return 0
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
