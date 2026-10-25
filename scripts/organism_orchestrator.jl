#!/usr/bin/env julia
"""
================================================================================
Organism IBM SIMULATION ORCHESTRATOR & CALIBRATOR
================================================================================
Scientific Agent-Based Modeling System for Biological Populations
File: scripts/organism_orchestrator.jl

Automates systematic exploration, sensitivity analysis, and multi-criteria
calibration of the organism individual-based model (IBM).

The orchestrator varies user-specified model parameters (over discrete grids,
intervals, or random samples) while keeping all other parameters fixed to baseline
values (from `organism_ibm.config`). Simulations are executed until predefined
multi-attribute biological and ecological criteria are achieved:
  1. Population Persistence: Non-extinction (living population >= N_min).
  2. Population Bounds: Non-explosion (living population <= N_max).
  3. Steady Size Structure: Relative drift in mean carapace width <= tolerance.
  4. Maturity Ogive Fidelity: Empirical maturity schedule matches empirical
     logistic curves (L50 ≈ 91.8 mm males, 51.6 mm females).
  5. Growth Schedule Fidelity: Stage sizes match empirical reference medians
     from Sainte-Marie and Alunno-Bruscia growth schedules.
================================================================================
"""

using TethysBio

using Printf
using Random
using Statistics
using DataFrames
using Agents

"""
    parse_vary_string(vary_str::String)

Parse parameter variation string into a dictionary mapping parameter symbols
to vectors of candidate values.

# Supported Formats:
- Comma-separated list: `"temp_base=3.5,4.5,5.5;food_max=8.0,10.0"`
- Range specification: `"temp_base=3.0:1.0:5.0;benthic_carrying_capacity=200.0,300.0"`

# Returns
`Dict{Symbol, Vector{Float64}}` of parameter candidates.
"""
function parse_vary_string(vary_str::String)
    vary_dict = Dict{Symbol, Vector{Float64}}()
    if isempty(strip(vary_str))
        return vary_dict
    end

    # Semicolon or double-pipe separated parameter groups
    groups = split(vary_str, r"[;\|]")
    for grp in groups
        s = strip(grp)
        if isempty(s)
            continue
        end
        parts = split(s, '=', limit = 2)
        if length(parts) != 2
            continue
        end
        param_name = Symbol(strip(parts[1]))
        val_spec = strip(parts[2])

        vals = Float64[]
        if contains(val_spec, ':')
            r_parts = split(val_spec, ':')
            if length(r_parts) == 3
                start_v = tryparse(Float64, strip(r_parts[1]))
                step_v = tryparse(Float64, strip(r_parts[2]))
                stop_v = tryparse(Float64, strip(r_parts[3]))
                if start_v !== nothing && step_v !== nothing && stop_v !== nothing
                    vals = collect(start_v:step_v:stop_v)
                end
            elseif length(r_parts) == 2
                start_v = tryparse(Float64, strip(r_parts[1]))
                stop_v = tryparse(Float64, strip(r_parts[2]))
                if start_v !== nothing && stop_v !== nothing
                    vals = collect(range(start_v, stop_v, length = 5))
                end
            end
        else
            for v_tok in split(val_spec, ',')
                pv = tryparse(Float64, strip(v_tok))
                if pv !== nothing
                    push!(vals, pv)
                end
            end
        end

        if !isempty(vals)
            vary_dict[param_name] = vals
        end
    end

    return vary_dict
end

"""
    build_parameter_grid(vary_dict::Dict{Symbol, Vector{Float64}})

Construct the full Cartesian product grid of parameter candidate combinations.

# Returns
`Vector{Dict{Symbol, Float64}}` where each element represents a trial parameter set.
"""
function build_parameter_grid(vary_dict::Dict{Symbol, Vector{Float64}})
    if isempty(vary_dict)
        return [Dict{Symbol, Float64}()]
    end

    keys_vec = collect(keys(vary_dict))
    val_arrays = [vary_dict[k] for k in keys_vec]

    combinations = [Dict{Symbol, Float64}() for _ in 1:prod(length.(val_arrays))]

    indices = CartesianIndices(Tuple(length.(val_arrays)))
    for (comb_idx, c_idx) in enumerate(indices)
        d = Dict{Symbol, Float64}()
        for (k_idx, p_key) in enumerate(keys_vec)
            d[p_key] = val_arrays[k_idx][c_idx[k_idx]]
        end
        combinations[comb_idx] = d
    end

    return combinations
end

"""
    parse_orchestrator_cli(args::Vector{String})

Parse command-line options for the organism simulation orchestrator.
"""
function parse_orchestrator_cli(args::Vector{String})
    config = Dict{Symbol, Any}(
        :config_file => "",
        :vary_str => "",
        :strategy => :grid,
        :max_trials => 20,
        :days => 730,
        :n_agents => 300,
        :stop_on_first_pass => true,
        :mode => :direct,
        :output_dir => projectdir("outputs"),
        :save_best => "orchestrated_run.toml",
        :plot_best => true,
        :help => false
    )

    function extract_token(idx::Int)
        tok = args[idx]
        if contains(tok, '=')
            parts = split(tok, '=', limit = 2)
            k = lstrip(parts[1], '-')
            v = String(parts[2])
            return (k, v, idx)
        elseif startswith(tok, '-')
            k = lstrip(tok, '-')
            if idx < length(args) && !startswith(args[idx + 1], '-')
                return (k, args[idx + 1], idx + 1)
            else
                return (k, "true", idx)
            end
        else
            return ("", "", idx)
        end
    end

    idx = 1
    while idx <= length(args)
        arg = args[idx]
        if arg in ("--help", "-h")
            config[:help] = true
            idx += 1
            continue
        elseif arg == "--stop_on_first_pass"
            config[:stop_on_first_pass] = true
            idx += 1
            continue
        elseif arg == "--no-stop_on_first_pass"
            config[:stop_on_first_pass] = false
            idx += 1
            continue
        elseif arg == "--plot_best"
            config[:plot_best] = true
            idx += 1
            continue
        elseif arg == "--no-plot_best"
            config[:plot_best] = false
            idx += 1
            continue
        end

        k, v, next_idx = extract_token(idx)
        idx = (next_idx > idx) ? (next_idx + 1) : (idx + 1)

        if isempty(k)
            continue
        end

        if k in ("config", "c")
            config[:config_file] = v
        elseif k in ("vary", "parameters", "params")
            config[:vary_str] = v
        elseif k in ("range",)
            current = config[:vary_str]
            config[:vary_str] = isempty(current) ? v : (current * ";" * v)
        elseif k in ("strategy",)
            config[:strategy] = Symbol(lowercase(v))
        elseif k in ("max_trials", "trials")
            parsed_t = tryparse(Int, v)
            if parsed_t !== nothing
                config[:max_trials] = parsed_t
            end
        elseif k in ("days",)
            parsed_d = tryparse(Int, v)
            if parsed_d !== nothing
                config[:days] = parsed_d
            end
        elseif k == "n_agents"
            parsed_a = tryparse(Int, v)
            if parsed_a !== nothing
                config[:n_agents] = parsed_a
            end
        elseif k in ("mode",)
            config[:mode] = Symbol(lowercase(v))
        elseif k in ("save_best",)
            config[:save_best] = v
        elseif k in ("output_dir",)
            config[:output_dir] = v
        elseif k in ("help", "h")
            config[:help] = true
        end
    end

    return config
end

"""
    display_orchestrator_help()

Print command-line help and usage examples for the simulation orchestrator.
"""
function display_orchestrator_help()
    println("""
================================================================================
Organism IBM SIMULATION ORCHESTRATOR & CALIBRATOR
================================================================================
Usage:
    julia --project=. scripts/organism_orchestrator.jl [OPTIONS]

Parameter Variation Syntax:
    --vary="<param1>=<v1>,<v2>,...;<param2>=<v3>,<v4>,..."
    --range="<param>=<min>:<step>:<max>"

Examples:
    # 1. Vary temperature baseline and food abundance over a 2x2 parameter grid:
    julia --project=. scripts/organism_orchestrator.jl --vary="temp_base=3.5,5.0;food_max=8.0,12.0"

    # 2. Step through temperature baseline across 3 to 6 °C:
    julia --project=. scripts/organism_orchestrator.jl --range="temp_base=3.0:1.0:6.0" --days=1095

    # 3. Explore multi-parameter space and continue across all trials without early stop:
    julia --project=. scripts/organism_orchestrator.jl \\
        --vary="temp_base=4.0,5.0;food_max=8.0,12.0;benthic_carrying_capacity=250.0,350.0" \\
        --max_trials=10 --no-stop_on_first_pass

Multi-Attribute Biological Stopping Criteria:
    - Persistence: Living population >= 10% of carrying capacity (no extinction).
    - Bounds: Living population <= 300% of carrying capacity (no explosion).
    - Steady Size Structure: Relative shift in mean carapace width <= 20%.
    - Maturity Ogive Fidelity: Empirical maturity schedule matches theoretical
      logistic curves (RMSE <= 0.35).
    - Growth Schedule Fidelity: Stage sizes match empirical reference medians (RMSE <= 15.0 mm).

Options:
    --config=<Path>, -c=<Path> Base configuration file (default: organism_ibm.config)
    --vary=<String>            Parameters to vary with candidate values
    --range=<String>           Parameter range specification (key=min:step:max)
    --strategy=<grid|random>   Search strategy (default: grid)
    --max_trials=<Int>         Maximum parameter trials to evaluate (default: 20)
    --days=<Int>               Simulation duration per trial in days (default: 730)
    --n_agents=<Int>           Initial population size per trial (default: 300)
    --stop_on_first_pass       Stop immediately when all criteria pass (default: true)
    --no-stop_on_first_pass    Evaluate all candidate combinations to find global optimum
    --mode=<direct|cli>        Execution mode: direct in-process (fastest) or CLI subprocess
    --save_best=<Filename>     Output filename for best configuration (default: organism_orchestrated.config)
    --plot_best                Export diagnostic figures for the winning model (default: true)
    --no-plot_best             Disable diagnostic figure generation
    --output_dir=<Path>        Target directory for results (default: outputs)
    --help, -h                 Show this help text
================================================================================
""")
end

"""
    execute_trial(base_config, trial_params, trial_num, total_trials, days, n_agents)

Execute a single simulation trial with `trial_params` overriding `base_config`,
and evaluate multi-attribute biological and ecological criteria.
"""
function execute_trial(
    base_config::Dict{Symbol, Any},
    trial_params::Dict{Symbol, Float64},
    trial_num::Int,
    total_trials::Int,
    days::Int,
    n_agents::Int
)
    # 1. Merge baseline configuration with trial overrides
    merged_config = copy(base_config)
    trial_keys = Set{Symbol}()
    for (k, v) in trial_params
        merged_config[k] = v
        push!(trial_keys, k)
    end
    merged_config[:days] = days
    merged_config[:n_agents] = n_agents
    merged_config[:save] = false
    merged_config[:plot] = false
    push!(trial_keys, :days, :n_agents, :save, :plot)
    # Trial parameters outrank the baseline, so name them as the explicit layer.
    sync_grid_dimensions!(merged_config; explicit = trial_keys)

    # 2. Build model parameters and initialize simulation
    params = IBMParameters(merged_config)
    seed = get(merged_config, :seed, 1234) + trial_num

    model = initialize_model(
        n_agents = n_agents,
        grid = merged_config[:grid],
        dt = merged_config[:dt],
        params = params,
        seed = seed
    )

    adata = default_agent_observations(params)
    mdata = [
        :total_eggs_spawned,
        :total_recruits_settled,
        :total_catch_numbers
    ]

    # 3. Execute simulation
    t_start = time()
    agent_df, model_df = run_simulation(model, days; adata = adata, mdata = mdata)
    elapsed = time() - t_start

    # 4. Evaluate multi-attribute criteria
    crit = evaluate_population_criteria(model, agent_df)

    # 5. Format parameters string for summary display
    p_str = join(["$(k)=$(v)" for (k, v) in trial_params], ", ")
    if isempty(p_str)
        p_str = "Baseline Settings"
    end

    status_str = crit.passed_all ? "PASSED" : "FAILED"
    @printf("[Trial %02d/%02d] %-35s | N: %4d | Drift: %4.1f%% | MatRMSE: %4.2f | GrwRMSE: %4.1f mm | %-6s (Score: %5.1f, %.1fs)\n",
            trial_num, total_trials, p_str,
            crit.n_living, crit.size_drift * 100.0, crit.maturity_rmse,
            crit.growth_rmse, status_str, crit.composite_score, elapsed)

    return (
        trial_num = trial_num,
        params = trial_params,
        merged_config = merged_config,
        model = model,
        agent_df = agent_df,
        model_df = model_df,
        criteria = crit,
        elapsed = elapsed
    )
end

"""
    main()

Main execution routine for the organism simulation orchestrator.
"""
function main_orchestrator()
    cli_opts = parse_orchestrator_cli(ARGS)

    if cli_opts[:help]
        display_orchestrator_help()
        return 0
    end

    println("="^85)
    println("      ORGANISM IBM SIMULATION ORCHESTRATOR & CALIBRATOR ENGINE       ")
    println("="^85)

    # 1. Locate and load base configuration
    base_cfg_path = cli_opts[:config_file]
    if isempty(base_cfg_path)
        candidates = [
            joinpath(pwd(), "organism_ibm.config"),
            projectdir("organism_ibm.config"),
            joinpath(@__DIR__, "organism_ibm.config"),
            projectdir("scripts", "organism_ibm.config")
        ]
    else
        candidates = [base_cfg_path]
    end

    for c in candidates
        if isfile(c)
            base_cfg_path = c
            break
        end
    end

    base_config = if !isempty(base_cfg_path) && isfile(base_cfg_path)
        println(" Base Configuration File: $(base_cfg_path)")
        load_ibm_configuration(base_cfg_path)
    else
        println(" Base Configuration File: Internal Package Defaults")
        default_ibm_configuration()
    end

    # 2. Parse parameter variation specifications
    vary_dict = parse_vary_string(cli_opts[:vary_str])
    if isempty(vary_dict)
        println("\n [Notice] No parameter variation specified via --vary or --range.")
        println(" Providing default sensitivity exploration on (temp_base, food_max)...")
        vary_dict = Dict(
            :temp_base => [4.0, 5.0, 6.0],
            :food_max => [8.0, 10.0, 12.0]
        )
    end

    println(" Parameters Under Orchestration:")
    for (k, vals) in vary_dict
        @printf("   - %-25s: %d candidate values %s\n", string(k), length(vals), string(vals))
    end

    # 3. Construct trial candidate combinations
    combinations = build_parameter_grid(vary_dict)
    total_grid_size = length(combinations)
    if cli_opts[:strategy] == :random
        Random.shuffle!(combinations)
    end

    max_trials = min(cli_opts[:max_trials], length(combinations))
    trials_to_run = combinations[1:max_trials]

    println(" Trial Strategy          : $(cli_opts[:strategy])")
    println(" Candidate Combinations  : $(total_grid_size) total ($(max_trials) scheduled)")
    println(" Duration Per Trial      : $(cli_opts[:days]) days (≈ $(round(cli_opts[:days]/365.0, digits=2)) years)")
    println(" Initial Population      : $(cli_opts[:n_agents]) organisms")
    println(" Stop on First Pass      : $(cli_opts[:stop_on_first_pass] ? "Yes (Early Termination)" : "No (Exhaustive Optimization)")")
    println("-"^85)
    println(" Executing Simulation Trials...")
    println("-"^85)

    trial_results = []
    best_result = nothing

    for (t_idx, p_set) in enumerate(trials_to_run)
        res = execute_trial(
            base_config,
            p_set,
            t_idx,
            max_trials,
            cli_opts[:days],
            cli_opts[:n_agents]
        )
        push!(trial_results, res)

        if best_result === nothing || res.criteria.composite_score < best_result.criteria.composite_score
            best_result = res
        end

        if cli_opts[:stop_on_first_pass] && res.criteria.passed_all
            println("-"^85)
            println(" [SUCCESS] Target criteria achieved at Trial $(t_idx)! Terminating early.")
            break
        end
    end

    println("="^85)
    println("                   ORCHESTRATION SUMMARY REPORT                       ")
    println("="^85)
    println(" Trial | Living N | Size Drift | Mat RMSE | Growth RMSE | Status | Score | Parameters")
    println("-------+----------+------------+----------+-------------+--------+-------+---------------------")
    for res in trial_results
        crit = res.criteria
        p_str = join(["$(k)=$(v)" for (k, v) in res.params], ", ")
        if isempty(p_str)
            p_str = "Baseline"
        end
        status_str = crit.passed_all ? "PASS" : "FAIL"
        @printf("  %02d   |   %4d   |   %5.1f%%   |   %4.2f   |  %5.1f mm  |  %-4s  | %5.1f | %s\n",
                res.trial_num, crit.n_living, crit.size_drift * 100.0,
                crit.maturity_rmse, crit.growth_rmse, status_str,
                crit.composite_score, p_str)
    end
    println("="^85)

    if best_result !== nothing
        crit = best_result.criteria
        println("\n [Optimal Parameter Set Identified]")
        p_str = join(["$(k)=$(v)" for (k, v) in best_result.params], ", ")
        println("   - Trial Number       : ", best_result.trial_num)
        println("   - Parameter Overrides: ", isempty(p_str) ? "None (Baseline)" : p_str)
        println("   - Composite Score    : ", round(crit.composite_score, digits = 2))
        println("   - All Criteria Passed: ", crit.passed_all ? "YES" : "NO")
        println("   - Component Status:")
        println("       * No Extinction (N >= N_min)        : ", crit.is_persistent ? "PASSED" : "FAILED", " (N = $(crit.n_living))")
        println("       * No Explosion (N <= N_max)         : ", crit.is_bounded ? "PASSED" : "FAILED")
        println("       * Steady Size Structure             : ", crit.is_size_steady ? "PASSED" : "FAILED", " (Drift = $(round(crit.size_drift * 100.0, digits=1))%)")
        println("       * Maturity Ogive Fidelity           : ", crit.is_maturity_realistic ? "PASSED" : "FAILED", " (RMSE = $(round(crit.maturity_rmse, digits=3)))")
        println("       * Growth Schedule Fidelity          : ", crit.is_growth_realistic ? "PASSED" : "FAILED", " (RMSE = $(round(crit.growth_rmse, digits=2)) mm)")

        # 4. Save best configuration to file
        mkpath(cli_opts[:output_dir])
        save_path = isabspath(cli_opts[:save_best]) ?
            cli_opts[:save_best] : joinpath(cli_opts[:output_dir], cli_opts[:save_best])
        save_ibm_configuration(save_path, best_result.merged_config)
        println("\n Saved calibrated configuration file to: $(save_path)")

        # 5. Export diagnostic figures if requested
        if cli_opts[:plot_best]
            println(" Exporting diagnostic figures for winning configuration...")
            export_all_plots(
                best_result.model,
                best_result.agent_df,
                cli_opts[:output_dir];
                model_df = best_result.model_df
            )
            println(" Diagnostic figures exported successfully to: $(cli_opts[:output_dir])")
        end
    end

    println("="^85)
    return (
        trials = trial_results,
        best = best_result
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_orchestrator()
end
