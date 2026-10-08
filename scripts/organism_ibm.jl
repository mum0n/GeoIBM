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
    --temp_base <Float> Mean bottom temperature in °C (default: 5.0)
    --temp_amplitude <Float>  Seasonal temperature amplitude in °C (default: 4.0)
    --dt <Float>        Simulation time step in days (default: 1.0)
    --seed <Int>        Random number generator seed (default: 1234)
    --plot <Bool>       Generate and save visualization plots
    --animate <Bool>    Generate and record a spatial animation
    --gui <Bool>        Launch interactive Leaflet spatial dashboard (default GUI)
    --gui_simple <Bool> Launch simple real-time GLMakie GUI dashboard
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

"""
    main()

Entry point for the organism IBM CLI simulation execution.
Delegates directly to `model_agent.run_ibm_cli(ARGS)` to guarantee canonical
argument parsing, parameter discovery, and dashboard dispatch.
"""
function main()
    return model_agent.run_ibm_cli(ARGS)
end

# If invoked as a standalone script from terminal, execute main()
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
