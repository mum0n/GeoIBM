"""
Test suite orchestrator for the TethysBio package.

Supports modular and targeted test segment execution to minimize compute time
and avoid recompilation overheads.

Usage:
    julia --project=. test/runtests.jl              # runs all segments
    julia --project=. test/runtests.jl biology      # Allometry and biological traits
    julia --project=. test/runtests.jl macro        # Macro-ecological population models
    julia --project=. test/runtests.jl config       # Configuration & parameter integrity
    julia --project=. test/runtests.jl simulation   # Agent bioenergetics, dispersal & steps
    julia --project=. test/runtests.jl vis          # Visualization, plots & GUI dashboard
    julia --project=. test/runtests.jl habitat      # Habitat suitability & Tethys integration
"""

using Test

const AVAILABLE_SEGMENTS = Dict{Symbol, NamedTuple{(:title, :file), Tuple{String, String}}}(
    :biology    => ("Allometry, Growth, and Empirical Biology", "test_allometry_biology.jl"),
    :macro      => ("Macro Population Dynamics and Leslie Matrices", "test_macro_dynamics.jl"),
    :config     => ("Configuration, CLI Parsing, and Parameter Validation", "test_config_parameters.jl"),
    :simulation => ("Agent Simulation, Energetics, Mortality, and Dispersal", "test_simulation_agents.jl"),
    :vis        => ("Visualization Tools, Plots, and GUI Dashboard", "test_visualization_gui.jl"),
    :habitat    => ("Habitat Suitability and Tethys Environmental Integration", "test_habitat_environment.jl"),
)

const DEFAULT_SEGMENTS = [:biology, :macro, :config, :simulation, :vis, :habitat]

function parse_requested_segments(args)
    if isempty(args)
        return DEFAULT_SEGMENTS
    end
    requested = Symbol[]
    for arg in args
        s = Symbol(lowercase(strip(arg)))
        if haskey(AVAILABLE_SEGMENTS, s)
            push!(requested, s)
        else
            matched = false
            for k in keys(AVAILABLE_SEGMENTS)
                if startswith(string(k), string(s))
                    push!(requested, k)
                    matched = true
                    break
                end
            end
            if !matched
                error("Unknown test segment: '$arg'. Available: $(join(keys(AVAILABLE_SEGMENTS), ", "))")
            end
        end
    end
    return unique(requested)
end

function run_selected_tests(segments)
    @testset "TethysBio Comprehensive Suite" begin
        for seg in segments
            info = AVAILABLE_SEGMENTS[seg]
            println("\n" * "="^80)
            println(">>> Running Segment: $(info.title) ($(info.file))")
            println("="^80)
            include(joinpath(@__DIR__, info.file))
        end
    end
end

selected_segments = parse_requested_segments(ARGS)
run_selected_tests(selected_segments)