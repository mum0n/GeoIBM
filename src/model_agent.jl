"""
    model_agent

End-to-end individual- and agent-based simulation model (IBM) for biological populations.

This package provides:
- Agent definitions and life history stage tracking with explicit developmental `stage`.
- Configurable biological rules (stage bounds, growth increments, logistic maturity
  schedules, allometric fecundity, and fishery selectivity).
- Bioenergetics and dynamic spatial simulation in discrete grid environments.
- High-level execution and statistical analysis utilities.
"""
module model_agent

using Agents
using DataFrames
using Distributions
using CairoMakie
using Random
using Statistics

# Include submodules and components
include("types.jl")
include("organism_dynamics.jl")
include("species.jl")
include("simulation.jl")
include("analysis.jl")
include("visualization.jl")
include("gui.jl")
include("cli.jl")

# Export types and enums
export LifeStage, Sex, zoea, megalopa, juvenile, adult, egg
export male, female, undifferentiated
export Organism, IBMParameters, ProcessToggles
export SpeciesBehavior, SpeciesParams, default_species_behavior
export AggregateGroup, AggregatePopulations
export is_adult, is_adult_male, is_adult_female, is_ovigerous
export is_legal_male, is_legal
export mass_if_adult_female, mass_if_adult_male, mass_if_legal_male
export StageDefinition
export LeslieMatrixFit, MacroModelComparison, EmpiricalBiologicalModel

# Export biological relationships
export llbm, llbf, get_stage_size, stage_size, stage_size_lower_bound
export growth_increment, predict_postgrowth_size
export growth_schedule, growth_simulate, maturity, fecundity, selectivity
export sample_population, calculate_mass
export build_stages, gaussian_kde, derive_stage_proportions_kde
export generate_mock_survey_data, fit_empirical_biology_model
export estimate_demographics, summarize_empirical_model
export sample_organism_traits
export FECUNDITY_PRIMIPAROUS_INTERCEPT, FECUNDITY_PRIMIPAROUS_SLOPE, FECUNDITY_PRIMIPAROUS_ERR
export FECUNDITY_MULTIPAROUS_INTERCEPT, FECUNDITY_MULTIPAROUS_SLOPE, FECUNDITY_MULTIPAROUS_ERR

# Export organism dynamics and bioenergetics
export calculate_energy_change, check_molt_conditions, perform_molt!
export molt_probability, calculate_mortality_probability
export shell_condition, min_intermolt_days, required_degree_days
export step_aggregate_groups!, perform_development!, perform_reproduction!

# Export simulation engine
export initialize_model, agent_step!, model_step!, run_simulation
export organism_disperse!, nagents, allagents, abmrng, safe_mean
export default_ibm_configuration, load_ibm_configuration, save_ibm_configuration

# Export analysis tools
export compute_empirical_ogive, summarize_population
export extract_stock_recruitment, fit_stock_recruitment, summarize_stock_recruitment
export save_simulation_results, load_simulation_results, extract_annual_production
export write_dataframe_csv, read_dataframe_csv
export fit_schaefer, fit_delay_difference, fit_population_dynamics
export summarize_population_dynamics
export mohn_rho, retrospective_analysis, summarize_retrospective
export extract_annual_stage_matrix, fit_generalized_leslie_model
export summarize_leslie_model, compare_macro_models, summarize_macro_comparison
export evaluate_population_criteria

# Export visualization tools
export plot_population_timeseries, plot_stage_distribution, plot_size_by_sex
export plot_stage_allometry, plot_reproduction_dynamics, plot_spatial_habitat
export plot_maturity_ogives, plot_simulation_dashboard, plot_stock_recruitment
export plot_population_dynamics_fits, plot_retrospective_analysis, export_all_plots
export plot_stage_specific_dynamics, plot_macro_model_comparison
export organism_color, organism_size, organism_marker, organism_heatarray
export plot_abm_summary, animate_organism_simulation
export plot_functional_relationships
export launch_organism_gui, IBMGuiState, step_gui_simulation!
export parse_cli_args, run_ibm_cli, julia_main
export Figure

end # module model_agent
