"""
    model_agent

End-to-end individual- and agent-based simulation model (IBM) for the snow crab
(*Chionoecetes opilio*).

This package provides:
- Agent definitions and life history stage tracking with explicit developmental `instar`.
- Empirical biological rules (instar bounds, Hiatt growth increments, logistic maturity
  schedules, allometric fecundity, and fishery selectivity).
- Bioenergetics and dynamic spatial simulation in discrete grid environments.
- High-level execution and statistical analysis utilities.
"""
module model_agent

using Agents
using DataFrames
using Distributions
using Plots
using Random
using Statistics

# Include submodules and components
include("types.jl")
include("empirical_biology.jl")
include("crab_dynamics.jl")
include("simulation.jl")
include("analysis.jl")
include("visualization.jl")

# Export types and enums
export CrabStage, CrabSex, zoea, megalopa, juvenile, adult, egg
export male, female, undifferentiated
export SnowCrab, IBMParameters
export is_adult, is_adult_male, is_adult_female, is_ovigerous
export is_legal_male, is_legal
export weight_if_adult_female, weight_if_adult_male, weight_if_legal_male
export InstarStageDefinition, LeslieMatrixFit, MacroModelComparison
export EmpiricalBiologicalModel

# Export empirical biological relationships
export llbm, llbf, carapacewidth, growth_increment, predict_postmolt_cw
export growth_schedule, growth_simulate, maturity, fecundity, selectivity
export sample_population, crab_weight
export build_instar_stages, gaussian_kde, derive_instar_proportions_kde
export generate_mock_survey_data, fit_empirical_biology_model
export sample_crab_traits, summarize_empirical_model

# Export crab dynamics and bioenergetics
export calculate_energy_change, check_molt_conditions, perform_molt!
export molt_probability, calculate_mortality_probability
export shell_condition, min_intermolt_days, required_degree_days

# Export simulation engine
export initialize_model, agent_step!, model_step!, run_simulation
export crab_disperse!, nagents, allagents, abmrng, safe_mean
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
export plot_population_timeseries, plot_instar_distribution, plot_size_by_sex
export plot_instar_allometry, plot_reproduction_dynamics, plot_spatial_habitat
export plot_maturity_ogives, plot_simulation_dashboard, plot_stock_recruitment
export plot_population_dynamics_fits, plot_retrospective_analysis, export_all_plots
export plot_stage_specific_dynamics, plot_macro_model_comparison
export crab_color, crab_size, crab_marker, crab_heatarray
export plot_abm_summary, animate_snowcrab_simulation, plot_functional_relationships

end # module model_agent
