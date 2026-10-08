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
using GeoData

# Include submodules and components
include("types.jl")
include("organism_dynamics.jl")
include("species.jl")
include("simulation.jl")
include("analysis.jl")
include("visualization.jl")
include("gui.jl")
include("leaflet_dashboard.jl")
include("leaflet_comparison.jl")
include("cli.jl")

# Export types and enums
export LifeStage, Sex, zoea, megalopa, juvenile, adult, egg
export male, female, undifferentiated
export Organism, IBMParameters, ProcessToggles
export SpeciesBehavior, SpeciesBehaviorImpl, SpeciesParams, default_species_behavior
export FIRST_COHORT_STAGE, larval_stage_count, shell_age, shell_age_years, stage_at_maturity_fraction
export PhenologyParams, MortalityParams, EnvironmentParams, LeslieParams
export CriteriaParams
export OrganismParams, ProcessToggles, SimulationParams, FisheryParams, OutputParams
export AggregateGroup, AggregatePopulations
export is_adult, is_adult_male, is_adult_female, is_ovigerous
export is_mature_male, is_mature_female
export is_legal_male, is_legal, fishery_active
export mass_if_mature_female, mass_if_mature_male, mass_if_legal_male
export named_predicate, NamedPredicate
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
export DEFAULT_FECUNDITY_PRIMIPAROUS_INTERCEPT, DEFAULT_FECUNDITY_PRIMIPAROUS_SLOPE, DEFAULT_FECUNDITY_PRIMIPAROUS_ERR
export DEFAULT_FECUNDITY_MULTIPAROUS_INTERCEPT, DEFAULT_FECUNDITY_MULTIPAROUS_SLOPE, DEFAULT_FECUNDITY_MULTIPAROUS_ERR

# Export organism dynamics and bioenergetics
export calculate_energy_change, check_molt_conditions, perform_molt!
export molt_probability, calculate_mortality_probability
export mortality_components, dominant_mortality_cause
export shell_age_years, individuation_stage
export NamedAggregate, total_sum
export total_benthic_abundance, benthic_density_feedback
export shell_condition, min_intermolt_days, required_degree_days
export step_aggregate_groups!, perform_development!, perform_reproduction!
export calculate_habitat_suitability

# Export simulation engine
export initialize_model, agent_step!, model_step!, run_simulation
export build_habitat_quality_field, build_geodata_environmental_fields
export default_agent_observations
export organism_disperse!, nagents, allagents, abmrng, safe_mean, safe_sum
export default_ibm_configuration, load_ibm_configuration, save_ibm_configuration
export parameter_struct_defaults
export config_field_types, config_field_type, coerce_config_value
export struct_from_config, coerce_to_declared, try_coerce_to_declared, to_number
export config_section_map, ibm_configuration_sections, parameter_structs
export known_config_keys, runtime_config_keys, unknown_config_keys, cli_switch_keys
export reconstruct_params, with_overrides
export sync_grid_dimensions!
export resolve_configuration_files

# Export analysis tools
export compute_empirical_ogive, summarize_population
export maturity_ogive_residuals, maturity_ogive_mismatch, cumulative_maturity_curves
export extract_stock_recruitment, fit_stock_recruitment, summarize_stock_recruitment
export save_simulation_results, load_simulation_results, extract_annual_production
export annual_production_targets, empty_annual_production, ANNUAL_PRODUCTION_SCHEMA
export write_dataframe_csv, read_dataframe_csv
export fit_schaefer, fit_delay_difference, fit_population_dynamics
export summarize_population_dynamics
export optimize_model, optimize_blocks, safe_objective, OBJECTIVE_PENALTY, safe_objective
export logistic_unit, logit
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
export fmt_param
export organism_color, organism_size, organism_marker, organism_heatarray
export plot_abm_summary, animate_organism_simulation
export plot_functional_relationships
export launch_organism_gui, IBMGuiState, step_gui_simulation!
export generate_leaflet_dashboard_html, launch_leaflet_dashboard
export generate_leaflet_comparison_dashboard_html, launch_leaflet_comparison_dashboard
export create_empty_annual_data
export COUNT_PLOT_FLOOR, log10_count, GUI_EARLY_STAGES
export kde_density, SIZE_DENSITY_BANDWIDTH_MM, SEX_DENSITY_ALPHA
export rescan_config!, flatten_toml, config_diff, RESCAN_IMMUTABLE
export calendar_month, CALENDAR_MONTH_END
export parse_cli_args, run_ibm_cli, julia_main
export print_simulation_report, print_save_step_outputs, generate_summary_plots
export display_cli_help
export Figure

end # module model_agent
