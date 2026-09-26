"""
Core data types, agent structures, and parameter hierarchies for the organism IBM.
"""

using Agents

@enum LifeStage zoea megalopa juvenile adult egg
@enum Sex male female undifferentiated

"""
    AggregateGroup

Represents an aggregate cohort group for pre-individuated life stages (egg
through stage 5, spanning approximately 4 years). Sex ratio is assumed 1:1
(undifferentiated) in accordance with empirical early developmental observations.
"""
Base.@kwdef mutable struct AggregateGroup
    id::Int = 0
    stage::Int = 0              # 0: aggregate egg / larval clutch; 1-5: benthic juvenile stages
    abundance::Float64 = 0.0     # Total number of individuals in the aggregate group
    size::Float64 = 0.0          # Mean organism size (carapace width / dimension in mm)
    mass::Float64 = 0.0          # Mean organism wet mass (g)
    age_days::Float64 = 0.0      # Chronological age in days
    degree_days::Float64 = 0.0   # Accumulated thermal degree-days
    last_growth_time::Float64 = 0.0 # Timestamp of most recent developmental transition
    pos::Tuple{Int, Int} = (1, 1)# Spatial reference coordinates on grid
end

"""
    AggregatePopulations

Container managing all aggregate cohorts across pre-individuation stages.
"""
Base.@kwdef mutable struct AggregatePopulations
    groups::Vector{AggregateGroup} = AggregateGroup[]
    next_group_id::Int = 1
end

abstract type SpeciesBehavior end

abstract type SpeciesParams end

# Generic species interface function declarations
function default_species_behavior end
function stage_size_lower_bound end
function get_stage_size end
function growth_increment end
function predict_postgrowth_size end
function growth_schedule end
function maturity end
function fecundity end
function selectivity end
function calculate_mass end
function build_stages end
function min_intermolt_days end
function required_degree_days end
function perform_development! end
function perform_reproduction! end


"""
    Organism(GridAgent{2})

Individual agent representation for post-individuation life stages (stage 6 onwards,
up to stage 16 in males and stage 14 in females).
"""
@agent struct Organism(GridAgent{2})
    stage::Int                      # Developmental stage index (6 through 16)
    sex::Sex
    behavior::SpeciesBehavior       # Modular behavior implementation
    age_days::Float64
    size::Float64                   # Organism size (carapace width in mm)
    mass::Float64                   # Body wet mass in grams
    energy_reserve::Float64
    last_growth_event_time::Float64 # Timestamp of last ecdysis (molt)
    degree_days::Float64            # Cumulative physiological thermal units
    is_mating_ready::Bool
    egg_count::Int                  # Number of eggs in current clutch
    hatch_day::Float64
    sperm_clutches::Int
    is_mature::Bool

    # Individual empirical biological trait quantiles in (0, 1)
    trait_growth_quantile::Float64
    trait_maturity_quantile::Float64
    trait_fecundity_quantile::Float64
    trait_mortality_quantile::Float64
end

# Primary outer constructors for Organism
function Organism(
    id::Int,
    pos::Tuple{Int, Int},
    stage::Integer,
    sex::Sex,
    behavior::SpeciesBehavior,
    age_days::Real,
    size::Real,
    mass::Real,
    energy_reserve::Real,
    last_growth_event_time::Real,
    degree_days::Real,
    is_mating_ready::Bool,
    egg_count::Integer,
    hatch_day::Real,
    sperm_clutches::Integer,
    is_mature::Bool = false;
    trait_growth_quantile::Real = 0.5,
    trait_maturity_quantile::Real = 0.5,
    trait_fecundity_quantile::Real = 0.5,
    trait_mortality_quantile::Real = 0.5
)
    return Organism(
        id, pos, Int(stage), sex, behavior,
        Float64(age_days), Float64(size), Float64(mass), Float64(energy_reserve),
        Float64(last_growth_event_time), Float64(degree_days), is_mating_ready,
        Int(egg_count), Float64(hatch_day), Int(sperm_clutches), is_mature,
        Float64(trait_growth_quantile), Float64(trait_maturity_quantile),
        Float64(trait_fecundity_quantile), Float64(trait_mortality_quantile)
    )
end

function Organism(
    id::Int,
    pos::Tuple{Int, Int},
    stage::Integer,
    sex::Sex,
    age_days::Real,
    size::Real,
    mass::Real,
    energy_reserve::Real,
    last_growth_event_time::Real,
    degree_days::Real,
    is_mating_ready::Bool,
    egg_count::Integer,
    hatch_day::Real,
    sperm_clutches::Integer,
    is_mature::Bool = false;
    behavior::SpeciesBehavior = default_species_behavior(),
    trait_growth_quantile::Real = 0.5,
    trait_maturity_quantile::Real = 0.5,
    trait_fecundity_quantile::Real = 0.5,
    trait_mortality_quantile::Real = 0.5
)
    return Organism(
        id, pos, Int(stage), sex, behavior,
        age_days, size, mass, energy_reserve,
        last_growth_event_time, degree_days, is_mating_ready,
        egg_count, hatch_day, sperm_clutches, is_mature;
        trait_growth_quantile = trait_growth_quantile,
        trait_maturity_quantile = trait_maturity_quantile,
        trait_fecundity_quantile = trait_fecundity_quantile,
        trait_mortality_quantile = trait_mortality_quantile
    )
end

# Forwarding method for legacy call signatures with stage_enum
function Organism(
    id::Int,
    pos::Tuple{Int, Int},
    stage_enum::LifeStage,
    sex::Sex,
    stage_int::Integer,
    age_days::Real,
    size::Real,
    mass::Real,
    energy_reserve::Real,
    last_growth_event_time::Real,
    degree_days::Real,
    is_mating_ready::Bool,
    egg_count::Integer,
    hatch_day::Real,
    sperm_clutches::Integer,
    is_mature::Bool = (stage_enum === adult);
    kwargs...
)
    return Organism(
        id, pos, Int(stage_int), sex, age_days, size, mass, energy_reserve,
        last_growth_event_time, degree_days, is_mating_ready, egg_count,
        hatch_day, sperm_clutches, is_mature;
        kwargs...
    )
end


# Interface for species-specific biological processes
# Concrete implementations in species.jl

is_adult(org::Organism) = org.is_mature || org.stage >= 10
is_adult_male(org::Organism) = is_adult(org) && org.sex === male
is_adult_female(org::Organism) = is_adult(org) && org.sex === female
is_ovigerous(org::Organism) = is_adult_female(org) && org.egg_count > 0
is_legal_male(org::Organism; legal_size::Float64 = 95.0) = is_adult_male(org) && org.size >= legal_size
is_legal(org::Organism; legal_size::Float64 = 95.0) = is_legal_male(org; legal_size=legal_size)

mass_if_adult_female(org::Organism) = is_adult_female(org) ? org.mass : 0.0
mass_if_adult_male(org::Organism) = is_adult_male(org) ? org.mass : 0.0
mass_if_legal_male(org::Organism; legal_size::Float64 = 95.0) = is_legal_male(org; legal_size=legal_size) ? org.mass : 0.0

"""
    StageDefinition

Defines discrete life-history stages for demographic matrix models.
"""
struct StageDefinition
    index::Int
    name::String
    stage::Int
    sex::Sex
    size_range::Tuple{Float64, Float64}
    size_mean::Float64
    mean_mass::Float64
    is_legal_male::Bool
    is_reproductive::Bool
end

function Base.getproperty(sd::StageDefinition, s::Symbol)
    if s === :cw_mean
        return getfield(sd, :size_mean)
    elseif s === :cw_range
        return getfield(sd, :size_range)
    elseif s === :mean_weight
        return getfield(sd, :mean_mass)
    elseif s === :instar
        return getfield(sd, :stage)
    else
        return getfield(sd, s)
    end
end

"""
    LeslieMatrixFit

Results and demographic diagnostics from fitting a generalized Leslie stage matrix.
"""
struct LeslieMatrixFit
    stages::Vector{StageDefinition}
    A::Matrix{Float64}
    eigenvalues::Vector{ComplexF64}
    dominant_eigenvalue::Float64
    intrinsic_growth_rate::Float64
    stable_stage_distribution::Vector{Float64}
    reproductive_values::Vector{Float64}
    damping_ratio::Float64
    elasticity_matrix::Matrix{Float64}
    observed_stages::Matrix{Float64}
    predicted_stages::Matrix{Float64}
    legal_abundance_obs::Vector{Float64}
    legal_abundance_pred::Vector{Float64}
    legal_biomass_obs_kg::Vector{Float64}
    legal_biomass_pred_kg::Vector{Float64}
    r2::Float64
    rmse::Float64
    aic::Float64
    years::Vector{Int}
end

"""
    MacroModelComparison

Comparative summary across Schaefer, Delay-Difference, and Leslie macro models.
"""
struct MacroModelComparison
    years::Vector{Int}
    biomass_obs_kg::Vector{Float64}
    biomass_schaefer_kg::Vector{Float64}
    biomass_delay_diff_kg::Vector{Float64}
    biomass_leslie_kg::Vector{Float64}
    cv_obs::Float64
    cv_schaefer::Float64
    cv_delay_diff::Float64
    cv_leslie::Float64
    sd_obs::Float64
    sd_schaefer::Float64
    sd_delay_diff::Float64
    sd_leslie::Float64
    var_delta_obs::Float64
    var_delta_schaefer::Float64
    var_delta_delay_diff::Float64
    var_delta_leslie::Float64
    r2_schaefer::Float64
    r2_delay_diff::Float64
    r2_leslie::Float64
    rmse_schaefer::Float64
    rmse_delay_diff::Float64
    rmse_leslie::Float64
    aic_schaefer::Float64
    aic_delay_diff::Float64
    aic_leslie::Float64
    preferred_model::Symbol
end

"""
    EmpiricalBiologicalModel

Container for empirical parameters, allometric scalings, and trait covariances.
"""
struct EmpiricalBiologicalModel
    mass_params::Dict{Symbol, Tuple{Float64, Float64, Float64}}
    maturity_params::Dict{Symbol, Tuple{Float64, Float64, Float64}}
    fecundity_params::Dict{Symbol, Tuple{Float64, Float64, Float64}}
    stage_modals::Dict{Int, Tuple{Float64, Float64}}
    trait_names::Vector{Symbol}
    trait_cov::Matrix{Float64}
    r2_scores::Dict{Symbol, Float64}
    sample_size::Int
end

# --- Parameter Hierarchies ---

"""
    ProcessToggles

Granular feature toggles controlling activation of specific biological,
environmental, and behavioral mechanisms. When a toggle is disabled, the process
reverts to a biologically sound stochastic default.
"""
Base.@kwdef struct ProcessToggles
    use_energetics::Bool = true
    use_thermal_stress::Bool = true
    use_cannibalism::Bool = false
    use_senescence::Bool = true
    use_spermatheca::Bool = true
    use_movement::Bool = true
    use_directed_movement::Bool = true
    use_fishery::Bool = true
    use_degree_days::Bool = true
    use_molting::Bool = true
    use_reproduction::Bool = true
end

Base.@kwdef struct SimulationParams
    dt::Float64 = 1.0
    seed::Int = 1234
    n_agents::Int = 300
    days::Int = 730
    grid_nx::Int = 100
    grid_ny::Int = 100
end

Base.@kwdef struct EnvironmentParams
    temp_base::Float64 = 5.0
    temp_amplitude::Float64 = 4.0
    temp_noise_sd::Float64 = 0.5
    temp_base_degreedays::Float64 = 0.0
    food_max::Float64 = 10.0
    food_replenishment_rate::Float64 = 0.1
    benthic_carrying_capacity::Float64 = 300.0
    half_saturation_food::Float64 = 2.0
    assimilation_efficiency::Float64 = 0.75
    habitat_quality_mean::Float64 = 1.0
    habitat_quality_sd::Float64 = 0.1
end

Base.@kwdef struct OrganismParams
    max_size_male::Float64 = 160.0
    max_size_female::Float64 = 95.0
    max_stage_male::Int = 16
    max_stage_female::Int = 14
    maturity_size_mid::Float64 = 60.0
    metabolic_rate_base::Float64 = 0.01
    intake_rate_base::Float64 = 0.15
    energy_molt_threshold::Float64 = 50.0
    energy_molt_slope::Float64 = 1.5
    molt_energy_retention::Float64 = 0.10
    min_mating_energy_fraction::Float64 = 0.1
    initial_energy::Float64 = 10.0
    molt_probability_scale::Float64 = 45.0
end


Base.@kwdef struct PhenologyParams
    molt_season_start::Float64 = 15.0
    molt_season_end::Float64 = 140.0
    mating_season_start::Float64 = 30.0
    mating_season_end::Float64 = 120.0
    mating_search_radius::Int = 5
    brood_duration_base::Float64 = 365.0
    egg_survival_base::Float64 = 5.0e-5
end

Base.@kwdef struct MortalityParams
    soft_shell_duration::Float64 = 30.0
    senescence_age_threshold::Float64 = 3.0
    temp_thermal_stress::Float64 = 8.0
end

Base.@kwdef struct FisheryParams
    legal_size_male::Float64 = 95.0
    fishing_probability_daily::Float64 = 0.0
    fishing_season_start::Float64 = 90.0
    fishing_season_end::Float64 = 150.0
end

Base.@kwdef struct OutputParams
    save::Bool = true
    save_file::String = "simulation_results.csv"
    snapshot_day::Float64 = 274.0
    frequency_days::Float64 = 365.0
    plot::Bool = false
    output_dir::String = "outputs"
    animate::Bool = false
    animation_file::String = "organism_animation.mp4"
    anim_frames::Int = 120
    anim_dt::Int = 2
    anim_fps::Int = 15
    anim_field::String = "temperature"
end

Base.@kwdef struct BinningParams
    use_bins::Bool = true
    n_age_bins::Int = 50
    n_size_bins::Int = 20
    min_size::Float64 = 1.0
    max_size::Float64 = 100.0
end

"""
    SpeciesParams

Generic container for species-specific biological parameters.
All values can be loaded from TOML configuration files.
"""
Base.@kwdef struct SpeciesParams
    # Maturity (Log-Logistic Ogives)
    maturity_l50_male::Float64 = 91.8
    maturity_l50_male_sd::Float64 = 1.2
    maturity_slope_male::Float64 = 6.9957
    maturity_slope_male_sd::Float64 = 0.25

    maturity_l50_female::Float64 = 51.6
    maturity_l50_female_sd::Float64 = 0.8
    maturity_slope_female::Float64 = 12.9812
    maturity_slope_female_sd::Float64 = 0.40

    # Fecundity (Empirical Models)
    fecundity_parameterization::String = "moncton"
    fecundity_primiparous_intercept::Float64 = -0.7229
    fecundity_primiparous_slope::Float64 = 2.94
    fecundity_primiparous_err::Float64 = 0.090
    fecundity_multiparous_intercept::Float64 = 0.0158
    fecundity_multiparous_slope::Float64 = 2.5811
    fecundity_multiparous_err::Float64 = 0.145

    # Growth (Bivariate Normal: Intercept & Slope)
    growth_int_female::Float64 = 0.1931
    growth_beta1_female::Float64 = 0.0545
    growth_slp_female::Float64 = 1.0372
    growth_beta2_female::Float64 = -0.017376
    growth_beta3_female::Float64 = 0.005068

    growth_int_male::Float64 = 0.30153
    growth_beta1_male::Float64 = 0.02198
    growth_slp_male::Float64 = 1.00027
    growth_beta2_male::Float64 = -0.00643655
    growth_beta3_male::Float64 = 0.0017492

    # Gear Selectivity
    selectivity_s50::Float64 = 95.0
    selectivity_s50_sigma::Float64 = 5.0

    # Allometric Mass Parameters
    mass_alpha_male_immature::Float64 = 0.0002665
    mass_beta_male_immature::Float64 = 3.098
    mass_alpha_male_mature::Float64 = 0.0001830
    mass_beta_male_mature::Float64 = 3.160
    mass_alpha_female_immature::Float64 = 0.0003440
    mass_beta_female_immature::Float64 = 2.980
    mass_alpha_female_mature::Float64 = 0.0004500
    mass_beta_female_mature::Float64 = 2.920
    mass_sigma::Float64 = 0.08
end

"""
    LeslieParams

Parameters for the generalized Leslie (Lefkovitch) stage-transition matrix model.
"""
Base.@kwdef struct LeslieParams
    # Stage 1 (Recruit, stage 4)
    survival_stage1::Float64 = 0.74
    gamma_stage1::Float64 = 0.78

    # Stage 2 (Juvenile, stage 5)
    survival_stage2::Float64 = 0.76
    gamma_stage2::Float64 = 0.75
    male_split_fraction::Float64 = 0.50
    female_split_fraction::Float64 = 0.50

    # Male stages (3-9)
    survival_male_legal::Float64 = 0.78
    survival_male_sublegal::Float64 = 0.80
    gamma_male_base::Float64 = 0.75
    gamma_male_decay::Float64 = 0.10
    p_term_male_base::Float64 = 0.10
    p_term_male_exponent::Float64 = 1.3

    # Female stages (3-9)
    survival_female::Float64 = 0.80
    gamma_female_base::Float64 = 0.70
    gamma_female_decay::Float64 = 0.12
    p_term_female_base::Float64 = 0.15
    p_term_female_exponent::Float64 = 1.4

    # Fecundity
    fecundity_base::Float64 = 0.85
    fecundity_size_exponent::Float64 = 2.5
    fecundity_size_reference::Float64 = 65.0

    # Trajectory refinement
    col_sum_cap::Float64 = 0.95
    col_sum_scale_low::Float64 = 0.85
    col_sum_scale_high::Float64 = 0.15
end

struct IBMParameters
    toggles::ProcessToggles
    simulation::SimulationParams
    environment::EnvironmentParams
    organism::OrganismParams
    phenology::PhenologyParams
    mortality::MortalityParams
    fishery::FisheryParams
    output::OutputParams
    binning::BinningParams
    species::SpeciesParams
    leslie::LeslieParams
end

function IBMParameters(; kwargs...)
    toggles_kwargs = Dict{Symbol, Any}()
    simulation_kwargs = Dict{Symbol, Any}()
    environment_kwargs = Dict{Symbol, Any}()
    organism_kwargs = Dict{Symbol, Any}()
    phenology_kwargs = Dict{Symbol, Any}()
    mortality_kwargs = Dict{Symbol, Any}()
    fishery_kwargs = Dict{Symbol, Any}()
    output_kwargs = Dict{Symbol, Any}()
    binning_kwargs = Dict{Symbol, Any}()
    species_kwargs = Dict{Symbol, Any}()
    leslie_kwargs = Dict{Symbol, Any}()

    for (k, v) in kwargs
        sym = Symbol(k)
        if sym in fieldnames(ProcessToggles)
            toggles_kwargs[sym] = v
        elseif sym in fieldnames(SimulationParams)
            simulation_kwargs[sym] = v
        elseif sym in fieldnames(EnvironmentParams)
            environment_kwargs[sym] = v
        elseif sym in fieldnames(OrganismParams)
            organism_kwargs[sym] = v
        elseif sym in fieldnames(PhenologyParams)
            phenology_kwargs[sym] = v
        elseif sym in fieldnames(MortalityParams)
            mortality_kwargs[sym] = v
        elseif sym in fieldnames(FisheryParams)
            fishery_kwargs[sym] = v
        elseif sym in fieldnames(OutputParams)
            output_kwargs[sym] = v
        elseif sym in fieldnames(BinningParams)
            binning_kwargs[sym] = v
        elseif sym in fieldnames(SpeciesParams)
            species_kwargs[sym] = v
        elseif sym in fieldnames(LeslieParams)
            leslie_kwargs[sym] = v
        end
    end

    toggles = get(kwargs, :toggles, ProcessToggles(; toggles_kwargs...))
    simulation = get(kwargs, :simulation, SimulationParams(; simulation_kwargs...))
    environment = get(kwargs, :environment, EnvironmentParams(; environment_kwargs...))
    organism = get(kwargs, :organism, OrganismParams(; organism_kwargs...))
    phenology = get(kwargs, :phenology, PhenologyParams(; phenology_kwargs...))
    mortality = get(kwargs, :mortality, MortalityParams(; mortality_kwargs...))
    fishery = get(kwargs, :fishery, FisheryParams(; fishery_kwargs...))
    output = get(kwargs, :output, OutputParams(; output_kwargs...))
    binning = get(kwargs, :binning, BinningParams(; binning_kwargs...))
    species = get(kwargs, :species, SpeciesParams(; species_kwargs...))
    leslie = get(kwargs, :leslie, LeslieParams(; leslie_kwargs...))

    @info "IBMParameters constructor" typeof(toggles) typeof(simulation) typeof(environment) typeof(organism) typeof(phenology) typeof(mortality) typeof(fishery) typeof(output) typeof(binning) typeof(species) typeof(leslie)

    return IBMParameters(
        toggles, simulation, environment, organism,
        phenology, mortality, fishery, output, binning, species, leslie
    )
end

"""
    IBMParameters(dict::AbstractDict)

Constructor mapping a nested dictionary (e.g. from TOML) to the hierarchical
`IBMParameters` structure.
"""
function IBMParameters(dict::AbstractDict)
    flat_kwargs = Dict{Symbol, Any}()
    for (k, v) in dict
        if isa(v, AbstractDict)
            for (sub_k, sub_v) in v
                flat_kwargs[Symbol(sub_k)] = sub_v
            end
        else
            flat_kwargs[Symbol(k)] = v
        end
    end
    return IBMParameters(; flat_kwargs...)
end

function Base.getproperty(p::IBMParameters, sym::Symbol)
    if sym in fieldnames(IBMParameters)
        return getfield(p, sym)
    elseif hasfield(ProcessToggles, sym)
        return getfield(getfield(p, :toggles), sym)
    elseif hasfield(FisheryParams, sym)
        return getfield(getfield(p, :fishery), sym)
    elseif hasfield(SimulationParams, sym)
        return getfield(getfield(p, :simulation), sym)
    elseif hasfield(EnvironmentParams, sym)
        return getfield(getfield(p, :environment), sym)
    elseif hasfield(OrganismParams, sym)
        return getfield(getfield(p, :organism), sym)

    elseif hasfield(PhenologyParams, sym)
        return getfield(getfield(p, :phenology), sym)
    elseif hasfield(MortalityParams, sym)
        return getfield(getfield(p, :mortality), sym)
    elseif hasfield(OutputParams, sym)
        return getfield(getfield(p, :output), sym)
    elseif hasfield(BinningParams, sym)
        return getfield(getfield(p, :binning), sym)
    elseif hasfield(SpeciesParams, sym)
        return getfield(getfield(p, :species), sym)
    elseif hasfield(LeslieParams, sym)
        return getfield(getfield(p, :leslie), sym)
    else
        return getfield(p, sym)
    end
end