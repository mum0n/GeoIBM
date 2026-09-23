using Agents

@enum LifeStage zoea megalopa juvenile adult egg end
@enum Sex male female undifferentiated end

"""
    StageBinDistribution

Represents a discretized population distribution over size or age for non-spatial life stages.
"""
@kwdef struct StageBinDistribution
    counts::Vector{Float64}      # Number of individuals in each bin (age or size)
    bin_edges::Vector{Float64}   # The boundaries for the bins
end

"""
    AggregatePopulations

A container for all non-spatial, aggregated life stages used before individuation.
Includes eggs, planktonic larvae/zoea, megalopa, and instars 1 through 5.
"""
@kwdef struct AggregatePopulations
    eggs::{?StageBinDistribution} = nothing
    larvae::{?StageBinDistribution} = nothing # e.g., zoea/planktonic larvae
    megalopa::{?StageBinDistribution} = nothing
    instars::Vector{StageBinDistribution} = StageBinDistribution[] # Instar 1 to 5 (length 5)
end

abstract type SpeciesBehavior end

"""
    SnowCrabBehavior <: SpeciesBehavior

Implementation of species-specific biological processes for Snow Crabs.
"""
struct SnowCrabBehavior <: SpeciesBehavior end

# Interface for species-specific biological processes
function perform_development!(behavior::SpeciesBehavior, org::Organism, model) error("Not implemented") end
function perform_reproduction!(behavior::SpeciesBehavior, org::Organism, model) error("Not implemented") end

@agent struct Organism(GridAgent{2})
    stage::LifeStage
    sex::Sex
    behavior::SpeciesBehavior  # Modular behavior definition (e.g., SnowCrabBehavior)
    developmental_step::Int    # e.g., instar for crustaceans, larval stage index
    age_days::Float64
    size::Float64              # morphometry (e.g., carapace width or length)
    mass::Float64              # body mass
    energy_reserve::Float64
    last_growth_event_time::Float64 # time of last metamorphosis/molt
    degree_days::Float64       # Accumulated thermal units (placeholder for simplicity)
    is_mating_ready::Bool
    egg_count::Int             # number of eggs in current clutch or total produced
    hatch_day::Float64
    sperm_clutches::Int

    # Individual biological traits (sampled at recruitment/birth)
    trait_l50::Float64                  # Morphometry for 50% maturity
    trait_maturity_slope::Float64       # Slope of the maturity ogive
    trait_fecundity_intercept::Float64  # intercept for log-scale fecundity
    trait_fecundity_slope::Float64      # slope for log-scale fecundity
    trait_growth_intercept::Float64     # ln(size) growth intercept
    trait_growth_slope::Float64         # ln(size) growth slope
end

is_adult(org::Organism) = org.stage === adult
is_adult_male(org::Organism) = is_adult(org) && org.sex === male
is_adult_female(org::Organism) = is_adult(org) && org.sex === female
is_ovigerous(org::Organism) = org.stage === adult && org.sex === female && org.egg_count > 0
is_legal_male(org::Organism) = is_adult_male(org) && org.size >= 95.0 # placeholder
is_legal(org::Organism) = is_adult_male(org) && org.size >= 95.0

weight_if_adult_female(org::Organism) = org.mass 
weight_if_adult_male(org::Organism) = org.mass  
weight_if_legal_male(org::Organism) = org.mass   

# --- Parameter Hierarchies ---

Base.@kwdef struct SimulationParams
    use_energetics::Bool = true
    use_thermal_stress::Bool = true
    use_food_dynamics::Bool = true
    use_habitat_taxis::Bool = true
    use_cannibalism::Bool = true
    use_senescence::Bool = true
    use_reproduction::Bool = true
    use_recruitment::Bool = true

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
end

Base.@kwdef struct OrganismParams
    max_size_male::Float64 = 160.0
    max_size_female::Float64 = 95.0
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

Base.@kwdef struct SnowCrabParams
    # --- Maturity (Log-Logistic Ogives) ---
    maturity_l50_male::Float64 = 91.8
    maturity_l50_male_sd::Float64 = 1.2
    maturity_slope_male::Float64 = 6.9957
    maturity_slope_male_sd::Float64 = 0.25

    maturity_l50_female::Float64 = 51.6
    maturity_l50_female_sd::Float64 = 0.8
    maturity_slope_female::Float64 = 12.9812
    maturity_slope_female_sd::Float64 = 0.40

    # --- Fecundity (Empirical Models) ---
    fecundity_parameterization::String = "moncton"

    # --- Growth (Bivariate Normal: Intercept & Slope) ---
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

    # --- Gear Selectivity ---
    selectivity_s50::Float64 = 95.0
    selectivity_s50_sigma::Float64 = 5.0
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
    animation_file::String = "snowcrab_animation.mp4"
    anim_frames::Int = 120
    anim_dt::Int = 2
    anim_fps::Int = 15
    anim_field::String = "temperature"
end

@kwdef struct BinningParams
    use_bins::Bool = true
    n_age_bins::Int = 50
    n_size_bins::Int = 20
    min_size::Float64 = 1.0
    max_size::Float64 = 100.0 # for early stages? or total range
end

Base.@kwdef struct IBMParameters
    simulation::SimulationParams = SimulationParams()
    environment::EnvironmentParams = EnvironmentParams()
    organism::OrganismParams = OrganismParams()
    species_snowcrab::SnowCrabParams = SnowCrabParams()
    phenology::PhenologyParams = PhenologyParams()
    mortality::MortalityParams = MortalityParams()
    fishery::FisheryParams = FisheryParams()
    output::OutputParams = OutputParams
    binning::BinningParams = BinningParams() 
end

"""
    IBMParameters(dict::AbstractDict)

Constructor that maps a nested dictionary (from TOML) to the hierarchical `IBMParameters` structure.
"""
function IBMParameters(dict::AbstractDict)
    # Helper to recursively build params from dict sections
    function build_params(struct_type, section_dict)
        p = struct_type()
        for (k, v) in section_dict
            sym_k = Symbol(k)
            if hasproperty(p, sym_k)
                setproperty!(p, sym_k, v)
            end
        end
        return p
    end

    # Initialize base structure
    params = IBMParameters()

    # Map top-level sections to their respective structs
    mapping = Dict(
        "simulation" => SimulationParams,
        "environment" => EnvironmentParams,
        "organism" => OrganismParams,
        "species_snowcrab" => SnowCrabParams,
        "phenology" => PhenologyParams,
        "mortality" => MortalityParams,
        "fishery" => FisheryParams,
        "output" => OutputParams
    )

    for (section_name, struct_type) in mapping
        if haskey(dict, section_name)
            # If the value is a dict, it's a sub-section from TOML
            sub_dict = dict[section_name]
            if sub_dict isa AbstractDict
                setproperty!(params, Symbol(section_name), build_params(struct_type, sub_dict))
            end
        end
    end

    return params
end