"""
Types and data structures for the Snow Crab Individual-Based Model (IBM).
"""

using Agents

"""
    CrabStage

Lifecycle stages for snow crab (*Chionoecetes opilio*):
- `zoea`: Planktonic larval instars (instars 1 and 2).
- `megalopa`: Transitional planktonic larval instar (instar 3).
- `juvenile`: Immature benthic instars (instars 4+).
- `adult`: Morphoterminally mature benthic crabs.
- `egg`: Brooded embryos (tracked primarily via egg count on mature females).
"""
@enum CrabStage zoea megalopa juvenile adult egg

"""
    CrabSex

Sex classification for snow crabs:
- `male`: Morphologically male crab.
- `female`: Morphologically female crab.
- `undifferentiated`: Sexually uncommitted early larval stage.
"""
@enum CrabSex male female undifferentiated

"""
    SnowCrab(GridAgent{2})

Individual agent representation of a snow crab in a 2D discrete grid space.

# Fields
- `id::Int`: Unique agent identifier.
- `pos::NTuple{2, Int}`: Spatial coordinates `(x, y)` on the grid.
- `stage::CrabStage`: Current developmental lifecycle stage.
- `sex::CrabSex`: Sex of the individual crab.
- `instar::Int`: Developmental molt stage (1: Zoea I, 2: Zoea II, 3: Megalopa,
  4..14+: Benthic instars).
- `age_days::Float64`: Chronological age in days since hatching.
- `size_mm::Float64`: Carapace width (CW) in millimeters.
- `weight_g::Float64`: Individual wet body mass in grams.
- `energy_reserve::Float64`: Internal accumulated metabolic energy budget.
- `last_molt_time::Float64`: Simulation timestamp of the most recent ecdysis.
- `degree_days::Float64`: Cumulative physiological thermal degree-days since last ecdysis.
- `is_mating_ready::Bool`: Flag indicating physiological receptivity for mating.
- `eggs_count::Int`: Number of extruded eggs currently brooded beneath the abdomen.
- `hatch_day::Float64`: Simulation timestamp when brooded embryos are scheduled to hatch.
- `sperm_clutches::Int`: Number of viable clutches remaining in spermathecal storage.
"""
@agent struct SnowCrab(GridAgent{2})
    stage::CrabStage
    sex::CrabSex
    instar::Int
    age_days::Float64
    size_mm::Float64
    weight_g::Float64
    energy_reserve::Float64
    last_molt_time::Float64
    degree_days::Float64
    is_mating_ready::Bool
    eggs_count::Int
    hatch_day::Float64
    sperm_clutches::Int
end

"""
    is_adult(crab::SnowCrab)

Return `true` if `crab` has completed terminal molt to `adult`.
"""
is_adult(crab::SnowCrab) = (crab.stage === adult)

"""
    is_adult_male(crab::SnowCrab)

Return `true` if `crab` is a morphometrically mature adult male.
"""
is_adult_male(crab::SnowCrab) = (crab.stage === adult && crab.sex === male)

"""
    is_adult_female(crab::SnowCrab)

Return `true` if `crab` is a morphometrically mature adult female.
"""
is_adult_female(crab::SnowCrab) = (crab.stage === adult && crab.sex === female)

"""
    is_ovigerous(crab::SnowCrab)

Return `true` if `crab` is an adult female carrying an egg clutch.
"""
is_ovigerous(crab::SnowCrab) = (crab.stage === adult && crab.sex === female &&
                                crab.eggs_count > 0)

"""
    is_legal_male(crab::SnowCrab)

Return `true` if `crab` is a commercial legal-sized adult male (CW >= 95.0 mm).
"""
is_legal_male(crab::SnowCrab) = (crab.stage === adult && crab.sex === male &&
                                crab.size_mm >= 95.0)

"""
    is_legal(crab::SnowCrab)

Return `true` if `crab` is male with carapace width meeting or exceeding
commercial legal threshold (CW >= 95.0 mm).
"""
is_legal(crab::SnowCrab) = (crab.sex === male && crab.size_mm >= 95.0)

"""
    weight_if_adult_female(crab::SnowCrab)

Return individual wet body mass in grams (`crab.weight_g`) if `crab` is a morphometrically
mature adult female (`stage === adult` and `sex === female`), or `0.0` otherwise.
Used in Agent data aggregation to calculate total mature female biomass.

# Arguments
- `crab::SnowCrab`: Individual snow crab agent.

# Returns
`Float64` body weight in grams (or `0.0`).
"""
weight_if_adult_female(crab::SnowCrab) = is_adult_female(crab) ? crab.weight_g : 0.0

"""
    weight_if_adult_male(crab::SnowCrab)

Return individual wet body mass in grams (`crab.weight_g`) if `crab` is a morphometrically
mature adult male (`stage === adult` and `sex === male`), or `0.0` otherwise.
Used in Agent data aggregation to calculate total mature male biomass.

# Arguments
- `crab::SnowCrab`: Individual snow crab agent.

# Returns
`Float64` body weight in grams (or `0.0`).
"""
weight_if_adult_male(crab::SnowCrab) = is_adult_male(crab) ? crab.weight_g : 0.0

"""
    weight_if_legal_male(crab::SnowCrab)

Return individual wet body mass in grams (`crab.weight_g`) if `crab` is a commercially
legal harvestable adult male (`stage === adult`, `sex === male`, `size_mm >= 95.0`), or
`0.0` otherwise. Used in Agent data aggregation to calculate total fishable male biomass.

# Arguments
- `crab::SnowCrab`: Individual snow crab agent.

# Returns
`Float64` body weight in grams (or `0.0`).
"""
weight_if_legal_male(crab::SnowCrab) = is_legal_male(crab) ? crab.weight_g : 0.0


"""
    IBMParameters

Configuration parameters controlling physical environment, bioenergetics,
molting thresholds, reproduction, and harvesting rules in the snow crab IBM.

# Fields
- `max_size_male::Float64`: Asymptotic maximum carapace width for males (mm).
- `max_size_female::Float64`: Asymptotic maximum carapace width for females (mm).
- `maturity_size_mid::Float64`: Nominal median carapace width at maturity (mm).
- `metabolic_rate_base::Float64`: Basal metabolic coefficient.
- `intake_rate_base::Float64`: Basal food ingestion coefficient.
- `energy_molt_threshold::Float64`: Intercept of energetic molt requirement.
- `energy_molt_slope::Float64`: Allometric slope of energetic molt requirement.
- `molt_energy_retention::Float64`: Fraction of energy reserve retained after ecdysis.
- `legal_size_male::Float64`: Minimum carapace width for commercial retention (mm).
- `fishing_probability_daily::Float64`: Daily harvest probability during open season.
- `fishing_season_start::Float64`: Day-of-year marking the start of the fishery.
- `fishing_season_end::Float64`: Day-of-year marking the close of the fishery.
- `temp_base::Float64`: Annual mean bottom temperature (°C).
- `temp_amplitude::Float64`: Seasonal temperature oscillation amplitude (°C).
- `temp_noise_sd::Float64`: Standard deviation of spatial temperature perturbations.
- `food_replenishment_rate::Float64`: Daily increment of regenerated food resource.
- `food_max::Float64`: Local maximum carrying capacity of food resource.
- `egg_survival_base::Float64`: Base survival rate from egg to benthic settler.
- `benthic_carrying_capacity::Float64`: Target benthic population carrying capacity.
- `mating_search_radius::Int`: Spatial radius (grid cells) for detecting mates.
- `mating_season_start::Float64`: Day-of-year for spring mating season onset.
- `mating_season_end::Float64`: Day-of-year for spring mating season end.
- `molt_season_start::Float64`: Day-of-year for spring adolescent molting window onset.
- `molt_season_end::Float64`: Day-of-year for spring adolescent molting window end.
- `temp_base_degreedays::Float64`: Base threshold temperature for physiological degree-days (°C).
- `brood_duration_base::Float64`: Base embryo brooding incubation duration in days.
- `half_saturation_food::Float64`: Holling Type II half-saturation food concentration.
- `assimilation_efficiency::Float64`: Fraction of ingested food assimilated into metabolic pool.
- `soft_shell_duration::Float64`: Duration in days of elevated post-ecdysis vulnerability.
- `senescence_age_threshold::Float64`: Adult carapace shell age (years) when senescence starts.
- `temp_thermal_stress::Float64`: Bottom-water temperature (°C) above which stenothermic stress mortality activates.
- `min_mating_energy_fraction::Float64`: Minimum energy reserve (as fraction of `energy_molt_threshold`) required for mating or clutch extrusion.
- `dt::Float64`: Simulation time step in days.
"""
Base.@kwdef struct IBMParameters
    max_size_male::Float64 = 160.0
    max_size_female::Float64 = 95.0
    maturity_size_mid::Float64 = 60.0
    metabolic_rate_base::Float64 = 0.01
    intake_rate_base::Float64 = 0.15  # I_max: saturation-limit ingestion coeff (g^0.75 d^-1)
    energy_molt_threshold::Float64 = 50.0
    energy_molt_slope::Float64 = 1.5
    molt_energy_retention::Float64 = 0.10
    legal_size_male::Float64 = 95.0
    fishing_probability_daily::Float64 = 0.0
    fishing_season_start::Float64 = 90.0
    fishing_season_end::Float64 = 150.0
    temp_base::Float64 = 5.0
    temp_amplitude::Float64 = 4.0
    temp_noise_sd::Float64 = 0.50
    food_replenishment_rate::Float64 = 0.10
    food_max::Float64 = 10.0
    egg_survival_base::Float64 = 5.0e-5
    benthic_carrying_capacity::Float64 = 300.0
    mating_search_radius::Int = 5
    mating_season_start::Float64 = 30.0
    mating_season_end::Float64 = 120.0
    molt_season_start::Float64 = 15.0
    molt_season_end::Float64 = 140.0
    temp_base_degreedays::Float64 = 0.0
    brood_duration_base::Float64 = 365.0
    half_saturation_food::Float64 = 2.0
    assimilation_efficiency::Float64 = 0.75
    soft_shell_duration::Float64 = 30.0
    senescence_age_threshold::Float64 = 3.0
    temp_thermal_stress::Float64 = 8.0    # °C onset for stenothermic mortality
    min_mating_energy_fraction::Float64 = 0.10  # fraction of energy_molt_threshold
    dt::Float64 = 1.0
end

"""
    IBMParameters(dict::AbstractDict)

Construct `IBMParameters` by extracting parameter keys from `dict`
(keyed by `Symbol` or `String`), falling back to default values for
any unspecified fields.
"""
function IBMParameters(dict::AbstractDict)
    kwargs = Dict{Symbol, Any}()
    for f in fieldnames(IBMParameters)
        target_type = fieldtype(IBMParameters, f)
        if haskey(dict, f)
            kwargs[f] = convert(target_type, dict[f])
        elseif haskey(dict, string(f))
            kwargs[f] = convert(target_type, dict[string(f)])
        end
    end
    return IBMParameters(; kwargs...)
end

"""
    InstarStageDefinition

Definition of a discrete developmental instar stage for generalized Leslie
stage-structured demographic modeling.

# Fields
- `index::Int`: Unique 1-based stage identifier (1 to S).
- `name::String`: Descriptive label (e.g. "Instar 4", "Instar 6 Male").
- `instar::Int`: Underlying developmental instar number (4 to 12+).
- `sex::CrabSex`: Sex categorization (`undifferentiated`, `male`, or `female`).
- `cw_range::Tuple{Float64, Float64}`: Carapace width boundary interval [min, max] (mm).
- `cw_mean::Float64`: Modal theoretical carapace width (mm).
- `mean_weight_g::Float64`: Mean wet body mass (g) derived from allometry.
- `is_legal_male::Bool`: True if commercially harvestable legal male (CW >= 95 mm).
- `is_reproductive::Bool`: True if mature female contributing to reproduction.
"""
struct InstarStageDefinition
    index::Int
    name::String
    instar::Int
    sex::CrabSex
    cw_range::Tuple{Float64, Float64}
    cw_mean::Float64
    mean_weight_g::Float64
    is_legal_male::Bool
    is_reproductive::Bool
end

"""
    LeslieMatrixFit

Results and demographic diagnostics from fitting a generalized Leslie (Lefkovitch)
stage-structured transition matrix model to multi-year instar stage abundances.

# Fields
- `stages::Vector{InstarStageDefinition}`: Life-history stage definitions.
- `A::Matrix{Float64}`: Estimated S x S transition matrix.
- `eigenvalues::Vector{ComplexF64}`: All complex eigenvalues of A.
- `dominant_eigenvalue::Float64`: Real dominant eigenvalue (lambda_1).
- `intrinsic_growth_rate::Float64`: Asymptotic annual growth rate r = ln(lambda_1).
- `stable_stage_distribution::Vector{Float64}`: Right eigenvector w_1 (sum = 1.0).
- `reproductive_values::Vector{Float64}`: Left eigenvector v_1 (relative to stage 1).
- `damping_ratio::Float64`: Asymptotic rate of convergence rho = lambda_1 / |lambda_2|.
- `elasticity_matrix::Matrix{Float64}`: Proportional sensitivity matrix E (sum = 1.0).
- `observed_stages::Matrix{Float64}`: S x T historical stage abundance matrix.
- `predicted_stages::Matrix{Float64}`: S x T projected stage abundance matrix.
- `legal_abundance_obs::Vector{Float64}`: Observed legal male abundance across years.
- `legal_abundance_pred::Vector{Float64}`: Predicted legal male abundance across years.
- `legal_biomass_obs_kg::Vector{Float64}`: Observed legal male biomass (kg).
- `legal_biomass_pred_kg::Vector{Float64}`: Predicted legal male biomass (kg).
- `r2::Float64`: Coefficient of determination on legal fishable abundance.
- `rmse::Float64`: Root mean squared error on legal fishable abundance.
- `aic::Float64`: Akaike Information Criterion score.
- `years::Vector{Int}`: Evaluated simulation years.
"""
struct LeslieMatrixFit
    stages::Vector{InstarStageDefinition}
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

Comparative summary of legal fishable stock biomass dynamics across three macro-ecological
modeling paradigms: Schaefer surplus production, Deriso-Schnute delay-difference,
and generalized Leslie stage transition matrix.

# Fields
- `years::Vector{Int}`: Evaluated annual time steps.
- `biomass_obs_kg::Vector{Float64}`: Observed legal male biomass (kg).
- `biomass_schaefer_kg::Vector{Float64}`: Schaefer predicted legal biomass (kg).
- `biomass_delay_diff_kg::Vector{Float64}`: Delay-difference predicted legal biomass (kg).
- `biomass_leslie_kg::Vector{Float64}`: Leslie stage-transition predicted legal biomass (kg).
- `cv_obs::Float64`: Observed coefficient of variation (sigma / mu).
- `cv_schaefer::Float64`: Schaefer predicted coefficient of variation.
- `cv_delay_diff::Float64`: Delay-difference predicted coefficient of variation.
- `cv_leslie::Float64`: Leslie predicted coefficient of variation.
- `sd_obs::Float64`: Observed standard deviation of biomass.
- `sd_schaefer::Float64`: Schaefer biomass standard deviation.
- `sd_delay_diff::Float64`: Delay-difference biomass standard deviation.
- `sd_leslie::Float64`: Leslie biomass standard deviation.
- `var_delta_obs::Float64`: Observed interannual biomass step variance Var(B_{t+1} - B_t).
- `var_delta_schaefer::Float64`: Schaefer interannual biomass step variance.
- `var_delta_delay_diff::Float64`: Delay-difference interannual biomass step variance.
- `var_delta_leslie::Float64`: Leslie interannual biomass step variance.
- `r2_schaefer::Float64`: Schaefer goodness of fit R^2.
- `r2_delay_diff::Float64`: Delay-difference goodness of fit R^2.
- `r2_leslie::Float64`: Leslie goodness of fit R^2.
- `rmse_schaefer::Float64`: Schaefer root mean squared error.
- `rmse_delay_diff::Float64`: Delay-difference root mean squared error.
- `rmse_leslie::Float64`: Leslie root mean squared error.
- `aic_schaefer::Float64`: Schaefer AIC score.
- `aic_delay_diff::Float64`: Delay-difference AIC score.
- `aic_leslie::Float64`: Leslie AIC score.
- `preferred_model::Symbol`: Model with minimum AIC score (:schaefer, :delay_difference, :leslie).
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

Unified empirical biological parameter and trait covariance model for the snow crab
(*Chionoecetes opilio*), calibrated directly from observational survey data.

# Fields
- `weight_params::Dict{Symbol, Tuple{Float64, Float64, Float64}}`: Mapping from
  stratum (`:male_imm`, `:male_mat`, `:female_imm`, `:female_mat`) to allometric
  triplet `(alpha, beta, sigma)` where ``\\ln(W) = \\ln(\\alpha) + \\beta \\ln(CW) + \\epsilon``.
- `maturity_params::Dict{Symbol, Tuple{Float64, Float64, Float64}}`: Mapping from sex
  (`:male`, `:female`) to `(l50, slope, temp_coef)` where logit maturity satisfies
  ``\\text{logit}(P) = \\beta (\\ln(CW) - \\ln(L_{50})) + \\kappa_{temp} (T - 4.0)``.
- `fecundity_params::Dict{Symbol, Tuple{Float64, Float64, Float64}}`: Mapping from parity
  (`:primiparous`, `:multiparous`) to `(a, b, sigma)` where
  ``\\log_{10}(E) = a + b \\log_{10}(CW) + \\epsilon``.
- `instar_modals::Dict{Int, Tuple{Float64, Float64}}`: Instar stage index mapped
  to empirical modal dimensions `(mean_cw, sd_cw)`.
- `trait_names::Vector{Symbol}`: Trait identifiers in residual covariance matrix.
- `trait_cov::Matrix{Float64}`: Empirical residual covariance matrix across traits.
- `r2_scores::Dict{Symbol, Float64}`: Goodness-of-fit coefficients for sub-models.
- `sample_size::Int`: Number of observational records utilized in model calibration.
"""
struct EmpiricalBiologicalModel
    weight_params::Dict{Symbol, Tuple{Float64, Float64, Float64}}
    maturity_params::Dict{Symbol, Tuple{Float64, Float64, Float64}}
    fecundity_params::Dict{Symbol, Tuple{Float64, Float64, Float64}}
    instar_modals::Dict{Int, Tuple{Float64, Float64}}
    trait_names::Vector{Symbol}
    trait_cov::Matrix{Float64}
    r2_scores::Dict{Symbol, Float64}
    sample_size::Int
end


