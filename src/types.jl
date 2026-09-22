using Agents [L5]

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

    # Individual biological traits (sampled at recruitment)
    trait_l50::Float64          # Carapace width for 50% maturity
    trait_maturity_slope::Float64 # Slope of the maturity ogive
    trait_fecundity_intercept::Float64 # intercept for log-scale fecundity
    trait_fecundity_slope::Float64     # slope for log-scale fecundity
    trait_growth_intercept::Float64    # ln(CW) growth intercept (Sainte-Marie et al.)
    trait_growth_slope::Float64        # ln(CW) growth slope
end

is_adult(crab::SnowCrab) = crab.stage === adult
is_adult_male(crab::SnowCrab) = is_adult(crab) && crab.sex === male
is_adult_female(crab::SnowCrab) = is_adult(crab) && crab.sex === female
is_ovigerous(crab::SnowCrab) = crab.stage === adult && crab.sex === female && crab.eggs_count > 0
is_legal_male(crab::SnowCrab) = is_adult_male(crab) && crab.size_mm >= 95.0 # placeholder for param access if needed, but usually handled in sim
is_legal(crab::SnowCrab) = is_adult_male(crab) && crab.size_mm >= 95.0

weight_if_adult_female(crab::SnowCrab) = crab.weight_g # Placeholder for logic from outline
weight_if_adult_male(crab::SnowCrab) = crab.weight_g  # Placeholder for logic from outline
weight_if_legal_male(crab::SnowCrab) = crab.weight_g   # Placeholder for logic from outline

struct InstarStageDefinition end
struct LeslieMatrixFit end
struct MacroModelComparison end
struct EmpiricalBiologicalModel end

Base.@kwdef struct IBMParameters
    # Feature Toggles (Modular Architecture)
    use_energetics::Bool = true
    use_thermal_stress::Bool = true
    use_food_dynamics::Bool = true
    use_habitat_taxis::Bool = true
    use_cannibalism::Bool = true
    use_senescence::Bool = true

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
    initial_energy::Float64 = 10.0
    molt_probability_scale::Float64 = 45.0

    # Maturity Parameters (L50 and slope for log-logistic ogives)
    maturity_l50_male::Float64 = 91.8
    maturity_l50_male_sd::Float64 = 1.2
    maturity_slope_male::Float64 = 6.9957
    maturity_slope_male_sd::Float64 = 0.25

    maturity_l50_female::Float64 = 51.6
    maturity_l50_female_sd::Float64 = 0.8
    maturity_slope_female::Float64 = 12.9812
    maturity_slope_female_sd::Float64 = 0.40

    # Fecundity Parameters (Moncton models)
    fecundity_moncton_prim_m_log10_intercept::Float64 = -0.7229
    fecundity_moncton_prim_m_log10_slope::Float64 = 2.94
    fecundity_moncton_prim_m_err::Float64 = 0.090

    fecundity_moncton_mult_m_log10_intercept::Float64 = 0.0158
    fecundity_moncton_mult_m_log10_slope::Float64 = 2.5811
    fecundity_moncton_mult_m_err::Float64 = 0.145

    # Selectivity Parameters
    # Growth Parameters (Bivariate Normal for intercept and slope)
    growth_int_f::Float64 = 0.1931
    growth_beta1_f::Float64 = 0.0545
    growth_slp_f::Float64 = 1.0372
    growth_beta2_f::Float64 = -0.017376
    growth_beta3_f::Float64 = 0.005068

    growth_int_m::Float64 = 0.30153
    growth_beta1_m::Float64 = 0.02198
    growth_slp_m::Float64 = 1.00027
    growth_beta2_m::Float64 = -0.00643655
    growth_beta3_m::Float64 = 0.0017492

    # Selectivity Parameters
    selectivity_s50::Float64 = 95.0
    selectivity_s50_sigma::Float64 = 5.0

    dt::Float64 = 1.0
end

function IBMParameters(dict::AbstractDict)
    params = IBMParameters()
    for (k, v) in dict
        sym_k = Symbol(k)
        if hasproperty(params, sym_k)
            setproperty!(params, sym_k, v)
        end
    end
    return params
end
