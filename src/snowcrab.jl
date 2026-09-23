# ==============================================================================
# Snow Crab (Chionoecetes opilio) Species Module
#
# Centralizes all species-specific parameters, structs, empirical allometry,
# developmental thresholds, phenology, and reproductive behaviors for the
# snow crab.
#
# Primary references:
# - Sainte-Marie, B. (1993). Reproductive cycle and fecundity of primiparous
#   and multiparous female snow crab. CJFAS, 50(10), 2147-2156.
# - Sainte-Marie, B., Raymond, S., & Brêthes, J. C. (1995). Growth and maturation
#   of the benthic stages of male snow crab. CJFAS, 52(5), 903-924.
# - DFO (2020). Assessment of Nova Scotia (4VWX) Snow Crab. DFO Can. Sci. Advis. Sec.
# ==============================================================================

"""
    SnowCrabParams <: SpeciesParams

Biological and allometric parameter container for snow crab (*Chionoecetes opilio*).
"""
Base.@kwdef struct SnowCrabParams <: SpeciesParams
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
end

"""
    SnowCrabBehavior <: SpeciesBehavior

Concrete species behavior implementation for the snow crab (*Chionoecetes opilio*).
Holds parameter definitions and dispatches all biological processes.
"""
struct SnowCrabBehavior <: SpeciesBehavior
    params::SnowCrabParams
end

SnowCrabBehavior() = SnowCrabBehavior(SnowCrabParams())

"""
    default_species_behavior()

Return the default concrete `SpeciesBehavior` implementation (`SnowCrabBehavior`).
"""
default_species_behavior() = SnowCrabBehavior()

"""
    const SnowCrab = Organism

Type alias designating individual `Organism` agents as snow crab agents.
"""
const SnowCrab = Organism

# --- Empirical Biological Invariants ---
const FECUNDITY_PRIMIPAROUS_INTERCEPT = -0.7229
const FECUNDITY_PRIMIPAROUS_SLOPE     = 2.94
const FECUNDITY_PRIMIPAROUS_ERR       = 0.090
const FECUNDITY_MULTIPAROUS_INTERCEPT = 0.0158
const FECUNDITY_MULTIPAROUS_SLOPE     = 2.5811
const FECUNDITY_MULTIPAROUS_ERR       = 0.145

# -----------------------------------------------------------------------------
# Morphometrics and Allometric Growth
# -----------------------------------------------------------------------------

"""
    stage_size_lower_bound(behavior::SnowCrabBehavior, stage::Real, sex)

Compute the empirical lower bound of morphometric size (carapace width in mm)
for snow crab developmental stage and sex.
"""
function stage_size_lower_bound(behavior::SnowCrabBehavior, stage::Real, sex)
    is_female = (sex == 1 || sex == :female || Symbol(sex) == :female)
    if is_female
        if stage < 4.0
            return (stage <= 1.5) ? 0.95 : (stage <= 2.5 ? 1.45 : 2.50)
        end
        i = stage - 4.0
        return exp(2.198848 + 0.315026 * i)
    else
        if stage < 4.0
            return (stage <= 1.5) ? 0.95 : (stage <= 2.5 ? 1.45 : 2.50)
        end
        i = stage - 3.0
        return exp(1.917564 + 0.298914 * i)
    end
end

"""
    llbm(stage)

Compute empirical size lower bound (mm) for male snow crab at developmental `stage`.
"""
llbm(stage) = stage_size_lower_bound(SnowCrabBehavior(), stage, :male)

"""
    llbf(stage)

Compute empirical size lower bound (mm) for female snow crab at developmental `stage`.
"""
llbf(stage) = stage_size_lower_bound(SnowCrabBehavior(), stage, :female)

"""
    get_stage_size(stage::Real, sex; kwargs...)

Convenience method computing or sampling snow crab carapace width (mm) for `stage` and `sex`.
"""
get_stage_size(stage::Real, sex; kwargs...) =
    get_stage_size(SnowCrabBehavior(), stage, sex; kwargs...)

"""
    growth_increment(behavior::SnowCrabBehavior, log_size::Real; kwargs...)

Predict the natural logarithm of post-molt carapace width, ln(CW_{t+1}),
given pre-molt natural logarithm ln(CW_t), sex, maturity transition status,
nutritional condition factor, and ambient bottom temperature for snow crab.

Mathematical formulation:
```math
\\ln(CW_{t+1}) = \\beta_0 + \\beta_1 \\ln(CW_t) + \\delta_{mat} + \\Delta_{cond} + \\Delta_{temp} + \\epsilon
```
"""
function growth_increment(
    behavior::SnowCrabBehavior,
    log_size::Real;
    sex = 0,
    mat::Integer = 0,
    temp = nothing,
    condition::Real = 1.0,
    rng::AbstractRNG = Random.default_rng(),
    growth_quantile::Union{Nothing, Real} = nothing
)
    is_female = (sex == 1 || sex == :female || Symbol(sex) == :female)
    z1 = if !isnothing(growth_quantile)
        if growth_quantile <= 0.0 || growth_quantile >= 1.0
            error("growth_quantile must be strictly in (0, 1), received: $(growth_quantile)")
        end
        quantile(Normal(0.0, 1.0), Float64(growth_quantile))
    else
        rand(rng, Normal(0.0, 1.0))
    end
    z2 = rand(rng, Normal(0.0, 1.0))

    params = behavior.params
    if !is_female
        intercept = params.growth_int_male + params.growth_beta1_male * z1
        slope = params.growth_slp_male + params.growth_beta2_male * z1 +
                params.growth_beta3_male * z2
        mat_effect = (mat == 1) ? rand(rng, Normal(-0.13564, 0.00922)) : 0.0
    else
        intercept = params.growth_int_female + params.growth_beta1_female * z1
        slope = params.growth_slp_female + params.growth_beta2_female * z1 +
                params.growth_beta3_female * z2
        mat_effect = (mat == 1) ? rand(rng, Normal(-0.1310, 0.0252)) : 0.0
    end

    cond_factor = Float64(condition)
    if cond_factor <= 0.0
        error("Nutritional condition factor must be positive, received: $(cond_factor)")
    end
    cond_mod = 0.05 * (cond_factor - 1.0)
    temp_mod = (temp !== nothing) ? -0.004 * (Float64(temp) - 4.0) : 0.0
    residual = rand(rng, Normal(0.0, 0.040))

    log_size_next = intercept + slope * log_size + mat_effect + cond_mod +
                    temp_mod + residual
    min_expansion = (mat == 1) ? 0.06 : 0.12
    return max(log_size_next, log_size + log(1.0 + min_expansion))
end

growth_increment(log_size::Real; kwargs...) =
    growth_increment(SnowCrabBehavior(), log_size; kwargs...)

predict_postgrowth_size(size::Real; kwargs...) =
    predict_postgrowth_size(SnowCrabBehavior(), size; kwargs...)

growth_schedule() = growth_schedule(SnowCrabBehavior())

# -----------------------------------------------------------------------------
# Maturity, Fecundity, and Fishery Selectivity
# -----------------------------------------------------------------------------

"""
    maturity(behavior::SnowCrabBehavior, sex, log_size::Real; temp=nothing, ...)

Calculate probability of morphometric maturity during molt for snow crab.
Minimum size gate: Stages 4-6 (size < 35 mm) cannot undergo terminal molt.
"""
function maturity(
    behavior::SnowCrabBehavior,
    sex,
    log_size::Real;
    temp = nothing,
    maturity_quantile::Union{Nothing, Real} = nothing,
    rng::AbstractRNG = Random.default_rng(),
    deterministic::Bool = false
)
    if log_size < log(35.0)
        return 0.0
    end

    is_female = (sex == 1 || sex == :female || Symbol(sex) == :female)
    l50_base = if deterministic
        !is_female ? behavior.params.maturity_l50_male :
                     behavior.params.maturity_l50_female
    elseif !isnothing(maturity_quantile)
        if maturity_quantile <= 0.0 || maturity_quantile >= 1.0
            error("maturity_quantile must be strictly in (0, 1), received: $(maturity_quantile)")
        end
        if !is_female
            quantile(Normal(behavior.params.maturity_l50_male,
                            behavior.params.maturity_l50_male_sd),
                     Float64(maturity_quantile))
        else
            quantile(Normal(behavior.params.maturity_l50_female,
                            behavior.params.maturity_l50_female_sd),
                     Float64(maturity_quantile))
        end
    else
        if !is_female
            rand(rng, Normal(behavior.params.maturity_l50_male,
                             behavior.params.maturity_l50_male_sd))
        else
            rand(rng, Normal(behavior.params.maturity_l50_female,
                             behavior.params.maturity_l50_female_sd))
        end
    end

    slope = if deterministic
        !is_female ? behavior.params.maturity_slope_male :
                     behavior.params.maturity_slope_female
    elseif !isnothing(maturity_quantile)
        if !is_female
            quantile(Normal(behavior.params.maturity_slope_male,
                            behavior.params.maturity_slope_male_sd),
                     Float64(maturity_quantile))
        else
            quantile(Normal(behavior.params.maturity_slope_female,
                            behavior.params.maturity_slope_female_sd),
                     Float64(maturity_quantile))
        end
    else
        if !is_female
            rand(rng, Normal(behavior.params.maturity_slope_male,
                             behavior.params.maturity_slope_male_sd))
        else
            rand(rng, Normal(behavior.params.maturity_slope_female,
                             behavior.params.maturity_slope_female_sd))
        end
    end

    l50 = if temp !== nothing
        t_val = Float64(temp)
        max(35.0, l50_base - 2.0 * (t_val - 4.0))
    else
        l50_base
    end

    linear_pred = slope * (log_size - log(l50))
    if linear_pred > 40.0
        return 1.0
    elseif linear_pred < -40.0
        return 0.0
    end
    return 1.0 / (1.0 + exp(-linear_pred))
end

maturity(sex, log_size::Real; kwargs...) =
    maturity(SnowCrabBehavior(), sex, log_size; kwargs...)

function maturity(
    size::Real;
    sex = 0,
    params::Union{Nothing, SpeciesParams} = nothing,
    kwargs...
)
    behavior = (params === nothing) ? SnowCrabBehavior() : SnowCrabBehavior(params)
    return maturity(behavior, sex, log(size); kwargs...)
end

"""
    fecundity(behavior::SnowCrabBehavior, size::Real; ...)

Estimate brood fecundity for female snow crab from carapace size (mm).
Supports 'moncton' (Sainte-Marie 1993) and 'st.marie' parameterizations.
"""
function fecundity(
    behavior::SnowCrabBehavior,
    size::Real;
    is_primiparous::Bool = true,
    method::Union{Nothing, String} = nothing,
    fecundity_parameterization::Union{Nothing, String} = nothing,
    shell_age::Real = 1.0,
    rng::AbstractRNG = Random.default_rng(),
    deterministic::Bool = false,
    fecundity_quantile::Union{Nothing, Real} = nothing
)
    if size <= 0.0
        error("Organism size must be positive for fecundity estimation, got: $(size)")
    end
    sz = Float64(size)
    param_type = if fecundity_parameterization !== nothing
        fecundity_parameterization
    elseif method !== nothing
        occursin("st.marie", method) ? "st.marie" : "moncton"
    else
        behavior.params.fecundity_parameterization
    end

    primiparous = if method !== nothing
        occursin("primiparous", method) ? true :
            (occursin("multiparous", method) ? false : is_primiparous)
    else
        is_primiparous
    end

    if param_type == "st.marie"
        return sz^2.725 * (10.0^0.7311)
    elseif param_type == "moncton"
        params = behavior.params
        if primiparous
            intercept = if !isnothing(fecundity_quantile)
                if fecundity_quantile <= 0.0 || fecundity_quantile >= 1.0
                    error("fecundity_quantile must be strictly in (0, 1), received: $(fecundity_quantile)")
                end
                quantile(Normal(params.fecundity_primiparous_intercept, 0.05),
                         Float64(fecundity_quantile))
            else
                params.fecundity_primiparous_intercept
            end
            slope = params.fecundity_primiparous_slope
            fec_log10 = log10(sz) * slope + intercept
            err = deterministic ? 0.0 :
                rand(rng, Normal(0.0, params.fecundity_primiparous_err))
            return 10.0^(fec_log10 + err)
        else
            intercept = if !isnothing(fecundity_quantile)
                if fecundity_quantile <= 0.0 || fecundity_quantile >= 1.0
                    error("fecundity_quantile must be strictly in (0, 1), received: $(fecundity_quantile)")
                end
                quantile(Normal(params.fecundity_multiparous_intercept, 0.05),
                         Float64(fecundity_quantile))
            else
                params.fecundity_multiparous_intercept
            end
            slope = params.fecundity_multiparous_slope
            fec_log10 = log10(sz) * slope + intercept
            err = deterministic ? 0.0 :
                rand(rng, Normal(0.0, params.fecundity_multiparous_err))
            senescence = (shell_age > 3.0) ?
                max(0.60, 1.0 - 0.08 * (shell_age - 3.0)) : 1.0
            return 10.0^(fec_log10 + err) * senescence
        end
    else
        error("Unknown fecundity parameterization: $(param_type). Valid: 'moncton', 'st.marie'")
    end
end

function fecundity(
    size::Real;
    params::Union{Nothing, SpeciesParams} = nothing,
    kwargs...
)
    behavior = (params === nothing) ? SnowCrabBehavior() : SnowCrabBehavior(params)
    return fecundity(behavior, size; kwargs...)
end

fecundity(size::AbstractArray; kwargs...) =
    fecundity.(Ref(SnowCrabBehavior()), size; kwargs...)

"""
    selectivity(behavior::SnowCrabBehavior, size::Real; s50=nothing, s50_sigma=nothing)

Calculate gear selectivity or harvest probability as a logistic function
of crab size (mm) for snow crab.
"""
function selectivity(
    behavior::SnowCrabBehavior,
    size::Real;
    s50::Union{Nothing, Real} = nothing,
    s50_sigma::Union{Nothing, Real} = nothing
)
    effective_s50 = s50 !== nothing ? Float64(s50) : behavior.params.selectivity_s50
    effective_sigma = s50_sigma !== nothing ? Float64(s50_sigma) :
                      behavior.params.selectivity_s50_sigma
    if effective_sigma <= 0.0
        error("Selectivity bandwidth s50_sigma must be positive, got: $(effective_sigma)")
    end
    diff = (Float64(size) - effective_s50) / effective_sigma
    if diff > 40.0
        return 1.0
    elseif diff < -40.0
        return 0.0
    end
    return 1.0 / (1.0 + exp(-diff))
end

function selectivity(
    behavior::SnowCrabBehavior,
    size::AbstractArray;
    s50::Union{Nothing, Real} = nothing,
    s50_sigma::Union{Nothing, Real} = nothing
)
    effective_s50 = s50 !== nothing ? Float64(s50) : behavior.params.selectivity_s50
    effective_sigma = s50_sigma !== nothing ? Float64(s50_sigma) :
                      behavior.params.selectivity_s50_sigma
    if effective_sigma <= 0.0
        error("Selectivity bandwidth s50_sigma must be positive, got: $(effective_sigma)")
    end
    diff = (Float64.(size) .- effective_s50) ./ effective_sigma
    return 1.0 ./ (1.0 .+ exp.(-diff))
end

selectivity(size::Real; kwargs...) = selectivity(SnowCrabBehavior(), size; kwargs...)
selectivity(size::AbstractArray; kwargs...) = selectivity(SnowCrabBehavior(), size; kwargs...)

"""
    calculate_mass(behavior::SnowCrabBehavior, size::Real, sex; mat=0, stochastic=false, rng=...)

Compute wet body mass (grams) from size (mm), sex, and maturity status
based on empirical allometric equations for snow crab (Sainte-Marie et al. 1995; DFO 2020).
"""
function calculate_mass(
    behavior::SnowCrabBehavior,
    size::Real,
    sex;
    mat::Integer = 0,
    stochastic::Bool = false,
    rng::AbstractRNG = Random.default_rng()
)
    if size <= 0.0
        return 0.0
    end
    is_fem = (sex == 1 || sex == :female || Symbol(sex) == :female)
    alpha, beta_val = if !is_fem
        (mat == 1) ? (0.0001830, 3.160) : (0.0002665, 3.098)
    else
        (mat == 1) ? (0.0004500, 2.920) : (0.0003440, 2.980)
    end
    base_wt = alpha * (Float64(size)^beta_val)
    if stochastic
        sigma = 0.08
        residual = rand(rng, Normal(0.0, sigma))
        return base_wt * exp(residual - 0.5 * sigma^2)
    end
    return base_wt
end

calculate_mass(size::Real, sex; kwargs...) =
    calculate_mass(SnowCrabBehavior(), size, sex; kwargs...)

"""
    build_stages(behavior::SnowCrabBehavior; max_male_stage=12, max_female_stage=11)

Construct the discrete stage hierarchy for generalized Leslie stage-transition
modeling guided by empirical snow crab growth-stage relationships.
"""
function build_stages(
    behavior::SnowCrabBehavior;
    max_male_stage::Int = 12,
    max_female_stage::Int = 11
)
    stages = StageDefinition[]
    idx = 1
    # Stage 1: Stage 4 (pooled sexes)
    push!(stages, StageDefinition(
        idx,
        "Stage 4 (Recruit)",
        4,
        undifferentiated,
        (7.0, 14.5),
        10.7,
        calculate_mass(behavior, 10.7, :male; mat = 0),
        false,
        false
    ))
    idx += 1

    # Stage 2: Stage 5 (pooled sexes)
    push!(stages, StageDefinition(
        idx,
        "Stage 5 (Juvenile)",
        5,
        undifferentiated,
        (14.5, 22.0),
        18.0,
        calculate_mass(behavior, 18.0, :male; mat = 0),
        false,
        false
    ))
    idx += 1

    # Male Stages: Stage 6 to max_male_stage
    male_bounds = [
        (6, (22.0, 31.0), 26.5),
        (7, (31.0, 42.5), 36.5),
        (8, (42.5, 58.0), 50.0),
        (9, (58.0, 78.0), 68.0),
        (10, (78.0, 95.0), 86.5),
        (11, (95.0, 115.0), 105.0),
        (12, (115.0, 160.0), 130.0)
    ]
    for (stg, rng_b, size_m) in male_bounds
        if stg <= max_male_stage
            is_legal = (size_m >= 95.0 || stg >= 11)
            is_term = (stg == max_male_stage)
            lbl = is_term ? "Stage $(stg)+ Male" : "Stage $(stg) Male"
            wt = calculate_mass(behavior, size_m, :male; mat = is_legal ? 1 : 0)
            push!(stages, StageDefinition(
                idx,
                lbl,
                stg,
                male,
                rng_b,
                size_m,
                wt,
                is_legal,
                false
            ))
            idx += 1
        end
    end

    # Female Stages: Stage 6 to max_female_stage
    female_bounds = [
        (6, (22.0, 30.5), 26.0),
        (7, (30.5, 41.5), 35.5),
        (8, (41.5, 54.5), 47.5),
        (9, (54.5, 68.0), 61.0),
        (10, (68.0, 82.0), 75.0),
        (11, (82.0, 110.0), 90.0)
    ]
    for (stg, rng_b, size_m) in female_bounds
        if stg <= max_female_stage
            is_repro = (stg >= 9)
            is_term = (stg == max_female_stage)
            lbl = is_term ? "Stage $(stg)+ Female" : "Stage $(stg) Female"
            wt = calculate_mass(behavior, size_m, :female; mat = is_repro ? 1 : 0)
            push!(stages, StageDefinition(
                idx,
                lbl,
                stg,
                female,
                rng_b,
                size_m,
                wt,
                false,
                is_repro
            ))
            idx += 1
        end
    end
    return stages
end

build_stages(; kwargs...) = build_stages(SnowCrabBehavior(); kwargs...)

# -----------------------------------------------------------------------------
# Life History and Ecdysis Thresholds
# -----------------------------------------------------------------------------

"""
    min_intermolt_days(behavior::SnowCrabBehavior, stage::Int)

Return the minimum required intermolt duration in days for snow crab at `stage`.
"""
function min_intermolt_days(behavior::SnowCrabBehavior, stage::Int)
    if stage <= 0
        return 60.0
    elseif stage == 1
        return 60.0
    elseif stage == 2
        return 75.0
    elseif stage == 3
        return 90.0
    elseif stage == 4
        return 105.0
    elseif stage == 5
        return 120.0
    elseif stage == 6
        return 150.0
    elseif stage == 7
        return 200.0
    else
        return 300.0
    end
end

min_intermolt_days(stage::Int) = min_intermolt_days(SnowCrabBehavior(), stage)
min_intermolt_days(stage_cat::LifeStage, stage::Int = 0) =
    min_intermolt_days(SnowCrabBehavior(), stage_cat, stage)

"""
    required_degree_days(behavior::SnowCrabBehavior, stage::Int)

Calculate cumulative physiological thermal degree-days (°C·days above 0°C)
required to complete pre-ecdysial preparation for snow crab at `stage`.
"""
function required_degree_days(behavior::SnowCrabBehavior, stage::Int)
    if stage <= 0
        return 150.0
    elseif stage == 1
        return 200.0
    elseif stage == 2
        return 250.0
    elseif stage == 3
        return 300.0
    elseif stage == 4
        return 320.0
    elseif stage == 5
        return 350.0
    elseif stage == 6
        return 550.0
    elseif stage == 7
        return 750.0
    else
        return 950.0
    end
end

required_degree_days(stage::Int) = required_degree_days(SnowCrabBehavior(), stage)
required_degree_days(stage_cat::LifeStage, stage::Int = 0) =
    required_degree_days(SnowCrabBehavior(), stage_cat, stage)

# -----------------------------------------------------------------------------
# Dynamic Behaviors: Development and Reproduction
# -----------------------------------------------------------------------------

"""
    perform_development!(behavior::SnowCrabBehavior, org::Organism, model)

Advance physiological development and ecdysis for a snow crab agent.
"""
function perform_development!(::SnowCrabBehavior, org::Organism, model)
    if !model.params.toggles.use_molting || org.is_mature
        return nothing
    end

    day_of_year = mod(model.current_time, 365.0)
    is_molt_season = (model.params.phenology.molt_season_start <= day_of_year <=
                      model.params.phenology.molt_season_end)

    if is_molt_season
        days_since_molt = model.current_time - org.last_growth_event_time
        min_days = min_intermolt_days(org.behavior, org.stage)
        if days_since_molt >= min_days
            current_temp = model.temperature_field[org.pos...]
            p_molt = molt_probability(org, current_temp, model.current_time;
                                      params = model.params)
            if rand(abmrng(model)) < p_molt
                perform_molt!(org, model.current_time, current_temp;
                              params = model.params, rng = abmrng(model))
                model.total_molts += 1
            end
        end
    end
    return nothing
end

"""
    perform_reproduction!(behavior::SnowCrabBehavior, org::Organism, model)

Execute annual reproductive behaviors (copulation, brood extrusion, hatching release)
for a female snow crab agent.
"""
function perform_reproduction!(::SnowCrabBehavior, org::Organism, model)
    if !model.params.toggles.use_reproduction
        return nothing
    end

    day_of_year = mod(model.current_time, 365.0)
    is_mating_season = (model.params.phenology.mating_season_start <= day_of_year <=
                        model.params.phenology.mating_season_end)

    if org.is_mature && org.sex === female
        # 1. Copulation & Clutch Extrusion
        if is_mating_season && org.is_mating_ready && org.egg_count == 0
            radius = model.params.phenology.mating_search_radius
            nearby_males = filter(
                m -> m.is_mature && m.sex === male,
                collect(nearby_agents(org, model, radius))
            )
            if !isempty(nearby_males)
                if model.params.toggles.use_spermatheca
                    male_weights = [m.size^2 for m in nearby_males]
                    selected_male = sample_weighted(nearby_males, male_weights,
                                                    abmrng(model))
                    org.sperm_clutches = (selected_male.size >= 95.0) ? 3 : 2
                    selected_male.energy_reserve = max(0.0,
                        selected_male.energy_reserve - 8.0)
                end

                can_extrude = !model.params.toggles.use_spermatheca ||
                              (org.sperm_clutches > 0)
                if can_extrude
                    if model.params.toggles.use_spermatheca
                        org.sperm_clutches -= 1
                    end

                    egg_estimate = fecundity(
                        org.behavior,
                        org.size;
                        is_primiparous = (org.sperm_clutches == 0),
                        fecundity_quantile = org.trait_fecundity_quantile,
                        rng = abmrng(model)
                    )
                    cx = min(max(org.pos[1], 1), size(model.temperature_field, 1))
                    cy = min(max(org.pos[2], 1), size(model.temperature_field, 2))
                    local_temp = model.temperature_field[cx, cy]
                    tau_incubation = 365.0 * (5.0 / max(1.0, local_temp))^0.30
                    incubation_noise = rand(abmrng(model), Normal(0.0, 5.0))

                    org.egg_count = max(round(Int, egg_estimate), 1000)
                    org.hatch_day = model.current_time + tau_incubation + incubation_noise
                    org.is_mating_ready = false

                    org.mass = calculate_mass(org;
                        include_eggs = true,
                        rng = abmrng(model)
                    )
                    model.total_eggs_spawned += org.egg_count
                    model.annual_eggs_spawned += org.egg_count
                end
            end
        end

        # 2. Clutch Release upon Hatching
        if org.egg_count > 0 && model.current_time >= org.hatch_day
            if haskey(abmproperties(model), :aggregate_populations)
                new_group = AggregateGroup(
                    id = model.aggregate_populations.next_group_id,
                    stage = 0,
                    abundance = Float64(org.egg_count),
                    size = 0.60,
                    mass = 0.00007,
                    age_days = 0.0,
                    degree_days = 0.0,
                    last_growth_time = model.current_time,
                    pos = org.pos
                )
                model.aggregate_populations.next_group_id += 1
                push!(model.aggregate_populations.groups, new_group)
            end

            org.egg_count = 0
            org.is_mating_ready = true
            org.mass = calculate_mass(org; include_eggs = false, rng = abmrng(model))
        end
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Convenience Methods and Aliases for Snow Crab Workflows
# -----------------------------------------------------------------------------


crab_color(args...; kwargs...) = organism_color(args...; kwargs...)
crab_size(args...; kwargs...) = organism_size(args...; kwargs...)
crab_marker(args...; kwargs...) = organism_marker(args...; kwargs...)
crab_heatarray(args...; kwargs...) = organism_heatarray(args...; kwargs...)
crab_disperse!(args...; kwargs...) = organism_disperse!(args...; kwargs...)
weight_if_adult_female(args...; kwargs...) = mass_if_adult_female(args...; kwargs...)
weight_if_adult_male(args...; kwargs...) = mass_if_adult_male(args...; kwargs...)
weight_if_legal_male(args...; kwargs...) = mass_if_legal_male(args...; kwargs...)

sample_population(year::Int = -1; kwargs...) =
    sample_population(default_species_behavior(), year; kwargs...)
sample_crab_traits(args...; kwargs...) = sample_organism_traits(args...; kwargs...)
