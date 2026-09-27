# ==============================================================================
# Generic Species Module
# ==============================================================================
# Provides a configurable, species-agnostic implementation of biological processes.
# Species-specific parameters are loaded from TOML configuration files.
#
# To implement a new species:
# 1. Create a config file (e.g., configs/myspecies.toml) overriding default.toml
# 2. The SpeciesBehavior will use those parameters automatically
# ==============================================================================

"""
    SpeciesBehaviorImpl <: SpeciesBehavior

Concrete species behavior implementation using config-driven parameters.
Holds parameter definitions and dispatches all biological processes.
"""
struct SpeciesBehaviorImpl <: SpeciesBehavior
    params::SpeciesParams
end

SpeciesBehaviorImpl() = SpeciesBehaviorImpl(SpeciesParams())

"""
    default_species_behavior()

Return the default concrete `SpeciesBehavior` implementation.
"""
default_species_behavior() = SpeciesBehaviorImpl()

# --- Empirical Biological Invariants (from config, with fallbacks) ---
# These constants provide backward compatibility and fallbacks
const DEFAULT_FECUNDITY_PRIMIPAROUS_INTERCEPT = -0.7229
const DEFAULT_FECUNDITY_PRIMIPAROUS_SLOPE     = 2.94
const DEFAULT_FECUNDITY_PRIMIPAROUS_ERR       = 0.090
const DEFAULT_FECUNDITY_MULTIPAROUS_INTERCEPT = 0.0158
const DEFAULT_FECUNDITY_MULTIPAROUS_SLOPE     = 2.5811
const DEFAULT_FECUNDITY_MULTIPAROUS_ERR       = 0.145

# -----------------------------------------------------------------------------
# Morphometrics and Allometric Growth
# -----------------------------------------------------------------------------

"""
    stage_size_lower_bound(behavior::SpeciesBehaviorImpl, stage::Real, sex)

Compute the empirical lower bound of morphometric size (carapace width in mm)
for a given developmental stage and sex.

Uses the theoretical growth curves defined by the species parameters.
"""
function stage_size_lower_bound(behavior::SpeciesBehaviorImpl, stage::Real, sex)
    p = behavior.params
    is_female = (sex == 1 || sex == :female || Symbol(sex) == :female)

    # Pre-benthic stages come from a small lookup table split by breakpoints.
    if stage < p.benthic_anchor_stage
        sizes = p.size_lb_immature_sizes
        isempty(sizes) && return 0.0
        for (index, breakpoint) in enumerate(p.size_lb_immature_breakpoints)
            stage <= breakpoint && return sizes[min(index, length(sizes))]
        end
        return sizes[end]
    end

    intercept = is_female ? p.size_lb_female_intercept : p.size_lb_male_intercept
    slope = is_female ? p.size_lb_female_slope : p.size_lb_male_slope
    anchor = is_female ? p.size_lb_female_anchor_stage : p.size_lb_male_anchor_stage
    return exp(intercept + slope * (stage - anchor))
end

"""
    llbm(stage)

Compute empirical size lower bound (mm) for male at developmental `stage`.
"""
llbm(stage) = stage_size_lower_bound(SpeciesBehaviorImpl(), stage, :male)

"""
    llbf(stage)

Compute empirical size lower bound (mm) for female at developmental `stage`.
"""
llbf(stage) = stage_size_lower_bound(SpeciesBehaviorImpl(), stage, :female)

"""
    get_stage_size(stage::Real, sex; kwargs...)

Convenience method computing or sampling carapace width (mm) for `stage` and `sex`.
"""
get_stage_size(stage::Real, sex; kwargs...) =
    get_stage_size(SpeciesBehaviorImpl(), stage, sex; kwargs...)

"""
    growth_increment(behavior::SpeciesBehaviorImpl, log_size::Real; kwargs...)

Predict the natural logarithm of post-molt carapace width, ln(CW_{t+1}),
given pre-molt natural logarithm ln(CW_t), sex, maturity transition status,
nutritional condition factor, and ambient bottom temperature.

Mathematical formulation:
```math
\$\\ln(CW_{t+1}) = \\beta_0 + \\beta_1 \\ln(CW_t) + \\delta_{mat} + \\Delta_{cond} + \\Delta_{temp} + \\epsilon\$
```
"""
function growth_increment(
    behavior::SpeciesBehaviorImpl,
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

    sp = behavior.params
    if !is_female
        intercept = sp.growth_int_male + sp.growth_beta1_male * z1
        slope = sp.growth_slp_male + sp.growth_beta2_male * z1 +
                sp.growth_beta3_male * z2
        mat_effect = (mat == 1) ?
            rand(rng, Normal(sp.growth_maturation_penalty_male_mean,
                             sp.growth_maturation_penalty_male_sd)) : 0.0
    else
        intercept = sp.growth_int_female + sp.growth_beta1_female * z1
        slope = sp.growth_slp_female + sp.growth_beta2_female * z1 +
                sp.growth_beta3_female * z2
        mat_effect = (mat == 1) ?
            rand(rng, Normal(sp.growth_maturation_penalty_female_mean,
                             sp.growth_maturation_penalty_female_sd)) : 0.0
    end

    cond_factor = Float64(condition)
    if cond_factor <= 0.0
        error("Nutritional condition factor must be positive, received: $(cond_factor)")
    end
    cond_mod = sp.growth_condition_coefficient * (cond_factor - 1.0)
    temp_mod = (temp !== nothing) ?
        sp.growth_temperature_coefficient *
        (Float64(temp) - sp.growth_temperature_reference) : 0.0
    residual = rand(rng, Normal(0.0, sp.growth_residual_sd))

    log_size_next = intercept + slope * log_size + mat_effect + cond_mod +
                    temp_mod + residual
    min_expansion = (mat == 1) ? sp.growth_min_expansion_mature :
                                 sp.growth_min_expansion_immature
    return max(log_size_next, log_size + log(1.0 + min_expansion))
end

growth_increment(log_size::Real; kwargs...) =
    growth_increment(SpeciesBehaviorImpl(), log_size; kwargs...)

predict_postgrowth_size(size::Real; kwargs...) =
    predict_postgrowth_size(SpeciesBehaviorImpl(), size; kwargs...)

growth_schedule() = growth_schedule(SpeciesBehaviorImpl())

# -----------------------------------------------------------------------------
# Maturity, Fecundity, and Fishery Selectivity
# -----------------------------------------------------------------------------

"""
    maturity(behavior::SpeciesBehaviorImpl, sex, log_size::Real; temp=nothing, ...)

Calculate probability of morphometric maturity during molt.
Individuals below the `maturity_size_gate` species parameter cannot attain
maturity regardless of the ogive.
"""
function maturity(
    behavior::SpeciesBehaviorImpl,
    sex,
    log_size::Real;
    temp = nothing,
    maturity_quantile::Union{Nothing, Real} = nothing,
    rng::AbstractRNG = Random.default_rng(),
    deterministic::Bool = false
)
    size_gate = behavior.params.maturity_size_gate
    if size_gate > 0.0 && log_size < log(size_gate)
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
        shifted = l50_base + behavior.params.maturity_l50_temp_slope *
                  (Float64(temp) - behavior.params.maturity_temp_reference)
        max(size_gate, shifted)
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
    maturity(SpeciesBehaviorImpl(), sex, log_size; kwargs...)

function maturity(
    size::Real;
    sex = 0,
    params::Union{Nothing, SpeciesParams} = nothing,
    kwargs...
)
    behavior = (params === nothing) ? SpeciesBehaviorImpl() : SpeciesBehaviorImpl(params)
    return maturity(behavior, sex, log(size); kwargs...)
end

"""
    fecundity(behavior::SpeciesBehaviorImpl, size::Real; ...)

Estimate brood fecundity for female from carapace size (mm).
Supports 'moncton' (Sainte-Marie 1993) and 'st.marie' parameterizations.
"""
function fecundity(
    behavior::SpeciesBehaviorImpl,
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

    params = behavior.params
    if param_type == "st.marie"
        return sz^params.fecundity_stmarie_exponent *
               (10.0^params.fecundity_stmarie_coefficient)
    elseif param_type == "moncton"
        if primiparous
            intercept = if !isnothing(fecundity_quantile)
                if fecundity_quantile <= 0.0 || fecundity_quantile >= 1.0
                    error("fecundity_quantile must be strictly in (0, 1), received: $(fecundity_quantile)")
                end
                quantile(Normal(params.fecundity_primiparous_intercept,
                                    params.fecundity_intercept_sd),
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
                quantile(Normal(params.fecundity_multiparous_intercept,
                                    params.fecundity_intercept_sd),
                         Float64(fecundity_quantile))
            else
                params.fecundity_multiparous_intercept
            end
            slope = params.fecundity_multiparous_slope
            fec_log10 = log10(sz) * slope + intercept
            err = deterministic ? 0.0 :
                rand(rng, Normal(0.0, params.fecundity_multiparous_err))
            senescence = (shell_age > params.fecundity_senescence_threshold) ?
                max(params.fecundity_senescence_floor,
                    1.0 - params.fecundity_senescence_rate *
                    (shell_age - params.fecundity_senescence_threshold)) : 1.0
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
    behavior = (params === nothing) ? SpeciesBehaviorImpl() : SpeciesBehaviorImpl(params)
    return fecundity(behavior, size; kwargs...)
end

fecundity(size::AbstractArray; kwargs...) =
    fecundity.(Ref(SpeciesBehaviorImpl()), size; kwargs...)

"""
    selectivity(behavior::SpeciesBehaviorImpl, size::Real; s50=nothing, s50_sigma=nothing)

Calculate gear selectivity or harvest probability as a logistic function
of organism size (mm).
"""
function selectivity(
    behavior::SpeciesBehaviorImpl,
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
    behavior::SpeciesBehaviorImpl,
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

"""
    selectivity(size; params = nothing, s50 = nothing, s50_sigma = nothing, ...)

Fishery retention probability at `size`, using `params` (a `SpeciesParams`) to
supply the defaults.

Mirrors the `maturity` and `fecundity` convenience wrappers. The previous
wrappers forwarded every keyword to the behavior method, which accepts only
`s50` and `s50_sigma`, so a caller that passed `params = ...` — the natural way to
compare parameter sets — got a `MethodError`.
"""
function selectivity(
    size::Real;
    params::Union{Nothing, SpeciesParams} = nothing,
    kwargs...
)
    behavior = (params === nothing) ? SpeciesBehaviorImpl() : SpeciesBehaviorImpl(params)
    return selectivity(behavior, size; kwargs...)
end

function selectivity(
    size::AbstractArray;
    params::Union{Nothing, SpeciesParams} = nothing,
    kwargs...
)
    behavior = (params === nothing) ? SpeciesBehaviorImpl() : SpeciesBehaviorImpl(params)
    return selectivity(behavior, size; kwargs...)
end

"""
    calculate_mass(behavior::SpeciesBehaviorImpl, size::Real, sex; mat=0, stochastic=false, rng=...)

Compute wet body mass (grams) from size (mm), sex, and maturity status
based on empirical allometric equations.
"""
function calculate_mass(
    behavior::SpeciesBehaviorImpl,
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
    params = behavior.params
    alpha, beta_val = if !is_fem
        (mat == 1) ? (params.mass_alpha_male_mature, params.mass_beta_male_mature) :
                     (params.mass_alpha_male_immature, params.mass_beta_male_immature)
    else
        (mat == 1) ? (params.mass_alpha_female_mature, params.mass_beta_female_mature) :
                     (params.mass_alpha_female_immature, params.mass_beta_female_immature)
    end
    base_wt = alpha * (Float64(size)^beta_val)
    if stochastic
        sigma = params.mass_sigma
        residual = rand(rng, Normal(0.0, sigma))
        return base_wt * exp(residual - 0.5 * sigma^2)
    end
    return base_wt
end

calculate_mass(size::Real, sex; kwargs...) =
    calculate_mass(SpeciesBehaviorImpl(), size, sex; kwargs...)

"""
    build_stages(behavior::SpeciesBehaviorImpl; max_male_stage=12, max_female_stage=11)

Construct the discrete stage hierarchy for generalized Leslie stage-transition
modeling guided by empirical growth-stage relationships.
"""
function build_stages(
    behavior::SpeciesBehaviorImpl;
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

build_stages(; kwargs...) = build_stages(SpeciesBehaviorImpl(); kwargs...)

# -----------------------------------------------------------------------------
# Life History and Ecdysis Thresholds
# -----------------------------------------------------------------------------

"""
    stage_schedule_lookup(schedule::AbstractVector, stage::Integer)

Index a stage-indexed schedule vector, clamping out-of-range stages to the
first or last entry so that terminal stages reuse the final schedule value.
"""
function stage_schedule_lookup(schedule::AbstractVector, stage::Integer)
    isempty(schedule) && error("stage schedule must not be empty")
    idx = clamp(Int(stage), 0, length(schedule) - 1) + 1
    return Float64(schedule[idx])
end

"""
    min_intermolt_days(behavior::SpeciesBehavior, stage::Int)

Return the minimum required intermolt duration in days for `stage`, read from
the `intermolt_days_schedule` species parameter.
"""
min_intermolt_days(behavior::SpeciesBehavior, stage::Int) =
    stage_schedule_lookup(behavior.params.intermolt_days_schedule, stage)

min_intermolt_days(stage::Int) = min_intermolt_days(SpeciesBehaviorImpl(), stage)

function min_intermolt_days(behavior::SpeciesBehavior, stage_cat::LifeStage, stage::Int = 0)
    larval = behavior.params.intermolt_days_larval
    if stage_cat === egg
        return stage_schedule_lookup(larval, 0)
    elseif stage_cat === zoea
        return stage_schedule_lookup(larval, 1)
    elseif stage_cat === megalopa
        return stage_schedule_lookup(larval, 2)
    end
    return min_intermolt_days(behavior, stage)
end

min_intermolt_days(stage_cat::LifeStage, stage::Int = 0) =
    min_intermolt_days(SpeciesBehaviorImpl(), stage_cat, stage)

"""
    required_degree_days(behavior::SpeciesBehavior, stage::Int)

Cumulative physiological thermal degree-days (°C·days above 0°C) required to
complete pre-ecdysial preparation at `stage`, read from the
`degree_days_schedule` species parameter.
"""
required_degree_days(behavior::SpeciesBehavior, stage::Int) =
    stage_schedule_lookup(behavior.params.degree_days_schedule, stage)

required_degree_days(stage::Int) = required_degree_days(SpeciesBehaviorImpl(), stage)

function required_degree_days(behavior::SpeciesBehavior, stage_cat::LifeStage, stage::Int = 0)
    larval = behavior.params.degree_days_larval
    if stage_cat === egg
        return stage_schedule_lookup(larval, 0)
    elseif stage_cat === zoea
        return stage_schedule_lookup(larval, 1)
    elseif stage_cat === megalopa
        return stage_schedule_lookup(larval, 2)
    end
    return required_degree_days(behavior, stage)
end

required_degree_days(stage_cat::LifeStage, stage::Int = 0) =
    required_degree_days(SpeciesBehaviorImpl(), stage_cat, stage)

# -----------------------------------------------------------------------------
# Dynamic Behaviors: Development and Reproduction
# -----------------------------------------------------------------------------

"""
    perform_development!(behavior::SpeciesBehavior, org::Organism, model)

Advance physiological development and ecdysis for an organism agent.
"""
function perform_development!(::SpeciesBehavior, org::Organism, model)
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
    perform_reproduction!(behavior::SpeciesBehavior, org::Organism, model)

Execute annual reproductive behaviors (copulation, brood extrusion, hatching release)
for a mature female organism agent.
"""
function perform_reproduction!(::SpeciesBehavior, org::Organism, model)
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
                    phen = model.params.phenology
                    tau_incubation = phen.brood_duration_base *
                                     (phen.brood_temperature_reference /
                                      max(1.0, local_temp))^phen.brood_temperature_exponent
                    incubation_noise = rand(abmrng(model),
                                            Normal(0.0, phen.brood_duration_sd))

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
                    size = model.params.organism.egg_size_mm,
                    mass = model.params.organism.egg_mass_g,
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