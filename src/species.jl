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
    FIRST_COHORT_STAGE

The lowest stage that exists as an `AggregateGroup`.

Stage 0 is the brooded egg, which is carried by a female rather than being a
cohort, so a clutch that hatches — and the seeded cohorts in `initialize_model` —
begin at stage 1. Both call sites derive it from here rather than writing `0`: with
a `0` at either, eggs and larvae shared a bin and every pelagic stage below was
offset by one.
"""
const FIRST_COHORT_STAGE = 1

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

For a sex-specific stage the bound follows a saturating curve in log size,

    ln S(stage) = ln L + (intercept - ln L) * exp(-decay * (stage - anchor))

which equals `exp(intercept)` at the anchor stage and tends to the asymptotic
size `L`. The curve therefore cannot produce an animal larger than the species
gets, which the previous `exp(intercept + slope * (stage - anchor))` form could
and did: it is unbounded for any positive slope.

Stages below `benthic_anchor_stage` come from a small lookup table, because size
there is far from the curve.
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

    if is_female
        intercept = p.size_lb_female_intercept
        decay = p.size_lb_female_decay
        anchor = p.size_lb_female_anchor_stage
        asymptote = p.max_size_female
    else
        intercept = p.size_lb_male_intercept
        decay = p.size_lb_male_decay
        anchor = p.size_lb_male_anchor_stage
        asymptote = p.max_size_male
    end
    asymptote > 0.0 ||
        error("asymptotic size must be positive, received: $(asymptote)")
    ln_l = log(asymptote)
    ln_s = ln_l + (intercept - ln_l) * exp(-decay * (stage - anchor))
    # A non-positive decay removes the saturation and restores the unbounded form,
    # so refuse it rather than silently reintroduce the defect.
    decay > 0.0 ||
        error("stage size decay must be positive, received: $(decay)")
    return exp(ln_s)
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
    build_stages(behavior::SpeciesBehaviorImpl; max_male_stage=nothing, max_female_stage=nothing)

Construct the discrete stage hierarchy for generalized Leslie stage-transition
modeling guided by empirical growth-stage relationships.

The stage bounds come from the species parameters, so a species change means
editing configuration rather than this function. `max_male_stage` and
`max_female_stage` override the configured terminal stages.
"""
function build_stages(
    behavior::SpeciesBehaviorImpl;
    max_male_stage::Union{Nothing, Int} = nothing,
    max_female_stage::Union{Nothing, Int} = nothing
)
    sp = behavior.params
    male_limit = max_male_stage === nothing ? sp.leslie_max_male_stage : max_male_stage
    female_limit = max_female_stage === nothing ? sp.leslie_max_female_stage :
                   max_female_stage
    # The four vectors of a stage table are read in parallel, so a mismatch in
    # length would silently pair a size range with the wrong stage's mean.
    pooled_lens = length.(
        (
            sp.stage_size_range_stage1,
            sp.stage_size_range_lower_stage1,
            sp.stage_size_range_upper_stage1,
            sp.stage_size_range_mean_stage1
        )
    )
    all(==(first(pooled_lens)), pooled_lens) || error(
        "pooled-sex stage size tables must be the same length; got $(pooled_lens)"
    )
    n = first(pooled_lens)

    stages = StageDefinition[]
    idx = 1
    # Pooled-sex stages: the recruit and juvenile bins, before the sexes diverge.
    for k in 1:n
        push!(
            stages,
            StageDefinition(
                idx,
                "Stage $(Int(sp.stage_size_range_stage1[k])) ($(k == 1 ? "Recruit" : "Juvenile"))",
                Int(sp.stage_size_range_stage1[k]),
                undifferentiated,
                (sp.stage_size_range_lower_stage1[k], sp.stage_size_range_upper_stage1[k]),
                sp.stage_size_range_mean_stage1[k],
                calculate_mass(behavior, sp.stage_size_range_mean_stage1[k], :male; mat = 0),
                false,
                false
            )
        )
        idx += 1
    end

    male_lens = length.(
        (
            sp.stage_size_range_stage_male,
            sp.stage_size_range_lower_male,
            sp.stage_size_range_upper_male,
            sp.stage_size_range_mean_male
        )
    )
    all(==(first(male_lens)), male_lens) ||
        error("male stage size tables must all be the same length; got $(male_lens)")
    nm = first(male_lens)
    for k in 1:nm
        stg = Int(sp.stage_size_range_stage_male[k])
        if stg <= male_limit
            size_m = sp.stage_size_range_mean_male[k]
            is_legal = (size_m >= sp.leslie_legal_size_male ||
                        stg >= sp.leslie_first_legal_stage_male)
            is_term = (stg == male_limit)
            lbl = is_term ? "Stage $(stg)+ Male" : "Stage $(stg) Male"
            wt = calculate_mass(behavior, size_m, :male; mat = is_legal ? 1 : 0)
            push!(
                stages,
                StageDefinition(
                    idx,
                    lbl,
                    stg,
                    male,
                    (sp.stage_size_range_lower_male[k], sp.stage_size_range_upper_male[k]),
                    size_m,
                    wt,
                    is_legal,
                    false
                )
            )
            idx += 1
        end
    end

    female_lens = length.(
        (
            sp.stage_size_range_stage_female,
            sp.stage_size_range_lower_female,
            sp.stage_size_range_upper_female,
            sp.stage_size_range_mean_female
        )
    )
    all(==(first(female_lens)), female_lens) ||
        error("female stage size tables must all be the same length; got $(female_lens)")
    nf = first(female_lens)
    for k in 1:nf
        stg = Int(sp.stage_size_range_stage_female[k])
        if stg <= female_limit
            is_repro = (stg >= sp.leslie_first_repro_stage_female)
            is_term = (stg == female_limit)
            lbl = is_term ? "Stage $(stg)+ Female" : "Stage $(stg) Female"
            wt = calculate_mass(
                behavior, sp.stage_size_range_mean_female[k], :female;
                mat = is_repro ? 1 : 0
            )
            push!(
                stages,
                StageDefinition(
                    idx,
                    lbl,
                    stg,
                    female,
                    (
                        sp.stage_size_range_lower_female[k],
                        sp.stage_size_range_upper_female[k]
                    ),
                    sp.stage_size_range_mean_female[k],
                    wt,
                    false,
                    is_repro
                )
            )
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
    stg = Int(stage)
    if stg < 0
        throw(DomainError(stg, "Developmental stage index cannot be negative"))
    end
    idx = min(stg, length(schedule) - 1) + 1
    return Float64(schedule[idx])
end

"""
    larval_stage_count(behavior::SpeciesBehavior)

Number of leading developmental stages that follow the pelagic larval schedule
rather than the benthic schedule.

The larval schedules are declared per pelagic stage, egg then zoea then megalopa,
so their own length states how far the pelagic block reaches. Everything from that
stage onward develops on `intermolt_days_schedule` and `degree_days_schedule`.

This is the accessor the simulation uses. It previously read only the benthic
schedule, so the eggs and larvae were advanced on the benthic numbers — 60 days
and 150 degree-days for an egg that is meant to hatch on nothing — and the pelagic
schedules were configurable but inert.
"""
larval_stage_count(behavior::SpeciesBehavior) =
    length(behavior.params.intermolt_days_larval)

"""
    min_intermolt_days(behavior::SpeciesBehavior, stage::Int)

Return the minimum required intermolt duration in days for `stage`.

Leading pelagic stages read `intermolt_days_larval`; the rest read
`intermolt_days_schedule`.
"""
function min_intermolt_days(behavior::SpeciesBehavior, stage::Int)
    stage < larval_stage_count(behavior) &&
        return stage_schedule_lookup(behavior.params.intermolt_days_larval, stage)
    return stage_schedule_lookup(behavior.params.intermolt_days_schedule, stage)
end

min_intermolt_days(stage::Int) = min_intermolt_days(SpeciesBehaviorImpl(), stage)

function min_intermolt_days(behavior::SpeciesBehavior, stage_cat::LifeStage, stage::Int = 0)
    # Route through the stage-indexed form so the pelagic mapping is written once.
    # This overload and the `Int` overload used to each restate that egg, zoea and
    # megalopa are stages 0, 1 and 2, and the stage-indexed form already reads the
    # larval schedules for exactly those stages.
    idx = life_stage_index(stage_cat)
    return min_intermolt_days(behavior, idx === nothing ? stage : idx)
end

min_intermolt_days(stage_cat::LifeStage, stage::Int = 0) =
    min_intermolt_days(SpeciesBehaviorImpl(), stage_cat, stage)

"""
    life_stage_index(stage_cat::LifeStage)

Developmental stage number a pelagic life stage occupies, which is the index into
`intermolt_days_larval` and `degree_days_larval`.

`nothing` for a stage that is not pelagic. `LifeStage` also names `juvenile` and
`adult`, and those have no fixed developmental stage number of their own: the
caller's own `stage` argument is the answer for them. Collapsing them into the
megalopa index, which is what a three-branch `?:` does, silently answered a
juvenile's question with the megalopa's numbers.
"""
life_stage_index(stage_cat::LifeStage) = stage_cat === egg ? 0 :
                                        stage_cat === zoea ? 1 :
                                        stage_cat === megalopa ? 2 : nothing

"""
    required_degree_days(behavior::SpeciesBehavior, stage::Int)

Cumulative physiological thermal degree-days (°C·days above 0°C) required to
complete pre-ecdysial preparation at `stage`.

Leading pelagic stages read `degree_days_larval`; the rest read
`degree_days_schedule`.
"""
function required_degree_days(behavior::SpeciesBehavior, stage::Int)
    stage < larval_stage_count(behavior) &&
        return stage_schedule_lookup(behavior.params.degree_days_larval, stage)
    return stage_schedule_lookup(behavior.params.degree_days_schedule, stage)
end

required_degree_days(stage::Int) = required_degree_days(SpeciesBehaviorImpl(), stage)

function required_degree_days(behavior::SpeciesBehavior, stage_cat::LifeStage, stage::Int = 0)
    idx = life_stage_index(stage_cat)
    return required_degree_days(behavior, idx === nothing ? stage : idx)
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
function perform_development!(
    ::SpeciesBehavior,
    org::Organism,
    model;
    temp::Union{Nothing, Real} = nothing
)
    params = model.params
    if !params.toggles.use_molting || org.is_mature
        return nothing
    end

    days_since_molt = model.current_time - org.last_growth_event_time
    min_days = min_intermolt_days(org.behavior, org.stage)
    if days_since_molt >= min_days
        # The caller normally has the organism's own cell temperature already, since
        # it reads it for energetics and mortality in the same step and nothing
        # between here and there moves the organism or changes the field.
        current_temp = temp === nothing ? model.temperature_field[org.pos...] : temp
        # The seasonal gate lives in `molt_probability`, which applies it only from
        # `molt_season_min_stage` upward. Repeating it here for every stage confined
        # all benthic stages to the season and made that parameter inert.
        p_molt = molt_probability(org, current_temp, model.current_time; params = params)
        if rand(abmrng(model)) < p_molt
            perform_molt!(org, model.current_time, current_temp; params = params,
                          rng = abmrng(model))
            model.total_molts += 1
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

    day_of_year = mod(model.current_time, model.params.simulation.days_per_year)
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
                    phen = model.params.phenology
                    size_exponent = phen.mating_size_weight_exponent
                    male_weights = [m.size^size_exponent for m in nearby_males]
                    selected_male = sample_weighted(nearby_males, male_weights,
                                                    abmrng(model))
                    # Whether the male is legal is a fishery question, so it reads the
                    # same `legal_size_male` the harvest gate uses rather than a copy
                    # of it. A store that is not spent leaves the female able to
                    # extrude further broods; the parity test below must not read that
                    # count, because a female holding three stores is not thereby
                    # multiparous.
                    org.sperm_clutches = (selected_male.size >=
                                          model.params.fishery.legal_size_male) ?
                                         phen.sperm_stores_legal :
                                         phen.sperm_stores_sublegal
                    selected_male.energy_reserve = max(
                        0.0,
                        selected_male.energy_reserve - phen.mating_energy_cost
                    )
                end

                can_extrude = !model.params.toggles.use_spermatheca ||
                              (org.sperm_clutches > 0)
                if can_extrude
                    if model.params.toggles.use_spermatheca
                        org.sperm_clutches -= 1
                    end

                    # Parity is determined from post-ecdysis shell age. Primiparous
                    # females have molted into maturity recently (shell age below the
                    # maximum primiparous threshold), whereas multiparous females have
                    # maintained their carapace across subsequent reproductive cycles.
                    local_shell_age = shell_age_years(org, model.params)
                    egg_estimate = fecundity(
                        org.behavior,
                        org.size;
                        is_primiparous =
                            (local_shell_age <=
                             model.params.species.primiparous_shell_age_max),
                        fecundity_quantile = org.trait_fecundity_quantile,
                        rng = abmrng(model)
                    )
                    cx = min(max(org.pos[1], 1), size(model.temperature_field, 1))
                    cy = min(max(org.pos[2], 1), size(model.temperature_field, 2))
                    local_temp = model.temperature_field[cx, cy]
                    phen = model.params.phenology
                    tau_incubation = phen.brood_duration_base *
                                     (phen.brood_temperature_reference /
                                      max(phen.brood_temperature_floor,
                                          local_temp))^phen.brood_temperature_exponent
                    incubation_noise = rand(abmrng(model),
                                            Normal(0.0, phen.brood_duration_sd))

                    org.egg_count = max(round(Int, egg_estimate),
                                        model.params.phenology.minimum_clutch_size)
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
                # Hatching produces the first *cohort* stage, which is one past the
                # egg. Stage 0 is the brooded egg — carried by the female, not a
                # cohort — so a clutch that hatches at stage 0 put eggs and larvae
                # in the same bin, and the pelagic stages below were offset by one.
                new_group = AggregateGroup(
                    id = model.aggregate_populations.next_group_id,
                    stage = FIRST_COHORT_STAGE,
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