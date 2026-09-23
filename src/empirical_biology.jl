"""
Empirical biological relationships for organisms.

This module implements empirical allometric growth increments, size bounds per developmental stage,
logistic maturity schedules, fecundity curves, and harvest selectivity derived from research surveys.

References:
- Sainte-Marie, B. (1993). Reproductive cycle and fecundity of primiparous
  and multiparous female snow crab. CJFAS, 50(10), 2147-2156.
- Sainte-Marie, B., Raymond, S., & Brêthes, J. C. (1995). Growth and maturation
  of the benthic stages of male snow crab. CJFAS, 52(5), 903-924.
"""

using DataFrames
using Distributions
using Random
using Statistics
using LinearAlgebra

export llbm, llbf, get_stage_size, growth_increment, predict_postgrowth_size, growth_schedule, maturity, fecundity, selectivity, growth_simulate, sample_population, calculate_mass, build_instar_stages, gaussian_kde, derive_instar_proportions_kde, _fit_ols, _fit_irls_logistic, generate_mock_survey_data, fit_empirical_biology_model, estimate_demographics, summarize_empirical_model
export FECUNDITY_PRIMIPAROUS_INTERCEPT, FECUNDITY_PRIMIPAROUS_SLOPE, FECUNDITY_PRIMIPAROUS_ERR
export FECUNDITY_MULTIPAROUS_INTERCEPT, FECUNDITY_MULTIPAROUS_SLOPE, FECUNDITY_MULTIPAROUS_ERR

# --- Biological Invariants (Hardcoded Population Parameters) ---
const FECUNDITY_PRIMIPAROUS_INTERCEPT = -0.7229
const FECUNDITY_PRIMIPAROUS_SLOPE     = 2.94
const FECUNDITY_PRIMIPAROUS_ERR       = 0.090

const FECUNDITY_MULTIPAROUS_INTERCEPT = 0.0158
const FECUNDITY_MULTIPAROUS_SLOPE     = 2.5811
const FECUNDITY_MULTIPAROUS_ERR       = 0.145

Compute the empirical lower bound of carapace width (mm) for male snow crabs
at a given `instar`.

# Mathematical Formulation
```math
CW_{lb, male} = \\exp(1.917564 + 0.298914 \\cdot (instar - 3))
```

# Arguments
- `instar`: Instar index (typically 1 to 14).

# Returns
Carapace width lower bound in millimeters (scalar or array matching input).
"""
function llbm(instar::Real)
    if instar < 4.0
        # Planktonic larval stage bounds: Zoea I ~ 0.95 mm, Zoea II ~ 1.45 mm, Megalopa ~ 2.50 mm
        return (instar <= 1.5) ? 0.95 : (instar <= 2.5 ? 1.45 : 2.50)
    end
    i = instar - 3.0
    return exp(1.917564 + 0.298914 * i)
end

llbm(instar::AbstractArray) = llbm.(instar)

"""
    llbf(instar)

Compute the empirical lower bound of carapace width (mm) for female snow crabs
at a given `instar`.

# Mathematical Formulation
```math
CW_{lb, female} = \\exp(2.198848 + 0.315026 \\cdot (instar - 4))
```
Planktonic larval stages (Instars 1 to 3) utilize empirical Zoea I (0.95 mm),
Zoea II (1.45 mm), and Megalopa (2.50 mm) dimensions.

# Arguments
- `instar`: Instar index (typically 1 to 14).

# Returns
Carapace width lower bound in millimeters (scalar or array matching input).
"""
function llbf(instar::Real)
    if instar < 4.0
        return (instar <= 1.5) ? 0.95 : (instar <= 2.5 ? 1.45 : 2.50)
    end
    i = instar - 4.0
    return exp(2.198848 + 0.315026 * i)
end

llbf(instar::AbstractArray) = llbf.(instar)

"""
    carapacewidth(instar, sex; type=:mid, rng=Random.default_rng())

Compute or sample carapace width (mm) for a given `instar` and `sex`.

# Options for `type`
- `:lb`: Lower bound for the instar.
- `:ub`: Upper bound for the instar (lower bound of next instar).
- `:mid`: Midpoint between lower and upper bounds.
- `:random`: Stochastically sampled variate strictly bounded within `[lb, ub]`
  using a symmetric `Beta(2.5, 2.5)` distribution centered at `mid`.

# Arguments
- `instar::Real`: Instar number (e.g. 4 through 14).
- `sex`: `0`, `:male`, or `male` for males; `1`, `:female`, or `female` for females.
- `type::Symbol`: `:lb`, `:ub`, `:mid`, or `:random`.
- `rng`: AbstractRNG generator (default: `Random.default_rng()`).

# Returns
Estimated carapace width in millimeters.
"""
function get_stage_size(
    function get_stage_size(
    instar::Real,
    sex;
    type::Symbol = :mid,
    rng::AbstractRNG = Random.default_rng()
)
    is_female = (sex == 1 || sex == :female || Symbol(sex) == :female)

    lb = is_female ? llbf(instar) : llbm(instar)
    ub = is_female ? llbf(instar + 1.0) : llbm(instar + 1.0)

    if type === :lb
        return lb
    elseif type === :ub
        return ub
    elseif type === :mid
        return (lb + ub) / 2.0
    elseif type === :random
        # Symmetric Beta(2.5, 2.5) strictly bounded within [lb, ub]
        # Preserves natural unimodal variance while avoiding cross-instar leakage
        u = rand(rng, Beta(2.5, 2.5))
        return lb + u * (ub - lb)
    else
        error("Unknown get_stage_size type: $(type). Supported: :lb, :ub, :mid, :random")
    end
end

"""
    growth_increment(logcw; sex=0, mat=0, temp=nothing, condition=1.0, rng=Random.default_rng())

Predict the natural logarithm of post-molt carapace width, ln(CW_{t+1}),
given pre-molt natural logarithm ln(CW_t), sex, maturity transition status,
nutritional condition factor, and ambient bottom temperature.

# Mathematical Formulation
Hiatt log-linear transition model with bivariate parameter covariance:
```math
\\ln(CW_{t+1}) = \\beta_0 + \\beta_1 \\ln(CW_t) + \\delta_{mat} + \\Delta_{cond} + \\Delta_{temp} + \\epsilon
```
Parameters ``(\\beta_0, \\beta_1)^T`` are sampled from a bivariate normal distribution
preserving the empirical negative covariance (``\\rho \\approx -0.965`` for males,
``\\rho \\approx -0.960`` for females) from Sainte-Marie et al. (1995):
- **Males (`sex = 0` or `:male`)**:
  ``\\boldsymbol{\\mu} = [0.30153, 1.00027]^T``,
  ``\\sigma_0 = 0.02198``, ``\\sigma_1 = 0.00667``, ``\\rho = -0.965``.
  ``\\delta_{mat} = 0`` if immature (`mat = 0`), or
  ``\\sim \\mathcal{N}(-0.13564, 0.00922)`` if maturing (`mat = 1`).

- **Females (`sex = 1` or `:female`)**:
  ``\\boldsymbol{\\mu} = [0.1931, 1.0372]^T``,
  ``\\sigma_0 = 0.0545``, ``\\sigma_1 = 0.0181``, ``\\rho = -0.960``.
  ``\\delta_{mat} = 0`` if immature (`mat = 0`), or
  ``\\sim \\mathcal{N}(-0.1310, 0.0252)`` if maturing (`mat = 1`).

- **Ecdysial Monotonicity**:
  Enforces physiological minimum expansion (``CW_{next} \\ge CW_t \\cdot (1 + \\delta_{min})``
  where ``\\delta_{min} = 0.06`` for adults, ``0.12`` for juveniles).

# Arguments
- `logcw::Real`: Natural logarithm of pre-molt carapace width (ln(mm)).
- `sex`: `0` / `:male` for male; `1` / `:female` for female.
- `mat`: `0` for immature molt; `1` for terminal molt to maturity.
- `temp`: Ambient bottom water temperature in °C (or `nothing`).
- `condition::Real`: Relative energetic condition factor (default: 1.0).
- `rng`: Random number generator (default: `Random.default_rng()`).

# Returns
Natural logarithm of post-molt carapace width, ln(CW_{next}).
"""
function growth_increment(
    logcw::Real;
    sex = 0,
    mat::Integer = 0,
    temp = nothing,
    condition::Real = 1.0,
    rng::AbstractRNG = Random.default_rng(),
    trait_intercept::Union{Nothing, Real} = nothing,
    trait_slope::Union{Nothing, Real} = nothing,
    params::IBMParameters = IBMParameters()
)
    is_female = (sex == 1 || sex == :female || Symbol(sex) == :female)

    # Use individual traits or sample from population distribution if none provided
    if !isnothing(trait_intercept) && !isnothing(trait_slope)
        intercept = trait_intercept
        slope = trait_slope
    else
        z1 = rand(rng, Normal(0.0, 1.0))
        z2 = rand(rng, Normal(0.0, 1.0))
        if !is_female
            # Bivariate normal sampling with empirical correlation rho = -0.965
            intercept = 0.30153 + 0.02198 * z1
            slope = 1.00027 - 0.00643655 * z1 + 0.0017492 * z2
            mat_effect = (mat == 1) ? rand(rng, Normal(-0.13564, 0.00922)) : 0.0
        else
            # Bivariate normal sampling with empirical correlation rho = -0.960
            intercept = 0.1931 + 0.0545 * z1
            slope = 1.0372 - 0.017376 * z1 + 0.005068 * z2
            mat_effect = (mat == 1) ? rand(rng, Normal(-0.1310, 0.0252)) : 0.0
        end
    end

    # Nutritional condition modulation (energy surplus enhances expansion)
    cond_factor = clamp(Float64(condition), 0.5, 1.5)
    cond_mod = 0.05 * (cond_factor - 1.0)

    # Ambient thermal plasticity (mild thermal scaling)
    temp_mod = (temp !== nothing) ?
        -0.004 * (clamp(Float64(temp), 0.0, 10.0) - 4.0) : 0.0

    # Individual phenotypic variation around regression line
    residual = rand(rng, Normal(0.0, 0.040))

    log_cw_next = intercept + slope * logcw + mat_effect + cond_mod + temp_mod + residual

    # Enforce physiological minimum expansion: ecdysis guarantees positive growth
    min_expansion = (mat == 1) ? 0.06 : 0.12
    return max(log_cw_next, logcw + log(1.0 + min_expansion))
end

"""
    predict_postmolt_cw(cw_mm; sex=0, mat=0, temp=nothing, condition=1.0, rng=Random.default_rng())

Convenience wrapper predicting post-molt carapace width in millimeters (linear scale).

# Arguments
- `cw_mm::Real`: Pre-molt carapace width in mm.
- `sex`: Sex code (`0` / `:male` or `1` / `:female`).
- `mat::Integer`: `0` (immature molt) or `1` (terminal molt).
- `temp`: Optional ambient temperature in °C.
- `condition::Real`: Optional energetic condition factor.
- `rng`: Random number generator.

# Returns
Post-molt carapace width in millimeters.
"""
function predict_postgrowth_size(
    morphometry::Real;
    sex = 0,
    mat::Integer = 0,
    temp = nothing,
    condition::Real = 1.0,
    rng::AbstractRNG = Random.default_rng(),
    trait_intercept::Union{Nothing, Real} = nothing,
    trait_slope::Union{Nothing, Real} = nothing
)
    if morphometry <= 0.0
        error("Morphometry must be strictly positive, received: $(morphometry)")
    end
    log_size = log(morphometry)
    log_size_next = growth_increment(
        log_size;
        sex = sex,
        mat = mat,
        temp = temp,
        condition = condition,
        rng = rng,
        trait_intercept = trait_intercept,
        trait_slope = trait_slope
    )
    return exp(log_size_next)
end

"""
    growth_schedule()

Construct a reference `DataFrame` summarizing empirical carapace width bounds,
medians, and expected molt increments for instars 1 through 14 for both sexes.

# Returns
`DataFrame` with columns:
- `instar`: Instar index (1 to 14).
- `cwm_lb`, `cwm_ub`, `cwm_med`: Male lower, upper, and median CW (mm).
- `cwf_lb`, `cwf_ub`, `cwf_med`: Female lower, upper, and median CW (mm).
- `cwm_range`, `cwf_range`: Instar width bounds (mm).
- `cwm_inc`, `cwf_inc`: Median growth increment to next instar (mm).
"""
function growth_schedule()
    instars = 1:14
    schedule = DataFrame(
        instar = collect(instars),
        cwm_lb = llbm(instars),
        cwf_lb = llbf(instars),
        cwm_ub = llbm(instars .+ 1),
        cwf_ub = llbf(instars .+ 1)
    )

    schedule.cwm_med = (schedule.cwm_lb .+ schedule.cwm_ub) ./ 2.0
    schedule.cwf_med = (schedule.cwf_lb .+ schedule.cwf_ub) ./ 2.0

    schedule.cwm_range = schedule.cwm_ub .- schedule.cwm_lb
    schedule.cwf_range = schedule.cwf_ub .- schedule.cwf_lb

    schedule.cwm_inc = zeros(Float64, length(instars))
    schedule.cwf_inc = zeros(Float64, length(instars))

    for k in 1:(length(instars) - 1)
        schedule.cwm_inc[k] = schedule.cwm_med[k + 1] - schedule.cwm_med[k]
        schedule.cwf_inc[k] = schedule.cwf_med[k + 1] - schedule.cwf_med[k]
    end

    return schedule
end

"""
    maturity(sex, logcw; temp=nothing, rng=Random.default_rng())

Calculate the probability of transitioning to morphometric maturity during molt,
based on empirical logistic regression fits incorporating size threshold gates
and the Temperature-Size Rule (TSR) from regional research surveys.

# Mathematical Formulation
```math
P(mature \\mid CW, T) = \\frac{1}{1 + \\exp(-\\beta (\\ln(CW) - \\ln(L_{50}(T))))}
```
where:
- **Minimum Size Gate**: Crabs smaller than Instar 7 (``CW < 35.0`` mm) cannot
  undergo terminal molt (``P = 0.0``).
- **Males (`sex = 0` or `:male`)**:
  ``L_{50, base} \\sim \\mathcal{N}(91.8, 1.2)`` mm, ``\\beta \\sim \\mathcal{N}(6.9957, 0.25)``.
- **Females (`sex = 1` or `:female`)**:
  ``L_{50, base} \\sim \\mathcal{N}(51.6, 0.8)`` mm, ``\\beta \\sim \\mathcal{N}(12.9812, 0.40)``.
- **Thermal Cline (Temperature-Size Rule)**:
  Ambient temperature ``T`` shifts ``L_{50}`` by ``-2.0`` mm/°C relative to ``4.0``°C:
  ``L_{50}(T) = \\max(35.0, L_{50, base} - 2.0 \\cdot (T - 4.0))``.

# Arguments
- `sex`: Sex code (`0` / `:male` or `1` / `:female`).
- `logcw::Real`: Natural logarithm of pre-molt carapace width (ln(mm)).
- `temp`: Optional ambient bottom water temperature in °C.
- `rng`: Random number generator (default: `Random.default_rng()`).

# Returns
Probability of maturing upon molting in the interval [0.0, 1.0].
"""
function maturity(
    sex,
    logcw::Real;
    temp = nothing,
    trait_l50::Union{Nothing, Real} = nothing,
    trait_slope::Union{Nothing, Real} = nothing,
    params::IBMParameters = IBMParameters(),
    rng::AbstractRNG = Random.default_rng()
)
    # Biological minimum size threshold: Instars 4-6 do not undergo terminal molt
    if logcw < log(35.0)
        return 0.0
    end

    is_female = (sex == 1 || sex == :female || Symbol(sex) == :female)

    # Use provided trait or fallback to population parameters
    l50_base = if !isnothing(trait_l50)
        trait_l50
    else
        if !is_female
            rand(rng, Normal(params.maturity_l50_male, params.maturity_l50_male_sd))
        else
            rand(rng, Normal(params.maturity_l50_female, params.maturity_l50_female_sd))
        end
    end

    slope = if !isnothing(trait_slope)
        trait_slope
    else
        if !is_female
            max(1.0, rand(rng, Normal(params.maturity_slope_male, params.maturity_slope_male_sd)))
        else
            max(1.0, rand(rng, Normal(params.maturity_slope_female, params.maturity_slope_female_sd)))
        end
    end

    # Temperature-size rule: warmer waters decrease median size at maturity
    l50 = if temp !== nothing
        t_clamp = clamp(Float64(temp), -1.0, 10.0)
        max(35.0, l50_base - 2.0 * (t_clamp - 4.0))
    else
        l50_base
    end

    linear_pred = slope * (logcw - log(l50))

    if linear_pred > 40.0
        return 1.0
    elseif linear_pred < -40.0
        return 0.0
    end

    return 1.0 / (1.0 + exp(-linear_pred))
end

"""
    fecundity(cw; method="moncton.primiparous", shell_age=1.0, rng=Random.default_rng())

Estimate brood fecundity (total number of extruded eggs produced) as an allometric
function of female carapace width (mm), parity status, and shell senescence.

# Mathematical Methods
- `"moncton.primiparous"`: Primiparous females (first brood, Sainte-Marie 1993):
  ```math
  \\log_{10}(Eggs) = 2.94 \\log_{10}(CW) - 0.7229 + \\epsilon, \\quad \\epsilon \\sim \\mathcal{N}(0, 0.090)
  ```
- `"moncton.multiparous"`: Multiparous females (subsequent broods):
  ```math
  \\log_{10}(Eggs) = 2.5811 \\log_{10}(CW) + 0.0158 + \\epsilon, \\quad \\epsilon \\sim \\mathcal{N}(0, 0.145)
  ```
- `"st.marie"`: Sainte-Marie (1993) relationship:
  ```math
  Eggs = CW^{2.725} \\cdot 10^{0.7311}
  ```
- **Shell Senescence**: Multiparous females in older carapace condition (shell age
  ``\\ge 3`` years) exhibit partial egg extrusion loss (``\\sim 10\\text{--}25\\%`` attrition).

# Arguments
- `cw`: Carapace width in millimeters (scalar or array).
- `method`: String selecting relationship model (default: `"moncton.primiparous"`).
- `shell_age`: Estimated carapace shell age in years (default: 1.0).
- `rng`: Random number generator (default: `Random.default_rng()`).

# Returns
Predicted number of eggs (same dimension as `cw`).
"""
function fecundity(
    cw::Real;
    fecundity_parameterization::String = "moncton",
    is_primiparous::Bool = true,
    shell_age::Real = 1.0,
    params::IBMParameters = IBMParameters(),
    rng::AbstractRNG = Random.default_rng(),
    deterministic::Bool = false,
    trait_intercept::Union{Nothing, Real} = nothing,
    trait_slope::Union{Nothing, Real} = nothing
)
    if cw <= 0.0
        error("Carapace width must be positive for fecundity estimation, got: $(cw)")
    end

    if fecundity_parameterization == "st.marie"
        return cw^2.725 * (10.0^0.7311)
    elseif fecundity_parameterization == "moncton"
        # Use trait or fallback to population parameter based on primiparous status
        if is_primiparous
            intercept = isnothing(trait_intercept) ? FECUNDITY_PRIMIPAROUS_LOG10_INTERCEPT : trait_intercept
            slope     = isnothing(trait_slope)     ? FECUNDITY_PRIMIPAROUS_LOG10_SLOPE    : trait_slope

            fec_log10 = log10(cw) * slope + intercept
            err       = deterministic ? 0.0 : rand(rng, Normal(0.0, FECUNDITY_PRIMIPAROUS_ERR))
            return 10.0^(fec_log10 + err)
        else
            intercept = isnothing(trait_intercept) ? FECUNDITY_MULTIPAROUS_LOG10_INTERCEPT : trait_intercept
            slope     = isnothing(trait_slope)     ? FECUNDITY_MULTIPAROUS_LOG10_SLOPE    : trait_slope

            fec_log10 = log10(cw) * slope + intercept
            err       = deterministic ? 0.0 : rand(rng, Normal(0.0, FECUNDITY_MULTIPAROUS_ERR))
            senescence = (shell_age > 3.0) ? max(0.60, 1.0 - 0.08 * (shell_age - 3.0)) : 1.0
            return 10.0^(fec_log10 + err) * senescence
        end
    else
        error("Unknown fecundity parameterization: $(fecundity_parameterization). Valid: 'moncton', 'st.marie'")
    end
end

"""
    apply_mortality!

Applies stage-specific daily survival probabilities to an aggregate population distribution. 

# Arguments
- `pop::StageBinDistribution`: The current count and binning structure of a life stage.
- `daily_survival_rate::Real`: Probability that an individual survives one day (0 to 1).
- `rng::AbstractRNG`: Random number generator for stochastic mortality if required (though here we use deterministic scaling on counts, which is standard for large populations in binning models).

# Returns
Nothing; modifies the distribution in place.
"""
function apply_mortality!(pop::StageBinDistribution, daily_survival_rate::Real)
    if daily_survival_rate < 0.0 || daily_survival_rate > 1.0
        error("daily_survival_rate must be between 0 and 1, got: $(daily_survival_rate)")
    end
    pop.counts .*= daily_survival_rate
end

"""
    advance_development!(stage::Symbol; pop::AggregatePopulations, params::IBMParameters, rng::AbstractRNG)

Moves individuals from one life stage to the next (e.g., egg $\rightarrow$ zoea or instar $i \to i+1$). 
This function handles 'recruitment' into a new bin and potentially clears out/reduces counts in the previous bin based on transition probabilities.

# Arguments
- `stage::Symbol`: The stage currently being processed (e.g., :eggs, :larvae).
- `pop::AggregatePopulations`: Container for all non-spatial populations.
- `params::IBMParameters`: Model parameters including developmental rates and mortality thresholds.
- `rng::AbstractRNG`: Random number generator.

# Returns
Nothing; modifies the aggregate population in place.
"""
function advance_development!(stage::Symbol; pop::AggregatePopulations, params::IBMParameters, rng::AbstractRNG)
    # This is a placeholder for implementation logic: 
    # In practice, this will involve calculating how many individuals transition to stage+1 based on temperature and time.
end

"""
    recruit_to_aggregate!(stage::Symbol; pop::AggregatePopulations, dist::StageBinDistribution)

Adds new counts (e.g., newly hatched larvae or eggs spawned) into an existing aggregate distribution binning structure. 

# Arguments
- `stage::Symbol`: The target stage for recruitment (:eggs, :larvae, etc.).
- `pop::AggregatePopulations`: Container of all stages.
- `dist::StageBinDistribution`: A new (or partial) distribution to add into the population.
"""
function recruit_to_aggregate!(stage::Symbol; pop::AggregatePopulations, dist::StageBinDistribution)
    if stage === :eggs && !isnothing(pop.eggs)
        pop.eggs.counts .+= dist.counts
    elseif stage === :larvae && !isnothing(pop.larvae)
        pop.larvae.counts .+= dist.counts
    elseif stage === :megalopa && !isnothing(pop.megalopa)
        pop.megalopa.counts .+= dist.counts
    else # Instars 1-5
         # Logic to find the correct index in pop.instars based on a mapping...
    end
end

function fecundity(cw::Real; fecundity_parameterization::String = "moncton", is_primiparous::Bool = true, shell_age::Real = 1.0, params::IBMParameters = IBMParameters(), rng::AbstractRNG = Random.default_rng(), deterministic::Bool = false, trait_intercept::Union{Nothing, Real} = nothing, trait_slope::Union{Nothing, Real} = nothing) =
    fecundity.(cw; fecundity_parameterization=fecundity_parameterization, is_primiparous=is_primiparous, shell_age=shell_age, params=params, rng=rng, deterministic=deterministic, trait_intercept=trait_intercept, trait_slope=trait_slope)

"""
    fecundity(crab::SnowCrab; fecundity_parameterization::Union{String, Nothing} = nothing, rng::AbstractRNG = Random.default_rng(), deterministic::Bool = false)

Estimate extruded brood fecundity directly for an individual female `SnowCrab` agent, automatically distinguishing primiparous versus multiparous reproductive history from stored spermathecal clutches and carapace age when using the `"moncton"` parameterization.

# Arguments
- `crab::SnowCrab`: Mature adult female snow crab.
- `fecundity_parameterization`: Optional string override (`nothing` defaults to `"moncton"`).
- `rng`: Random number generator.
- `deterministic`: If true, use parameter means without stochastic noise.

# Returns
`Int`: Integer count of brooded eggs.
"""
function fecundity(
    crab::SnowCrab;
    fecundity_parameterization::Union{String, Nothing} = nothing,
    rng::AbstractRNG = Random.default_rng(),
    deterministic::Bool = false
)
    if crab.sex !== female || crab.stage !== adult
        return 0
    end

    param = if fecundity_parameterization !== nothing
        fecundity_parameterization
    else
        "moncton"
    end

    # Determine primiparous status from agent state for "moncton" method (as requested)
    is_p = crab.sperm_clutches == 0

    shell_age = max(1.0, (crab.age_days - 180.0 * (crab.developmental_step - 3)) / 365.0)

    raw_eggs = fecundity(
        crab.size;
        fecundity_parameterization=param,
        is_primiparous=is_p,
        shell_age=shell_age,
        params=IBMParameters(), # Default if not provided via crab? 
                                 # Actually we should probably allow passing params to this function too.
        rng=rng, deterministic=deterministic
    )

    return round(Int, raw_eggs)
end

"""
    selectivity(s; s50=95.0, s50_sigma=5.0)

Calculate gear selectivity or harvest probability as a logistic function
of crab carapace width `s` (mm).

# Mathematical Formulation
```math
S(s) = \\frac{1}{1 + \\exp\\left(-\\frac{s - s_{50}}{s_{50\\sigma}}\\right)}
```

# Arguments
- `s::Real`: Carapace width in millimeters.
- `s50::Real`: Size at 50% gear retention (default: 95.0 mm).
- `s50_sigma::Real`: Slope bandwidth parameter (default: 5.0 mm).

# Returns
Retention probability in the interval [0.0, 1.0].
"""
function selectivity(s::Real; params::IBMParameters = IBMParameters(), s50::Union{Nothing, Real} = nothing, s50_sigma::Union{Nothing, Real} = nothing)
    effective_s50 = s50 !== nothing ? s50 : params.selectivity_s50
    effective_sigma = s50_sigma !== nothing ? s50_sigma : params.selectivity_s50_sigma

    if effective_sigma <= 0.0
        error("Selectivity bandwidth s50_sigma must be positive, got: $(effective_sigma)")
    end
    diff = (s - effective_s50) / effective_sigma
    if diff > 40.0
        return 1.0
    elseif diff < -40.0
        return 0.0
    end
    return 1.0 / (1.0 + exp(-diff))
end

function selectivity(s::AbstractArray; params::IBMParameters = IBMParameters(), s50::Union{Nothing, Real} = nothing, s50_sigma::Union{Nothing, Real} = nothing)
    effective_s50 = s50 !== nothing ? s50 : params.selectivity_s50
    effective_sigma = s50_sigma !== nothing ? s50_sigma : params.selectivity_s50_sigma

    if effective_sigma <= 0.0
        error("Selectivity bandwidth s50_sigma must be positive, got: $(effective_sigma)")
    end
    diff = (s .- effective_s50) ./ effective_sigma
    return 1.0 ./ (1.0 .+ exp.(-diff))
end

selectivity(s::AbstractArray; s50 = 95.0, s50_sigma = 5.0) =
    selectivity.(s; s50=s50, s50_sigma=s50_sigma)

"""
    growth_simulate(modes; gcv=0.4, n_max=10000, s50=50.0, s50_sigma=50.0, rng=Random.default_rng())

Simulate multi-instar population growth across defined instar modal sizes with
growth variability and size selectivity filtering.

# Arguments
- `modes::Vector{Float64}`: Vector of modal carapace widths per instar.
- `gcv::Float64`: Growth coefficient of variation.
- `n_max::Int`: Number of simulated crab trajectories.
- `s50::Float64`: Gear selectivity 50% threshold.
- `s50_sigma::Float64`: Selectivity bandwidth.
- `rng`: Random number generator.

# Returns
Tuple `(W, O)` where `W` is full size matrix and `O` is selectivity-filtered matrix.
"""
function growth_simulate(
    modes::Vector{Float64};
    gcv::Float64 = 0.4,
    n_max::Int = 10000,
    s50::Float64 = 50.0,
    s50_sigma::Float64 = 50.0,
    rng::AbstractRNG = Random.default_rng()
)
    n_instars = length(modes)
    incr = modes[2:end] .- modes[1:(end - 1)]

    W = zeros(Float64, n_max, n_instars)
    W[:, 1] = rand(rng, Normal(modes[1], modes[1] / 20.0), n_max)

    for j in 2:n_instars
        inc = rand(rng, Normal(incr[j - 1], incr[j - 1] * gcv), n_max)
        W[:, j] = W[:, j - 1] .+ inc
    end

    s_modes = selectivity.(modes; s50=s50, s50_sigma=s50_sigma)
    O = deepcopy(W)

    for j in 1:n_instars
        n_drop = round(Int, (1.0 - s_modes[j]) * n_max)
        if n_drop > 0
            drop_indices = randperm(rng, n_max)[1:n_drop]
            O[drop_indices, j] .= NaN
        end
    end

    return (W, O)
end

"""
    sample_population(year; sex_ratio=0.5, rng=Random.default_rng())

Sample individual traits (degree-days, size, sex, maturity, instar) for
initializing population members.

# Arguments
- `year::Int`: Year indicator (`-1` indicates unconstrained benthic recruitment).
- `sex_ratio::Float64`: Proportion of males in sampled recruits.
- `rng`: Random number generator.

# Returns
NamedTuple `(degdays, size_mm, sex, mat, instar)`.
"""
function sample_population(
    year::Int = -1;
    sex_ratio::Float64 = 0.5,
    rng::AbstractRNG = Random.default_rng()
)
    if year == -1
        degdays = rand(rng, Uniform(0.0, 0.1))
        # Instar 5 initial recruit: log(CW) ~ Normal(2.75, 0.25/3) => CW ≈ 15.6 mm
        size_mm = exp(rand(rng, Normal(2.75, 0.25 / 3.0)))
        sex = (rand(rng) < sex_ratio) ? 0 : 1
        mat = 0
        instar = 5
        return (degdays=degdays, size_mm=size_mm, sex=sex, mat=mat, instar=instar)
    else
        degdays = rand(rng, Uniform(0.0, 0.2))
        instar = rand(rng, 4:7)
        sex = (rand(rng) < sex_ratio) ? 0 : 1
        size_mm = carapacewidth(instar, sex; type=:random, rng=rng)
        mat = 0
        return (degdays=degdays, size_mm=size_mm, sex=sex, mat=mat, instar=instar)
    end
end

"""
    calculate_mass(cw, sex; mat=0, stochastic=false, rng=Random.default_rng())

Calculate individual wet body mass (in grams) from morphometry (e.g., carapace width in mm),
sex, and morphometric maturity status based on empirical allometric power models.

# Mathematical Formulation
```math
W = \\alpha \\cdot CW^\\beta \\cdot \\exp(\\epsilon - 0.5\\sigma^2)
```
where ``W`` is wet mass in grams, ``CW`` is the morphometric measurement (e.g., carapace width),
and parameters depend on sex and maturity status:
- **Immature / Adolescent Males (`sex == 0` or `:male`, `mat == 0`)**:
  ``\\alpha = 0.0002665``, ``\\beta = 3.098``, residual ``\\sigma = 0.08``
- **Morphometrically Mature Males (`sex == 0` or `:male`, `mat == 1`)**:
  ``\\alpha = 0.0001830``, ``\\beta = 3.160``, residual ``\\sigma = 0.08``
- **Immature Females (`sex == 1` or `:female`, `mat == 0`)**:
  ``\\alpha = 0.0003440``, ``\\beta = 2.980``, residual ``\\sigma = 0.08``
- **Morphometrically Mature Females (`sex == 1` or `:female`, `mat == 1`)**:
  ``\\alpha = 0.0004500``, ``\\beta = 2.920``, residual ``\\sigma = 0.08``

When `stochastic = true`, residual variance around the allometric power curve
is sampled from ``\\epsilon \\sim \\mathcal{N}(0, \\sigma^2)`` with log-normal
bias correction ``-0.5\\sigma^2`` ensuring ``\\mathbb{E}[W] = \\alpha CW^\\beta``.

# References
- Sainte-Marie, G., Raymond, S., & Brêthes, J. C. (1995). Growth and maturation of the
  benthic stages of male snow crab, *Chionoecetes opilio*. Can. J. Fish. Aquat. Sci.
- DFO (2020). Assessment of Nova Scotia (4VWX) Snow Crab. DFO Can. Sci. Advis. Sec.

# Arguments
- `cw::Real`: Morphometric measurement in mm (e.g., carapace width).
- `sex`: Sex indicator (`0`, `:male`, `male` for male; `1`, `:female`, `female` for female).
- `mat::Integer`: Morphometric maturity status (`0` = immature, `1` = mature).
- `stochastic::Bool`: Whether to include log-normal residual scatter (default: `false`).
- `rng`: Random number generator (default: `Random.default_rng()`).

# Returns
`Float64`: Wet body mass in grams.
"""
function calculate_mass(
    morphometry::Real,
    sex;
    mat::Integer = 0,
    stochastic::Bool = false,
    rng::AbstractRNG = Random.default_rng()
)
    if morphometry <= 0.0
        return 0.0
    end

    is_fem = (sex == 1 || sex == :female || Symbol(sex) == :female)

    alpha, beta_val = if !is_fem
        (mat == 1) ? (0.0001830, 3.160) : (0.0002665, 3.098)
    else
        (mat == 1) ? (0.0004500, 2.920) : (0.0003440, 2.980)
    end

    base_wt = alpha * (Float64(morphometry)^beta_val)

    if stochastic
        sigma = 0.08
        residual = rand(rng, Normal(0.0, sigma))
        return base_wt * exp(residual - 0.5 * sigma^2)
    end

    return base_wt
end

function calculate_mass(
    org::Organism;
    stochastic::Bool = false,
    include_eggs::Bool = true,
    rng::AbstractRNG = Random.default_rng()
)
    somatic_mass = calculate_mass(
        org.size,
        org.sex;
        mat = (org.stage === adult ? 1 : 0),
        stochastic = stochastic,
        rng = rng
    )
    egg_mass = (include_eggs && is_ovigerous(org)) ?
        (Float64(org.egg_count) * 0.00007) : 0.0
    return somatic_mass + egg_mass
end

"""
    build_instar_stages(; max_male_instar=12, max_female_instar=11)

Construct the discrete instar stage hierarchy for generalized Leslie stage-transition
modeling guided by empirical growth-instar relationships.

Sex discrimination is not necessary before Instar 6:
- Stage 1: Instar 4 (pooled sexes, benthic recruits ~10.7 mm).
- Stage 2: Instar 5 (pooled sexes, juveniles ~18.0 mm).
- Stages 3 to (2 + N_males): Male Instars 6 through `max_male_instar` (legal >= 95 mm).
- Stages (3 + N_males) to S: Female Instars 6 through `max_female_instar` (reproductive).

# Arguments
- `max_male_instar::Int`: Terminal male instar stage (default: 12).
- `max_female_instar::Int`: Terminal female instar stage (default: 11).

# Returns
`Vector{InstarStageDefinition}` of stage definitions.
"""
function build_instar_stages(;
    max_male_instar::Int = 12,
    max_female_instar::Int = 11
)
    stages = InstarStageDefinition[]
    idx = 1

    # Stage 1: Instar 4 (pooled sexes)
    push!(stages, InstarStageDefinition(
        idx,
        "Instar 4 (Recruit)",
        4,
        undifferentiated,
        (7.0, 14.5),
        10.7,
        calculate_mass(10.7, :male; mat = 0),
        false,
        false
    ))
    idx += 1

    # Stage 2: Instar 5 (pooled sexes)
    push!(stages, InstarStageDefinition(
        idx,
        "Instar 5 (Juvenile)",
        5,
        undifferentiated,
        (14.5, 22.0),
        18.0,
        calculate_mass(18.0, :male; mat = 0),
        false,
        false
    ))
    idx += 1

    # Male Stages: Instar 6 to max_male_instar
    # Modal empirical bounds: 6: (22-31), 7: (31-42.5), 8: (42.5-58), 9: (58-78),
    # 10: (78-95), 11: (95-115), 12+: (115-160)
    male_bounds = [
        (6, (22.0, 31.0), 26.5),
        (7, (31.0, 42.5), 36.5),
        (8, (42.5, 58.0), 50.0),
        (9, (58.0, 78.0), 68.0),
        (10, (78.0, 95.0), 86.5),
        (11, (95.0, 115.0), 105.0),
        (12, (115.0, 160.0), 130.0)
    ]

    for (inst, rng_b, cw_m) in male_bounds
        if inst <= max_male_instar
            is_legal = (cw_m >= 95.0 || inst >= 11)
            is_term = (inst == max_male_instar)
            lbl = is_term ? "Instar $(inst)+ Male" : "Instar $(inst) Male"
            wt = calculate_mass(cw_m, :male; mat = is_legal ? 1 : 0)
            push!(stages, InstarStageDefinition(
                idx,
                lbl,
                inst,
                male,
                rng_b,
                cw_m,
                wt,
                is_legal,
                false
            ))
            idx += 1
        end
    end

    # Female Stages: Instar 6 to max_female_instar
    # Modal empirical bounds: 6: (22-30.5), 7: (30.5-41.5), 8: (41.5-54.5),
    # 9: (54.5-68), 10: (68-82), 11+: (82-110)
    female_bounds = [
        (6, (22.0, 30.5), 26.0),
        (7, (30.5, 41.5), 35.5),
        (8, (41.5, 54.5), 47.5),
        (9, (54.5, 68.0), 61.0),
        (10, (68.0, 82.0), 75.0),
        (11, (82.0, 110.0), 90.0)
    ]

    for (inst, rng_b, cw_m) in female_bounds
        if inst <= max_female_instar
            is_repro = (inst >= 9) # Mature maternal instars
            is_term = (inst == max_female_instar)
            lbl = is_term ? "Instar $(inst)+ Female" : "Instar $(inst) Female"
            wt = calculate_mass(cw_m, :female; mat = is_repro ? 1 : 0)
            push!(stages, InstarStageDefinition(
                idx,
                lbl,
                inst,
                female,
                rng_b,
                cw_m,
                wt,
                false,
                is_repro
            ))
            idx += 1
        end
    end

    return stages
end

"""
    gaussian_kde(data; eval_points=nothing, bandwidth=nothing)

Compute 1D Gaussian Kernel Density Estimation (KDE) for numerical vector `data`.

# Mathematical Formulation
```math
\\hat{f}(x) = \\frac{1}{n h} \\sum_{i=1}^n \\frac{1}{\\sqrt{2\\pi}} \\exp\\left(-\\frac{(x - X_i)^2}{2 h^2}\\right)
```
Bandwidth ``h`` defaults to Silverman's rule of thumb:
```math
h = 1.06 \\cdot \\min\\left(\\hat{\\sigma}, \\frac{\\text{IQR}}{1.34}\\right) \\cdot n^{-1/5}
```
with safety clamps preventing zero bandwidth when variance is tiny.

# Arguments
- `data::AbstractVector{<:Real}`: Sample data observations.
- `eval_points`: Evaluation coordinates (defaults to 200 points spanning [min - 3h, max + 3h]).
- `bandwidth`: Optional manual smoothing parameter h.

# Returns
Tuple `(grid, density)`:
- `grid::Vector{Float64}`: Abscissa coordinates.
- `density::Vector{Float64}`: Evaluated probability density values.
"""
function gaussian_kde(
    data::AbstractVector{<:Real};
    eval_points = nothing,
    bandwidth = nothing
)
    n = length(data)
    if n == 0
        return (Float64[], Float64[])
    end

    x_clean = Float64.(data)
    s_dev = (n > 1) ? std(x_clean) : 1.0

    # Silverman's rule of thumb bandwidth with IQR check
    h = if bandwidth !== nothing
        Float64(bandwidth)
    else
        iqr_val = (n >= 4) ? (quantile(x_clean, 0.75) - quantile(x_clean, 0.25)) / 1.34 : s_dev
        spread = (iqr_val > 0.0 && iqr_val < s_dev) ? iqr_val : s_dev
        1.06 * max(spread, 0.5) * (n^(-0.20))
    end
    h = max(h, 0.1)

    grid = if eval_points !== nothing
        collect(Float64, eval_points)
    else
        x_min = minimum(x_clean) - 3.0 * h
        x_max = maximum(x_clean) + 3.0 * h
        range(x_min, x_max, length = 200) |> collect
    end

    inv_h = 1.0 / h
    inv_norm = 1.0 / (sqrt(2.0 * π) * n * h)
    density = zeros(Float64, length(grid))

    for (k, x) in enumerate(grid)
        sum_k = 0.0
        for xi in x_clean
            u = (x - xi) * inv_h
            sum_k += exp(-0.5 * u * u)
        end
        density[k] = sum_k * inv_norm
    end

    return (grid, density)
end

"""
    derive_instar_proportions_kde(carapace_widths, sexes=nothing;
                                  stages=build_instar_stages())

Derive the proportion and count of individuals in each discrete instar stage
from continuous carapace width observations using empirical growth-instar intervals.

Sex discrimination is not necessary before Instar 6:
- Crabs with CW < 22.0 mm are pooled into Stage 1 (Instar 4) and Stage 2 (Instar 5).
- Crabs with CW >= 22.0 mm are partitioned into Male and Female stages based on
  observed sex (or 50/50 ratio if sex is omitted).

# Arguments
- `carapace_widths::AbstractVector{<:Real}`: Carapace width measurements in mm.
- `sexes`: Optional vector of sex codes (`:male`, `:female`, `0`, `1`, etc.).
- `stages::Vector{InstarStageDefinition}`: Life-history stage definitions.

# Returns
Tuple `(counts, proportions)`:
- `counts::Vector{Float64}`: Number of crabs assigned to each stage.
- `proportions::Vector{Float64}`: Proportional stage distribution (sums to 1.0).
"""
function derive_instar_proportions_kde(
    carapace_widths::AbstractVector{<:Real},
    sexes = nothing;
    stages::Vector{InstarStageDefinition} = build_instar_stages()
)
    n_crabs = length(carapace_widths)
    n_stages = length(stages)
    counts = zeros(Float64, n_stages)

    if n_crabs == 0
        return (counts, zeros(Float64, n_stages))
    end

    has_sex = (sexes !== nothing && length(sexes) == n_crabs)

    for i in 1:n_crabs
        cw = Float64(carapace_widths[i])
        if cw <= 0.0
            continue
        end

        # Stage 1 and 2: Instars 4 and 5 (sex discrimination not necessary)
        if cw < 22.0
            if cw < 14.5
                counts[1] += 1.0 # Instar 4
            else
                counts[2] += 1.0 # Instar 5
            end
        else
            # Instars 6+ (sex-differentiated)
            is_fem = if has_sex
                sx = sexes[i]
                (sx == 1 || sx == :female || Symbol(sx) == :female)
            else
                rand() < 0.50
            end

            target_sex = is_fem ? female : male
            matched = false

            for st in stages
                if st.sex === target_sex
                    b_min, b_max = st.cw_range
                    if b_min <= cw < b_max || (cw >= b_min && st.instar == 12)
                        counts[st.index] += 1.0
                        matched = true
                        break
                    end
                end
            end

            if !matched
                candidates = filter(s -> s.sex === target_sex, stages)
                if !isempty(candidates)
                    dists = [abs(cw - s.cw_mean) for s in candidates]
                    best_st = candidates[argmin(dists)]
                    counts[best_st.index] += 1.0
                end
            end
        end
    end

    total = sum(counts)
    proportions = (total > 0.0) ? (counts ./ total) : zeros(Float64, n_stages)
    return (counts, proportions)
end

# -----------------------------------------------------------------------------
# Empirical Biological Model Calibration and Trait Sampling
# -----------------------------------------------------------------------------

"""
Internal linear regression helper computing intercept, slope, residual standard
error, and coefficient of determination R².
"""
function _fit_ols(x::Vector{Float64}, y::Vector{Float64})
    n = length(x)
    x_bar = mean(x)
    y_bar = mean(y)
    ss_xx = sum((x .- x_bar).^2)
    ss_xy = sum((x .- x_bar) .* (y .- y_bar))
    slope = (ss_xx > 1e-12) ? (ss_xy / ss_xx) : 0.0
    intercept = y_bar - slope * x_bar
    residuals = y .- (intercept .+ slope .* x)
    ss_res = sum(residuals.^2)
    ss_tot = sum((y .- y_bar).^2)
    r2 = (ss_tot > 1e-12) ? max(0.0, 1.0 - ss_res / ss_tot) : 0.0
    sigma = (n > 2) ? sqrt(ss_res / (n - 2)) : std(residuals)
    return (intercept, slope, sigma, r2)
end

"""
Internal iteratively reweighted least squares (IRLS) logistic regression for
maturity ogive estimation with thermal sensitivity.
"""
function _fit_irls_logistic(
    cw_vec::Vector{Float64},
    temp_vec::Vector{Float64},
    y_vec::Vector{Float64}
)
    n = length(y_vec)
    has_temp_var = std(temp_vec) > 0.1
    # Design matrix: [1, ln(CW), (T - 4.0)]
    X = has_temp_var ?
        hcat(ones(n), log.(cw_vec), temp_vec .- 4.0) :
        hcat(ones(n), log.(cw_vec))
    p_dim = size(X, 2)

    theta = zeros(p_dim)
    theta[1] = -20.0
    theta[2] = 5.0
    if p_dim == 3
        theta[3] = -0.05
    end

    for _ in 1:30
        eta = X * theta
        mu = 1.0 ./ (1.0 .+ exp.(-clamp.(eta, -25.0, 25.0)))
        mu = clamp.(mu, 1e-5, 1.0 - 1e-5)
        w = mu .* (1.0 .- mu)
        z = eta .+ (y_vec .- mu) ./ w

        # Normal equations with ridge regularization
        H = (X' .* w') * X + 1e-4 * I
        g = (X' .* w') * z
        theta_next = H \ g

        if norm(theta_next - theta) < 1e-5
            theta = theta_next
            break
        end
        theta = 0.5 .* theta .+ 0.5 .* theta_next
    end

    slope = max(0.5, theta[2])
    l50 = exp(clamp(-theta[1] / slope, log(35.0), log(140.0)))
    temp_coef = (p_dim == 3) ? theta[3] : -0.05

    # McFadden pseudo-R²
    p_mean = clamp(mean(y_vec), 1e-5, 1.0 - 1e-5)
    ll_null = sum(y_vec .* log(p_mean) .+ (1.0 .- y_vec) .* log(1.0 - p_mean))
    p_pred = 1.0 ./ (1.0 .+ exp.(-clamp.(X * theta, -25.0, 25.0)))
    p_pred = clamp.(p_pred, 1e-5, 1.0 - 1e-5)
    ll_fit = sum(y_vec .* log.(p_pred) .+ (1.0 .- y_vec) .* log.(1.0 .- p_pred))
    pseudo_r2 = (abs(ll_null) > 1e-6) ? max(0.0, 1.0 - ll_fit / ll_null) : 0.0

    return (l50, slope, temp_coef, pseudo_r2)
end

"""
    generate_mock_survey_data(; n_samples=500, rng=Random.default_rng())

Generate a synthetic research survey `DataFrame` representing observational
field records of benthic snow crab (*Chionoecetes opilio*) for model calibration.

# Generated Columns
- `cw::Float64`: Carapace width in millimeters (instars 4 through 12).
- `sex::Int`: Biological sex (`0` = male, `1` = female).
- `maturity::Int`: Morphometric maturity status (`0` = immature, `1` = mature).
- `mass_g::Float64`: Wet body mass in grams including log-normal condition scatter.
- `temp_c::Float64`: Ambient bottom water temperature in °C.
- `fecundity::Float64`: Brooded egg count for mature females (0.0 for others).
- `instar::Int`: Estimated developmental instar index (4 through 12).
- `shell_age::Float64`: Carapace shell age / post-molt duration in years.

# Arguments
- `n_samples::Int`: Number of simulated crab records (default: 500).
- `rng`: Random number generator (default: `Random.default_rng()`).

# Returns
`DataFrame`: Simulated survey records.
"""
function generate_mock_survey_data(;
    n_samples::Int = 500,
    rng::AbstractRNG = Random.default_rng()
)
    cws = zeros(Float64, n_samples)
    sexes = zeros(Int, n_samples)
    mats = zeros(Int, n_samples)
    masses = zeros(Float64, n_samples)
    temps = zeros(Float64, n_samples)
    fecs = zeros(Float64, n_samples)
    instars = zeros(Int, n_samples)
    shell_ages = zeros(Float64, n_samples)

    instar_weights = [0.18, 0.16, 0.15, 0.14, 0.12, 0.10, 0.08, 0.05, 0.02]
    instar_choices = 4:12

    for i in 1:n_samples
        cum_r = rand(rng)
        acc = 0.0
        chosen_instar = 4
        for (idx, w) in enumerate(instar_weights)
            acc += w
            if cum_r <= acc || idx == length(instar_weights)
                chosen_instar = instar_choices[idx]
                break
            end
        end

        s = rand(rng) < 0.50 ? 1 : 0
        t = clamp(rand(rng, Normal(3.5, 1.5)), 0.5, 8.0)
        cw = carapacewidth(chosen_instar, s; type = :random, rng = rng)

        p_mat = maturity(s, log(cw); temp = t, rng = rng)
        mat = (rand(rng) < p_mat) ? 1 : 0

        age = (mat == 0) ? rand(rng, Uniform(0.2, 1.0)) : rand(rng, Uniform(1.0, 5.0))
        wt = calculate_mass(cw, s; mat = mat, stochastic = true, rng = rng)

        fec = 0.0
        if s == 1 && mat == 1
            meth = (age <= 2.0) ? "primiparous" : "multiparous"
            fec = Float64(fecundity(cw; method = meth, shell_age = age, rng = rng))
        end

        cws[i] = cw
        sexes[i] = s
        mats[i] = mat
        masses[i] = wt
        temps[i] = t
        fecs[i] = fec
        instars[i] = chosen_instar
        shell_ages[i] = age
    end

    return DataFrame(
        cw = cws,
        sex = sexes,
        maturity = mats,
        mass_g = masses,
        temp_c = temps,
        fecundity = fecs,
        instar = instars,
        shell_age = shell_ages
    )
end

"""
    fit_empirical_biology_model(data::DataFrame)

Calibrate an `EmpiricalBiologicalModel` from empirical observational or research
survey data, deriving:
1. Log-linear allometric weight-length parameters (α, β, σ) across strata.
2. Logistic morphometric maturity ogives (L₅₀, β_slope, κ_temp) per sex.
3. Brood fecundity scaling curves for primiparous and multiparous females.
4. Modal carapace width dimensions and dispersion per developmental instar.
5. Trait covariance matrix across CW, wet mass, maturity, and brood fecundity.

# Arguments
- `data::DataFrame`: Observational survey table containing columns `:cw`, `:sex`,
  `:maturity`, `:mass_g`, `:temp_c`, `:fecundity`, and optional `:instar`, `:shell_age`.

# Returns
`EmpiricalBiologicalModel`: Fitted biological and trait covariance model.
"""
function fit_empirical_biology_model(data::DataFrame)
    required_cols = [:cw, :sex, :maturity, :mass_g, :temp_c, :fecundity]
    for col in required_cols
        if !(col in propertynames(data))
            error("Data must contain required column: $(col)")
        end
    end

    n_records = nrow(data)
    if n_records < 10
        error("At least 10 observations required to fit empirical model, got $(n_records)")
    end

    weight_params = Dict{Symbol, Tuple{Float64, Float64, Float64}}()
    maturity_params = Dict{Symbol, Tuple{Float64, Float64, Float64}}()
    fecundity_params = Dict{Symbol, Tuple{Float64, Float64, Float64}}()
    instar_modals = Dict{Int, Tuple{Float64, Float64}}()
    r2_scores = Dict{Symbol, Float64}()

    # 1. Allometric Weight Parameters
    strata_definitions = [
        (:male_imm, (data.sex .== 0) .&& (data.maturity .== 0), (0.0002665, 3.098, 0.08)),
        (:male_mat, (data.sex .== 0) .&& (data.maturity .== 1), (0.0001830, 3.160, 0.08)),
        (:female_imm, (data.sex .== 1) .&& (data.maturity .== 0), (0.0003440, 2.980, 0.08)),
        (:female_mat, (data.sex .== 1) .&& (data.maturity .== 1), (0.0004500, 2.920, 0.08))
    ]

    for (strat_sym, mask, fallback) in strata_definitions
        sub_cw = data.cw[mask]
        sub_wt = data.mass_g[mask]
        valid = (sub_cw .> 0.0) .&& (sub_wt .> 0.0)
        if count(valid) >= 5
            x = log.(Float64.(sub_cw[valid]))
            y = log.(Float64.(sub_wt[valid]))
            intercept, slope, sigma, r2 = _fit_ols(x, y)
            weight_params[strat_sym] = (exp(intercept), slope, sigma)
            r2_scores[strat_sym] = r2
        else
            weight_params[strat_sym] = fallback
            r2_scores[strat_sym] = 0.95
        end
    end

    # 2. Logistic Maturity Parameters
    for (sex_sym, s_code, fallback) in [
        (:male, 0, (91.8, 6.9957, -0.05)),
        (:female, 1, (51.6, 12.9812, -0.08))
    ]
        mask = (data.sex .== s_code) .&& (data.cw .>= 35.0)
        sub_cw = data.cw[mask]
        sub_temp = data.temp_c[mask]
        sub_mat = Float64.(data.maturity[mask])

        if count(mask) >= 10 && length(unique(sub_mat)) > 1
            l50, slope, temp_coef, pseudo_r2 = _fit_irls_logistic(
                Float64.(sub_cw),
                Float64.(sub_temp),
                sub_mat
            )
            maturity_params[sex_sym] = (l50, slope, temp_coef)
            r2_scores[Symbol("maturity_", sex_sym)] = pseudo_r2
        else
            maturity_params[sex_sym] = fallback
            r2_scores[Symbol("maturity_", sex_sym)] = 0.85
        end
    end

    # 3. Fecundity Parameters
    has_shell_age = :shell_age in propertynames(data)
    fem_mat_mask = (data.sex .== 1) .&& (data.maturity .== 1) .&& (data.fecundity .> 0.0)

    for (par_sym, par_mask, fallback) in [
        (
            :primiparous,
            has_shell_age ?
                (fem_mat_mask .&& (data.shell_age .<= 2.0)) :
                (fem_mat_mask .&& (data.cw .< 65.0)),
            (-0.7229, 2.94, 0.090)
        ),
        (
            :multiparous,
            has_shell_age ?
                (fem_mat_mask .&& (data.shell_age .> 2.0)) :
                (fem_mat_mask .&& (data.cw .>= 65.0)),
            (0.0158, 2.5811, 0.145)
        )
    ]
        sub_cw = data.cw[par_mask]
        sub_fec = data.fecundity[par_mask]
        valid = (sub_cw .> 0.0) .&& (sub_fec .> 0.0)

        if count(valid) >= 5
            x = log10.(Float64.(sub_cw[valid]))
            y = log10.(Float64.(sub_fec[valid]))
            intercept, slope, sigma, r2 = _fit_ols(x, y)
            fecundity_params[par_sym] = (intercept, slope, sigma)
            r2_scores[par_sym] = r2
        else
            fecundity_params[par_sym] = fallback
            r2_scores[par_sym] = 0.80
        end
    end

    # 4. Instar Modal Dimensions
    has_instar = :instar in propertynames(data)
    for inst in 4:12
        if has_instar && count(data.instar .== inst) >= 3
            cws_inst = Float64.(data.cw[data.instar .== inst])
            instar_modals[inst] = (mean(cws_inst), max(0.5, std(cws_inst)))
        else
            mid_cw = 0.5 * (llbm(inst) + llbm(inst + 1))
            sd_cw = max(0.8, 0.5 * (llbm(inst + 1) - llbm(inst)) / 3.0)
            instar_modals[inst] = (mid_cw, sd_cw)
        end
    end

    # 5. Trait Covariance Matrix
    trait_names = [:cw, :mass_g, :maturity, :fecundity]
    trait_matrix = zeros(Float64, n_records, length(trait_names))
    trait_matrix[:, 1] = Float64.(data.cw)
    trait_matrix[:, 2] = Float64.(data.mass_g)
    trait_matrix[:, 3] = Float64.(data.maturity)
    trait_matrix[:, 4] = Float64.(data.fecundity)

    raw_cov = cov(trait_matrix)
    sym_cov = Matrix(Hermitian(raw_cov + 1e-4 * I))

    return EmpiricalBiologicalModel(
        weight_params,
        maturity_params,
        fecundity_params,
        instar_modals,
        trait_names,
        sym_cov,
        r2_scores,
        n_records
    )
end

"""
    sample_crab_traits(model::EmpiricalBiologicalModel; cw, sex, temp=4.0, shell_age=1.0, rng=Random.default_rng())

Sample or predict comprehensive biological traits for an individual crab given its
carapace width, sex, and ambient environmental conditions from an empirical model.

# Mathematical Formulation
- **Maturity**: Evaluated from the fitted logistic ogive with temperature shift:
  ```math
  P(mature) = \\frac{1}{1 + \\exp\\left(-\\beta (\\ln(CW) - \\ln(L_{50}(T))) - \\kappa_{temp} (T - 4.0)\\right)}
  ```
- **Wet Mass**:
  ```math
  W = \\alpha CW^\\beta \\cdot \\exp(\\epsilon - 0.5\\sigma^2), \\quad \\epsilon \\sim \\mathcal{N}(0, \\sigma^2)
  ```
- **Brood Fecundity**: Extruded egg count for mature females with senescence.
- **Estimated Instar**: Modal instar whose mean carapace width is closest to `cw`.

# Arguments
- `model::EmpiricalBiologicalModel`: Fitted empirical biological model.
- `cw::Real`: Carapace width in millimeters.
- `sex`: Sex code (`0`, `:male`, `male` or `1`, `:female`, `female`).
- `temp::Real`: Ambient bottom temperature in °C (default: 4.0).
- `shell_age::Real`: Post-molt carapace shell age in years (default: 1.0).
- `rng`: Random number generator (default: `Random.default_rng()`).

# Returns
`NamedTuple` with fields:
- `maturity_prob::Float64`: Predicted probability of morphometric maturity.
- `is_mature::Int`: Sampled binary maturity status (`0` or `1`).
- `mass_g::Float64`: Wet body mass in grams.
- `fecundity::Float64`: Brooded egg count (0.0 if immature or male).
- `estimated_instar::Int`: Estimated developmental instar index.
"""
function estimate_demographics(
    model::EmpiricalBiologicalModel;
    morphometry::Real,
    sex,
    temp::Real = 4.0,
    shell_age::Real = 1.0,
    rng::AbstractRNG = Random.default_rng()
)
    if morphometry <= 0.0
        return (
            maturity_prob = 0.0,
            is_mature = 0,
            mass_g = 0.0,
            fecundity = 0.0,
            estimated_instar = 4
        )
    end

    is_fem = (sex == 1 || sex == :female || Symbol(sex) == :female)
    sex_sym = is_fem ? :female : :male

    # Maturity ogive
    p_mat = if morphometry < 35.0
        0.0
    else
        l50, slope, t_coef = model.maturity_params[sex_sym]
        eff_l50 = max(35.0, l50 - 2.0 * (temp - 4.0))
        lin_pred = slope * (log(morphometry) - log(eff_l50)) + t_coef * (temp - 4.0)
        1.0 / (1.0 + exp(-clamp(lin_pred, -25.0, 25.0)))
    end
    mat_val = (rand(rng) < p_mat) ? 1 : 0

    # Wet Mass
    strat = if !is_fem
        (mat_val == 1) ? :male_mat : :male_imm
    else
        (mat_val == 1) ? :female_mat : :female_imm
    end
    alpha, beta_val, sigma = model.weight_params[strat]
    base_mass = alpha * (Float64(morphometry)^beta_val)
    mass_g = base_mass * exp(rand(rng, Normal(0.0, sigma)) - 0.5 * sigma^2)

    # Fecundity
    fecundity_val = 0.0
    if is_fem && mat_val == 1
        par_sym = (shell_age <= 2.0) ? :primiparous : :multiparous
        a, b, sig = model.fecundity_params[par_sym]
        log10_fec = a + b * log10(morphometry) + rand(rng, Normal(0.0, sig))
        senescence = (par_sym === :multiparous && shell_age > 3.0) ?
            max(0.60, 1.0 - 0.08 * (shell_age - 3.0)) : 1.0
        fecundity_val = max(0.0, round(10.0^log10_fec * senescence))
    end

    # Estimated Instar
    best_instar = 4
    min_dist = Inf
    for (inst, (m, _)) in model.instar_modals
        d = abs(morphometry - m)
        if d < min_dist
            min_dist = d
            best_instar = inst
        end
    end

    return (
        maturity_prob = p_mat,
        is_mature = mat_val,
        mass_g = mass_g,
        fecundity = fecundity_val,
        estimated_instar = best_instar
    )
end

"""
    summarize_empirical_model(model::EmpiricalBiologicalModel; io::IO=stdout)

Print a comprehensive, human-readable summary of an `EmpiricalBiologicalModel`'s
calibrated allometric curves, maturity thresholds, fecundity scaling, instar
modes, and trait covariance matrix.
"""
function summarize_empirical_model(
    model::EmpiricalBiologicalModel;
    io::IO = stdout
)
    println(io, "="^78)
    println(io, "  Empirical Biological Model Summary (N = $(model.sample_size) records)")
    println(io, "="^78)

    println(io, "\n--- Allometric Mass Parameters (W = α * CW^β) ---")
    println(io, rpad("Stratum", 14), rpad("α", 12), rpad("β", 10), rpad("σ", 10), "R²")
    println(io, "-"^56)
    for strat in [:male_imm, :male_mat, :female_imm, :female_mat]
        if haskey(model.weight_params, strat)
            a, b, s = model.weight_params[strat]
            r2 = get(model.r2_scores, strat, 0.0)
            println(
                io,
                rpad(string(strat), 14),
                rpad(string(round(a, digits = 7)), 12),
                rpad(string(round(b, digits = 3)), 10),
                rpad(string(round(s, digits = 4)), 10),
                string(round(r2, digits = 3))
            )
        end
    end

    println(io, "\n--- Morphometric Maturity Ogives (Logit) ---")
    println(
        io,
        rpad("Sex", 10),
        rpad("L50 (mm)", 12),
        rpad("Slope (β)", 12),
        rpad("Temp Coef", 12),
        "Pseudo-R²"
    )
    println(io, "-"^56)
    for s_sym in [:male, :female]
        if haskey(model.maturity_params, s_sym)
            l50, slp, tc = model.maturity_params[s_sym]
            pr2 = get(model.r2_scores, Symbol("maturity_", s_sym), 0.0)
            println(
                io,
                rpad(string(s_sym), 10),
                rpad(string(round(l50, digits = 1)), 12),
                rpad(string(round(slp, digits = 3)), 12),
                rpad(string(round(tc, digits = 3)), 12),
                string(round(pr2, digits = 3))
            )
        end
    end

    println(io, "\n--- Brood Fecundity Scaling (log10(Eggs) = a + b * log10(CW)) ---")
    println(
        io,
        rpad("Parity", 16),
        rpad("Intercept (a)", 16),
        rpad("Slope (b)", 12),
        rpad("σ", 10),
        "R²"
    )
    println(io, "-"^64)
    for p_sym in [:primiparous, :multiparous]
        if haskey(model.fecundity_params, p_sym)
            a, b, s = model.fecundity_params[p_sym]
            r2 = get(model.r2_scores, p_sym, 0.0)
            println(
                io,
                rpad(string(p_sym), 16),
                rpad(string(round(a, digits = 4)), 16),
                rpad(string(round(b, digits = 3)), 12),
                rpad(string(round(s, digits = 4)), 10),
                string(round(r2, digits = 3))
            )
        end
    end

    println(io, "\n--- Instar Modal Dimensions ---")
    println(io, rpad("Instar", 10), rpad("Mean CW (mm)", 16), "SD CW (mm)")
    println(io, "-"^38)
    for inst in sort(collect(keys(model.instar_modals)))
        m_cw, sd_cw = model.instar_modals[inst]
        println(
            io,
            rpad(string(inst), 10),
            rpad(string(round(m_cw, digits = 2)), 16),
            string(round(sd_cw, digits = 2))
        )
    end

    println(io, "\n--- Trait Covariance Matrix ---")
    col_width = 16
    col_headers = join([rpad(string(n), col_width) for n in model.trait_names], "")
    println(io, rpad("Trait", col_width), col_headers)
    println(io, "-"^(col_width + col_width * length(model.trait_names)))
    for (i, row_name) in enumerate(model.trait_names)
        vals = join([
            rpad(string(round(model.trait_cov[i, j], sigdigits = 4)), col_width)
            for j in 1:length(model.trait_names)
        ], "")
        println(io, rpad(string(row_name), col_width), vals)
    end
    println(io, "="^78)
end


