"""
    organism_dynamics.jl

Biological physiological dynamics, bioenergetics, ecdysis (molting), mortality,
and reproduction for hybrid aggregate and individual-based organism models.
"""

using DataFrames
using Distributions
using Random
using LinearAlgebra

"""
    _logistic_safe(x::Real)

Numerically stable logistic sigmoid function:
```math
\\sigma(x) = \\frac{1}{1 + e^{-x}}
```

# Arguments
- `x::Real`: Logit or linear predictor value.

# Returns
`Float64`: Probability in the open interval (0, 1).
"""
function _logistic_safe(x::Real)
    if x >= 0.0
        z = exp(-x)
        return 1.0 / (1.0 + z)
    else
        z = exp(x)
        return z / (1.0 + z)
    end
end

"""
    stage_size_lower_bound(behavior::SpeciesBehavior, stage::Real, sex)

Compute the empirical lower bound of morphometric size for a given species, stage,
and sex.
"""
function stage_size_lower_bound(behavior::SpeciesBehavior, stage::Real, sex)
    # Fallback - concrete implementation in species.jl
    return stage_size_lower_bound(SpeciesBehaviorImpl(), stage, sex)
end

stage_size_lower_bound(behavior::SpeciesBehavior, stage::AbstractArray, sex) =
    stage_size_lower_bound.(Ref(behavior), stage, Ref(sex))

"""
    get_stage_size(behavior::SpeciesBehavior, stage::Real, sex; type=:mid, rng=Random.default_rng())

Compute or sample carapace size (mm) for a given developmental `stage` and `sex`.
Alias: `stage_size`.

# Options for `type`
- `:lb`: Lower bound for the stage.
- `:ub`: Upper bound for the stage (lower bound of next stage).
- `:mid`: Midpoint between lower and upper bounds.
- `:random`: Stochastically sampled variate strictly bounded within `[lb, ub]`
  using a symmetric `Beta(2.5, 2.5)` distribution centered at `mid`.

# Arguments
- `behavior::SpeciesBehavior`: Species-specific behavioral dispatch.
- `stage::Real`: Stage index.
- `sex`: Sex indicator (`0`/`:male`/`male` or `1`/`:female`/`female`).
- `type::Symbol`: `:lb`, `:ub`, `:mid`, or `:random`.
- `rng::AbstractRNG`: Random number generator.

# Returns
Estimated carapace size in millimeters.
"""
function get_stage_size(
    behavior::SpeciesBehavior,
    stage::Real,
    sex;
    type::Symbol = :mid,
    rng::AbstractRNG = Random.default_rng(),
    params::IBMParameters = IBMParameters()
)
    lb = stage_size_lower_bound(behavior, stage, sex)
    ub = stage_size_lower_bound(behavior, stage + 1.0, sex)
    if type === :lb
        return lb
    elseif type === :ub
        return ub
    elseif type === :mid
        return (lb + ub) / 2.0
    elseif type === :random
        u = rand(rng, Beta(params.organism.stage_size_sample_a, params.organism.stage_size_sample_b))
        return lb + u * (ub - lb)
    else
        error("Unknown get_stage_size type: $(type). Supported: :lb, :ub, :mid, :random")
    end
end

const stage_size = get_stage_size

"""
    growth_increment(behavior::SpeciesBehavior, log_size::Real; kwargs...)

Predict the natural logarithm of post-molt size given pre-molt natural logarithm.
"""
function growth_increment(
    behavior::SpeciesBehavior,
    log_size::Real;
    kwargs...
)
    # Dispatch to concrete implementation
    return growth_increment(SpeciesBehaviorImpl(), log_size; kwargs...)
end

"""
    predict_postgrowth_size(behavior, size; sex=0, mat=0, temp=nothing, condition=1.0, rng=Random.default_rng(), growth_quantile=nothing)

Predict post-molt size in millimeters (linear scale) from pre-molt size, sex,
maturity status, ambient temperature, condition, and individual growth quantile.
"""
function predict_postgrowth_size(
    behavior::SpeciesBehavior,
    size::Real;
    sex = 0,
    mat::Integer = 0,
    temp = nothing,
    condition::Real = 1.0,
    rng::AbstractRNG = Random.default_rng(),
    growth_quantile::Union{Nothing, Real} = nothing
)
    if size <= 0.0
        error("Size must be strictly positive, received: $(size)")
    end
    log_size = log(size)
    log_size_next = growth_increment(
        behavior,
        log_size;
        sex = sex,
        mat = mat,
        temp = temp,
        condition = condition,
        rng = rng,
        growth_quantile = growth_quantile
    )
    return exp(log_size_next)
end

"""
    growth_schedule(behavior::SpeciesBehavior)

Construct a reference `DataFrame` summarizing empirical size bounds, medians,
and expected molt increments across developmental stages for both sexes.
"""
function growth_schedule(behavior::SpeciesBehavior)
    stages = 1:14
    schedule = DataFrame(
        stage = collect(stages),
        size_male_lb = stage_size_lower_bound.(Ref(behavior), stages, :male),
        size_female_lb = stage_size_lower_bound.(Ref(behavior), stages, :female),
        size_male_ub = stage_size_lower_bound.(Ref(behavior), stages .+ 1, :male),
        size_female_ub = stage_size_lower_bound.(Ref(behavior), stages .+ 1, :female)
    )
    schedule.size_male_med = (schedule.size_male_lb .+ schedule.size_male_ub) ./ 2.0
    schedule.size_female_med = (schedule.size_female_lb .+ schedule.size_female_ub) ./ 2.0
    schedule.size_male_range = schedule.size_male_ub .- schedule.size_male_lb
    schedule.size_female_range = schedule.size_female_ub .- schedule.size_female_lb
    schedule.size_male_inc = zeros(Float64, length(stages))
    schedule.size_female_inc = zeros(Float64, length(stages))
    for k in 1:(length(stages) - 1)
        schedule.size_male_inc[k] = schedule.size_male_med[k + 1] - schedule.size_male_med[k]
        schedule.size_female_inc[k] = schedule.size_female_med[k + 1] - schedule.size_female_med[k]
    end
    return schedule
end

"""
    maturity(behavior::SpeciesBehavior, sex, log_size::Real; kwargs...)

Calculate the probability of transitioning to morphometric maturity during molt.
"""
function maturity(
    behavior::SpeciesBehavior,
    sex,
    log_size::Real;
    kwargs...
)
    error("maturity not implemented for $(typeof(behavior))")
end

"""
    fecundity(behavior::SpeciesBehavior, size::Real; kwargs...)

Estimate brood fecundity as an allometric function of female size.
"""
function fecundity(
    behavior::SpeciesBehavior,
    size::Real;
    kwargs...
)
    error("fecundity not implemented for $(typeof(behavior))")
end

fecundity(behavior::SpeciesBehavior, size::AbstractArray; kwargs...) =
    fecundity.(Ref(behavior), size; kwargs...)

"""
    individuation_stage(params) → Int

Stage at which an aggregate cohort transitions into individuated `Organism` agents.

`OrganismParams.juvenile_max_stage`: aggregate groups cover stages
`0:(juvenile_max_stage - 1)`, and the cohort that completes the last of them
individuates. Derived from configuration rather than the literal `6`, so
reconfiguring the juvenile span actually moves the boundary instead of leaving the
aggregate cohort to advance past a stage nothing was watching for — in which case
no individual was ever individuated and the benthic population silently received
no recruits.
"""
individuation_stage(params::IBMParameters) = params.organism.juvenile_max_stage

"""
    shell_age_years(org::Organism, params::IBMParameters) → Float64

Years since metamorphosis, inferred as

```
(age_days - shell_age_stage_offset * (stage - shell_age_first_stage)) / days_per_year
```

floored at one year.

Single definition, used by both fecundity senescence and mortality senescence.

Those two paths previously used different quantities. Fecundity used this
age-based formula; mortality used *time since the last ecdysis*. That is not a
coarser version of the same thing, it is a different thing, and the difference is
not cosmetic:

- Senescence became a function of molt recency rather than age. A ten-year-old
  that had molted yesterday had **zero** mortality senescence, and a one-year-old
  that last molted 2000 days ago had exactly the same hazard as the ten-year-old.
  The one variable senescence is supposed to depend on was not read at all.
- A terminal-molt individual never molts again, so its time since ecdysis grows
  without limit and `exp(rate * (years - threshold))` diverges. The daily hazard
  reached 6.6e3 before `hazard_cap` clamped it, which the cap then reported as
  certain death.
- A female could be reproductively senescent on the age-based path while carrying
  no mortality senescence whatsoever.
"""
function shell_age_years(org::Organism, params::IBMParameters)
    op = params.organism
    inferred = (org.age_days - op.shell_age_stage_offset *
            (org.stage - op.shell_age_first_stage)) / params.simulation.days_per_year
    return max(1.0, inferred)
end

"""
    fecundity(org::Organism; rng, deterministic, params)

Estimate extruded brood fecundity directly for an individual female `Organism` agent.
"""
function fecundity(
    org::Organism;
    rng::AbstractRNG = Random.default_rng(),
    deterministic::Bool = false,
    params::IBMParameters = IBMParameters()
)
    if org.sex !== female || !org.is_mature
        return 0
    end
    is_p = (org.sperm_clutches <= 1)
    shell_age = shell_age_years(org, params)
    raw_eggs = fecundity(
        org.behavior,
        org.size;
        is_primiparous = is_p,
        shell_age = shell_age,
        fecundity_quantile = org.trait_fecundity_quantile,
        rng = rng,
        deterministic = deterministic
    )
    return round(Int, raw_eggs)
end

"""
    selectivity(behavior::SpeciesBehavior, size::Real; kwargs...)

Calculate gear selectivity or harvest probability as a function of size.
"""
function selectivity(
    behavior::SpeciesBehavior,
    size::Real;
    kwargs...
)
    error("selectivity not implemented for $(typeof(behavior))")
end

function selectivity(
    behavior::SpeciesBehavior,
    size::AbstractArray;
    kwargs...
)
    return selectivity.(Ref(behavior), size; kwargs...)
end

"""
    growth_simulate(modes; gcv=0.4, n_max=10000, s50=50.0, s50_sigma=50.0, rng=Random.default_rng())

Simulate multi-stage population growth across defined stage modal sizes with
growth variability and size selectivity filtering.
"""
function growth_simulate(
    modes::Vector{Float64};
    gcv::Float64 = 0.4,
    n_max::Int = 10000,
    s50::Float64 = 50.0,
    s50_sigma::Float64 = 50.0,
    rng::AbstractRNG = Random.default_rng()
)
    n_stages = length(modes)
    incr = modes[2:end] .- modes[1:(end - 1)]
    W = zeros(Float64, n_max, n_stages)
    W[:, 1] = rand(rng, Normal(modes[1], modes[1] / 20.0), n_max)
    for j in 2:n_stages
        inc = rand(rng, Normal(incr[j - 1], incr[j - 1] * gcv), n_max)
        W[:, j] = W[:, j - 1] .+ inc
    end
    s_modes = selectivity.(modes; s50=s50, s50_sigma=s50_sigma)
    O = deepcopy(W)
    for j in 1:n_stages
        n_drop = round(Int, (1.0 - s_modes[j]) * n_max)
        if n_drop > 0
            drop_indices = randperm(rng, n_max)[1:n_drop]
            O[drop_indices, j] .= NaN
        end
    end
    return (W, O)
end

"""
    sample_population(behavior::SpeciesBehavior, year::Int = -1; sex_ratio=0.5, rng=Random.default_rng())

Sample individual traits (degree-days, size, sex, maturity, stage) for
initializing population members.
"""
function sample_population(
    behavior::SpeciesBehavior,
    year::Int = -1;
    sex_ratio::Float64 = 0.5,
    rng::AbstractRNG = Random.default_rng()
)
    if year == -1
        degdays = rand(rng, Uniform(0.0, 0.1))
        sz = exp(rand(rng, Normal(2.75, 0.25 / 3.0)))
        sex = (rand(rng) < sex_ratio) ? 0 : 1
        mat = 0
        stage_val = 5
        return (degdays=degdays, size=sz, sex=sex, mat=mat, stage=stage_val)
    else
        degdays = rand(rng, Uniform(0.0, 0.2))
        stage_val = rand(rng, 4:7)
        sex = (rand(rng) < sex_ratio) ? 0 : 1
        sz = get_stage_size(behavior, stage_val, sex; type=:random, rng=rng)
        mat = 0
        return (degdays=degdays, size=sz, sex=sex, mat=mat, stage=stage_val)
    end
end

"""
    calculate_mass(behavior::SpeciesBehavior, size::Real, sex; kwargs...)

Compute wet body mass from size, sex, and maturity status for a species.
"""
function calculate_mass(
    behavior::SpeciesBehavior,
    size::Real,
    sex;
    kwargs...
)
    error("calculate_mass not implemented for $(typeof(behavior))")
end

function calculate_mass(
    org::Organism;
    stochastic::Bool = false,
    include_eggs::Bool = true,
    rng::AbstractRNG = Random.default_rng(),
    params::IBMParameters = IBMParameters()
)
    somatic_mass = calculate_mass(
        org.behavior,
        org.size,
        org.sex;
        mat = (is_adult(org) ? 1 : 0),
        stochastic = stochastic,
        rng = rng
    )
    egg_mass = (include_eggs && is_ovigerous(org)) ?
        (Float64(org.egg_count) * params.organism.egg_mass_g) : 0.0
    return somatic_mass + egg_mass
end

"""
    build_stages(behavior::SpeciesBehavior; kwargs...)

Construct discrete stage hierarchy for stage-transition modeling.
"""
function build_stages(
    behavior::SpeciesBehavior;
    kwargs...
)
    # Dispatch to concrete implementation
    return build_stages(SpeciesBehaviorImpl(); kwargs...)
end


"""
    shell_condition(org::Organism, current_time::Real)

Determine the standard fisheries carapace shell condition index (1 through 5)
for an individual organism based on time elapsed since the most recent ecdysis.

# Arguments
- `org::Organism`: Individual organism agent.
- `current_time::Real`: Current simulation timestamp (days).

# Returns
`Int`: Shell condition category (1 to 5).
"""
function shell_condition(
    org::Organism,
    current_time::Real;
    params::IBMParameters = IBMParameters()
)
    elapsed_days = max(0.0, current_time - org.last_growth_event_time)
    breaks = params.organism.shell_condition_days
    category = length(breaks) + 1
    for (index, threshold) in enumerate(breaks)
        if elapsed_days < threshold
            category = index
            break
        end
    end
    return category
end

"""
    calculate_energy_change(temp, food, size; params=IBMParameters(), mass=nothing)

Compute daily metabolic energy expenditure and realized food intake for an individual
organism given local environmental conditions and body size or mass.

# Arguments
- `temp::Real`: Local ambient temperature (°C).
- `food::Real`: Local available food resource concentration.
- `size::Real`: Organism size in mm.
- `params::IBMParameters`: Simulation parameters.
- `mass`: Optional body wet mass in grams.

# Returns
Tuple `(dE_dt, actual_intake)` in energy units / day.
"""
function calculate_energy_change(
    temp::Real,
    food::Real,
    size::Real;
    params::IBMParameters = IBMParameters(),
    mass::Union{Real, Nothing} = nothing
)
    if size <= 0.0 && (mass === nothing || mass <= 0.0)
        return (0.0, 0.0)
    end

    effective_mass = if mass !== nothing && mass > 0.0
        Float64(mass)
    else
        calculate_mass(size, :male; mat = 0)
    end

    org = params.organism
    # Kleiber-style allometric metabolic consumption
    mass_scaling = effective_mass^org.metabolic_mass_exponent
    metabolic_temp_factor = exp(org.metabolic_temperature_coefficient * temp)
    t_opt = org.metabolic_optimal_temperature
    t_sigma = org.metabolic_optimal_sigma
    thermal_modifier = exp(-((temp - t_opt)^2) / (2.0 * t_sigma^2))
    maintenance_cost = org.metabolic_rate_base * mass_scaling *
                       metabolic_temp_factor * thermal_modifier

    # Holling Type II functional feeding response
    half_sat = params.environment.half_saturation_food
    intake_demand = org.intake_rate_base * (food / (food + half_sat)) * mass_scaling
    actual_intake = min(intake_demand, max(0.0, food))

    # Net energy assimilation. A fraction `specific_dynamic_action` of the
    # assimilated intake is spent on processing, leaving the rest for growth.
    assim_eff = params.environment.assimilation_efficiency
    usable_fraction = clamp(1.0 - org.specific_dynamic_action, 0.0, 1.0)
    net_assimilation = (actual_intake * assim_eff * usable_fraction) - maintenance_cost

    return (net_assimilation, actual_intake)
end

function calculate_energy_change(
    org::Organism,
    temp::Real,
    food::Real;
    params::IBMParameters = IBMParameters()
)
    return calculate_energy_change(temp, food, org.size; params = params, mass = org.mass)
end

"""
    total_benthic_abundance(model)

Total benthic population size, counting both spatial individuals and the
aggregate juvenile cohorts that have not yet individuated.

Carrying capacity must be evaluated against the whole benthic population. Using
only the individuated agents understates density by an order of magnitude during
the juvenile phase, and once the aggregate cohorts are exhausted it makes
carrying capacity completely inert.
"""
function total_benthic_abundance(model)
    total = float(nagents(model))
    aggs = get(abmproperties(model), :aggregate_populations, nothing)
    if aggs !== nothing
        for grp in aggs.groups
            total += grp.abundance
        end
    end
    return total
end

"""
    benthic_density_feedback(model)

Density-dependent multiplier for mortality, equal to
`1 + (N / K)^exponent` where `N` is the total benthic abundance and `K` is
`benthic_carrying_capacity`.

The exponent comes from the configuration. At the default value of 2 the
feedback is gentle below `K` and rises steeply above it, so the population is
actually regulated near carrying capacity; a purely linear response stays finite
however far `N` exceeds `K` and therefore does not bound the population.
"""
function benthic_density_feedback(model)
    env = model.params.environment
    k_cap = env.benthic_carrying_capacity
    k_cap > 0.0 || return 1.0
    ratio = total_benthic_abundance(model) / k_cap
    return 1.0 + ratio^env.benthic_carrying_capacity_exponent
end

"""
    molt_probability(org, current_temp, current_time=0.0; params=IBMParameters())

Calculate the daily probability of undergoing ecdysis for an individual organism.
"""
function molt_probability(
    org::Organism,
    current_temp::Real,
    current_time::Real = 0.0;
    params::IBMParameters = IBMParameters()
)
    if org.is_mature || !params.toggles.use_molting
        return 0.0
    end

    min_intermolt = min_intermolt_days(org.behavior, org.stage)
    days_since_molt = current_time - org.last_growth_event_time
    if days_since_molt < min_intermolt
        return 0.0
    end

    if params.toggles.use_degree_days
        min_dd = required_degree_days(org.behavior, org.stage)
        if org.degree_days < min_dd
            return 0.0
        end
    end

    # Seasonal gate. `molt_season_min_stage` sets the first stage that must molt
    # within the season; earlier stages molt as soon as their physiological
    # requirements are met.
    if org.stage >= params.phenology.molt_season_min_stage
        day_of_year = mod(current_time, 365.0)
        in_molt_season = (params.phenology.molt_season_start <= day_of_year <=
                          params.phenology.molt_season_end)
        if !in_molt_season
            return 0.0
        end
    end

    if params.toggles.use_energetics
        required_energy = max(params.organism.energy_molt_floor,
                              params.organism.energy_molt_slope * org.size)
        if org.energy_reserve < required_energy
            return 0.0
        end
    end

    # Ecdysial thermal inhibition. The onset and bandwidth are configurable so
    # that inhibition can be placed above the warmest temperature reached during
    # the molt season; thresholds hardcoded below the seasonal maximum would zero
    # the molt probability for part of the season and stall development.
    onset = params.phenology.molt_thermal_onset
    bandwidth = params.phenology.molt_thermal_bandwidth
    thermal_inhibition = if params.toggles.use_thermal_stress && bandwidth > 0.0 && current_temp > onset
        max(0.0, 1.0 - ((current_temp - onset) / bandwidth)^2)
    else
        1.0
    end

    if thermal_inhibition <= 0.0
        return 0.0
    end

    progress_ratio = if params.toggles.use_degree_days
        org.degree_days / required_degree_days(org.behavior, org.stage)
    else
        days_since_molt / min_intermolt
    end

    lambda_molt = params.organism.molt_hazard_coefficient *
                     min(progress_ratio, params.organism.molt_progress_saturation) * thermal_inhibition
    p_molt = 1.0 - exp(-lambda_molt * params.simulation.dt)

    if p_molt < 0.0 || p_molt > 1.0
        error("Calculated molt probability $(p_molt) outside valid [0, 1] range.")
    end

    return p_molt
end

"""
    check_molt_conditions(org, current_temp, current_time=0.0; params=IBMParameters(), rng=nothing)

Determine whether an individual organism satisfies developmental and seasonal
conditions to initiate ecdysis.
"""
function check_molt_conditions(
    org::Organism,
    current_temp::Real,
    current_time::Real = 0.0;
    params::IBMParameters = IBMParameters(),
    rng::Union{AbstractRNG, Nothing} = nothing
)
    p_molt = molt_probability(org, current_temp, current_time; params = params)
    if p_molt <= 0.0
        return false
    end

    if rng === nothing
        return true
    else
        return rand(rng) < p_molt
    end
end

"""
    perform_molt!(org, current_time, temp=nothing; params=IBMParameters(), rng=Random.default_rng())

Execute ecdysis on `org`: increments `stage`, evaluates terminal morphometric maturity,
predicts post-growth size, updates wet mass, resets degree-days, and deducts energetic
molting expenditure.
"""
function perform_molt!(
    org::Organism,
    current_time::Real,
    temp::Union{Real, Nothing} = nothing;
    params::IBMParameters = IBMParameters(),
    rng::AbstractRNG = Random.default_rng()
)
    if org.is_mature || !params.toggles.use_molting
        return nothing
    end

    sex_code = (org.sex === female) ? 1 : 0
    prob_mat = maturity(
        org.behavior,
        sex_code,
        log(org.size);
        temp = temp,
        maturity_quantile = org.trait_maturity_quantile,
        rng = rng
    )
    mat_status = (rand(rng) < prob_mat) ? 1 : 0

    if mat_status == 1
        org.is_mature = true
        org.is_mating_ready = true
    end

    condition_factor = if params.toggles.use_energetics
        req_e = max(params.organism.energy_molt_floor,
                     params.organism.energy_molt_slope * org.size)
        cond_ratio = org.energy_reserve / req_e
        base_fraction = params.organism.molt_energy_base_fraction
        base_fraction + (1.0 - base_fraction) * tanh(cond_ratio - 1.0)
    else
        1.0
    end

    org.size = predict_postgrowth_size(
        org.behavior,
        org.size;
        sex = sex_code,
        mat = mat_status,
        temp = temp,
        condition = condition_factor,
        rng = rng,
        growth_quantile = org.trait_growth_quantile
    )
    org.stage += 1

    # Stage ceilings: 16 for males, 14 for females
    max_stage = (org.sex === female) ? params.organism.max_stage_female : params.organism.max_stage_male
    if org.stage >= max_stage
        org.stage = max_stage
        org.is_mature = true
    end

    org.mass = calculate_mass(org; stochastic = true, rng = rng)
    if params.toggles.use_energetics
        org.energy_reserve *= params.organism.molt_energy_retention
    end
    org.last_growth_event_time = Float64(current_time)
    org.degree_days = 0.0

    return nothing
end

"""
    mortality_components(
        org::Organism,
        temp::Real,
        local_density::Real,
        current_time::Real;
        params::IBMParameters = IBMParameters(),
        density_feedback::Real = 1.0,
        on_clamp = nothing
    )

Per-cause daily mortality hazards for one organism, as a `NamedTuple`.

Returns the seven component hazards (`baseline`, `predation`, `soft_shell`,
`starvation`, `thermal`, `senescence`, `cannibalism`) plus `raw_total` before the
cap and `total` after it.

The breakdown exists so a death can be attributed to the hazard that actually
produced it. The previous code computed a single summed probability and then
attributed the death by a rule unrelated to that sum — `is_mature → senescence,
else → predation` — so a juvenile that died of background baseline mortality was
recorded as a predation death, and a mature individual that died of anything at
all was recorded as a senescence death. The published death-cause counts were
therefore fiction, and `deaths_starvation` and `deaths_thermal` were never
incremented at all despite being declared and exported in the run output.

See [`dominant_mortality_cause`](@ref) for the attribution convention.

`density_feedback` multiplies the density-independent baseline hazards. Pass
`benthic_density_feedback(model)` so that `benthic_carrying_capacity` regulates
the individuated population; without it the baseline mortality is independent of
density and carrying capacity has no effect once a cohort is fully individuated.
"""
function mortality_components(
    org::Organism,
    temp::Real,
    local_density::Real,
    current_time::Real;
    params::IBMParameters = IBMParameters(),
    density_feedback::Real = 1.0,
    on_clamp = nothing
)
    dt = params.simulation.dt
    m = params.mortality
    dpy = params.simulation.days_per_year
    feedback = max(1.0, Float64(density_feedback))
    # Report, rather than silently absorb, a hazard that reaches the cap. A
    # capped probability is not a probability the model predicted: the individual
    # was certain to die, and the reported value is an artifact of the guard. That
    # indicates a mis-specified rate, and without a warning a run could clamp
    # every step and still look clean.
    #
    # `maxlog = 1` keeps this to one line per session: a run where the cap binds
    # daily would otherwise emit a warning per agent per day.
    default_on_clamp = function (source, raw, cap)
        @warn "daily mortality probability reached hazard_cap; the modelled hazard is at or above a certainty, so the reported probability is the cap, not the model" source =
            source raw = raw cap = cap maxlog = 1
        return nothing
    end
    report_clamp = (on_clamp === nothing) ? default_on_clamp : on_clamp

    # Every hazard is capped: senescence and thermal stress are unbounded in their
    # raw form, and a terminal-molt individual never molts again, so its time since
    # ecdysis grows without limit and its senescence hazard would eventually exceed
    # a daily probability of one.
    cap = clamp(m.hazard_cap, 0.0, 1.0)

    # 1. Baseline natural mortality modulated by individual mortality trait
    #    quantile and by population density relative to carrying capacity
    q_mort = org.trait_mortality_quantile
    if q_mort <= 0.0 || q_mort >= 1.0
        error("trait_mortality_quantile must be strictly in (0, 1), received: $(q_mort)")
    end
    individual_mort_factor = exp(
        m.baseline_trait_sigma * quantile(Normal(0.0, 1.0), q_mort)
    )
    base_daily = ((m.rate_baseline / dpy) * individual_mort_factor * feedback) * dt
    stage_multiplier = (org.stage < params.organism.juvenile_max_stage) ?
        m.baseline_juvenile_multiplier : 1.0

    # 2. Predation risk attenuated by size
    size_excess = max(0.0, org.size - m.predation_size_reference)
    predation_risk = ((m.rate_predation / dpy) * exp(-m.predation_size_decay * size_excess)) * dt

    # 3. Soft-shell vulnerability post-ecdysis.
    # Vulnerable only within `soft_shell_duration` days of the ecdysis, decaying
    # with e-folding time `soft_shell_decay_days`. The window used to be ignored,
    # so the hazard was nonzero forever: a long-lived individual kept accruing
    # soft-shell risk at a rate set only by how long ago it last molted.
    days_since_molt = max(0.0, current_time - org.last_growth_event_time)
    soft_shell_risk = if days_since_molt > m.soft_shell_duration
        0.0
    else
        tau = max(m.soft_shell_decay_days, eps())
        ((m.rate_soft_shell / dpy) * exp(-days_since_molt / tau)) * dt
    end

    # 4. Starvation mortality
    starvation_risk = if params.toggles.use_energetics && org.energy_reserve < 0.0
        deficit = -org.energy_reserve
        crit_pool = max(0.5, org.mass * m.starvation_critical_fraction)
        ((m.rate_starvation / dpy) * (1.0 - exp(-deficit / crit_pool))) * dt
    else
        0.0
    end

    # 5. Thermal stress mortality
    thermal_stress = if params.toggles.use_thermal_stress && temp > m.temp_thermal_stress
        raw = ((m.rate_thermal / dpy) *
               ((temp - m.temp_thermal_stress)^m.thermal_exponent)) * dt
        raw > cap && report_clamp(:thermal_stress, raw, cap)
        clamp(raw, 0.0, cap)
    else
        0.0
    end

    # 6. Senescence mortality for terminal molt adults.
    #    Keyed on inferred shell age, not on time since the last ecdysis: those are
    #    different quantities, and senescence is an age process. See
    #    `shell_age_years` for what the previous formulation got wrong.
    senescence_risk = if params.toggles.use_senescence && org.is_mature
        age_years = shell_age_years(org, params)
        thresh = m.senescence_age_threshold
        if age_years > thresh
            raw = ((m.rate_senescence / dpy) *
                   exp(m.senescence_exp_rate * (age_years - thresh))) * dt
            raw > cap && report_clamp(:senescence, raw, cap)
            clamp(raw, 0.0, cap)
        else
            0.0
        end
    else
        0.0
    end

    # 7. Cannibalism risk on small juveniles
    cannibalism_risk = if params.toggles.use_cannibalism &&
                         org.size < m.cannibalism_size_threshold && local_density > 0.0
        ((m.rate_cannibalism / dpy) *
         min(local_density / m.cannibalism_density_reference, m.cannibalism_density_cap)) * dt
    else
        0.0
    end

    raw_total = (base_daily * stage_multiplier) +
        predation_risk + soft_shell_risk + starvation_risk +
        thermal_stress + senescence_risk + cannibalism_risk
    total_prob = clamp(raw_total, 0.0, cap)
    raw_total > cap && report_clamp(:total, raw_total, cap)

    return (
        baseline = base_daily * stage_multiplier,
        predation = predation_risk,
        soft_shell = soft_shell_risk,
        starvation = starvation_risk,
        thermal = thermal_stress,
        senescence = senescence_risk,
        cannibalism = cannibalism_risk,
        raw_total = raw_total,
        total = total_prob
    )
end

"""
    dominant_mortality_cause(components) → Symbol

Which hazard most plausibly caused a death, given [`mortality_components`](@ref).

Dominant-cause attribution: the largest single component wins, rather than a
fractional split across causes. A competing-risks decomposition would apportion
one death across several causes, which the integer per-cause counters cannot
represent and which would be a modelling claim stronger than the model supports —
these hazards are summed as an approximation, not as mutually exclusive causes.

Returns `:none` when every component is zero.
"""
function dominant_mortality_cause(components)
    causes = (
        :baseline, :predation, :soft_shell, :starvation,
        :thermal, :senescence, :cannibalism
    )
    best = :none
    best_value = 0.0
    for cause in causes
        v = getfield(components, cause)
        if v > best_value
            best = cause
            best_value = v
        end
    end
    return best
end

"""
    calculate_mortality_probability(org, temp, local_density, current_time;
                                    params=IBMParameters(), density_feedback=1.0)

Total daily mortality probability for one organism: baseline natural mortality,
predation, soft-shell post-molt vulnerability, starvation, thermal stress,
senescence and cannibalism, summed and capped.

This is [`mortality_components`](@ref) reduced to its total. Call that directly
when the per-cause breakdown is needed, so a death can be attributed to the
hazard that produced it rather than inferred from the organism's maturity.
"""
function calculate_mortality_probability(
    org::Organism,
    temp::Real,
    local_density::Real,
    current_time::Real;
    params::IBMParameters = IBMParameters(),
    density_feedback::Real = 1.0,
    on_clamp = nothing
)
    return mortality_components(
        org, temp, local_density, current_time;
        params = params,
        density_feedback = density_feedback,
        on_clamp = on_clamp
    ).total
end


"""
    step_aggregate_groups!(model)

Advance aggregate cohort groups (stage 0 / egg through stage 5) by one time step `dt`.
Applies continuous stochastic survival (binomial trials) and probabilistic developmental
stage transitions (binomial draws scaling with accumulated physiological development).

Upon completing stage 5 (~4 years cumulative development), surviving individuals
undergo individuation into spatial `Organism` agents at stage 6 with 1:1 sex allocation.
"""
function step_aggregate_groups!(
    model;
    behavior::SpeciesBehavior = get(abmproperties(model), :species_behavior, default_species_behavior())
)
    dt = model.dt
    params = model.params
    groups = model.aggregate_populations.groups
    isempty(groups) && return nothing

    nx, ny = size(model.temperature_field)
    surviving_groups = AggregateGroup[]

    for grp in groups
        grp.age_days += dt
        cx = min(max(grp.pos[1], 1), nx)
        cy = min(max(grp.pos[2], 1), ny)
        temp = model.temperature_field[cx, cy]
        dd_inc = max(0.0, temp - model.params.environment.temp_base_degreedays) * dt
        grp.degree_days += dd_inc

        # 1. Stochastic aggregate mortality
        base_m = if grp.stage == 0
            0.002
        elseif grp.stage <= 2
            0.015
        elseif grp.stage == 3
            0.010
        elseif grp.stage == 4
            0.003
        else
            0.0015
        end

        # Density feedback uses the whole benthic population, so carrying
        # capacity constrains the juveniles as well as the individuals.
        density_feedback = benthic_density_feedback(model)
        hazard_rate = base_m * density_feedback
        cumulative_hazard = hazard_rate * dt

        if cumulative_hazard < 0.0
            error("Aggregate cumulative mortality hazard $(cumulative_hazard) cannot be negative.")
        end

        surv_prob = exp(-cumulative_hazard)
        n_curr = round(Int, grp.abundance)
        n_surv = if n_curr > 0
            rand(abmrng(model), Binomial(n_curr, surv_prob))
        else
            0
        end

        if n_surv <= 0
            continue
        end

        # 2. Continuous Probabilistic Transition (Degree-Days or Chronological Days)
        days_in_stage = max(0.0, model.current_time - grp.last_growth_time)
        d_metric, d_req, d_scale = if model.params.toggles.use_degree_days
            req_dd = required_degree_days(behavior, grp.stage)
            (grp.degree_days, req_dd, max(1.0, params.mortality.aggregate_hazard_degree_days * req_dd))
        else
            min_days = min_intermolt_days(behavior, grp.stage)
            (days_in_stage, min_days, max(1.0, params.mortality.aggregate_hazard_intermolt * min_days))
        end

        diff = (d_metric - d_req) / d_scale
        p_trans = if diff < -params.mortality.aggregate_transition_saturation
            0.0
        elseif diff > params.mortality.aggregate_transition_saturation
            1.0 - exp(-params.mortality.aggregate_transition_hazard * dt)
        else
            hazard = params.mortality.aggregate_transition_hazard / (1.0 + exp(-diff))
            1.0 - exp(-hazard * dt)
        end

        n_trans = (p_trans > 0.0 && n_surv > 0) ? rand(abmrng(model), Binomial(n_surv, p_trans)) : 0
        n_remain = n_surv - n_trans

        if n_remain > 0
            grp.abundance = Float64(n_remain)
            push!(surviving_groups, grp)
        end

        if n_trans > 0
            next_stage = grp.stage + 1
            if next_stage <= 5
                next_sz = (next_stage < 4) ? stage_size_lower_bound(behavior, Float64(next_stage), :male) : stage_size(behavior, next_stage, :male; type = :random, rng = abmrng(model))
                next_wt = calculate_mass(behavior, next_sz, :male; mat = 0, stochastic = true, rng = abmrng(model))
                new_grp = AggregateGroup(
                    id = model.aggregate_populations.next_group_id,
                    stage = next_stage,
                    abundance = Float64(n_trans),
                    size = next_sz,
                    mass = next_wt,
                    age_days = grp.age_days,
                    degree_days = 0.0,
                    last_growth_time = model.current_time,
                    pos = grp.pos
                )
                model.aggregate_populations.next_group_id += 1
                push!(surviving_groups, new_grp)
                if next_stage == 1 ||
                   next_stage == Int(params.species.benthic_anchor_stage)
                    model.total_recruits_settled += n_trans
                end
            elseif next_stage == individuation_stage(params)
                # Individuation into spatial `Organism` agents, at the configured
                # juvenile->benthic boundary. This was the literal `6`, so
                # configuring `juvenile_max_stage` to anything else meant the
                # aggregate cohort advanced to a stage nothing was watching for and
                # no individual was ever individuated: the benthic population
                # received no recruits at all, silently.
                ind_stage = next_stage
                for _ in 1:n_trans
                    rec_sex = (rand(abmrng(model)) < 0.5) ? male : female
                    mean_log_sz = log(
                        stage_size(
                            behavior, ind_stage, rec_sex;
                            type = :random, rng = abmrng(model)
                        )
                    )
                    rec_size = exp(rand(abmrng(model), Normal(mean_log_sz, 0.04)))
                    rec_mass = calculate_mass(behavior, rec_size, rec_sex; mat = 0, stochastic = true, rng = abmrng(model))

                    dx = rand(abmrng(model), -2:2)
                    dy = rand(abmrng(model), -2:2)
                    rx = min(max(grp.pos[1] + dx, 1), nx)
                    ry = min(max(grp.pos[2] + dy, 1), ny)

                    new_org = add_agent!(
                        (rx, ry),
                        model,
                        ind_stage,
                        rec_sex,
                        grp.age_days,
                        rec_size,
                        rec_mass,
                        model.params.organism.initial_energy,
                        model.current_time,
                        0.0,
                        false,
                        0,
                        0.0,
                        0,
                        false;
                        behavior = behavior
                    )
                    sample_individual_traits!(new_org, model, abmrng(model))
                end
            end
        end
    end

    model.aggregate_populations.groups = surviving_groups
    return nothing
end
