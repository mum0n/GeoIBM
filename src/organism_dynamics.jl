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

The stage-size curve is a property of the species, so it lives on the behaviour.
A behaviour that does not provide it is an error — it used to build a
`SpeciesBehaviorImpl()` and call that, discarding the argument, so a second species
would silently be measured against snow crab stage bounds.
"""
function stage_size_lower_bound(behavior::SpeciesBehavior, stage::Real, sex)
    return error(
        """
        $(typeof(behavior)) does not implement stage_size_lower_bound.
        The stage-size curve must be defined for the behaviour, not defaulted to one \
        species' curve here.
        """,
    )
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
    # The stage size curve saturates at the asymptotic size, so it already
    # respects the ceiling. Clamping keeps the invariant true even if the curve
    # is re-fitted by hand against a different asymptote, which is the one way a
    # caller could otherwise reintroduce an impossible animal.
    is_female = (sex == 1 || sex == :female || Symbol(sex) == :female)
    size_cap = is_female ? params.species.max_size_female : params.species.max_size_male
    lb = min(stage_size_lower_bound(behavior, stage, sex), size_cap)
    ub = min(stage_size_lower_bound(behavior, stage + 1.0, sex), size_cap)
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

The implementation is a property of the species, so it lives on the behaviour
rather than here. A behaviour that does not provide it is an error.

It used to build a `SpeciesBehaviorImpl()` and call that, discarding the argument.
With one species that is invisible, because every behaviour *is* a
`SpeciesBehaviorImpl` and dispatch already picks the right method. The cost only
appears when a second species is written: it would be evaluated against snow crab
growth coefficients and produce plausible, wrong animals. Failing at the call site
is the only safe behaviour.
"""
function growth_increment(
    behavior::SpeciesBehavior,
    log_size::Real;
    kwargs...
)
    return error(
        """
        $(typeof(behavior)) does not implement growth_increment.
        Species-dependent life-history rules must be defined for the behaviour, not \
        defaulted to one species' rules here.
        """,
    )
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
    growth_schedule(behavior::SpeciesBehavior; params = IBMParameters())

Construct a reference `DataFrame` summarizing empirical size bounds, medians,
and expected molt increments across developmental stages for both sexes.

The stage range comes from `OrganismParams.max_stage_*` rather than the literal
`1:14`, which had stopped at 14 while individuals could reach 16 — the table was
quietly missing the oldest stages, and after the renumbering it covered barely a
third of the life history.
"""
function growth_schedule(behavior::SpeciesBehavior;
    params::IBMParameters = IBMParameters())
    last_stage = max(params.organism.max_stage_male, params.organism.max_stage_female) - 1
    stages = 1:last_stage
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
    shell_age(org::Organism, params) -> Float64

Years elapsed since the individual's last moult, fractional and unclamped.

`age_days` is chronological age since settlement; `shell_age_stage_offset` per
stage is the accumulated intermolt time, so the difference is time since the
current moult. For a terminally moulted adult — which moults no more — that is its
whole post-maturity lifespan, which is what senescence should be keyed to.

`shell_age_years` is the same quantity floored at one year. The floor suits
parity inference, which asks "how many broods has she been through" and has no
meaning below one. It is wrong for senescence: an adult that moulted last month
reads as a year old, which shifts the whole senescence curve by a year.
"""
function shell_age(org::Organism, params::IBMParameters)
    op = params.organism
    return (org.age_days - op.shell_age_stage_offset *
            (org.stage - op.shell_age_first_stage)) / params.simulation.days_per_year
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
    # Parity is inferred from shell age, not from `sperm_clutches`. The latter is a
    # count of stored sperm, which a female gains at every mating and spends one
    # clutch per brood, so it is not a record of how many broods she has carried:
    # a first-time brooder holding two stores would otherwise be scored as
    # multiparous.
    is_p = shell_age_years(org, params) <= params.species.primiparous_shell_age_max
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
    growth_simulate(modes; behavior, gcv, size_cv, n_max, rng)

Simulate multi-stage population growth across defined stage modal sizes with
growth variability and size selectivity filtering.

`gcv` and `size_cv` are coefficients of variation, not sizes. Selectivity is taken
from `behavior` rather than passed in as `s50`/`s50_sigma`: those were defaulting
to a literal `50.0`, which is a second source of truth for the same quantity the
species declares as `maturity_l50_male` / `maturity_l50_female`. A caller that
changed the species ogive left this function filtering at 50 mm regardless.
"""
function growth_simulate(
    modes::Vector{Float64};
    behavior::SpeciesBehavior = default_species_behavior(),
    gcv::Float64 = 0.4,
    size_cv::Float64 = 0.05,
    n_max::Int = 10000,
    rng::AbstractRNG = Random.default_rng()
)
    n_stages = length(modes)
    incr = modes[2:end] .- modes[1:(end - 1)]
    W = zeros(Float64, n_max, n_stages)
    W[:, 1] = rand(rng, Normal(modes[1], modes[1] * size_cv), n_max)
    for j in 2:n_stages
        inc = rand(rng, Normal(incr[j - 1], incr[j - 1] * gcv), n_max)
        W[:, j] = W[:, j - 1] .+ inc
    end
    s_modes = selectivity.(modes; behavior = behavior)
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
    sample_population(behavior::SpeciesBehavior, year::Int = -1;
                      sex_ratio = 0.5, params = IBMParameters(),
                      rng = Random.default_rng())

Sample individual traits (degree-days, size, sex, maturity, stage) for
initializing population members.

The stage it samples is derived from `OrganismParams.juvenile_max_stage` rather
than written as a literal. The literals were `5` and `4:7`, which were correct
only while the individuation boundary sat at 6; a renumbering silently turned them
into stages three below where they belonged, and nothing failed because they were
still *valid* stage numbers — they just sampled the wrong part of the life history.
"""
function sample_population(
    behavior::SpeciesBehavior,
    year::Int = -1;
    sex_ratio::Float64 = 0.5,
    params::IBMParameters = IBMParameters(),
    rng::AbstractRNG = Random.default_rng()
)
    # The last aggregate stage is the one just below individuation; agents begin at
    # `juvenile_max_stage` itself.
    last_cohort = params.organism.juvenile_max_stage - 1
    if year == -1
        degdays = rand(rng, Uniform(0.0, params.simulation.initial_degree_days_min))
        sz = exp(rand(rng, Normal(params.organism.stage_size_sample_a,
            params.organism.stage_size_sample_b / 3.0)))
        sex = (rand(rng) < sex_ratio) ? 0 : 1
        mat = 0
        stage_val = last_cohort
        return (degdays=degdays, size=sz, sex=sex, mat=mat, stage=stage_val)
    else
        degdays = rand(rng, Uniform(0.0, params.simulation.initial_degree_days_min))
        stage_val = rand(rng, max(1, last_cohort - 1):(last_cohort + 2))
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

The stage table is species-specific, so it is defined on the behaviour. A behaviour
that does not provide it is an error rather than a silent substitution of the snow
crab table.
"""
function build_stages(
    behavior::SpeciesBehavior;
    kwargs...
)
    return error(
        """
        $(typeof(behavior)) does not implement build_stages.
        The Leslie stage table must be defined for the behaviour, not defaulted to \
        one species' table here.
        """,
    )
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
        # No sex is known on this (scalar) code path, so average the two
        # sex-specific allometries rather than silently assuming one sex.
        0.5 * (calculate_mass(size, :male; mat = 0) +
               calculate_mass(size, :female; mat = 0))
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
    actual_intake = min(intake_demand, max(0.0, isfinite(food) ? food : 0.0))

    # Net energy assimilation. A fraction `specific_dynamic_action` of the
    # assimilated intake is spent on processing, leaving the rest for growth.
    assim_eff = params.environment.assimilation_efficiency
    sda = org.specific_dynamic_action
    if sda < 0.0 || sda > 1.0
        error("specific_dynamic_action must be within [0, 1], received: $(sda)")
    end
    usable_fraction = 1.0 - sda
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
    density_feedback(model)

The density multiplier to use for this step, computed once in `model_step!` and
cached on the model.

`benthic_density_feedback` depends only on the population, not on the individual,
so evaluating it inside `agent_step!` recomputed the same value once per organism
per day, walking the aggregate cohorts each time. Profiling put a quarter of the
run's time there. The value is refreshed once per step instead, which is exactly
as current: the cohorts and the agent count do not change between `model_step!`
and the agent phase of the same step.

The cached value does go stale as soon as the agent phase removes organisms, so
read `benthic_density_feedback` directly outside a step — the GUI, the reporting
and the criteria evaluation do — rather than reading the cached value.
"""
function density_feedback(model)
    props = abmproperties(model)
    cached = get(props, :density_feedback, nothing)
    return cached === nothing ? benthic_density_feedback(model) : cached
end

"""
    current_day_of_year(model)

Day of the model's calendar at `model.current_time`.

`model_step!` computes it for the seasonal cycle and publishes it, so the agent
phase reads it rather than repeating the modulo for every organism. When there is
no step in progress — a GUI tick driving `agent_step!` directly, or a test — it is
computed here.
"""
function current_day_of_year(model)
    props = abmproperties(model)
    cached_time = get(props, :day_of_year_time, nothing)
    if cached_time !== nothing && cached_time == model.current_time
        return get(props, :day_of_year, mod(model.current_time, model.params.simulation.days_per_year))
    end
    return mod(model.current_time, model.params.simulation.days_per_year)
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
        day_of_year = mod(current_time, params.simulation.days_per_year)
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

    Execute ecdysis on `org`: increments `stage`, predicts post-growth size, updates
    wet mass, resets degree-days, and deducts energetic molting expenditure.

    Maturity is deliberately **not** evaluated here. It used to be, and that made
    maturation an annual coin flip: ecdysis for any stage at or above
    `molt_season_min_stage` is confined to `molt_season_start..molt_season_end`, so
    an individual that failed its maturity draw in one spring waited a full year for
    the next. Two routes into maturity — this draw and the terminal-stage rule below
    — were both step functions aligned to the same ~125-day window, which quantised
    legal-male biomass into annual pulses of several-fold amplitude. Maturity is now
    evaluated every step in `agent_step!`, where the size-dependent probability can
    actually accumulate.
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
    condition_factor = if params.toggles.use_energetics
        req_e = max(params.organism.energy_molt_floor,
                     params.organism.energy_molt_slope * org.size)
        cond_ratio = org.energy_reserve / req_e
        base_fraction = params.organism.molt_energy_base_fraction
        base_fraction + (1.0 - base_fraction) * tanh(cond_ratio - 1.0)
    else
        1.0
    end

    next_size = predict_postgrowth_size(
        org.behavior,
        org.size;
        sex = sex_code,
        # Always immature here: the guard above returns for a mature animal, and
        # maturity is no longer decided inside this function.
        mat = 0,
        temp = temp,
        condition = condition_factor,
        rng = rng,
        growth_quantile = org.trait_growth_quantile
    )
    # The ceiling is the species' asymptotic size, the same limit the stage size
    # curve saturates at, so the curve and the cap cannot disagree.
    size_cap = (org.sex === female) ? params.species.max_size_female :
               params.species.max_size_male
    org.size = min(next_size, size_cap)
    org.stage += 1

    # Model stage ceilings: 19 for males (instar 16), 16 for females (instar 13)
    max_stage = (org.sex === female) ? params.organism.max_stage_female : params.organism.max_stage_male
    if org.stage >= max_stage
        org.stage = max_stage
        # A female that reaches her terminal stage by molting rather than by
        # maturing was never `is_mating_ready`, and readiness is what the
        # reproductive step tests, so she could never carry a brood. Maturity
        # makes her ready for the same reason it makes a molting male ready.
        org.is_mature = true
        org.is_mating_ready = true
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

    # 1. Background natural mortality, modulated by the individual's mortality trait
    #    quantile, by stage and by population density relative to carrying capacity.
    #
    #    `baseline_rate_by_stage` replaces a single rate plus a juvenile multiplier.
    #    The multiplier was binary — one value below the individuation boundary and
    #    one above — so pelagic cohorts, first-instar crabs and terminal adults all
    #    shared a rate despite being three very different animals. The schedule is
    #    indexed by stage like every other, and holds its last value beyond its end,
    #    so it says the thing directly: high while pelagic, lower once benthic, lower
    #    again as the animal grows.
    q_mort = org.trait_mortality_quantile
    if q_mort <= 0.0 || q_mort >= 1.0
        error("trait_mortality_quantile must be strictly in (0, 1), received: $(q_mort)")
    end
    individual_mort_factor = exp(
        m.baseline_trait_sigma * quantile(Normal(0.0, 1.0), q_mort)
    )
    base_rate = stage_schedule_lookup(m.baseline_rate_by_stage, org.stage)
    base_daily = (base_rate / dpy) * individual_mort_factor * feedback * dt

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

    # 6. Senescence mortality.
    #
    # Keyed on inferred shell age, not on time since the last ecdysis: those are
    # different quantities, and senescence is an age process.
    #
    # Restricted to **terminally moulted** adults. `is_mature` is too weak a gate:
    # it is set as soon as an individual passes the maturity ogive, and for a crab
    # that is several years before it has finished growing. Senescence is wear-out
    # on a grown animal, so the gate is the terminal stage, and an animal that has
    # not got there pays no senescence at however old it is.
    #
    # The annual rate is formed first and converted to a daily probability by
    # `1 - exp(-λ·dt/dpy)`. The previous form, `λ/dpy * dt` clamped at
    # `hazard_cap`, was linear: at a shell age where λ reached 107/yr it computed a
    # daily hazard of 1.87, which the cap turned into certainty — one day's
    # arithmetic overflow presented as a modelled death. The exponential form cannot
    # exceed one however large λ becomes, so a steep ramp now degrades to "almost
    # certain" instead of "certain", which is what a large λ actually means.
    terminal = (org.sex === female) ? params.organism.max_stage_female :
               params.organism.max_stage_male
    senescence_risk = if params.toggles.use_senescence &&
                         org.is_mature &&
                         org.stage >= terminal
        # Fractional years since the last moult, unfloored: for a terminally
        # moulted adult that is the post-maturity lifespan, and it is 0 in the
        # year it reaches the terminal moult.
        age_years = shell_age(org, params)
        thresh = m.senescence_age_threshold
        if age_years > thresh
            lambda_per_year = m.rate_senescence *
                              exp(m.senescence_exp_rate * (age_years - thresh))
            raw = -expm1(-lambda_per_year * dt / dpy)
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

    raw_total = base_daily +
        predation_risk + soft_shell_risk + starvation_risk +
        thermal_stress + senescence_risk + cannibalism_risk
    total_prob = clamp(raw_total, 0.0, cap)
    raw_total > cap && report_clamp(:total, raw_total, cap)

    return (
        baseline = base_daily,
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

Advance aggregate cohort groups (the pelagic stages from zoea1 through the last benthic cohort stage) by one time step `dt`.
Applies continuous stochastic survival (binomial trials) and probabilistic developmental
stage transitions (binomial draws scaling with accumulated physiological development).

Upon completing the last cohort stage (~4 years cumulative development), surviving individuals
undergo individuation into spatial `Organism` agents at `juvenile_max_stage` with 1:1 sex allocation.
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
        # Pelagic stages (egg, zoea 1, zoea 2, megalopa) occupy the upper water column;
        # benthic juveniles settle to the seabed and experience bottom temperature.
        is_pelagic = grp.stage < larval_stage_count(behavior)
        props = abmproperties(model)
        temp_field = (is_pelagic && haskey(props, :surface_temperature_field)) ?
                     model.surface_temperature_field : model.temperature_field
        temp = temp_field[cx, cy]
        dd_inc = max(0.0, temp - model.params.environment.temp_base_degreedays) * dt
        grp.degree_days += dd_inc

        # 1. Stochastic aggregate mortality. Indexed by stage so a shorter
        # schedule holds its last value for the higher stages.
        base_m = stage_schedule_lookup(
            params.mortality.aggregate_baseline_hazards, grp.stage
        )

        # Density feedback uses the whole benthic population, so carrying
        # capacity constrains the juveniles as well as the individuals.
        density_feedback = benthic_density_feedback(model)

        # Thermal survival modulation: for pelagic larvae, survival is maximal at
        # 10-12°C (Ouellet & Sainte-Marie 2018), with concave-down viability:
        # h_temp(T) = exp(((T - T_opt)/sigma)^2 / 2)
        thermal_mult = if is_pelagic
            t_opt = params.mortality.larval_temp_opt
            t_sig = params.mortality.larval_temp_sigma
            exp(0.5 * ((temp - t_opt) / max(0.5, t_sig))^2)
        else
            1.0
        end

        # Over-age developmental penalty for pelagic larvae exceeding maximum duration (84 days)
        days_in_stage = max(0.0, model.current_time - grp.last_growth_time)
        overage_hazard = if is_pelagic && days_in_stage > 84.0
            params.mortality.larval_overage_hazard * (days_in_stage - 84.0) / 10.0
        else
            0.0
        end

        hazard_rate = (base_m * thermal_mult * density_feedback) + overage_hazard
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

        # 2. Continuous probabilistic transition.
        #
        # A cohort's members do not share a molt threshold. Each one draws its own
        # from a Poisson distribution whose mean is the stage's requirement, so the
        # requirement is the degree-days at which *half* the cohort has become
        # eligible — a 50% transition point, not a gate everyone crosses together.
        # That is what desynchronises a cohort, and it is why the requirement has to
        # be read as a distribution rather than as a threshold.
        #
        # The daily hazard is `h0 * P(threshold already met)`:
        #
        #     h(DD) = h0 * PoissonCDF(X; DD)
        #
        # which is zero at DD = 0, `h0 / 2` at DD = X by construction, and saturates
        # at `h0`. `h0` is `aggregate_transition_hazard`, a daily rate.
        #
        # There are deliberately no hard gates here. The previous form was a logistic
        # in `(DD - X) / (0.15 X)`, which reached its ceiling a third of the way to the
        # next stage and then held it there: the cohort left at full rate while still
        # short of the requirement, which is how a stage configured for half a year
        # was cleared in days.
        #
        # `required_degree_days` and `min_intermolt_days` are *per stage*, not
        # cumulative, and the cohort's `degree_days` is reset on transition.
        days_in_stage = max(0.0, model.current_time - grp.last_growth_time)
        (metric, req, h0) = if model.params.toggles.use_degree_days
            (grp.degree_days, required_degree_days(behavior, grp.stage),
             params.mortality.aggregate_transition_hazard)
        else
            (days_in_stage, min_intermolt_days(behavior, grp.stage),
             params.mortality.aggregate_transition_hazard)
        end

        # Fraction of the cohort whose own threshold has been reached. `req` may be
        # zero, in which case every member is eligible immediately.
        # `cdf(Poisson(req), ...)`, not `poiscdf`: `Distributions` does not export
        # `poiscdf` — it lives in `StatsFuns` — so the shorter name compiles and then
        # throws `UndefVarError` on the first cohort that reaches a transition.
        ready = if req > 0.0
            cdf(Poisson(req), floor(metric))
        else
            1.0
        end
        p_trans = ready > 0.0 ? 1 - exp(-h0 * ready * dt) : 0.0

        n_trans = (p_trans > 0.0 && n_surv > 0) ? rand(abmrng(model), Binomial(n_surv, p_trans)) : 0
        n_remain = n_surv - n_trans

        if n_remain > 0
            grp.abundance = Float64(n_remain)
            push!(surviving_groups, grp)
        end

        if n_trans > 0
            next_stage = grp.stage + 1
            benthic_anchor = Int(params.species.benthic_anchor_stage)
            if next_stage < individuation_stage(params)
                # The boundary comes from `juvenile_max_stage`. This was the literal
                # `5`, which agreed with the default `juvenile_max_stage = 6` only by
                # coincidence: configuring `juvenile_max_stage` to anything else put a
                # cohort into the aggregate branch at the stage the individuation
                # branch should have claimed, and it never became an agent.
                next_sz = (next_stage < benthic_anchor) ?
                          stage_size_lower_bound(behavior, Float64(next_stage), :male) :
                          stage_size(behavior, next_stage, :male;
                                     type = :random, rng = abmrng(model), params = params)
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
                if next_stage == benthic_anchor
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
                    rec_sex = (rand(abmrng(model)) < params.leslie.male_split_fraction) ? male :
                          female
                    mean_log_sz = log(
                        stage_size(
                            behavior, ind_stage, rec_sex;
                            type = :random, rng = abmrng(model), params = params
                        )
                    )
                    rec_size = exp(
                        rand(
                            abmrng(model),
                            Normal(mean_log_sz, params.environment.settlement_size_log_sd)
                        )
                    )
                    rec_mass = calculate_mass(behavior, rec_size, rec_sex; mat = 0, stochastic = true, rng = abmrng(model))

                    radius = params.environment.settlement_dispersal_radius
                    dx = rand(abmrng(model), -radius:radius)
                    dy = rand(abmrng(model), -radius:radius)
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

    model.aggregate_populations.groups = merge_cohort_groups(surviving_groups)
    return nothing
end

"""
    merge_cohort_groups(groups)

Combine aggregate cohorts that share a stage and a cell.

A cohort that both survives and transitions leaves its survivor group in place and
emits a *new* group for the transitions, at the same cell and the same stage it will
occupy tomorrow. Those children fragment in turn, so the number of groups grew
geometrically with the run: a 500-day run reached 1135 groups for six stages, and
the per-step cost is linear in the group count.

Pelagic cohorts carry no individual identity, so two groups at the same stage and
cell are interchangeable. Merging them is exact in expectation: the per-individual
hazard and transition probability are both monotone in `degree_days` and in time
in stage, so a binomial draw on the merged count at the abundance-weighted mean
rate has the same expected number of survivors and transitions as separate draws
on the parts. Abundances sum; size, mass, age, degree-days and last-growth time
become abundance-weighted means.

Keyed by `(stage, cell)`, so cohorts in different cells stay separate.
"""
function merge_cohort_groups(groups::AbstractVector{AggregateGroup})
    length(groups) <= 1 && return collect(groups)

    order = Tuple{Int, Int, Int}[]
    index = Dict{Tuple{Int, Int, Int}, AggregateGroup}()
    merged = AggregateGroup[]

    for grp in groups
        grp.abundance <= 0.0 && continue
        key = (grp.stage, Int(grp.pos[1]), Int(grp.pos[2]))
        existing = get(index, key, nothing)
        if existing === nothing
            # Copy, so the caller's array and the previous cohort objects are not
            # aliased into the new state.
            push!(merged, AggregateGroup(
                id = grp.id,
                stage = grp.stage,
                abundance = grp.abundance,
                size = grp.size,
                mass = grp.mass,
                age_days = grp.age_days,
                degree_days = grp.degree_days,
                last_growth_time = grp.last_growth_time,
                pos = grp.pos
            ))
            push!(order, key)
            index[key] = merged[end]
            continue
        end

        total = existing.abundance + grp.abundance
        w_existing = existing.abundance
        w_new = grp.abundance
        existing.size = (w_existing * existing.size + w_new * grp.size) / total
        existing.mass = (w_existing * existing.mass + w_new * grp.mass) / total
        existing.age_days = (w_existing * existing.age_days + w_new * grp.age_days) / total
        existing.degree_days =
            (w_existing * existing.degree_days + w_new * grp.degree_days) / total
        existing.last_growth_time =
            (w_existing * existing.last_growth_time + w_new * grp.last_growth_time) / total
        existing.abundance = total
    end
    return merged
end

"""
    calculate_habitat_suitability(
        temperature::Real,
        depth::Real;
        temp_half::Real = 5.0,
        temp_steepness::Real = 1.0,
        depth_opt::Real = 150.0,
        depth_sigma::Real = 50.0,
        depth_min::Real = 20.0,
        depth_max::Real = 400.0,
        temp_zero_threshold::Real = 10.0
    )::Float64

Compute the Habitat Suitability Index (HSI) for snow crab (*Chionoecetes opilio*)
as the product of thermal and bathymetric suitability components:

```math
HSI = S_T(T) \\times S_z(z)
```

# Thermal Suitability \$S_T(T)\$
A decreasing logistic response representing cold-water preference:
```math
S_T(T) = \\begin{cases}
0, & T \\ge T_{\\text{zero}} \\\\
\\frac{1}{1 + \\exp(k_T (T - T_{50}))} \\cdot w_T(T), & T > T_{50} \\\\
\\frac{1}{1 + \\exp(k_T (T - T_{50}))}, & T \\le T_{50}
\\end{cases}
```
where \$w_T(T) = \\frac{1}{2}\\left(1 + \\cos\\left(\\pi \\frac{T - T_{50}}{T_{\\text{zero}} - T_{50}}\\right)\\right)\$
smoothly tapers the probability to zero at \$T_{\\text{zero}}\$ (default 10.0 °C).

# Bathymetric Depth Suitability \$S_z(z)\$
A Gaussian response centered around optimal depth \$z_{\\text{opt}}\$ (140 m, \$\\sigma_z = 90\$ m)
with smooth cosine boundary window tapers at shallow (\$z_{\\min} = 20\$ m) and deep (\$z_{\\max} = 400\$ m)
limits:
```math
S_z(z) = \\exp\\left(-\\frac{1}{2}\\left(\\frac{z - z_{\\text{opt}}}{\\sigma_z}\\right)^2\\right) \\cdot w_{\\text{shallow}}(z) \\cdot w_{\\text{deep}}(z)
```
where \$w_{\\text{shallow}}(z)\$ and \$w_{\\text{deep}}(z)\$ smoothly taper to zero at \$z_{\\min}\$ and \$z_{\\max}\$.

# Arguments
- `temperature`: Bottom water temperature in °C.
- `depth`: Bathymetric seabed depth in meters (positive down).
- `temp_half`: Temperature of 50% thermal suitability (default: 5.0 °C).
- `temp_steepness`: Logistic slope parameter (default: 1.0).
- `depth_opt`: Optimal modal depth (default: 140.0 m).
- `depth_sigma`: Gaussian dispersion around modal depth (default: 90.0 m).
- `depth_min`: Shallow depth limit below which HSI is zero (default: 20.0 m).
- `depth_max`: Deep depth limit beyond which HSI is zero (default: 400.0 m).
- `temp_zero_threshold`: Temperature at which HSI tapers strictly to zero (default: 10.0 °C).

# Returns
Habitat suitability index as a `Float64` in `[0.0, 1.0]`.
"""
function calculate_habitat_suitability(
    temperature::Real,
    depth::Real;
    temp_half::Real = 5.0,
    temp_steepness::Real = 1.0,
    depth_opt::Real = 140.0,
    depth_sigma::Real = 90.0,
    depth_min::Real = 20.0,
    depth_max::Real = 400.0,
    temp_zero_threshold::Real = 10.0
)::Float64
    T = Float64(temperature)
    z = abs(Float64(depth))

    # Thermal suitability S_T
    s_t = if T >= temp_zero_threshold
        0.0
    else
        raw_prob = 1.0 / (1.0 + exp(temp_steepness * (T - temp_half)))
        if T > temp_half
            w_t = 0.5 * (1.0 + cos(π * (T - temp_half) / (temp_zero_threshold - temp_half)))
            raw_prob * w_t
        else
            raw_prob
        end
    end

    # Bathymetric depth suitability S_z:
    # A Gaussian response around optimal depth (140 m, σ = 90 m) with smooth boundary
    # window tapers approaching zero at the shallow (depth_min = 20 m) and deep (depth_max = 400 m) boundaries.
    s_z = if z <= depth_min || z >= depth_max
        0.0
    else
        g_raw = exp(-0.5 * ((z - depth_opt) / depth_sigma)^2)
        taper_width = min(40.0, depth_opt - depth_min)
        w_shallow = if z < (depth_min + taper_width)
            0.5 * (1.0 - cos(π * (z - depth_min) / taper_width))
        else
            1.0
        end
        w_deep = if z > (depth_max - taper_width)
            0.5 * (1.0 + cos(π * (z - (depth_max - taper_width)) / taper_width))
        else
            1.0
        end
        g_raw * w_shallow * w_deep
    end

    val = s_t * s_z
    if val < 0.0 || val > 1.0
        error("Analytical bounds violated in calculate_habitat_suitability: value $(val) outside [0.0, 1.0]")
    end
    return val
end

