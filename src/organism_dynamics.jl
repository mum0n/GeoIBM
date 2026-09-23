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
    error("stage_size_lower_bound not implemented for $(typeof(behavior))")
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
    rng::AbstractRNG = Random.default_rng()
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
        u = rand(rng, Beta(2.5, 2.5))
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
    error("growth_increment not implemented for behavior type $(typeof(behavior))")
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
    fecundity(org::Organism; rng=Random.default_rng(), deterministic=false)

Estimate extruded brood fecundity directly for an individual female `Organism` agent.
"""
function fecundity(
    org::Organism;
    rng::AbstractRNG = Random.default_rng(),
    deterministic::Bool = false
)
    if org.sex !== female || !org.is_mature
        return 0
    end
    is_p = (org.sperm_clutches <= 1)
    shell_age = max(1.0, (org.age_days - 180.0 * (org.stage - 3)) / 365.0)
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
    rng::AbstractRNG = Random.default_rng()
)
    somatic_mass = calculate_mass(
        org.behavior,
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
    build_stages(behavior::SpeciesBehavior; kwargs...)

Construct discrete stage hierarchy for stage-transition modeling.
"""
function build_stages(
    behavior::SpeciesBehavior;
    kwargs...
)
    error("build_stages not implemented for $(typeof(behavior))")
end

"""
    min_intermolt_days(stage::Int)
    min_intermolt_days(stage_category::LifeStage, stage::Int = 0)

Calculate the minimum biological intermolt duration (in days) required before
an individual or cohort can undergo subsequent ecdysis.

# Stage Structure
In the current snow crab parameterization, developmental `stage` directly indexes
post-larval benthic stages (stages 1 through 16):
- Stage 0: Pre-benthic aggregate groups, where the remaining life-history groups
  (egg, embryo, zoea I, zoea II, megalopa) are discriminated.
- Stages 1 to 5: Early benthic juvenile cohorts (~3.0 to ~18.0 mm carapace width),
  represented as aggregate cohort groups (`AggregateGroup`). Stage 1 is the first
  benthic recruit (~3.0 mm carapace width).
- Stages 6 to 16: Individuated subadult and adult spatial agents (`Organism`).
  Stage 6 (~21.0 mm carapace width) marks the onset of individual agent representation.

# Schedule (Sainte-Marie et al. 1995)
- Stage 0 (discriminated by `LifeStage`):
  - `egg`: 0.0 days
  - `zoea`: 5.0 days
  - `megalopa`: 10.0 days
- Benthic Stages (stages 1+):
  - Stage 1: 60.0 days
  - Stage 2: 90.0 days
  - Stage 3: 120.0 days
  - Stage 4: 160.0 days
  - Stage 5: 200.0 days
  - Stage 6: 250.0 days
  - Stage 7: 280.0 days
  - Stages 8+: 300.0 days

# Arguments
- `stage::Int`: Developmental stage index (0 to 16).
- `stage_category::LifeStage`: Life-history stage category for discriminating stage 0.

# Returns
`Float64`: Minimum required intermolt duration in days.
"""
function min_intermolt_days(behavior::SpeciesBehavior, stage::Int)
    error("min_intermolt_days not implemented for $(typeof(behavior))")
end


function min_intermolt_days(
    behavior::SpeciesBehavior,
    stage_cat::LifeStage,
    stage::Int = 0
)
    if stage_cat === egg
        return 0.0
    elseif stage_cat === zoea
        return 5.0
    elseif stage_cat === megalopa
        return 10.0
    else
        return min_intermolt_days(behavior, stage)
    end
end


"""
    required_degree_days(stage::Int)
    required_degree_days(stage_category::LifeStage, stage::Int = 0)

Calculate cumulative physiological thermal degree-days (°C·days above 0°C)
required to complete physiological pre-ecdysial preparation for a given stage.

# Arguments
- `stage::Int`: Developmental stage index.
- `stage_category::LifeStage`: Life-history stage category for discriminating stage 0.

# Returns
`Float64`: Required cumulative degree-days.
"""
function required_degree_days(behavior::SpeciesBehavior, stage::Int)
    error("required_degree_days not implemented for $(typeof(behavior))")
end


function required_degree_days(
    behavior::SpeciesBehavior,
    stage_cat::LifeStage,
    stage::Int = 0
)
    if stage_cat === egg
        return 0.0
    elseif stage_cat === zoea
        return 20.0
    elseif stage_cat === megalopa
        return 40.0
    else
        return required_degree_days(behavior, stage)
    end
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
function shell_condition(org::Organism, current_time::Real)
    elapsed_days = max(0.0, current_time - org.last_growth_event_time)
    if elapsed_days < 30.0
        return 1
    elseif elapsed_days < 365.0
        return 2
    elseif elapsed_days < 730.0
        return 3
    elseif elapsed_days < 1277.5
        return 4
    else
        return 5
    end
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

    # Kleiber metabolic consumption (allometric exponent 0.75)
    mass_scaling = effective_mass^0.75
    metabolic_temp_factor = exp(0.08 * temp)
    t_opt = 3.0
    t_sigma = 2.5
    thermal_modifier = exp(-((temp - t_opt)^2) / (2.0 * t_sigma^2))
    maintenance_cost = params.organism.metabolic_rate_base * mass_scaling *
                       metabolic_temp_factor * thermal_modifier

    # Holling Type II functional feeding response
    half_sat = params.environment.half_saturation_food
    intake_demand = params.organism.intake_rate_base * (food / (food + half_sat)) * mass_scaling
    actual_intake = min(intake_demand, max(0.0, food))

    # Net energy assimilation (Specific Dynamic Action SDA = 0.15)
    assim_eff = params.environment.assimilation_efficiency
    net_assimilation = (actual_intake * assim_eff * 0.85) - maintenance_cost

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

    # Seasonality gate for benthic subadults (stage >= 7)
    if org.stage >= 7
        day_of_year = mod(current_time, 365.0)
        in_molt_season = (params.phenology.molt_season_start <= day_of_year <=
                          params.phenology.molt_season_end)
        if !in_molt_season
            return 0.0
        end
    end

    if params.toggles.use_energetics
        required_energy = max(2.0, params.organism.energy_molt_slope * org.size)
        if org.energy_reserve < required_energy
            return 0.0
        end
    end

    # Thermal ecdysial inhibition for warm waters
    thermal_inhibition = if params.toggles.use_thermal_stress && current_temp > 5.0
        max(0.0, 1.0 - ((current_temp - 5.0) / 3.0)^2)
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

    lambda_molt = 0.15 * min(progress_ratio, 3.0) * thermal_inhibition
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
        req_e = max(2.0, params.organism.energy_molt_slope * org.size)
        cond_ratio = org.energy_reserve / req_e
        0.85 + 0.15 * tanh(cond_ratio - 1.0)
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
    calculate_mortality_probability(org, temp, local_density, current_time; params=IBMParameters())

Calculate comprehensive daily mortality probability across baseline predation,
soft-shell post-molt vulnerability, starvation, thermal stress, senescence, and cannibalism.
"""
function calculate_mortality_probability(
    org::Organism,
    temp::Real,
    local_density::Real,
    current_time::Real;
    params::IBMParameters = IBMParameters()
)
    dt = params.simulation.dt

    # 1. Baseline natural mortality modulated by individual mortality trait quantile
    q_mort = org.trait_mortality_quantile
    if q_mort <= 0.0 || q_mort >= 1.0
        error("trait_mortality_quantile must be strictly in (0, 1), received: $(q_mort)")
    end
    individual_mort_factor = exp(0.25 * quantile(Normal(0.0, 1.0), q_mort))
    base_daily = ((0.20 / 365.0) * individual_mort_factor) * dt
    stage_multiplier = (org.stage < 6) ? 2.5 : 1.0

    # 2. Predation risk attenuated by size
    size_excess = max(0.0, org.size - 5.0)
    predation_risk = ((0.15 / 365.0) * exp(-0.03 * size_excess)) * dt

    # 3. Soft-shell vulnerability post-ecdysis
    days_since_molt = max(0.0, current_time - org.last_growth_event_time)
    soft_shell_risk = ((0.12 / 365.0) * exp(-days_since_molt / 15.0)) * dt

    # 4. Starvation mortality
    starvation_risk = if params.toggles.use_energetics && org.energy_reserve < 0.0
        deficit = -org.energy_reserve
        crit_pool = max(0.5, org.mass * 0.05)
        ((0.20 / 365.0) * (1.0 - exp(-deficit / crit_pool))) * dt
    else
        0.0
    end

    # 5. Thermal stress mortality
    thermal_stress = if params.toggles.use_thermal_stress &&
                        temp > params.mortality.temp_thermal_stress
        ((0.05 / 365.0) * ((temp - params.mortality.temp_thermal_stress)^1.5)) * dt
    else
        0.0
    end

    # 6. Senescence mortality for terminal molt adults
    senescence_risk = if params.toggles.use_senescence && org.is_mature
        shell_age_years = days_since_molt / 365.0
        thresh = params.mortality.senescence_age_threshold
        if shell_age_years > thresh
            ((0.08 / 365.0) * exp(0.80 * (shell_age_years - thresh))) * dt
        else
            0.0
        end
    else
        0.0
    end

    # 7. Cannibalism risk on small juveniles
    cannibalism_risk = if params.toggles.use_cannibalism && org.size < 35.0 && local_density > 0.0
        ((0.05 / 365.0) * min(local_density / 10.0, 2.0)) * dt
    else
        0.0
    end

    total_prob = (base_daily * stage_multiplier) +
                 predation_risk + soft_shell_risk + starvation_risk +
                 thermal_stress + senescence_risk + cannibalism_risk

    if total_prob < 0.0 || total_prob > 1.0
        error("Total mortality probability $(total_prob) outside [0.0, 1.0]. Check mortality assumptions.")
    end

    return total_prob
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
    groups = model.aggregate_populations.groups
    isempty(groups) && return nothing

    nx, ny = size(model.temperature_field)
    k_cap = model.params.environment.benthic_carrying_capacity
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

        n_benthic = get(abmproperties(model), :n_benthic_organisms, 0)
        density_feedback = 1.0 + (n_benthic / max(1.0, k_cap))
        daily_hazard = (base_m * density_feedback) * dt

        if daily_hazard < 0.0 || daily_hazard > 1.0
            error("Aggregate daily mortality hazard $(daily_hazard) out of [0, 1] bounds.")
        end

        surv_prob = exp(-daily_hazard)
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
            (grp.degree_days, req_dd, max(1.0, 0.15 * req_dd))
        else
            min_days = min_intermolt_days(behavior, grp.stage)
            (days_in_stage, min_days, max(1.0, 0.20 * min_days))
        end

        diff = (d_metric - d_req) / d_scale
        p_trans = if diff < -10.0
            0.0
        elseif diff > 10.0
            1.0 - exp(-0.35 * dt)
        else
            hazard = 0.35 / (1.0 + exp(-diff))
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
                if next_stage == 1 || next_stage == 4
                    model.total_recruits_settled += n_trans
                end
            elseif next_stage == 6
                # Individuation into spatial Organism agents at stage 6
                for _ in 1:n_trans
                    rec_sex = (rand(abmrng(model)) < 0.5) ? male : female
                    mean_log_sz = log(stage_size(behavior, 6, rec_sex; type = :random, rng = abmrng(model)))
                    rec_size = exp(rand(abmrng(model), Normal(mean_log_sz, 0.04)))
                    rec_mass = calculate_mass(behavior, rec_size, rec_sex; mat = 0, stochastic = true, rng = abmrng(model))

                    dx = rand(abmrng(model), -2:2)
                    dy = rand(abmrng(model), -2:2)
                    rx = min(max(grp.pos[1] + dx, 1), nx)
                    ry = min(max(grp.pos[2] + dy, 1), ny)

                    new_org = add_agent!(
                        (rx, ry),
                        model,
                        6,
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
