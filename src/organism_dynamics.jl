"""
    organism_dynamics.jl

Biological physiological dynamics, bioenergetics, ecdysis (molting), and mortality for organisms.
"""

using Distributions
using Random

"""
    min_intermolt_days(stage::LifeStage, instar::Int)

Calculate the minimum biological intermolt duration (in days) required before
an individual can undergo subsequent ecdysis.

# Arguments
- `stage::LifeStage`: Current life history stage.
- `instar::Int`: Developmental molt instar index.

# Returns
`Float64`: Minimum required intermolt duration in days.
"""
# Implement SnowCrab Behavior implementation
function perform_development!(::SnowCrabBehavior, org::Organism, model)
    # 2. Molting check and execution (Developmental event for snow crabs)
    day_of_year = mod(model.current_time, 365.0)
    is_molt_season = (model.params.phenology.molt_season_start <= day_of_year <=
                      model.params.phenology.molt_season_end)

    if org.stage === juvenile && is_molt_season
        days_since_molt = model.current_time - org.last_growth_event_time
        min_days = min_intermolt_days(org.stage, org.developmental_step)
        if days_since_molt >= min_days
            p_molt = clamp(
                (days_since_molt - min_days + 1.0) / model.params.organism.molt_probability_scale,
                0.05,
                0.35
            )
            if rand(abmrng(model)) < p_molt
                sex_code = (org.sex === female) ? 1 : 0
                prob_mat = maturity(sex_code, log(org.size);
                                    trait_l50 = org.trait_l50,
                                    trait_slope = org.trait_maturity_slope,
                                    rng = abmrng(model))
                mat_status = (rand(abmrng(model)) < prob_mat) ? 1 : 0

                if mat_status == 1
                    org.stage = adult
                    org.is_mating_ready = true
                end

                org.size = predict_postmolt_cw(
                    org.size;
                    sex = sex_code,
                    mat = mat_status,
                    temp = nothing,
                    condition = 1.0,
                    rng = abmrng(model),
                    trait_intercept = org.trait_growth_intercept,
                    trait_slope = org.trait_growth_slope
                )
                org.developmental_step += 1
                org.last_growth_event_time = model.current_time
                org.mass = calculate_mass(org; stochastic = false, rng = abmrng(model))

                model.total_molts += 1
            end
        end
    end
end

function perform_reproduction!(::SnowCrabBehavior, org::Organism, model)
    # 6. Reproduction (Mating and Hatching/Recruitment)
    day_of_year = mod(model.current_time, 365.0)
    is_mating_season = (model.params.phenology.mating_season_start <= day_of_year <=
                        model.params.phenology.mating_season_end)

    if org.stage === adult && org.sex === female
        # Mating
        if is_mating_season && org.is_mating_ready && org.egg_count == 0
            radius = model.params.phenology.mating_search_radius
            nearby_males = filter(
                m -> m.stage === adult && m.sex === male,
                collect(nearby_agents(org, model, radius))
            )
            if !isempty(nearby_males)
                egg_estimate = fecundity(
                    org.size;
                    fecundity_parameterization="moncton",
                    is_primiparous=(org.sperm_clutches == 0),
                    trait_intercept=org.trait_fecundity_intercept,
                    trait_slope=org.trait_fecundity_slope,
                    rng = abmrng(model)
                )
                org.egg_count = max(round(Int, egg_estimate), 1000)
                org.hatch_day = model.current_time + model.params.phenology.brood_duration_base
                org.is_mating_ready = false

                org.mass = calculate_mass(org;
                    include_eggs = true,
                    rng = abmrng(model)
                )
                model.total_eggs_spawned += org.egg_count
                model.annual_eggs_spawned += org.egg_count
            end
        end

        # Hatching/Recruitment
        if org.egg_count > 0 && model.current_time >= org.hatch_day
            n_benthic = get(abmproperties(model), :n_benthic_organisms, 0)
            k_cap = model.params.benthic_carrying_capacity
            density_feedback = 1.0 + (n_benthic / k_cap)

            expected_recruits = (model.params.egg_survival_base * org.egg_count) / density_feedback
            num_recruits = rand(abmrng(model), Poisson(expected_recruits))

            nx, ny = size(model.temperature_field)
            for _ in 1:num_recruits
                recruit_sex = rand(abmrng(model), (male, female))
                sex_code = (recruit_sex === female) ? 1 : 0
                recruit_size = carapacewidth(4, recruit_sex; type=:random, rng=abmrng(model))
                dx = rand(abmrng(model), -5:5); dy = rand(abmrng(model), -5:5)
                rx = clamp(org.pos[1] + dx, 1, nx); ry = clamp(org.pos[2] + dy, 1, ny)

                recruit_mass = calculate_mass(recruit_size, recruit_sex; mat=0)

                # Add the new agent with sampled traits
                new_org = add_agent!(
                    (rx, ry), model, juvenile, recruit_sex, 4, 180.0, recruit_size,
                    recruit_mass, model.params.initial_energy, model.current_time,
                    0.0, false, 0, 0.0, 0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                )

                # Sample the recruit's unique personality traits
                sample_individual_traits!(new_org, model, abmrng(model))
            end
        end
    end
end

"""
    required_degree_days(stage::LifeStage, instar::Int)

Calculate the cumulative physiological thermal degree-days (°C·days above 0°C)
required to complete physiological pre-ecdysial preparation for a given stage/instar.

# Arguments
- `stage::LifeStage`: Life history stage.
- `instar::Int`: Instar index.

# Returns
`Float64`: Required cumulative degree-days.
"""
function required_degree_days(stage::LifeStage, instar::Int)
    if stage === zoea
        return 20.0
    elseif stage === megalopa
        return 40.0
    elseif instar <= 4
        return 250.0
    elseif instar == 5
        return 350.0
    elseif instar == 6
        return 550.0
    elseif instar == 7
        return 750.0
    else
        return 950.0
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
    elapsed_days = max(0.0, current_time - org.last_molt_time)
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

# Mathematical Formulation
- **Mass-Based Metabolic Scaling (Kleiber's Law)**:
  ```math
  M = M_{base} \\cdot W^{0.75} \\cdot \\exp(0.08 \\cdot T) \\cdot \\psi_T(T)
  ```
  where ``W`` is wet body mass (g) and ``\\psi_T(T) = \\exp(-0.01 \\cdot \\max(0, T - 6.0)^2)``
  moderates metabolic efficiency above thermal optimum.
- **Holling Type II Functional Ingestion Response**:
  ```math
  I_{demand} = I_{max} \\cdot \\frac{F_{local}}{F_{local} + K_{half}} \\cdot W^{0.75}
  ```
  where ``I_{max}`` = `intake_rate_base` (default: `0.15`) is the maximum ingestion coefficient at
  food saturation (units: g energy g^{-0.75} d^{-1}).
- **Realized Ingestion and Net Assimilation**:
  ```math
  I_{actual} = \\min(I_{demand}, F_{local})
  ```
  ```math
  \\frac{dE}{dt} = \\alpha_{assim} \\cdot (1 - \\text{SDA}) \\cdot I_{actual} - M
  ```
  where ``\\alpha_{assim} = 0.75`` and Specific Dynamic Action ``\\text{SDA} = 0.10``.

# Arguments
- `temp::Real`: Local ambient temperature (°C).
- `food::Real`: Local available food resource concentration.
- `size::Real`: Organism morphometry (e.g., carapace width) in mm.
- `params::IBMParameters`: Simulation parameters (default: `IBMParameters()`).
- `mass`: Optional wet mass in grams (defaults to allometric estimate from `size`).

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
    allometric_factor = max(0.01, effective_mass)^0.75

    # Thermal metabolic scaling with warm-temperature efficiency depression
    t_stress = params.temp_thermal_stress
    thermal_exp = exp(0.08 * temp)
    thermal_penalty = (temp > t_stress) ? exp(-0.01 * (temp - t_stress)^2) : 1.0
    metabolism = params.metabolic_rate_base * allometric_factor * thermal_exp * thermal_penalty

    # Holling Type II saturating functional response
    food_avail = max(0.0, Float64(food))
    k_half = params.half_saturation_food
    functional_response = (food_avail > 0.0) ?
        (food_avail / (food_avail + k_half)) : 0.0

    # Demand: I_max * FR * W^0.75, where I_max = intake_rate_base is the
    # maximum (food-saturated) ingestion coefficient (g energy g^-0.75 d^-1)
    desired_intake = params.intake_rate_base * functional_response * allometric_factor
    actual_intake = min(desired_intake, food_avail)

    # Net assimilated energy after Specific Dynamic Action (SDA ~ 10%)
    sda_factor = 0.90
    assimilated_energy = actual_intake * params.assimilation_efficiency * sda_factor
    dE_dt = assimilated_energy - metabolism

    return (dE_dt, actual_intake)
end

"""
    calculate_energy_change(org::Organism, temp::Real, food::Real; params=IBMParameters())

Dispatch bioenergetic calculation directly on an individual `Organism` agent using
its actual individual wet mass (`org.mass`).
"""
function calculate_energy_change(
    org::Organism,
    temp::Real,
    food::Real;
    params::IBMParameters = IBMParameters()
) = calculate_energy_change(
    temp,
    food,
    org.size;
    params = params,
    mass = org.mass
)

"""
    molt_probability(org, current_temp, current_time=0.0; params=IBMParameters())

Calculate the conditional daily probability of initiating ecdysis for `org`.

# Biological Verification Gates
1. **Terminal Molt Gate**: Adults do not molt (`p = 0.0`).
2. **Intermolt Duration Gate**: Elapsed time must exceed `min_intermolt_days(stage, instar)`.
3. **Thermal Degree-Day Gate**: Cumulative physiological degree-days must exceed
   `required_degree_days(stage, instar)`.
4. **Molt Seasonality Window**: Benthic subadults (Instar >= 7) molt exclusively during
   the spring window (`params.molt_season_start` to `params.molt_season_end`).
5. **Energetic Condition Gate**: Must hold minimum reserves (`params.energy_molt_slope * size`).
6. **Thermal Ecdysial Inhibition**: Water temperatures exceeding 6.0°C reduce transition hazard:
   ```math
   \\phi_T(T) = \\max\\left(0.0, 1.0 - \\left(\\frac{\\max(0, T - 5.0)}{3.0}\\right)^2\\right)
   ```

# Arguments
- `org::Organism`: Agent under inspection.
- `current_temp::Real`: Local water temperature (°C).
- `current_time::Real`: Simulation timestamp in days (default: `0.0`).
- `params::IBMParameters`: Simulation configuration parameters.
- `rng`: Optional random number generator for stochastic transition.

# Returns
Daily transition probability in the interval [0.0, 1.0].
"""
function molt_probability(
    org::Organism,
    current_temp::Real,
    current_time::Real = 0.0;
    params::IBMParameters = IBMParameters()
)
    if org.stage === adult
        return 0.0
    end

    min_intermolt = min_intermolt_days(org.stage, org.instar)
    if (current_time - org.last_molt_time) < min_intermolt
        return 0.0
    end

    min_dd = required_degree_days(org.stage, org.instar)
    if org.degree_days < min_dd
        return 0.0
    end

    # Seasonality gate for benthic subadults
    if org.instar >= 7
        day_of_year = mod(current_time, 365.0)
        in_molt_season = (params.molt_season_start <= day_of_year <=
                          params.molt_season_end)
        if !in_molt_season
            return 0.0
        end
    end

    required_energy = max(2.0, params.energy_molt_slope * org.size)
    if org.energy_reserve < required_energy
        return 0.0
    end

    # Thermal ecdysial inhibition for warm waters
    thermal_inhibition = if current_temp > 5.0
        max(0.0, 1.0 - ((current_temp - 5.0) / 3.0)^2)
    else
        1.0
    end

    if thermal_inhibition <= 0.0
        return 0.0
    end

    # Hazard rate increases with degree-day accumulation above requirement
    dd_ratio = org.degree_days / min_dd
    lambda_molt = 0.15 * min(dd_ratio, 3.0) * thermal_inhibition
    p_molt = 1.0 - exp(-lambda_molt * params.dt)

    return clamp(p_molt, 0.0, 1.0)
end

"""
    check_molt_conditions(org, current_temp, current_time=0.0; params=IBMParameters(), rng=nothing)

Determine whether an individual organism satisfies the physiological, developmental,
and seasonal conditions required to initiate ecdysis (molting).

If `rng` is provided (e.g. `rng = abmrng(model)`), ecdysis initiation is evaluated
probabilistically via Bernoulli trial on `molt_probability(org, ...)`.
If `rng === nothing`, returns whether `molt_probability(org, ...) > 0.0`.

# Arguments
- `org::Organism`: Agent under inspection.
- `current_temp::Real`: Local water temperature (°C).
- `current_time::Real`: Simulation timestamp in days (default: `0.0`).
- `params::IBMParameters`: Simulation configuration parameters.
- `rng`: Optional random number generator for stochastic transition.

# Returns
`true` if the organism initiates ecdysis, `false` otherwise.
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

""",
    perform_molt!(org, current_time, temp=nothing; params=IBMParameters(), rng=Random.default_rng())

Execute ecdysis on `org`: increments `instar`, advances developmental stage,
evaluates terminal morphometric maturity, predicts post-molt carapace width,
updates wet mass, resets degree-days, and deducts energetic molting expenditure.

# Biological Transitions
- **Zoea I -> Zoea II (Instar 1 -> 2)**: Larval increment bounded by empirical bounds (~1.45 mm).
- **Zoea II -> Megalopa (Instar 2 -> 3)**: Post-larval transition (~2.50 mm).
- **Megalopa -> Instar 4 Juvenile (Instar 3 -> 4)**: Settlement to benthos (~10.7 mm).
- **Benthic Juvenile (Instar 4+)**:
  - Evaluates `maturity(sex, log(CW); temp=temp, rng=rng)`.
  - Calculates energetic condition factor from reserves to modulate Hiatt increment.
  - Predicts new carapace width via `predict_postmolt_cw`.
  - Recalculates wet mass via `calculate_mass(org; stochastic=true, rng=rng)`.
  - Transitions to `adult` if terminal molt occurs, setting `is_mating_ready = true`.

# Arguments
- `org::Organism`: Individual agent undergoing ecdysis.
- `current_time::Real`: Current simulation timestamp (days).
- `temp`: Optional ambient bottom temperature in °C (for Temperature-Size Rule).
- `params::IBMParameters`: Configuration parameters.
- `rng`: Random number generator.

# Returns
Nothing.
"""
function perform_molt!(
    org::Organism,
    current_time::Real,
    temp::Union{Real, Nothing} = nothing;
    params::IBMParameters = IBMParameters(),
    rng::AbstractRNG = Random.default_rng()
)
    if org.stage === adult
        return nothing
    end

    sex_code = (org.sex === female) ? 1 : 0

    # Developmental progression
    if org.stage === zoea
        if org.instar == 1
            org.instar = 2
            # Zoea II empirical size: 1.45 to 1.85 mm
            org.size = llbm(2.0) + rand(rng, Uniform(0.0, 0.35))
        else
            org.instar = 3
            org.stage = megalopa
            # Megalopa empirical size: 2.50 to 3.10 mm
            org.size = llbm(3.0) + rand(rng, Uniform(0.0, 0.50))
        end
    elseif org.stage === megalopa
        # Settlement to benthic juvenile habitat at instar 4 (~10.7 mm)
        org.instar = 4
        org.stage = juvenile
        org.size = carapacewidth(4, sex_code; type = :random, rng = rng)
    elseif org.stage === juvenile
        # Benthic juvenile molt: check empirical maturity ogive with TSR
        prob_mat = maturity(sex_code, log(org.size); temp = temp, rng = rng)
        mat_status = (rand(rng) < prob_mat) ? 1 : 0

        if mat_status == 1
            org.stage = adult
            org.is_mating_ready = true
        end

        # Modulate Hiatt increment with energetic condition
        req_e = max(2.0, params.energy_molt_slope * org.size)
        cond_ratio = clamp(org.energy_reserve / req_e, 0.50, 2.0)
        condition_factor = clamp(0.85 + 0.30 * cond_ratio, 0.70, 1.30)

        org.size = predict_postmolt_cw(
            org.size;
            sex = sex_code,
            mat = mat_status,
            temp = temp,
            condition = condition_factor,
            rng = rng,
            trait_intercept = org.trait_growth_intercept,
            trait_slope = org.trait_growth_slope
        )
        org.instar += 1

        # Clamp to empirical size and instar limits
        max_cw = (org.sex === female) ? params.max_size_female : params.max_size_male
        org.size = min(org.size, max_cw)
        org.instar = min(org.instar, 15)  # empirical upper limit: males ~12-13, females ~10-11
    end

    # Update body mass following growth
    org.mass = calculate_mass(org; stochastic = true, rng = rng)

    # Molt energetic cost: retain only fraction of stored reserves
    org.energy_reserve *= params.molt_energy_retention
    org.last_molt_time = Float64(current_time)
    org.degree_days = 0.0

    return nothing
end

"""
    calculate_mortality_probability(org, temp, food, current_time=0.0; params=IBMParameters(), local_density=0.0)

Compute aggregate daily instantaneous natural mortality probability for an individual
organism, integrating:
1. Baseline adult physiological mortality (~0.20/year).
2. Early larval stage multipliers (Zoea: 5x, Megalopa: 3x).
3. Size-attenuated predation risk (decaying exponentially with morphometry).
4. Post-molt soft-shell ecdysial vulnerability (elevated during the 30 days post-molt).
5. Progressive starvation mortality scaling with accumulated energy deficit.
6. Stenothermic thermal stress mortality accelerating above 6.0°C.
7. Terminal adult carapace senescence mortality for shell ages >= 3.0 years.
8. Optional localized conspecific cannibalism risk.

# Mathematical Formulation
```math
Z_{daily} = (Z_{base} \\cdot \\mu_{stage}) + Z_{pred} + Z_{soft} + Z_{starve} + Z_{thermal} + Z_{senescence} + Z_{cannibal}
```

# Arguments
- `org::Organism`: Agent evaluated.
- `temp::Real`: Local ambient water temperature (°C).
- `food::Real`: Local food concentration.
- `params::IBMParameters`: Configuration parameters.
- `local_density::Real`: Optional local conspecific density for cannibalism (default: 0.0).

# Returns
`Float64`: Daily mortality probability in the interval [0.0, 1.0].
"""
function calculate_mortality_probability(
    org::Organism,
    temp::Real,
    food::Real,
    current_time::Real = 0.0;
    params::IBMParameters = IBMParameters(),
    local_density::Real = 0.0
)
    # 1. Base physiological mortality
    base_daily = 0.20 / 365.0
    stage_multiplier = if org.stage === zoea
        5.0
    elseif org.stage === megalopa
        3.0
    else
        1.0
    end

    # 2. Size-dependent predation risk (exponential attenuation with size)
    size_excess = max(0.0, org.size - 5.0)
    predation_risk = (0.15 / 365.0) * exp(-0.03 * size_excess)

    # 3. Post-molt soft-shell ecdysial vulnerability
    days_since_molt = max(0.0, current_time - org.last_molt_time)
    soft_shell_risk = (0.12 / 365.0) * exp(-days_since_molt / 15.0)

    # 4. Progressive starvation hazard scaling with energetic deficit
    starvation_risk = if params.use_energetics && org.energy_reserve < 0.0
        deficit = -org.energy_reserve
        crit_pool = max(0.5, org.mass * 0.05)
        (0.20 / 365.0) * (1.0 - exp(-deficit / crit_pool))
    else
        0.0
    end

    # 5. Stenothermic thermal stress mortality above params.temp_thermal_stress
    thermal_stress = if params.use_thermal_stress && temp > params.temp_thermal_stress
        (0.05 / 365.0) * ((temp - params.temp_thermal_stress)^1.5)
    else
        0.0
    end

    # 6. Terminal adult carapace senescence mortality (Shell ages >= 3 years)
    senescence_risk = if params.use_senescence && org.stage === adult
        shell_age_years = days_since_molt / 365.0
        thresh = params.senescence_age_threshold
        if shell_age_years > thresh
            (0.08 / 365.0) * exp(0.80 * (shell_age_years - thresh))
        else
            0.0
        end
    else
        0.0
    end

    # 7. Local conspecific cannibalism on small juveniles (size < 35 mm)
    cannibalism_risk = if params.use_cannibalism && org.size < 35.0 && local_density > 0.0
        (0.05 / 365.0) * min(local_density / 10.0, 2.0)
    else
        0.0
    end

    total_prob = (base_daily * stage_multiplier) +
                 predation_risk + soft_shell_risk + starvation_risk +
                 thermal_stress + senescence_risk + cannibalism_risk

    return clamp(total_prob, 0.0, 1.0)
end
