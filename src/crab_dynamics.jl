"""
Crab physiological dynamics, bioenergetics, molting, and mortality.

This module implements:
- Mass-based metabolic allometry (Kleiber's Law) and Holling Type II functional feeding.
- Developmental degree-day requirements and physiological molt scheduling.
- Ecdysis execution with Temperature-Size Rule maturity and condition scaling.
- Multi-component mortality: predation, soft-shell post-molt vulnerability,
  progressive starvation, stenothermic thermal stress, and adult senescence.
- Standard fisheries carapace shell condition classification (1 through 5).
"""

using Distributions
using Random

"""
    min_intermolt_days(stage::CrabStage, instar::Int)

Calculate the minimum biological intermolt duration (in days) required before
an individual crab can undergo subsequent ecdysis.

# Biological Schedule
- Zoea stages (Instars 1–2): 5.0 days.
- Megalopa stage (Instar 3): 10.0 days.
- Benthic Juveniles (Instar 4+): Increases allometrically with instar:
  Instar 4: 60 d, Instar 5: 90 d, Instar 6: 150 d, Instar 7: 200 d, Instars 8+: 300 d.

# Arguments
- `stage::CrabStage`: Current life history stage.
- `instar::Int`: Developmental molt instar index.

# Returns
`Float64`: Minimum required intermolt duration in days.
"""
function min_intermolt_days(stage::CrabStage, instar::Int)
    if stage === zoea
        return 5.0
    elseif stage === megalopa
        return 10.0
    elseif instar <= 4
        return 60.0
    elseif instar == 5
        return 90.0
    elseif instar == 6
        return 150.0
    elseif instar == 7
        return 200.0
    else
        return 300.0
    end
end

"""
    required_degree_days(stage::CrabStage, instar::Int)

Calculate the cumulative physiological thermal degree-days (°C·days above 0°C)
required to complete physiological pre-ecdysial preparation for a given stage/instar.

# Formulation
- Zoea (Instars 1–2): 20.0 °C·days.
- Megalopa (Instar 3): 40.0 °C·days.
- Benthic Juveniles (Instar 4+):
  ```math
  \\text{DD}_{req} = 150.0 \\cdot (instar - 2.0)^{1.15}
  ```
  yielding ~250 for Instar 4, ~350 for Instar 5, ~550 for Instar 6,
  ~750 for Instar 7, and ~950 for Instar 8+.

# Arguments
- `stage::CrabStage`: Life history stage.
- `instar::Int`: Instar index.

# Returns
`Float64`: Required cumulative degree-days.
"""
function required_degree_days(stage::CrabStage, instar::Int)
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
    shell_condition(crab::SnowCrab, current_time::Real)

Determine the standard fisheries carapace shell condition index (1 through 5)
for an individual snow crab based on time elapsed since the most recent ecdysis.

# Classification Criteria (DFO / Sainte-Marie et al. 1995)
- **Condition 1 (Soft / New Soft)**: Post-ecdysial carapace hardening (< 30 days).
  Soft or paper shell, uncalcified chelae; highly vulnerable.
- **Condition 2 (New Hard / Clean)**: Fully hardened clean carapace (30 days to 1 year).
  White ventral surface, no epibiont fouling, sharp dactyli.
- **Condition 3 (Intermediate Hard)**: Hard carapace with light wear (1 to 2 years).
  Yellowish-brown ventrum, light scratches, minor fouling.
- **Condition 4 (Old Shell)**: Weathered carapace (2 to 3.5 years).
  Dark brown/yellow ventrum, barnacles, leeches, worn teeth/dactyli.
- **Condition 5 (Very Old / Decaying)**: Senescent carapace (>= 3.5 years).
  Black/dark brown ventral coloration, heavy epibionts, chitinolytic erosion.

# Arguments
- `crab::SnowCrab`: Individual snow crab agent.
- `current_time::Real`: Current simulation timestamp (days).

# Returns
`Int`: Shell condition category (1 to 5).
"""
function shell_condition(crab::SnowCrab, current_time::Real)
    elapsed_days = max(0.0, current_time - crab.last_molt_time)
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
    calculate_energy_change(temp, food, size; params=IBMParameters(), weight=nothing)

Compute daily metabolic energy expenditure and realized food intake for an individual
crab given local environmental conditions and body size or wet mass.

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
- `size::Real`: Crab carapace width in millimeters.
- `params::IBMParameters`: Simulation parameters (default: `IBMParameters()`).
- `weight`: Optional wet mass in grams (defaults to allometric estimate from `size`).

# Returns
Tuple `(dE_dt, actual_intake)` in energy units / day.
"""
function calculate_energy_change(
    temp::Real,
    food::Real,
    size::Real;
    params::IBMParameters = IBMParameters(),
    weight::Union{Real, Nothing} = nothing
)
    if size <= 0.0 && (weight === nothing || weight <= 0.0)
        return (0.0, 0.0)
    end

    effective_mass = if weight !== nothing && weight > 0.0
        Float64(weight)
    else
        crab_weight(size, :male; mat = 0)
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
    calculate_energy_change(crab::SnowCrab, temp::Real, food::Real; params=IBMParameters())

Dispatch bioenergetic calculation directly on an individual `SnowCrab` agent using
its actual individual wet mass (`crab.weight_g`).
"""
calculate_energy_change(
    crab::SnowCrab,
    temp::Real,
    food::Real;
    params::IBMParameters = IBMParameters()
) = calculate_energy_change(
    temp,
    food,
    crab.size_mm;
    params = params,
    weight = crab.weight_g
)

"""
    molt_probability(crab, current_temp, current_time=0.0; params=IBMParameters())

Calculate the conditional daily probability of initiating ecdysis for `crab`.

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

# Returns
Daily transition probability in the interval [0.0, 1.0].
"""
function molt_probability(
    crab::SnowCrab,
    current_temp::Real,
    current_time::Real = 0.0;
    params::IBMParameters = IBMParameters()
)
    if crab.stage === adult
        return 0.0
    end

    min_intermolt = min_intermolt_days(crab.stage, crab.instar)
    if (current_time - crab.last_molt_time) < min_intermolt
        return 0.0
    end

    min_dd = required_degree_days(crab.stage, crab.instar)
    if crab.degree_days < min_dd
        return 0.0
    end

    # Seasonality gate for benthic subadults
    if crab.instar >= 7
        day_of_year = mod(current_time, 365.0)
        in_molt_season = (params.molt_season_start <= day_of_year <=
                          params.molt_season_end)
        if !in_molt_season
            return 0.0
        end
    end

    required_energy = max(2.0, params.energy_molt_slope * crab.size_mm)
    if crab.energy_reserve < required_energy
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
    dd_ratio = crab.degree_days / min_dd
    lambda_molt = 0.15 * min(dd_ratio, 3.0) * thermal_inhibition
    p_molt = 1.0 - exp(-lambda_molt * params.dt)

    return clamp(p_molt, 0.0, 1.0)
end

"""
    check_molt_conditions(crab, current_temp, current_time=0.0; params=IBMParameters(), rng=nothing)

Determine whether an individual crab satisfies the physiological, developmental,
and seasonal conditions required to initiate ecdysis (molting).

If `rng` is provided (e.g. `rng = abmrng(model)`), ecdysis initiation is evaluated
probabilistically via Bernoulli trial on `molt_probability(crab, ...)`.
If `rng === nothing`, returns whether `molt_probability(crab, ...) > 0.0`.

# Arguments
- `crab::SnowCrab`: Agent under inspection.
- `current_temp::Real`: Local water temperature (°C).
- `current_time::Real`: Simulation timestamp in days (default: `0.0`).
- `params::IBMParameters`: Simulation configuration parameters.
- `rng`: Optional random number generator for stochastic transition.

# Returns
`true` if the crab initiates ecdysis, `false` otherwise.
"""
function check_molt_conditions(
    crab::SnowCrab,
    current_temp::Real,
    current_time::Real = 0.0;
    params::IBMParameters = IBMParameters(),
    rng::Union{AbstractRNG, Nothing} = nothing
)
    p_molt = molt_probability(crab, current_temp, current_time; params = params)
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
    perform_molt!(crab, current_time, temp=nothing; params=IBMParameters(), rng=Random.default_rng())

Execute ecdysis on `crab`: increments `instar`, advances developmental stage,
evaluates terminal morphometric maturity, predicts post-molt carapace width,
updates wet mass, resets degree-days, and deducts energetic molting expenditure.

# Biological Transitions
- **Zoea I -> Zoea II (Instar 1 -> 2)**: Larval increment bounded by empirical bounds (`~1.45` mm).
- **Zoea II -> Megalopa (Instar 2 -> 3)**: Post-larval transition (`~2.50` mm).
- **Megalopa -> Instar 4 Juvenile (Instar 3 -> 4)**: Settlement to benthos (`~10.7` mm).
- **Benthic Juvenile (Instar 4+)**:
  - Evaluates `maturity(sex, log(CW); temp=temp, rng=rng)`.
  - Calculates energetic condition factor from reserves to modulate Hiatt increment.
  - Predicts new carapace width via `predict_postmolt_cw`.
  - Recalculates wet mass via `crab_weight(crab; stochastic=true, rng=rng)`.
  - Transitions to `adult` if terminal molt occurs, setting `is_mating_ready = true`.

# Arguments
- `crab::SnowCrab`: Individual agent undergoing ecdysis.
- `current_time::Real`: Current simulation timestamp (days).
- `temp`: Optional ambient bottom temperature in °C (for Temperature-Size Rule).
- `params::IBMParameters`: Configuration parameters.
- `rng`: Random number generator.
"""
function perform_molt!(
    crab::SnowCrab,
    current_time::Real,
    temp::Union{Real, Nothing} = nothing;
    params::IBMParameters = IBMParameters(),
    rng::AbstractRNG = Random.default_rng()
)
    if crab.stage === adult
        return nothing
    end

    sex_code = (crab.sex === female) ? 1 : 0

    if crab.stage === zoea
        if crab.instar == 1
            crab.instar = 2
            # Zoea II empirical size: 1.45 to 1.85 mm
            crab.size_mm = llbm(2.0) + rand(rng, Uniform(0.0, 0.35))
        else
            crab.instar = 3
            crab.stage = megalopa
            # Megalopa empirical size: 2.50 to 3.10 mm
            crab.size_mm = llbm(3.0) + rand(rng, Uniform(0.0, 0.50))
        end
    elseif crab.stage === megalopa
        # Settlement to benthic juvenile habitat at instar 4 (~10.7 mm)
        crab.instar = 4
        crab.stage = juvenile
        crab.size_mm = carapacewidth(4, sex_code; type = :random, rng = rng)
    elseif crab.stage === juvenile
        # Benthic juvenile molt: check empirical maturity ogive with TSR
        prob_mat = maturity(sex_code, log(crab.size_mm); temp = temp, rng = rng)
        mat_status = (rand(rng) < prob_mat) ? 1 : 0

        if mat_status == 1
            crab.stage = adult
            crab.is_mating_ready = true
        end

        # Modulate Hiatt increment with energetic condition
        req_e = max(2.0, params.energy_molt_slope * crab.size_mm)
        cond_ratio = clamp(crab.energy_reserve / req_e, 0.50, 2.0)
        condition_factor = clamp(0.85 + 0.30 * cond_ratio, 0.70, 1.30)

        crab.size_mm = predict_postmolt_cw(
            crab.size_mm;
            sex = sex_code,
            mat = mat_status,
            temp = temp,
            condition = condition_factor,
            rng = rng,
            trait_intercept = crab.trait_growth_intercept,
            trait_slope = crab.trait_growth_slope
        )
        crab.instar += 1

        # Clamp to empirical size and instar limits
        max_cw = (crab.sex === female) ? params.max_size_female : params.max_size_male
        crab.size_mm = min(crab.size_mm, max_cw)
        crab.instar = min(crab.instar, 15)  # empirical upper limit: males ~12-13, females ~10-11
    end

    # Update body mass following growth
    crab.weight_g = crab_weight(crab; stochastic = true, rng = rng)

    # Molt energetic cost: retain only fraction of stored reserves
    crab.energy_reserve *= params.molt_energy_retention
    crab.last_molt_time = Float64(current_time)
    crab.degree_days = 0.0

    return nothing
end

"""
    calculate_mortality_probability(crab, temp, food, current_time=0.0; params=IBMParameters(), local_density=0.0)

Compute aggregate daily instantaneous natural mortality probability for an individual
crab, integrating:
1. Baseline adult physiological mortality (~0.20/year).
2. Early larval stage multipliers (Zoea: 5x, Megalopa: 3x).
3. Size-attenuated predation risk (decaying exponentially with carapace width).
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
- `crab::SnowCrab`: Agent evaluated.
- `temp::Real`: Local ambient water temperature (°C).
- `food::Real`: Local food concentration.
- `params::IBMParameters`: Configuration parameters.
- `local_density::Real`: Optional local conspecific density for cannibalism (default: 0.0).

# Returns
`Float64`: Daily mortality probability in the interval [0.0, 1.0].
"""
function calculate_mortality_probability(
    crab::SnowCrab,
    temp::Real,
    food::Real,
    current_time::Real = 0.0;
    params::IBMParameters = IBMParameters(),
    local_density::Real = 0.0
)
    # 1. Base physiological mortality
    base_daily = 0.20 / 365.0
    stage_multiplier = if crab.stage === zoea
        5.0
    elseif crab.stage === megalopa
        3.0
    else
        1.0
    end

    # 2. Size-dependent predation risk (exponential attenuation with CW)
    cw_excess = max(0.0, crab.size_mm - 5.0)
    predation_risk = (0.15 / 365.0) * exp(-0.03 * cw_excess)

    # 3. Post-molt soft-shell ecdysial vulnerability
    days_since_molt = max(0.0, current_time - crab.last_molt_time)
    soft_shell_risk = (0.12 / 365.0) * exp(-days_since_molt / 15.0)

    # 4. Progressive starvation hazard scaling with energetic deficit
    starvation_risk = if params.use_energetics && crab.energy_reserve < 0.0
        deficit = -crab.energy_reserve
        crit_pool = max(0.5, crab.weight_g * 0.05)
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
    senescence_risk = if params.use_senescence && crab.stage === adult
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

    # 7. Local conspecific cannibalism on small juveniles (CW < 35 mm)
    cannibalism_risk = if params.use_cannibalism && crab.size_mm < 35.0 && local_density > 0.0
        (0.05 / 365.0) * min(local_density / 10.0, 2.0)
    else
        0.0
    end

    total_prob = (base_daily * stage_multiplier) +
                 predation_risk + soft_shell_risk + starvation_risk +
                 thermal_stress + senescence_risk + cannibalism_risk

    return clamp(total_prob, 0.0, 1.0)
end
