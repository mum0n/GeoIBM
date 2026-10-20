"""
Core data types, agent structures, and parameter hierarchies for the organism IBM.
"""

using Agents

@enum LifeStage zoea megalopa juvenile adult egg
@enum Sex male female undifferentiated

"""
    AggregateGroup

Represents an aggregate cohort group for pre-individuated life stages (egg
through stage 5, spanning approximately 4 years). Sex ratio is assumed 1:1
(undifferentiated) in accordance with empirical early developmental observations.
"""
Base.@kwdef mutable struct AggregateGroup
    id::Int = 0
    stage::Int = 0              # 0: aggregate egg / larval clutch; 1-5: benthic juvenile stages
    abundance::Float64 = 0.0     # Total number of individuals in the aggregate group
    size::Float64 = 0.0          # Mean organism size (carapace width / dimension in mm)
    mass::Float64 = 0.0          # Mean organism wet mass (g)
    age_days::Float64 = 0.0      # Chronological age in days
    degree_days::Float64 = 0.0   # Accumulated thermal degree-days
    last_growth_time::Float64 = 0.0 # Timestamp of most recent developmental transition
    pos::Tuple{Int, Int} = (1, 1)# Spatial reference coordinates on grid
end

"""
    AggregatePopulations

Container managing all aggregate cohorts across pre-individuation stages.
"""
Base.@kwdef mutable struct AggregatePopulations
    groups::Vector{AggregateGroup} = AggregateGroup[]
    next_group_id::Int = 1
end

abstract type SpeciesBehavior end

abstract type SpeciesParams end

# Generic species interface function declarations
function default_species_behavior end
function stage_size_lower_bound end
function get_stage_size end
function growth_increment end
function predict_postgrowth_size end
function growth_schedule end
function maturity end
function fecundity end
function selectivity end
function calculate_mass end
function build_stages end
function min_intermolt_days end
function required_degree_days end
function perform_development! end
function perform_reproduction! end


"""
    Organism(GridAgent{2})

Individual agent representation for post-individuation life stages (stage 6 onwards,
up to stage 16 in males and stage 14 in females).
"""
@agent struct Organism(GridAgent{2})
    stage::Int                      # Developmental stage index (6 through 16)
    sex::Sex
    behavior::SpeciesBehavior       # Modular behavior implementation
    age_days::Float64
    size::Float64                   # Organism size (carapace width in mm)
    mass::Float64                   # Body wet mass in grams
    energy_reserve::Float64
    last_growth_event_time::Float64 # Timestamp of last ecdysis (molt)
    degree_days::Float64            # Cumulative physiological thermal units
    is_mating_ready::Bool
    egg_count::Int                  # Number of eggs in current clutch
    hatch_day::Float64
    sperm_clutches::Int
    is_mature::Bool

    # Individual empirical biological trait quantiles in (0, 1)
    trait_growth_quantile::Float64
    trait_maturity_quantile::Float64
    trait_fecundity_quantile::Float64
    trait_mortality_quantile::Float64
end

# Primary outer constructors for Organism
function Organism(
    id::Int,
    pos::Tuple{Int, Int},
    stage::Integer,
    sex::Sex,
    behavior::SpeciesBehavior,
    age_days::Real,
    size::Real,
    mass::Real,
    energy_reserve::Real,
    last_growth_event_time::Real,
    degree_days::Real,
    is_mating_ready::Bool,
    egg_count::Integer,
    hatch_day::Real,
    sperm_clutches::Integer,
    is_mature::Bool = false;
    trait_growth_quantile::Real = 0.5,
    trait_maturity_quantile::Real = 0.5,
    trait_fecundity_quantile::Real = 0.5,
    trait_mortality_quantile::Real = 0.5
)
    return Organism(
        id, pos, Int(stage), sex, behavior,
        Float64(age_days), Float64(size), Float64(mass), Float64(energy_reserve),
        Float64(last_growth_event_time), Float64(degree_days), is_mating_ready,
        Int(egg_count), Float64(hatch_day), Int(sperm_clutches), is_mature,
        Float64(trait_growth_quantile), Float64(trait_maturity_quantile),
        Float64(trait_fecundity_quantile), Float64(trait_mortality_quantile)
    )
end

function Organism(
    id::Int,
    pos::Tuple{Int, Int},
    stage::Integer,
    sex::Sex,
    age_days::Real,
    size::Real,
    mass::Real,
    energy_reserve::Real,
    last_growth_event_time::Real,
    degree_days::Real,
    is_mating_ready::Bool,
    egg_count::Integer,
    hatch_day::Real,
    sperm_clutches::Integer,
    is_mature::Bool = false;
    behavior::SpeciesBehavior = default_species_behavior(),
    trait_growth_quantile::Real = 0.5,
    trait_maturity_quantile::Real = 0.5,
    trait_fecundity_quantile::Real = 0.5,
    trait_mortality_quantile::Real = 0.5
)
    return Organism(
        id, pos, Int(stage), sex, behavior,
        age_days, size, mass, energy_reserve,
        last_growth_event_time, degree_days, is_mating_ready,
        egg_count, hatch_day, sperm_clutches, is_mature;
        trait_growth_quantile = trait_growth_quantile,
        trait_maturity_quantile = trait_maturity_quantile,
        trait_fecundity_quantile = trait_fecundity_quantile,
        trait_mortality_quantile = trait_mortality_quantile
    )
end

# Forwarding method for legacy call signatures with stage_enum
function Organism(
    id::Int,
    pos::Tuple{Int, Int},
    stage_enum::LifeStage,
    sex::Sex,
    stage_int::Integer,
    age_days::Real,
    size::Real,
    mass::Real,
    energy_reserve::Real,
    last_growth_event_time::Real,
    degree_days::Real,
    is_mating_ready::Bool,
    egg_count::Integer,
    hatch_day::Real,
    sperm_clutches::Integer,
    is_mature::Bool = (stage_enum === adult);
    kwargs...
)
    return Organism(
        id, pos, Int(stage_int), sex, age_days, size, mass, energy_reserve,
        last_growth_event_time, degree_days, is_mating_ready, egg_count,
        hatch_day, sperm_clutches, is_mature;
        kwargs...
    )
end


# Interface for species-specific biological processes
# Concrete implementations in species.jl

"""
    is_mature_male(org)

Male that has passed the maturity process. This is the reproductive criterion:
only mature females extrude broods, so only mature biomass belongs in spawning
stock.
"""
is_mature_male(org::Organism) = org.is_mature && org.sex === male

"""
    is_mature_female(org)

Female that has passed the maturity process, and therefore the only class that
contributes eggs. Used for spawning stock biomass, so a large individual that
never matured is excluded rather than inflating the stock.
"""
is_mature_female(org::Organism) = org.is_mature && org.sex === female

"""
    is_ovigerous(org)

Mature female currently carrying a brood.
"""
is_ovigerous(org::Organism) = is_mature_female(org) && org.egg_count > 0

"""
    is_adult(org)

Individual that has passed the maturity process. This is a reproductive
criterion, not a size/allometry one: only mature individuals are considered
adults for reproductive purposes. Size-based thresholds are handled separately
via `adult_size_threshold` for allometry purposes.
"""
is_adult(org::Organism) = org.is_mature

is_adult_male(org::Organism) = is_adult(org) && org.sex === male

is_adult_female(org::Organism) = is_adult(org) && org.sex === female

"""
    is_legal_male(org, legal_size)

Male at or above the regulated legal size. Fisheries regulations are size-based,
so this does not require maturity, but it does require the individual to be past
the juvenile phase. The adult size threshold comes from
`OrganismParams.adult_size_threshold`, and the legal size comes from
`FisheryParams.legal_size_male`. Both thresholds are mandatory: they come from
`FisheryParams.legal_size_male` and `OrganismParams.adult_size_threshold`, and a
hardcoded default previously caused the demographic cache and the GUI colour
ramp to disagree with the configured value.
"""
is_legal_male(org::Organism, legal_size::Real, adult_size::Real) =
    org.sex === male && org.size >= legal_size && org.size >= adult_size

is_legal(org::Organism, legal_size::Real, adult_size::Real) =
    is_legal_male(org, legal_size, adult_size)

mass_if_mature_female(org::Organism) = is_mature_female(org) ? org.mass : 0.0
mass_if_mature_male(org::Organism) = is_mature_male(org) ? org.mass : 0.0
mass_if_legal_male(org::Organism, legal_size::Real, adult_size::Real) =
    is_legal_male(org, legal_size, adult_size) ? org.mass : 0.0

"""
    NamedPredicate(f, name, thresholds...)

Single-argument callable that applies a threshold-requiring predicate to one
organism. Built by [`named_predicate`](@ref).
"""
struct NamedPredicate{F,T} <: Function
    f::F
    name::Symbol
    thresholds::T
end

function (p::NamedPredicate)(org::Organism)
    return p.f(org, p.thresholds...)
end

Base.show(io::IO, p::NamedPredicate) = print(io, p.name)
Base.propertynames(::NamedPredicate) = ()
Base.propertynames(::NamedPredicate, ::Bool) = ()

"""
    NamedAggregate(name, f)

One-argument callable that aggregates like `f` but presents a fixed `name` to
`Agents.dataname`.

`dataname` derives a column name from the *function object*, so substituting a
total aggregator for a partial one — `safe_sum` for `sum`, say — silently renames
the column from `sum_mass` to `safe_sum_mass` and every downstream reader of
`sum_mass` starts resolving to `nothing`. This wrapper keeps the schema stable
while still being defined over an empty input.
"""
struct NamedAggregate{F} <: Function
    name::Symbol
    f::F
end

(a::NamedAggregate)(x) = a.f(x)

Base.show(io::IO, a::NamedAggregate) = print(io, a.name)
Base.propertynames(::NamedAggregate) = ()
Base.propertynames(::NamedAggregate, ::Bool) = ()

"""
    named_predicate(f, name::Symbol, thresholds...)

Bind `thresholds` to a predicate that requires them, producing a one-argument
function suitable for an `adata` aggregation list.

`Agents.dataname` derives the DataFrame column name from the function object
itself, and appends the values of any captured properties. An anonymous closure
would therefore yield a column such as `count_#7`, and a functor with readable
fields would embed the configured thresholds in the name. Either way the output
schema would depend on how the call site was written or on the configuration
values, so downstream column lookups would silently read `nothing` and report
zeros. This wrapper prints as `name` and reports no properties, so the column
name is exactly the same regardless of which thresholds were bound.
"""
named_predicate(f, name::Symbol, thresholds...) = NamedPredicate(f, name, thresholds)

"""
    StageDefinition

Defines discrete life-history stages for demographic matrix models.
"""
struct StageDefinition
    index::Int
    name::String
    stage::Int
    sex::Sex
    size_range::Tuple{Float64, Float64}
    size_mean::Float64
    mean_mass::Float64
    is_legal_male::Bool
    is_reproductive::Bool
end

function Base.getproperty(sd::StageDefinition, s::Symbol)
    if s === :cw_mean
        return getfield(sd, :size_mean)
    elseif s === :cw_range
        return getfield(sd, :size_range)
    elseif s === :mean_weight
        return getfield(sd, :mean_mass)
    elseif s === :instar
        return getfield(sd, :stage)
    else
        return getfield(sd, s)
    end
end

"""
    LeslieMatrixFit

Results and demographic diagnostics from fitting a generalized Leslie stage matrix.
"""
struct LeslieMatrixFit
    stages::Vector{StageDefinition}
    A::Matrix{Float64}
    eigenvalues::Vector{ComplexF64}
    dominant_eigenvalue::Float64
    intrinsic_growth_rate::Float64
    stable_stage_distribution::Vector{Float64}
    reproductive_values::Vector{Float64}
    damping_ratio::Float64
    elasticity_matrix::Matrix{Float64}
    observed_stages::Matrix{Float64}
    predicted_stages::Matrix{Float64}
    legal_abundance_obs::Vector{Float64}
    legal_abundance_pred::Vector{Float64}
    legal_biomass_obs_kg::Vector{Float64}
    legal_biomass_pred_kg::Vector{Float64}
    r2::Float64
    rmse::Float64
    aic::Float64
    # Number of free coefficients the fit estimated. Carried so a cross-model AIC
    # comparison can charge the correct penalty: the AIC inside the fit is computed
    # from this count, and a comparison that hardcodes its own, smaller, number would
    # rank the model on a different criterion than the one it reports.
    n_free_parameters::Int
    years::Vector{Int}
end

"""
    MacroModelComparison

Comparative summary across Schaefer, Delay-Difference, and Leslie macro models.
"""
struct MacroModelComparison
    years::Vector{Int}
    biomass_obs_kg::Vector{Float64}
    biomass_schaefer_kg::Vector{Float64}
    biomass_delay_diff_kg::Vector{Float64}
    biomass_leslie_kg::Vector{Float64}
    cv_obs::Float64
    cv_schaefer::Float64
    cv_delay_diff::Float64
    cv_leslie::Float64
    sd_obs::Float64
    sd_schaefer::Float64
    sd_delay_diff::Float64
    sd_leslie::Float64
    var_delta_obs::Float64
    var_delta_schaefer::Float64
    var_delta_delay_diff::Float64
    var_delta_leslie::Float64
    r2_schaefer::Float64
    r2_delay_diff::Float64
    r2_leslie::Float64
    rmse_schaefer::Float64
    rmse_delay_diff::Float64
    rmse_leslie::Float64
    aic_schaefer::Float64
    aic_delay_diff::Float64
    aic_leslie::Float64
    preferred_model::Symbol
    # Legend entries carrying the key estimated parameters, built once in
    # `compare_macro_models` where the fits are in hand. A legend reading only
    # "Schaefer" identifies the curve but not what was estimated, and the
    # estimates are the reason the model was fitted.
    label_observed::String
    label_schaefer::String
    label_delay_diff::String
    label_leslie::String
    # The fits are in number, so the raw count trajectories are carried alongside
    # the rescaled biomass ones. A consumer that knows the average weight of the
    # component being plotted can then do its own conversion, rather than being
    # handed a biomass series built with one pooled constant.
    numbers_observed::Vector{Float64}
    numbers_schaefer::Vector{Float64}
    numbers_delay_diff::Vector{Float64}
    numbers_leslie::Vector{Float64}
    # The single pooled mean legal mass `biomass_*_kg` were built with, kept so the
    # rescale is auditable rather than implicit.
    mean_component_weight_kg::Float64
end

"""
    EmpiricalBiologicalModel

Container for empirical parameters, allometric scalings, and trait covariances.
"""
struct EmpiricalBiologicalModel
    mass_params::Dict{Symbol, Tuple{Float64, Float64, Float64}}
    maturity_params::Dict{Symbol, Tuple{Float64, Float64, Float64}}
    fecundity_params::Dict{Symbol, Tuple{Float64, Float64, Float64}}
    stage_modals::Dict{Int, Tuple{Float64, Float64}}
    trait_names::Vector{Symbol}
    trait_cov::Matrix{Float64}
    r2_scores::Dict{Symbol, Float64}
    sample_size::Int
end

# --- Parameter Hierarchies ---

"""
    ProcessToggles

Granular feature toggles controlling activation of specific biological,
environmental, and behavioral mechanisms. When a toggle is disabled, the process
reverts to a biologically sound stochastic default.
"""
Base.@kwdef struct ProcessToggles
    use_energetics::Bool = true
    use_thermal_stress::Bool = true
    use_cannibalism::Bool = false
    use_senescence::Bool = true
    use_spermatheca::Bool = true
    use_movement::Bool = true
    use_directed_movement::Bool = true
    use_degree_days::Bool = true
    use_molting::Bool = true
    use_reproduction::Bool = true
    # There is no `use_fishery`. Harvest is switched by
    # `fishery.fishing_probability_daily`, and a probability of zero means no
    # fishery. It used to also be gated on a toggle, which made two settings
    # express one thing: with the toggle on and the probability at zero, the model
    # reported the fishery as enabled while no animal could ever be caught, and a
    # user who raised the probability expecting harvesting had to know which of the
    # two was the real switch. See `fishery_active`.
end

Base.@kwdef struct SimulationParams
    dt::Float64 = 1.0
    seed::Int = 1234

    days::Int = 730
    grid_nx::Int = 100
    grid_ny::Int = 100
    # Calendar bookkeeping. The seasonal temperature cycle peaks on
    # `temp_seasonal_phase_day`, and day-of-year is taken modulo
    # `days_per_year`, so a leap-year calendar is expressed by raising it.
    days_per_year::Float64 = 365.0
    temp_seasonal_phase_day::Float64 = 80.0

    # --- Initial abundance distribution ---
    #
    # Three vectors, element `i` being model stage `i - 1`. They are the whole of
    # it: seeding reads these and nothing else, so the initial age and sex structure
    # is *stated* rather than implied by a decay factor and a total.
    #
    # `initial_aggregate_cohort` seeds the undifferentiated cohorts below
    # `OrganismParams.juvenile_max_stage`. `AggregateGroup` carries no `sex` field,
    # so there is one cohort vector and not two: a cohort's members are unassigned
    # until individuation, which draws them 1:1 from `leslie.male_split_fraction`.
    #
    # The other two seed individuals, per sex, from `juvenile_max_stage` up to each
    # sex's terminal stage. They are separate because an organism *is* sexed and a
    # cohort is not.
    #
    # The default is the whole clutch at the first cohort stage and nothing else —
    # eggs only, no seeded individuals. Seeding previously always added `n_agents`
    # individuals on top of the cohorts, so a run began with a populated spawning and
    # fishable stock that it was supposed to earn, and the recruitment time series
    # started from an artificial population.
    #
    # Element 1 is stage 0 and is **not seeded**: stage 0 is the brooded egg, which a
    # female carries rather than a cohort holding it. The clutch is stage 1, so the
    # default sits at element 2. Putting it at element 1 reads as a configuration that
    # is ignored, and did: every shipped config reported a non-zero sum and seeded
    # nothing at all.
    initial_aggregate_cohort::Vector{Float64} = [0.0, 2000.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    initial_female_cohort::Vector{Float64} = fill(0.0, 12)
    initial_male_cohort::Vector{Float64} = fill(0.0, 12)
    # Age attributed to a freshly seeded cohort at a given stage, in days.
    initial_cohort_days_per_stage::Float64 = 90.0

    # --- Initial benthic agents ---
    # Age attributed to a seeded individual of a given stage, in days, is
    # `initial_age_days_per_stage * (stage - initial_age_first_stage)` plus a
    # uniform draw on `[0, initial_age_jitter_days]`.
    initial_age_days_per_stage::Float64 = 180.0
    initial_age_first_stage::Float64 = 3.0
    initial_age_jitter_days::Float64 = 90.0
    # Pre-ecdysial thermal requirement a seeded individual starts with, drawn
    # uniformly on `[initial_degree_days_min, initial_degree_days_max]`.
    initial_degree_days_min::Float64 = 50.0
    initial_degree_days_max::Float64 = 300.0
end

Base.@kwdef struct CriteriaParams
    # Persistence floor for `evaluate_population_criteria`, as an absolute count.
    # The criterion this feeds asks "did the population avoid extinction?", which
    # is a statement about the run, not about the configured carrying capacity.
    # It was previously `max(15, 0.10 * benthic_carrying_capacity)`, so a
    # population that self-regulated at a fraction of K was reported as not
    # persistent — the bound was derived from the same K the model regulates
    # around, and was therefore self-referential. 1 means "not extinct".
    min_population::Int = 1

    # Explosion guard, as a multiple of `benthic_carrying_capacity`. This one
    # legitimately refers to K: "the population did not explode past the scale
    # it was configured for" is exactly a statement about K. A transient
    # overshoot is normal, hence a multiple rather than K itself.
    max_population_multiple::Float64 = 3.0

    # --- Fidelity thresholds for `evaluate_population_criteria` ---
    # Each bounds one component criterion; a run passes when every one holds.
    # Largest allowable fractional shift in mean size across the recent window.
    max_size_drift::Float64 = 0.20
    # Largest allowable RMSE against the cumulative maturity curve, on [0, 1].
    max_maturity_rmse::Float64 = 0.35
    # Largest allowable RMSE of mean stage size against `growth_schedule()`, in mm.
    max_growth_rmse::Float64 = 15.0
    # Recent time steps considered for the size-stability window.
    window_steps::Int = 10
    # Minimum individuals in a size bin for its maturity proportion to count.
    maturity_ogive_min_count::Int = 10

    # --- Composite loss weights ---
    # The composite score is a weighted sum, not a pass/fail, so that an optimizer
    # has something to descend. Each failed criterion contributes its fixed
    # penalty; the continuous errors contribute proportionally.
    extinction_penalty::Float64 = 150.0
    explosion_penalty::Float64 = 150.0
    size_drift_weight::Float64 = 25.0
    maturity_rmse_weight::Float64 = 15.0
    growth_rmse_weight::Float64 = 1.0

    # --- Maturity ogive size bins ---
    # Lower edge, step and upper edge in mm. Bins with fewer than
    # `maturity_ogive_min_count` individuals are dropped.
    ogive_bin_start::Float64 = 20.0
    ogive_bin_step::Float64 = 10.0
    ogive_bin_stop::Float64 = 140.0
end

Base.@kwdef struct EnvironmentParams
    temp_offset::Float64 = 0.0
    temp_multiplier::Float64 = 1.0
    temp_noise_sd::Float64 = 0.5
    temp_base_degreedays::Float64 = -1.5
    food_max::Float64 = 10.0
    food_replenishment_rate::Float64 = 0.1
    use_geodata::Bool = false
    domain_lon::Vector{Float64} = [-65.0, -63.0]
    domain_lat::Vector{Float64} = [43.0, 45.0]
    time_range::Vector{String} = ["1993-01-01", "1993-12-31"]
    pelagic_depth::Float64 = 15.0
    temperature_source::String = "woa23"
    temperature_month::Int = 0
    bathymetry_source::String = "etopo2022"
    geodata_input_dir::String = "inputs"
    # The assimilated environmental store: the run reads its bathymetry and hydrography from
    # here, and `assimilate_environmental_inputs` writes them here. Relative to the workspace
    # so it resolves the same way from a notebook and from the CLI. An empty string means
    # `geodata_input_dir`, which is a run-local cache and not a store shared with the other
    # halves.
    geodata_store_dir::String = "../lakehouse/GeoIBM"
    # Density-dependent crowding. Total benthic abundance is
    # `individuated + aggregate`, and the mortality feedback is
    # `1 + (N / K)^exponent`.
    #
    # K must exceed the seeded stock, which at the defaults is 300 individuals
    # plus the aggregate cohorts: 2000 eggs and 500 per stage over stages 1 to 5,
    # for 4800. At K = 300 the multiplier at time zero was 257, which drove every
    # hazard to certainty on the first day: nothing survived long enough to reach
    # the benthic stages, so individuation never ran and the run reported an
    # extinction it had not modelled. At K = 12000 the multiplier is 1.16, so
    # crowding corrects the configured mortality by about two points a year
    # instead of replacing it.
    #
    # `initialize_model` warns when the seeded stock already exceeds K, because
    # that is a silent failure rather than a slow one.
    benthic_carrying_capacity::Float64 = 12_000.0
    half_saturation_food::Float64 = 2.0
    assimilation_efficiency::Float64 = 0.75
    habitat_offset::Float64 = 0.0
    habitat_multiplier::Float64 = 1.0
    habitat_quality_mean::Float64 = 0.5
    habitat_quality_sd::Float64 = 0.1
    # --- Habitat taxis ---
    # `organism_disperse!` scores each candidate cell by how well it matches the
    # organism's preferred temperature and its food density relative to the grid
    # mean, then samples a heading from a softmax over those scores.
    # `habitat_temperature_optimum` is the centre of the thermal preference and
    # `habitat_temperature_sigma` its width in degrees Celsius; they are separate
    # from `metabolic_optimal_temperature`, which sets the energetic cost curve
    # rather than the direction of travel.
    habitat_temperature_optimum::Float64 = 2.5
    habitat_temperature_sigma::Float64 = 1.5
    # Weight of food density in the combined suitability score, relative to the
    # thermal term. Zero makes dispersal purely thermal.
    habitat_food_weight::Float64 = 0.5
    # Softmax temperature of the heading distribution. Larger values spread
    # movement over more directions; smaller values concentrate it on the best.
    movement_softmax_temperature::Float64 = 0.5
    # --- Settlement ---
    # Radius, in grid cells, within which a newly individuated recruit settles
    # around its cohort.
    settlement_dispersal_radius::Int = 2
    # Standard deviation of the lognormal from which an individuated recruit's
    # size is drawn around its cohort's mean log size.
    settlement_size_log_sd::Float64 = 0.04
    # Exponent of the density-dependent mortality feedback. A value above 1 makes
    # the response accelerate past carrying capacity instead of remaining linear.
    benthic_carrying_capacity_exponent::Float64 = 2.0
end

# Backward compatibility accessors for renamed temperature controls
function Base.getproperty(env::EnvironmentParams, sym::Symbol)
    if sym === :temp_base
        return getfield(env, :temp_offset)
    elseif sym === :temp_amplitude
        return getfield(env, :temp_multiplier)
    end
    return getfield(env, sym)
end

Base.@kwdef struct OrganismParams
    # Terminal developmental stage, the hard ceiling a molt clamps to. Males reach
    # two stages further than females, as they grow larger and keep moulting.
    #
    # These are the model's own stage numbers, in which stage 1 is benthic instar
    # 1 and the three pelagic sub-stages — zoea1, zoea2, megalopa — occupy model
    # stages -2 to 0, so a cohort created at stage 0 on hatching is already zoea1.
    # `gui.jl` maps these to instar labels through `cohort_stage_label`.
    #
    # The species stage tables stop at 12, which bounds the Leslie projection, not
    # the animals: individuals molt on past it and their sizes keep growing along
    # the saturating lower-bound curve towards `max_size_*`, so the ceiling has to
    # sit above the table for the oldest cohorts to exist at all.
    max_stage_male::Int = 19
    max_stage_female::Int = 16
    # Plot name for each developmental stage, in stage order: element `i` is model
    # stage `i - 1`. Kept here so the figures, the configuration file and the code
    # cannot disagree about what a stage is called; the dashboard used to derive
    # these in code, and a row of bare `0 1 2 3` on an axis left it genuinely
    # ambiguous whether a number was a model stage or a benthic instar.
    #
    # A vector rather than a `[stage_labels]` table because the configuration
    # loader flattens one level of nesting and discards the section name, which
    # would strip the very stage keys this is meant to preserve.
    #
    # Entry 1 is zoea1: a cohort is created at stage 1 on hatching. The pelagic
    # block follows the snow crab sequence — two zoeal instars and a single
    # megalopa — so it occupies stages 1 to 3. From stage 4 on, each entry is a
    # benthic instar number counting from one, so model stage = instar + 3.
    # Benthic instars 1 through 16 reach model stage 19 (egg + 3 pelagic + 16 benthic).
    stage_labels::Vector{String} = ["Egg", "z1", "z2", "m", "1", "2", "3",
        "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"]
    metabolic_rate_base::Float64 = 0.01
    intake_rate_base::Float64 = 0.05

    # --- Bioenergetics (Kleiber-style allometric budget) ---
    # Metabolic cost scales as `mass^metabolic_mass_exponent`.
    metabolic_mass_exponent::Float64 = 0.75
    # Multiplies the metabolic cost by `exp(coefficient * temperature)`.
    metabolic_temperature_coefficient::Float64 = 0.08
    # Performance curve centre and width, in degrees Celsius.
    metabolic_optimal_temperature::Float64 = 3.0
    metabolic_optimal_sigma::Float64 = 2.5
    # Fraction of assimilated intake spent on processing (specific dynamic
    # action); the remainder, `1 - sda`, is available for growth and storage.
    specific_dynamic_action::Float64 = 0.15

    # --- Ecdysis energetics ---
    # Ecdysial requirement is `max(energy_molt_floor, energy_molt_slope * size)`.
    # The slope must be commensurate with the daily energy surplus, which the
    # bioenergetics above produce at roughly 0.03-0.2 per day depending on size.
    # A slope of order 1 per mm demands tens to hundreds of energy units, far more
    # than an animal accumulates in a year, so molting becomes impossible and the
    # whole population stalls at the first gated instar.
    energy_molt_slope::Float64 = 0.03
    energy_molt_floor::Float64 = 2.0
    molt_energy_retention::Float64 = 0.10
    # Share of the ecdysial requirement pre-committed before the conditional
    # bonus is applied: `base + (1 - base) * tanh(condition - 1)`.
    molt_energy_base_fraction::Float64 = 0.85

    # --- Molt hazard ---
    # Daily hazard is `molt_hazard_coefficient * min(progress, saturation)`,
    # converted to a probability with `1 - exp(-hazard * dt)`.
    molt_hazard_coefficient::Float64 = 0.15
    # Progress ratio beyond which the hazard stops increasing.
    molt_progress_saturation::Float64 = 3.0

    initial_energy::Float64 = 10.0

    # --- Stage and shell bookkeeping ---
    # Stage sizes are sampled from a symmetric Beta of these two parameters.
    stage_size_sample_a::Float64 = 2.5
    stage_size_sample_b::Float64 = 2.5
    # Shell age is inferred as `(age_days - offset * (stage - first)) / days_per_year`.
    shell_age_stage_offset::Float64 = 180.0
    shell_age_first_stage::Int = 6
    # Newmass-free shell condition index breakpoints, in days since ecdysis.
    shell_condition_days::Vector{Float64} = [30.0, 365.0, 730.0, 1277.5]
    # Wet mass and carapace size of a single egg.
    egg_mass_g::Float64 = 0.00007
    egg_size_mm::Float64 = 0.60
    # Probability of removing an individual in a day once its reserve is depleted.
    energy_deficit_removal_probability::Float64 = 0.75
    # Largest stage still treated as juvenile by the baseline mortality multiplier.
    juvenile_max_stage::Int = 12
    # Size (mm) at or above which an individual is treated as past the juvenile
    # phase for allometry and for display, even if it never matured. This
    # replaces a hardcoded stage threshold that was specific to one species.
        adult_size_threshold::Float64 = 68.0
    # --- Trait heterogeneity ---
    #
    # Bounds of the quantile drawn for each individual's four traits — growth,
    # maturity, fecundity and mortality. They are *quantiles*, not values: each
    # individual draws one and is then read off the corresponding distribution at that
    # position, which is how an ogive is given between-individual spread without
    # being resampled daily.
    #
    # Excluded at the ends because `maturity` rejects a quantile of 0 or 1 as
    # degenerate, and because an individual at exactly the median has no spread on
    # either side of it.
    trait_quantile_min::Float64 = 0.01
    trait_quantile_max::Float64 = 0.99
end


Base.@kwdef struct PhenologyParams
    molt_season_start::Float64 = 15.0
    molt_season_end::Float64 = 140.0
    mating_season_start::Float64 = 30.0
    mating_season_end::Float64 = 120.0
    mating_search_radius::Int = 5
    # Energy a male spends on a successful mating attempt, deducted from its
    # reserve. A male that cannot pay this does not mate.
    mating_energy_cost::Float64 = 8.0
    # Relative mating likelihood of a nearby male, raised to this power of its
    # carapace width. An exponent of 2 makes large males preferred.
    mating_size_weight_exponent::Float64 = 2.0
    # Spermatheca stores a successful mating deposits, split into this many usable
    # clutches. A legal-size male stores `sperm_stores_legal`, a smaller one
    # `sperm_stores_sublegal`; each clutch is extruded as one brood, so this sets
    # how many broods a single mating can support.
    sperm_stores_legal::Int = 3
    sperm_stores_sublegal::Int = 2
    # Floor on the extrapolated clutch size. Spermatheca deposition is modelled as
    # a single large mass rather than individual sperm, so a female very near the
    # size of maturity would otherwise carry an implausibly small clutch.
    minimum_clutch_size::Int = 1000
    # Lower bound on the temperature used for the incubation scaling, so a cell at
    # or below freezing cannot make the brood duration diverge.
    brood_temperature_floor::Float64 = 1.0

    # Incubation length is `brood_duration_base * (reference / T)^exponent`, so
    # warmer water shortens the brood.
    brood_duration_base::Float64 = 365.0
    brood_temperature_reference::Float64 = 5.0
    brood_temperature_exponent::Float64 = 0.30
    brood_duration_sd::Float64 = 5.0

    # First developmental stage subject to the seasonal molt gate. Stages below
    # this molt whenever their intermolt and degree-day requirements are met,
    # while stages at or above it are confined to the molt season. This asymmetry
    # concentrates a cohort at the first gated stage, so it is exposed rather
    # than hardcoded; set to 1 to make every benthic stage follow one annual cycle.
    molt_season_min_stage::Int = 13
    # Ecdysial thermal inhibition: the molt probability is scaled by
    # `max(0, 1 - ((T - onset) / bandwidth)^2)`. The onset must sit above the
    # warmest temperature reached during the molt season, otherwise inhibition
    # silently shuts molting off for part or all of that season.
    molt_thermal_onset::Float64 = 10.0
    molt_thermal_bandwidth::Float64 = 3.0
end

Base.@kwdef struct MortalityParams
    # Ceiling applied to every individual hazard and to their sum. Senescence and
    # thermal stress grow without bound in their raw form, and a terminal-molt
    # individual never moults again, so its time since ecdysis grows without
    # limit. Saturation is the correct behaviour: mortality cannot exceed
    # certainty.
    hazard_cap::Float64 = 1.0

    # --- Annual instantaneous mortality rates ---
    # Daily hazards are derived as `rate / days_per_year`, using
    # `SimulationParams.days_per_year`. That field used to be duplicated here,
    # and because the flat configuration maps a key by *name*, a user setting
    # `days_per_year` under `[simulation]` (a leap-year calendar, say) was
    # silently configuring mortality instead while the temperature field stayed
    # where they put it. One calendar, one field.
        # Background natural mortality by stage, annual, indexed by stage like the
    # ecdysis schedules and clamped at the last entry.
    #
    # This replaces a single rate plus a binary juvenile multiplier, which gave
    # pelagic cohorts, first-instar crabs and terminal adults the same mortality
    # despite being very different animals. A pelagic larva is small, poorly
    # insulated and has no shelter; a large benthic adult is the opposite. The
    # decline continues with size rather than stopping at the boundary.
    baseline_rate_by_stage::Vector{Float64} = [
        0.35, 0.35, 0.35, 0.35,
        0.15, 0.15, 0.15, 0.15,
        0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08,
        0.06,
    ]
    rate_predation::Float64 = 0.05
    rate_soft_shell::Float64 = 0.12
    rate_starvation::Float64 = 0.20
    rate_thermal::Float64 = 0.05
    rate_senescence::Float64 = 0.05
    rate_cannibalism::Float64 = 0.05
    # Mortality from handling a crab that is caught and returned to the sea, annual
    # rate. Charged on capture, and only during the window below — the damage is done
    # at the event, so a crab already handled before is not charged again and a
    # fishery that catches the same animal every season does not compound the rate.
    rate_handling::Float64 = 0.05
    # How long handling mortality is in force, in days from the start of the run.
    # After this it is zero: the rate is not a permanent property of the fishery but a
    # term for a bounded episode of it, so a long run does not accumulate handling
    # deaths indefinitely.
    handling_mortality_days::Float64 = 365.0

    # --- Shape modifiers ---
    baseline_trait_sigma::Float64 = 0.25
    predation_size_reference::Float64 = 5.0
    predation_size_decay::Float64 = 0.03
    # Window over which a post-ecdysis individual is considered soft-shelled,
    # and the e-folding time of that vulnerability.
    soft_shell_duration::Float64 = 30.0
    soft_shell_decay_days::Float64 = 15.0
    starvation_critical_fraction::Float64 = 0.05
    thermal_exponent::Float64 = 1.5
    temp_thermal_stress::Float64 = 8.0
    # Shell age (years) beyond which senescence begins, and its exponential rate.
    senescence_age_threshold::Float64 = 3.0
    senescence_exp_rate::Float64 = 0.80
    cannibalism_size_threshold::Float64 = 35.0
    cannibalism_density_reference::Float64 = 10.0
    cannibalism_density_cap::Float64 = 2.0

    # --- Aggregate (pre-individuation) cohort transition ---
    #
    # The daily transition hazard, `h0` in `h(DD) = h0 * PoissonCDF(X; DD)`. It is
    # the rate at which an individual that has already met its own molt threshold
    # moults, so a stage's requirement `X` — 50% of the cohort's Poisson thresholds —
    # is reached at roughly `X / DD-per-day` days and the stage takes a little longer.
    #
    # `aggregate_hazard_degree_days`, `aggregate_hazard_intermolt` and
    # `aggregate_transition_saturation` used to shape this as a logistic in
    # `(DD - X) / (k X)` with hard gates either side. They are gone: the logistic
    # reached its ceiling a third of the way to the next stage and held it there, so
    # a cohort left at full rate while still short of the requirement. The
    # distribution replaced them, and with them any need for a saturation constant.
    aggregate_transition_hazard::Float64 = 0.35
    # Baseline daily hazard for an aggregate cohort, indexed by stage through
    # `stage_schedule_lookup` so a shorter list holds its last value for the
    # higher stages. These cohorts predate individuation, so they carry the
    # pelagic and early-benthic mortality that the individual hazard terms above
    # do not represent.
    aggregate_baseline_hazards::Vector{Float64} = [
        0.002, 0.020, 0.020, 0.015, 0.0015,
    ]
    # Pelagic larval over-age developmental mortality hazard (daily excess rate
    # when larvae exceed maximum viable intermolt duration ~84 days).
    larval_overage_hazard::Float64 = 0.05
    # Parabolic optimal temperature for larval survival (Ouellet & Sainte-Marie 2018: 10-12°C).
    larval_temp_opt::Float64 = 11.0
    larval_temp_sigma::Float64 = 3.5
end

Base.@kwdef struct FisheryParams
    legal_size_male::Float64 = 95.0
    fishing_probability_daily::Float64 = 0.0
    fishing_season_start::Float64 = 90.0
    fishing_season_end::Float64 = 150.0
    no_take_proportion::Float64 = 0.0
end

"""
    fishery_active(params)

Whether the model harvests. Accepts an `IBMParameters` or a `FisheryParams`.

`fishing_probability_daily` is the switch: a probability of zero means no fishery,
whatever else is configured. There was once a second condition for this,
`toggles.use_fishery`, so the model could report the fishery as enabled while no
animal could ever be caught — and a user raising the probability expecting harvest
had to know which of the two settings was the real one. That toggle is gone.

The season window is deliberately not part of this test: a fishery can be
configured and still be closed on any given day.
"""
fishery_active(params) = fishery_active(params.fishery)
fishery_active(params::FisheryParams) = params.fishing_probability_daily > 0.0

Base.@kwdef struct OutputParams
    save::Bool = true
    save_file::String = "simulation_results.csv"
    snapshot_day::Float64 = 274.0
    frequency_days::Float64 = 365.0
    plot::Bool = false
    output_dir::String = "outputs"
    animate::Bool = false
    animation_file::String = "organism_animation.mp4"
    anim_frames::Int = 120
    anim_dt::Int = 2
    anim_fps::Int = 15
    anim_field::String = "temperature"
end

"""
    SpeciesParams

Generic container for species-specific biological parameters.
All values can be loaded from TOML configuration files.
"""
Base.@kwdef struct SpeciesParams
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
    # Standard deviation of the intercept when a fecundity draw is requested
    # without an explicit quantile.
    fecundity_intercept_sd::Float64 = 0.05
    # Sainte-Marie allometric form: `size^exponent * 10^coefficient`.
    fecundity_stmarie_exponent::Float64 = 2.725
    fecundity_stmarie_coefficient::Float64 = 0.7311
    # Reproductive senescence: fecundity is floored and then declines linearly in
    # shell age beyond the threshold.
    fecundity_senescence_threshold::Float64 = 3.0
    fecundity_senescence_floor::Float64 = 0.60
    fecundity_senescence_rate::Float64 = 0.08
    # Shell age (years) at or below which a female is treated as primiparous, i.e.
    # extruding her first brood. Parity is a property of the female's reproductive
    # history, not of how much sperm she happens to be holding.
    primiparous_shell_age_max::Float64 = 2.0
    # Carapace width (mm) that separates the primiparous and multiparous fecundity
    # forms. This is the split used only when shell age is unavailable: the two
    # forms are distinguished by the female's reproductive history, and size is the
    # best proxy available in a survey that did not record it. Females at or below
    # this size are scored primiparous.
    fecundity_parity_size_split::Float64 = 65.0

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

    # --- Growth modifiers applied to the post-molt size prediction ---
    # Maturation carries a cost in mean and variance, drawn separately by sex.
    growth_maturation_penalty_male_mean::Float64 = -0.13564
    growth_maturation_penalty_male_sd::Float64 = 0.00922
    growth_maturation_penalty_female_mean::Float64 = -0.1310
    growth_maturation_penalty_female_sd::Float64 = 0.0252
    # Nutritional condition and ambient temperature contributions.
    growth_condition_coefficient::Float64 = 0.05
    growth_temperature_coefficient::Float64 = -0.004
    growth_temperature_reference::Float64 = 4.0
    growth_residual_sd::Float64 = 0.040
    # Lower bound on proportional size increase per ecdysis, by maturity status.
    growth_min_expansion_immature::Float64 = 0.12
    growth_min_expansion_mature::Float64 = 0.06

    # --- Stage size bounds ---
    # `max_size_*` is the asymptotic carapace width for each sex. It lives here
    # rather than on the organism because it is the saturating limit of the stage
    # size curve below, and a limit the curve cannot see is a limit it cannot obey.
    max_size_male::Float64 = 160.0
    max_size_female::Float64 = 95.0
    # The empirical lower bound of carapace width per stage saturates at the
    # asymptotic size:
    #
    #     ln S(stage) = ln L + (intercept - ln L) * exp(-decay * (stage - anchor))
    #
    # so it equals `exp(intercept)` at the anchor stage and tends to `L`. The
    # earlier form, `exp(intercept + slope * (stage - anchor))`, is unbounded for
    # every positive slope: it put a stage-12 female at 112 mm against a 95 mm
    # ceiling and a stage-14 female at 210 mm. `intercept` and `decay` are least
    # squares fitted to the lower edge of each tabulated stage.
    #
    # A pre-benthic table covers the stages below the benthic anchor, where size is
    # small enough that the curve does not apply.
    size_lb_immature_sizes::Vector{Float64} = [0.95, 1.45, 2.50]
    size_lb_immature_breakpoints::Vector{Float64} = [1.5, 2.5]
    size_lb_male_intercept::Float64 = 1.102422
    size_lb_male_decay::Float64 = 0.235833
    size_lb_male_anchor_stage::Float64 = 4.0
    size_lb_female_intercept::Float64 = 1.930054
    size_lb_female_decay::Float64 = 0.308476
    size_lb_female_anchor_stage::Float64 = 5.0
    benthic_anchor_stage::Float64 = 5.0

    # Gear Selectivity
    selectivity_s50::Float64 = 95.0
    selectivity_s50_sigma::Float64 = 5.0

    # Allometric Mass Parameters
    mass_alpha_male_immature::Float64 = 0.0002665
    mass_beta_male_immature::Float64 = 3.098
    mass_alpha_male_mature::Float64 = 0.0001830
    mass_beta_male_mature::Float64 = 3.160
    mass_alpha_female_immature::Float64 = 0.0003440
    mass_beta_female_immature::Float64 = 2.980
    mass_alpha_female_mature::Float64 = 0.0004500
    mass_beta_female_mature::Float64 = 2.920
    mass_sigma::Float64 = 0.08

    # Ecdysis Schedules
    # Minimum intermolt duration (days) by developmental stage. Index 0 maps to
    # stage 0, index `i` to stage `i`; the final entry applies to all stages
    # at or beyond its index.
    # Stage durations, in days. Stages 0-5 are pelagic and read
    # `intermolt_days_larval`; these entries are their placeholders here.
    #
    # Instars 1-4 take half a year each, so an animal covers two instars a year, and
    # every instar from 5 on takes a year. Together with the pelagic block — zoea1
    # through the end of the megalopa in one year — that is a growth schedule of
    # roughly one year per two instars early and one per year after.
    #
    # The degree-day vectors below state the same timings as thermal requirements,
    # converted at the configured thermal rate, so the two schedules cannot drift
    # apart. The degree-days are what actually bind: `toggles.use_degree_days` makes
    # the transition test `accumulated_degree_days >= requirement`, so a day-based
    # requirement only takes effect if that toggle is off.
    intermolt_days_schedule::Vector{Float64} = [
        0.0, 0.0, 0.0, 0.0,
        182.5, 182.5, 182.5, 182.5,
        365.0, 365.0, 365.0, 365.0, 365.0, 365.0, 365.0, 365.0, 365.0, 365.0,
        365.0, 365.0, 365.0, 365.0,
    ]
    # Cumulative pre-ecdysial thermal requirement (degree-days) by stage, same
    # indexing convention as `intermolt_days_schedule`. Converted at 5 degree-days
    # per day, which is `temp_base - temp_base_degreedays` at the configured
    # temperatures; changing either rescale these in proportion.
    degree_days_schedule::Vector{Float64} = [
        0.0, 0.0, 0.0, 0.0,
        912.5, 912.5, 912.5, 912.5,
        1825.0, 1825.0, 1825.0, 1825.0, 1825.0, 1825.0, 1825.0, 1825.0, 1825.0,
        1825.0, 1825.0, 1825.0, 1825.0, 1825.0,
    ]
    # Pelagic schedules, ordered as [egg, zoea1, zoea2, megalopa] —
    # two zoeal instars and a single megalopa for Chionoecetes opilio.
    # Their length is what defines how far the pelagic block reaches: stages below it
    # are developed on these numbers, and every stage from
    # `length(intermolt_days_larval)` upward uses the benthic schedules above.
    #
    # Durations (36-42 days minimum, degree-days 270 DD at T0 = -1.5 °C) align
    # with empirical larval development studies (Ouellet & Sainte-Marie 2018, Incze et al. 1987).
    # The egg's entry is zero: no cohort ever exists at stage 0 (brooded clutch).
    intermolt_days_larval::Vector{Float64} = [0.0, 36.0, 36.0, 42.0]
    degree_days_larval::Vector{Float64} = [0.0, 270.0, 270.0, 270.0]

    # Minimum size gate (mm) below which maturity cannot be attained.
    maturity_size_gate::Float64 = 35.0
    # Sensitivity of the maturity size-at-50 to ambient temperature (mm per °C).
    maturity_l50_temp_slope::Float64 = -2.0
    # Reference temperature for the maturity temperature response (°C).
    maturity_temp_reference::Float64 = 4.0

    # --- Stage table used by the Leslie projection and the maturity analysis ---
    # Per stage: size range lower edge (mm), size range upper edge, and the
    # representative size used for the stage. Indexed in parallel; a stage
    # definition is emitted only while it does not exceed the corresponding
    # `leslie_max_male_stage` / `leslie_max_female_stage`.
    stage_size_range_stage1::Vector{Float64} = [7.0, 8.0]
    stage_size_range_lower_stage1::Vector{Float64} = [7.0, 14.5]
    stage_size_range_upper_stage1::Vector{Float64} = [14.5, 22.0]
    stage_size_range_mean_stage1::Vector{Float64} = [10.7, 18.0]
    stage_size_range_stage_male::Vector{Float64} = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    stage_size_range_lower_male::Vector{Float64} = [22.0, 31.0, 42.5, 58.0, 78.0, 95.0, 115.0]
    stage_size_range_upper_male::Vector{Float64} = [31.0, 42.5, 58.0, 78.0, 95.0, 115.0, 160.0]
    stage_size_range_mean_male::Vector{Float64} = [26.5, 36.5, 50.0, 68.0, 86.5, 105.0, 130.0]
    stage_size_range_stage_female::Vector{Float64} = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
    stage_size_range_lower_female::Vector{Float64} = [22.0, 30.5, 41.5, 54.5, 68.0, 82.0]
    stage_size_range_upper_female::Vector{Float64} = [30.5, 41.5, 54.5, 68.0, 82.0, 110.0]
    stage_size_range_mean_female::Vector{Float64} = [26.0, 35.5, 47.5, 61.0, 75.0, 90.0]
    # Terminal stages of the table above, in developmental stage numbers.
    leslie_max_male_stage::Int = 15
    leslie_max_female_stage::Int = 14
    # Size (mm) at or above which a male stage counts as legal in the projection.
    # The stage criterion below is the backstop for the largest stages, whose
    # representative size is above it already.
    leslie_legal_size_male::Float64 = 95.0
    leslie_first_legal_stage_male::Int = 14
    # First female stage counted as reproductive in the projection.
    leslie_first_repro_stage_female::Int = 12
    # The stage counted as *recruitment*: the cohort immediately before maturity
    # onset, one stage short of the first reproductive stage. This is the
    # classical stock-recruit quantity — recruits per spawner — expressed as an
    # abundance at the last stage before the animals are counted as spawners.
    #
    # It must be an individuated stage, above `OrganismParams.juvenile_max_stage`,
    # because it is sex-specific and `AggregateGroup` carries no `sex` field. The
    # counter used to run off the aggregate transition into
    # `benthic_anchor_stage`, which is a size-model threshold ("first stage with
    # sex-specific bounds") rather than any biological event.
    recruitment_stage::Int = 13
end

"""
    LeslieParams

Parameters for the generalized Leslie (Lefkovitch) stage-transition matrix model.
"""
Base.@kwdef struct LeslieParams
    # Stage 1 (Recruit, stage 4)
    survival_stage1::Float64 = 0.74
    gamma_stage1::Float64 = 0.78

    # Stage 2 (Juvenile, stage 5)
    survival_stage2::Float64 = 0.76
    gamma_stage2::Float64 = 0.75
    # Sex ratio applied to a cohort as it individuates. The female share is the
    # complement, so there is deliberately no second parameter: two independent
    # shares could be configured to sum to something other than one.
    male_split_fraction::Float64 = 0.50

    # Male stages (3-9)
    survival_male_legal::Float64 = 0.78
    survival_male_sublegal::Float64 = 0.80
    gamma_male_base::Float64 = 0.75
    gamma_male_decay::Float64 = 0.10
    p_term_male_base::Float64 = 0.10
    p_term_male_exponent::Float64 = 1.3

    # Female stages (3-9)
    survival_female::Float64 = 0.80
    gamma_female_base::Float64 = 0.70
    gamma_female_decay::Float64 = 0.12
    p_term_female_base::Float64 = 0.15
    p_term_female_exponent::Float64 = 1.4

    # Fecundity
    fecundity_base::Float64 = 0.85
    fecundity_size_exponent::Float64 = 2.5
    fecundity_size_reference::Float64 = 65.0

    # --- Fitting controls for `fit_generalized_leslie_model` ---
    # Weight on the recursive trajectory term relative to the one-step process
    # error in the objective.
    trajectory_error_weight::Float64 = 0.5
    # Developmental stage at which the transition-rate decay and the maturation
    # term are both zero, i.e. the first sex-specific stage. Stating it here keeps
    # the decay anchored to the stage the male and female series actually start at,
    # rather than to a literal that has to be changed with the stage table.
    first_sex_specific_stage::Int = 9
end

struct IBMParameters
    toggles::ProcessToggles
    simulation::SimulationParams
    environment::EnvironmentParams
    organism::OrganismParams
    phenology::PhenologyParams
    mortality::MortalityParams
    fishery::FisheryParams
    output::OutputParams
    species::SpeciesParams
    leslie::LeslieParams
    criteria::CriteriaParams
end

function IBMParameters(; kwargs...)
    toggles_kwargs = Dict{Symbol, Any}()
    simulation_kwargs = Dict{Symbol, Any}()
    environment_kwargs = Dict{Symbol, Any}()
    organism_kwargs = Dict{Symbol, Any}()
    phenology_kwargs = Dict{Symbol, Any}()
    mortality_kwargs = Dict{Symbol, Any}()
    fishery_kwargs = Dict{Symbol, Any}()
    output_kwargs = Dict{Symbol, Any}()
    species_kwargs = Dict{Symbol, Any}()
    leslie_kwargs = Dict{Symbol, Any}()
    criteria_kwargs = Dict{Symbol, Any}()

    for (k, v) in kwargs
        sym = Symbol(k)
        if sym === :temp_base
            environment_kwargs[:temp_offset] = v
        elseif sym === :temp_amplitude
            environment_kwargs[:temp_multiplier] = v
        elseif sym in fieldnames(ProcessToggles)
            toggles_kwargs[sym] = v
        elseif sym in fieldnames(SimulationParams)
            simulation_kwargs[sym] = v
        elseif sym in fieldnames(EnvironmentParams)
            environment_kwargs[sym] = v
        elseif sym in fieldnames(OrganismParams)
            organism_kwargs[sym] = v
        elseif sym in fieldnames(PhenologyParams)
            phenology_kwargs[sym] = v
        elseif sym in fieldnames(MortalityParams)
            mortality_kwargs[sym] = v
        elseif sym in fieldnames(FisheryParams)
            fishery_kwargs[sym] = v
        elseif sym in fieldnames(OutputParams)
            output_kwargs[sym] = v
                elseif sym in fieldnames(SpeciesParams)
            species_kwargs[sym] = v
        elseif sym in fieldnames(LeslieParams)
            leslie_kwargs[sym] = v
        elseif sym in fieldnames(CriteriaParams)
            criteria_kwargs[sym] = v
        end
    end

    toggles = get(kwargs, :toggles, ProcessToggles(; toggles_kwargs...))
    simulation = get(kwargs, :simulation, SimulationParams(; simulation_kwargs...))
    environment = get(kwargs, :environment, EnvironmentParams(; environment_kwargs...))
    organism = get(kwargs, :organism, OrganismParams(; organism_kwargs...))
    phenology = get(kwargs, :phenology, PhenologyParams(; phenology_kwargs...))
    mortality = get(kwargs, :mortality, MortalityParams(; mortality_kwargs...))
    fishery = get(kwargs, :fishery, FisheryParams(; fishery_kwargs...))
    output = get(kwargs, :output, OutputParams(; output_kwargs...))
    species = get(kwargs, :species, SpeciesParams(; species_kwargs...))
    leslie = get(kwargs, :leslie, LeslieParams(; leslie_kwargs...))
    criteria = get(kwargs, :criteria, CriteriaParams(; criteria_kwargs...))

    return IBMParameters(
        toggles, simulation, environment, organism,
        phenology, mortality, fishery, output, species, leslie, criteria
    )
end

"""
    IBMParameters(dict::AbstractDict)

Constructor mapping a nested dictionary (e.g. from TOML) to the hierarchical
`IBMParameters` structure.
"""
function IBMParameters(dict::AbstractDict)
    flat_kwargs = Dict{Symbol, Any}()
    for (k, v) in dict
        if isa(v, AbstractDict)
            for (sub_k, sub_v) in v
                flat_kwargs[Symbol(sub_k)] = sub_v
            end
        else
            flat_kwargs[Symbol(k)] = v
        end
    end
    return IBMParameters(; flat_kwargs...)
end

"""
    IBM_FIELD_OWNER

Maps every declared parameter name to the `IBMParameters` section that holds it,
built once from the section structs.

`Base.getproperty(p::IBMParameters, sym)` resolves a nested access such as
`p.fishery.legal_size_male` to a `getfield`. It used to do that with
`fieldnames(IBMParameters)` followed by a chain of `hasfield` checks, and both
allocate: `fieldnames` returns a fresh `Vector{Symbol}` on every call. In a loop
the compiler hoists that when it can prove the parameters are invariant, but it
cannot across a loop body containing opaque calls, so the per-organism loops
allocated this vector once per parameter read per agent per day. One dictionary
lookup replaces the lot.
"""
const IBM_FIELD_OWNER = let owner = Dict{Symbol, Symbol}()
    for section in fieldnames(IBMParameters)
        T = fieldtype(IBMParameters, section)
        for f in fieldnames(T)
            # Field names are unique across the sections, which the test suite
            # asserts. `haskey` keeps first-seen order should that ever change.
            haskey(owner, f) || (owner[f] = section)
        end
    end
    owner[:temp_base] = :environment
    owner[:temp_amplitude] = :environment
    owner
end

function Base.getproperty(p::IBMParameters, sym::Symbol)
    actual_sym = sym === :temp_base ? :temp_offset :
                 sym === :temp_amplitude ? :temp_multiplier : sym
    owner = get(IBM_FIELD_OWNER, actual_sym, nothing)
    owner === nothing && return getfield(p, sym)
    return getfield(getfield(p, owner), actual_sym)
end
