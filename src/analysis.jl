"""
Data aggregation and statistical analysis tools for organism IBM outputs.

Provides functions to compute empirical maturity ogives, population snapshots,
and growth metrics across stages and life stages.
"""

using DataFrames
using Distributions
using LinearAlgebra
using Optim
using Printf
using Random
using Statistics

"""
    compute_empirical_ogive(df::DataFrame, target_sex::Sex, bins::AbstractRange)

Calculate the empirical proportion of mature organisms within discrete carapace
width bins for a specified sex.

This is a **cross-sectional** measure: it pools every individual of the given
size regardless of age or molt history, so it estimates the probability that an
individual of that size *has ever* matured. The theoretical maturity function
`maturity(sex, log(size))` is instead the probability at a *single ecdysis*.
These are different quantities, related by `1 - (1 - p)^k` where `k` is the
number of ecdyses an individual has had at that size, so the two curves are not
expected to coincide and should not be compared as though they were.

Bins with few individuals are returned with their counts so callers can weight or
discard them; without the counts a proportion computed from two individuals looks
identical to one computed from two hundred.

# Arguments
- `df::DataFrame`: Agent data table containing `:size`, `:sex`, and `:is_mature`.
- `target_sex::Sex`: Sex to evaluate (`male` or `female`).
- `bins::AbstractRange`: Size bin boundaries (e.g. `20.0:5.0:120.0`).
- `min_count::Int`: Bins holding fewer than this many individuals are skipped
  entirely, since their proportions are not interpretable.

# Returns
NamedTuple `(bin_centers, prop_mature, counts)`:
- `bin_centers::Vector{Float64}`: Midpoint size of each reported bin.
- `prop_mature::Vector{Float64}`: Proportion mature in each reported bin.
- `counts::Vector{Int}`: Number of individuals behind each proportion.
"""
function compute_empirical_ogive(
    df::DataFrame,
    target_sex::Union{Sex, Integer},
    bins::AbstractRange;
    min_count::Int = 1
)
    s_target = (target_sex isa Integer) ? ((target_sex == 0) ? male : female) : target_sex
    sex_df = filter(row -> row.sex == s_target, df)
    bin_centers = Float64[]
    prop_mature = Float64[]
    counts = Int[]

    sz_col = :size
    n_bins = length(bins) - 1
    for k in 1:n_bins
        b_min = bins[k]
        b_max = bins[k + 1]

        in_bin = filter(row -> b_min <= row[sz_col] < b_max, sex_df)
        n_in_bin = nrow(in_bin)
        n_in_bin < min_count && continue
        push!(bin_centers, (b_min + b_max) / 2.0)
        if :is_mature in propertynames(in_bin)
            n_mature = count(row -> row.is_mature == true, eachrow(in_bin))
        else
            n_mature = count(s -> s >= 12, in_bin.stage)
        end
        push!(prop_mature, n_mature / n_in_bin)
        push!(counts, n_in_bin)
    end

    return (bin_centers = bin_centers, prop_mature = prop_mature, counts = counts)
end

"""
    compute_empirical_ogive(model, target_sex, bins; min_count=1)

Cross-sectional maturity ogive for `model` and `target_sex`. Accepts `target_sex`
as a `Sex` enum (`male`, `female`) or integer (0 = male, 1 = female). See the
`DataFrame` method for the definition of the quantity returned and the meaning of
`min_count`.
"""
function compute_empirical_ogive(
    model,
    target_sex::Union{Sex, Integer},
    bins::AbstractRange;
    min_count::Int = 1
)
    organisms = collect(allagents(model))
    if isempty(organisms)
        return (bin_centers = Float64[], prop_mature = Float64[], counts = Int[])
    end

    df = DataFrame(
        size = [c.size for c in organisms],
        sex = [c.sex for c in organisms],
        stage = [c.stage for c in organisms],
        is_mature = [c.is_mature for c in organisms]
    )

    actual_sex = if target_sex isa Integer
        (target_sex == 0) ? male : female
    else
        target_sex
    end

    return compute_empirical_ogive(df, actual_sex, bins; min_count = min_count)
end


"""
    cumulative_maturity_curves(sizes, target_sex=nothing; stages=build_stages())

Cumulative ("has ever matured") probability at each size in `sizes`, from the
per-ecdysis `maturity` schedule. With `target_sex` given, returns a single vector
for that sex; with `target_sex === nothing`, returns `(male, female)`.

Each size is taken to have accumulated one maturity opportunity per stage whose
representative size it is at or above, so the value is
``1 - \\prod_{lo \\leq size} (1 - p(lo))``. This is the curve a cross-sectional ogive
estimates, so it is the correct reference for comparing against simulated data.
See [`maturity_ogive_residuals`](@ref).
"""
function cumulative_maturity_curves(
    sizes::AbstractVector{<:Real},
    target_sex::Union{Nothing, Sex, Integer} = nothing;
    stages::Vector{StageDefinition} = build_stages()
)
    sexes = (target_sex === nothing) ? (male, female) : (target_sex,)
    curves = map(sexes) do sx
        s_target = (sx isa Integer) ? ((sx == 0) ? male : female) : sx
        sex_code = (s_target === male) ? 0 : 1
        # Each ecdysis happens at a stage's representative size, so the maturity
        # draw for that stage is evaluated there. Stage *lower* bounds would
        # systematically understate the accumulated probability, because every
        # bound sits below the current size and the per-ecdysis curve rises with
        # size.
        midpoints = sort!([
            Float64(st.size_mean) for st in stages
            if st.sex === s_target || st.sex === undifferentiated
        ])

        # The ecdysis count is an integer, so the raw product is a step function
        # of size. A cross-sectional ogive is smooth in size, and bins fall
        # between the stage midpoints, so interpolate the cumulative values
        # between midpoints. Monotonicity is preserved because the underlying
        # values increase with midpoint.
        discrete = Float64[]
        surv = 1.0
        for m in midpoints
            surv *= (1.0 - maturity(sex_code, log(m); deterministic = true))
            push!(discrete, 1.0 - surv)
        end

        map(sizes) do sz_v
            sz = Float64(sz_v)
            sz <= midpoints[1] && return discrete[1]
            sz >= midpoints[end] && return discrete[end]
            j = searchsortedlast(midpoints, sz)
            lo, hi = midpoints[j], midpoints[j + 1]
            w = (sz - lo) / (hi - lo)
            return discrete[j] + w * (discrete[j + 1] - discrete[j])
        end
    end

    return (target_sex === nothing) ? (curves[1], curves[2]) : curves[1]
end

"""
    maturity_ogive_residuals(ogive, target_sex; stages=build_stages()) → Vector{Float64}

Squared differences between a cross-sectional ogive from
[`compute_empirical_ogive`](@ref) and the matching *cumulative* theoretical curve.

`maturity(sex, log(size))` is the probability of maturing at a **single ecdysis**,
whereas the empirical ogive pools every individual of a size regardless of age and
therefore estimates the probability of having **ever** matured. Comparing the two
directly is biased against realistic values by construction: an individual at size
`s` has already ecdysed out of every earlier stage, so it has had one maturity
opportunity per stage at or below `s`. With per-ecdysis probabilities `p_i` over
those stages the cumulative probability is ``1 - \\prod_i (1 - p_i)``.

The opportunities are derived from the stage size ranges rather than supplied as a
constant, so the comparison stays like-for-like as the stage model changes. A bin
with no completed stage yet falls back to the single-ecdysis probability, which is
the only opportunity that exists at that size.
"""
function maturity_ogive_residuals(
    ogive,
    target_sex::Union{Sex, Integer};
    stages::Vector{StageDefinition} = build_stages()
)
    isempty(ogive.bin_centers) && return Float64[]
    expected = cumulative_maturity_curves(
        ogive.bin_centers,
        target_sex;
        stages = stages
    )
    return [(ogive.prop_mature[i] - expected[i])^2 for i in eachindex(expected)]
end

"""
    maturity_ogive_mismatch(ogive, target_sex; stages=build_stages()) → Float64

RMSE between a cross-sectional maturity ogive and the matching cumulative
theoretical curve. See [`maturity_ogive_residuals`](@ref) for why the cumulative
curve is the like-for-like comparison. Returns `0.0` for an empty ogive, matching
the previous treatment of "no data" as "no error".
"""
function maturity_ogive_mismatch(
    ogive,
    target_sex::Union{Sex, Integer};
    stages::Vector{StageDefinition} = build_stages()
)
    sq_errors = maturity_ogive_residuals(ogive, target_sex; stages = stages)
    return isempty(sq_errors) ? 0.0 : sqrt(mean(sq_errors))
end

"""
    summarize_population(model)

Compute a statistical snapshot of the current living population in `model`,
including demographic stage distributions and stock biomass indicators.

# Returns
NamedTuple with fields:
- `total_organisms::Int`: Count of alive organisms.
- `total_biomass_kg::Float64`: Total wet biomass of living organisms in kilograms.
- `female_ssb_kg::Float64`: Spawning stock biomass of mature females (kg).
- `male_mature_biomass_kg::Float64`: Biomass of mature males (kg).
- `total_ssb_kg::Float64`: Total spawning stock biomass (mature males + females) in kg.
- `legal_biomass_kg::Float64`: Commercially fishable biomass of legal males in kg.
- `mean_mass::Float64`: Mean body mass across the population in grams.
- `n_males::Int`: Total males.
- `n_females::Int`: Total females.
- `n_adults::Int`: Total mature adults.
- `n_adult_males::Int`: Total adult males.
- `n_adult_females::Int`: Total adult females.
- `n_juveniles::Int`: Total benthic juveniles.
- `n_larvae::Int`: Total planktonic larvae (zoea + megalopa).
- `mean_size::Float64`: Mean size (mm).
- `size_by_stage::Dict{Int, Float64}`: Mean size mapped by stage.
- `count_by_stage::Dict{Int, Int}`: Organism count mapped by stage.
"""
function summarize_population(model)
    orgs = collect(allagents(model))
    total = length(orgs)

    props = abmproperties(model)
    tot_spawned = get(props, :total_eggs_spawned, 0)
    tot_settled = get(props, :total_recruits_settled, 0)

    if total == 0
        return (
            total_organisms = 0,
            total_biomass_kg = 0.0,
            female_ssb_kg = 0.0,
            male_mature_biomass_kg = 0.0,
            total_ssb_kg = 0.0,
            legal_biomass_kg = 0.0,
            total_mature_females = 0,
            total_mature_males = 0,
            total_fishable_male_mass_kg = 0.0,
            total_mature_male_mass_kg = 0.0,
            total_mature_female_mass_kg = 0.0,
            total_egg_production = tot_spawned,
            mean_mass = 0.0,
            n_males = 0,
            n_females = 0,
            n_adults = 0,
            n_adult_males = 0,
            n_adult_females = 0,
            n_ovigerous_females = 0,
            total_eggs_brooded = 0,
            total_eggs_spawned = tot_spawned,
            total_recruits_settled = tot_settled,
            n_juveniles = 0,
            n_larvae = 0,
            mean_size = 0.0,
            size_by_stage = Dict{Int, Float64}(),
            count_by_stage = Dict{Int, Int}(),
            mass_by_stage = Dict{Int, Float64}(),
            size_by_stage_male = Dict{Int, Float64}(),
            count_by_stage_male = Dict{Int, Int}(),
            mass_by_stage_male = Dict{Int, Float64}(),
            size_by_stage_female = Dict{Int, Float64}(),
            count_by_stage_female = Dict{Int, Int}(),
            mass_by_stage_female = Dict{Int, Float64}()
        )
    end

    n_m = count(c -> c.sex === male, orgs)
    n_f = count(c -> c.sex === female, orgs)
    # Maturity-based counts. These are the reproductive classes, so they exclude
    # large individuals that never passed the maturity process.
    n_adm = count(is_mature_male, orgs)
    n_adf = count(is_mature_female, orgs)
    n_ad = n_adm + n_adf
    n_ovig = count(is_ovigerous, orgs)
    tot_eggs = sum(c.egg_count for c in orgs; init = 0)
    adult_sz = model.params.organism.adult_size_threshold
    n_juv = count(c -> !is_adult(c) && c.stage >= 4, orgs)
    n_larv = count(c -> c.stage < 4, orgs)

    count_by_aggregate_stage = Dict{Int, Int}()
    size_by_aggregate_stage = Dict{Int, Float64}()
    mass_by_aggregate_stage = Dict{Int, Float64}()
    n_stage_0 = 0
    n_stages_1_to_5 = 0

    if haskey(props, :aggregate_populations)
        agg_groups = props[:aggregate_populations].groups
        n_stage_0 = round(Int, sum(g.abundance for g in agg_groups if g.stage == 0; init = 0.0))
        n_stages_1_to_5 = round(Int, sum(g.abundance for g in agg_groups if g.stage in 1:5; init = 0.0))
        tot_eggs += n_stage_0
        n_larv += round(Int, sum(g.abundance for g in agg_groups if g.stage in (1, 2, 3); init = 0.0))
        n_juv += round(Int, sum(g.abundance for g in agg_groups if g.stage in (4, 5); init = 0.0))

        for stg in 0:5
            stg_grps = filter(g -> g.stage == stg, agg_groups)
            tot_ab = sum(g.abundance for g in stg_grps; init = 0.0)
            count_by_aggregate_stage[stg] = round(Int, tot_ab)
            if tot_ab > 0
                size_by_aggregate_stage[stg] = sum(g.size * g.abundance for g in stg_grps) / tot_ab
                mass_by_aggregate_stage[stg] = sum(g.mass * g.abundance for g in stg_grps) / tot_ab
            else
                size_by_aggregate_stage[stg] = 0.0
                mass_by_aggregate_stage[stg] = 0.0
            end
        end
    end

    mean_size = isempty(orgs) ? 0.0 : mean(c.size for c in orgs)
    mean_wt = isempty(orgs) ? 0.0 : mean(c.mass for c in orgs)

    # Stock biomass metrics (converted from grams to kilograms).
    # Spawning stock uses the *mature* classes, because only mature females
    # extrude broods. Counting large-but-immature individuals here inflated the
    # stock and biased the surplus-production term that divides by it.
    tot_biomass = sum(c.mass for c in orgs) / 1000.0
    f_ssb = sum(mass_if_mature_female, orgs) / 1000.0
    m_mat_b = sum(mass_if_mature_male, orgs) / 1000.0
    tot_ssb = f_ssb + m_mat_b

    legal_sz = model.params.fishery.legal_size_male
    n_leg = count(c -> is_legal_male(c, legal_sz, adult_sz), orgs)
    leg_b = sum(c -> mass_if_legal_male(c, legal_sz, adult_sz), orgs) / 1000.0

    all_stages = sort(unique(vcat(collect(keys(count_by_aggregate_stage)), [c.stage for c in orgs])))
    size_by_stage = Dict{Int, Float64}()
    count_by_stage = Dict{Int, Int}()
    mass_by_stage = Dict{Int, Float64}()

    size_by_stage_male = Dict{Int, Float64}()
    count_by_stage_male = Dict{Int, Int}()
    mass_by_stage_male = Dict{Int, Float64}()

    size_by_stage_female = Dict{Int, Float64}()
    count_by_stage_female = Dict{Int, Int}()
    mass_by_stage_female = Dict{Int, Float64}()

    for stg in all_stages
        stg_orgs = filter(c -> c.stage == stg, orgs)
        stg_m = filter(c -> c.stage == stg && c.sex === male, orgs)
        stg_f = filter(c -> c.stage == stg && c.sex === female, orgs)

        agg_cnt = get(count_by_aggregate_stage, stg, 0)
        agg_sz = get(size_by_aggregate_stage, stg, 0.0)
        agg_wt = get(mass_by_aggregate_stage, stg, 0.0)

        n_ind = length(stg_orgs)
        tot_cnt = n_ind + agg_cnt
        count_by_stage[stg] = tot_cnt

        tot_sz = sum(c.size for c in stg_orgs; init = 0.0) + (agg_sz * agg_cnt)
        size_by_stage[stg] = (tot_cnt > 0) ? (tot_sz / tot_cnt) : 0.0

        tot_wt = sum(c.mass for c in stg_orgs; init = 0.0) + (agg_wt * agg_cnt)
        mass_by_stage[stg] = (tot_cnt > 0) ? (tot_wt / tot_cnt) : 0.0

        agg_m = round(Int, agg_cnt / 2)
        agg_f = agg_cnt - agg_m

        tot_m = length(stg_m) + agg_m
        tot_f = length(stg_f) + agg_f
        count_by_stage_male[stg] = tot_m
        count_by_stage_female[stg] = tot_f

        sz_m = sum(c.size for c in stg_m; init = 0.0) + (agg_sz * agg_m)
        sz_f = sum(c.size for c in stg_f; init = 0.0) + (agg_sz * agg_f)
        size_by_stage_male[stg] = (tot_m > 0) ? (sz_m / tot_m) : 0.0
        size_by_stage_female[stg] = (tot_f > 0) ? (sz_f / tot_f) : 0.0

        wt_m = sum(c.mass for c in stg_m; init = 0.0) + (agg_wt * agg_m)
        wt_f = sum(c.mass for c in stg_f; init = 0.0) + (agg_wt * agg_f)
        mass_by_stage_male[stg] = (tot_m > 0) ? (wt_m / tot_m) : 0.0
        mass_by_stage_female[stg] = (tot_f > 0) ? (wt_f / tot_f) : 0.0
    end

    return (
        total_organisms = total,
        total_biomass_kg = tot_biomass,
        female_ssb_kg = f_ssb,
        male_mature_biomass_kg = m_mat_b,
        total_ssb_kg = tot_ssb,
        legal_biomass_kg = leg_b,
        legal_numbers = n_leg,
        n_legal_males = n_leg,
        total_mature_females = n_adf,
        total_mature_males = n_adm,
        total_fishable_male_mass_kg = leg_b,
        total_mature_male_mass_kg = m_mat_b,
        total_mature_female_mass_kg = f_ssb,
        total_egg_production = tot_spawned,
        mean_mass = mean_wt,
        n_males = n_m,
        n_females = n_f,
        n_adults = n_ad,
        n_adult_males = n_adm,
        n_adult_females = n_adf,
        n_ovigerous_females = n_ovig,
        total_eggs_brooded = tot_eggs,
        total_eggs_spawned = tot_spawned,
        total_recruits_settled = tot_settled,
        n_juveniles = n_juv,
        n_larvae = n_larv,
        mean_size = mean_size,
        n_stage_0 = n_stage_0,
        n_stages_1_to_5 = n_stages_1_to_5,
        count_by_aggregate_stage = count_by_aggregate_stage,
        size_by_aggregate_stage = size_by_aggregate_stage,
        mass_by_aggregate_stage = mass_by_aggregate_stage,
        size_by_stage = size_by_stage,
        count_by_stage = count_by_stage,
        mass_by_stage = mass_by_stage,
        size_by_stage_male = size_by_stage_male,
        count_by_stage_male = count_by_stage_male,
        mass_by_stage_male = mass_by_stage_male,
        size_by_stage_female = size_by_stage_female,
        count_by_stage_female = count_by_stage_female,
        mass_by_stage_female = mass_by_stage_female
    )
end

"""
    extract_stock_recruitment(model)

Extract empirical stock-recruitment observations recorded within `model`.

If individual clutch recruitment records exist in
`abmproperties(model)[:clutch_recruitment_records]`, they are returned as a `DataFrame`.

# Returns
`DataFrame` with columns:
- `time`: Hatch timestamp (days).
- `spawners`: Maternal egg proxy in thousands (`eggs_count / 1000.0`).
- `eggs_count`: Total eggs in clutch.
- `mother_mass`: Maternal body mass in grams.
- `recruits`: Settled Stage 4 benthic recruits from the clutch.
- `recruits_per_spawner`: Ratio of recruits produced per unit spawner.
- `density_feedback`: Density-dependent mortality multiplier at the time of
  settlement, `1 + (N/K)^exponent`.
"""
function extract_stock_recruitment(model)
    props = abmproperties(model)
    if haskey(props, :clutch_recruitment_records)
        raw_df = props[:clutch_recruitment_records]
        if nrow(raw_df) > 0
            df = DataFrame(
                time = raw_df.hatch_time,
                spawners = raw_df.egg_count ./ 1000.0,
                egg_count = raw_df.egg_count,
                mother_size = raw_df.mother_size,
                mother_mass = raw_df.mother_mass,
                recruits = raw_df.recruits_settled,
                density_feedback = raw_df.density_feedback
            )
            df.recruits_per_spawner = [
                (df.spawners[i] > 0.0) ? (df.recruits[i] / df.spawners[i]) : 0.0
                for i in 1:nrow(df)
            ]
            return df
        end
    end

    return DataFrame(
        time = Float64[],
        spawners = Float64[],
        egg_count = Int[],
        mother_size = Float64[],
        mother_mass = Float64[],
        recruits = Int[],
        density_feedback = Float64[],
        recruits_per_spawner = Float64[]
    )
end

"""
    extract_stock_recruitment(agent_df::DataFrame, model_df::DataFrame; lag_days=nothing, window_days=nothing, days_per_year=365.0)

Derive population-level annual stock-recruitment timeseries from simulation dataframes.

For each annual cohort window `[t, t + window_days]`, maternal spawner proxy (eggs spawned)
is paired with the resulting benthic recruitment settling in the subsequent lagged window
`[t + lag_days, t + lag_days + window_days]`.

# Arguments
- `agent_df::DataFrame`: Time-stepped agent data collected during simulation.
- `model_df::DataFrame`: Time-stepped model-level data (`:total_eggs_spawned`, `:total_recruits_settled`).
- `lag_days::Real`: Egg incubation plus larval development lag. Defaults to
  `days_per_year`, so the cohort windows stay aligned with the model's calendar
  under a leap-year configuration.
- `window_days::Real`: Annual aggregation window (default: `days_per_year`).
- `days_per_year::Real`: Calendar length to use when either window is not given.

# Returns
`DataFrame` with columns:
- `year::Int`: Spawning cohort index (1, 2, ...).
- `time_spawn::Float64`: Spawning cohort start time.
- `spawners::Float64`: Thousands of eggs spawned in the cohort window.
- `eggs_spawned::Int`: Total eggs extruded in the cohort window.
- `recruits::Int`: Net new Stage 4 recruits settled during the lagged window.
- `recruits_per_spawner::Float64`: Ratio of recruits per unit spawner.
"""
function extract_stock_recruitment(
    agent_df::DataFrame,
    model_df::DataFrame;
    lag_days::Union{Nothing, Real} = nothing,
    window_days::Union{Nothing, Real} = nothing,
    days_per_year::Real = 365.0
)
    # The cohort windows are calendar-year windows, so they have to be the
    # model's calendar. Both defaults used to be the literal 365.0, which silently
    # misaligned every cohort under a leap-year configuration.
    lag = lag_days === nothing ? Float64(days_per_year) : Float64(lag_days)
    window = window_days === nothing ? Float64(days_per_year) : Float64(window_days)
    if nrow(model_df) < 2
        return DataFrame(
            year = Int[],
            time_spawn = Float64[],
            spawners = Float64[],
            eggs_spawned = Int[],
            recruits = Int[],
            recruits_per_spawner = Float64[]
        )
    end

    t_max = maximum(model_df.time)
    n_cohorts = floor(Int, (t_max - lag) / window)

    res_year = Int[]
    res_t_spawn = Float64[]
    res_spawners = Float64[]
    res_eggs = Int[]
    res_recruits = Int[]
    res_rps = Float64[]

    for y in 1:n_cohorts
        t_spawn_start = (y - 1) * window
        t_spawn_end = y * window
        t_rec_start = t_spawn_start + lag
        t_rec_end = t_spawn_end + lag

        sub_spawn = filter(r -> t_spawn_start <= r.time <= t_spawn_end, model_df)
        sub_rec = filter(r -> t_rec_start <= r.time <= t_rec_end, model_df)

        if nrow(sub_spawn) >= 1 && nrow(sub_rec) >= 1
            eggs_start = first(sub_spawn).total_eggs_spawned
            eggs_end = last(sub_spawn).total_eggs_spawned
            delta_eggs = max(0, eggs_end - eggs_start)

            rec_start = first(sub_rec).total_recruits_settled
            rec_end = last(sub_rec).total_recruits_settled
            delta_rec = max(0, rec_end - rec_start)

            s_val = delta_eggs / 1000.0
            rps_val = (s_val > 0.0) ? (delta_rec / s_val) : 0.0

            if delta_eggs > 0 || delta_rec > 0
                push!(res_year, y)
                push!(res_t_spawn, t_spawn_start)
                push!(res_spawners, s_val)
                push!(res_eggs, delta_eggs)
                push!(res_recruits, delta_rec)
                push!(res_rps, rps_val)
            end
        end
    end

    return DataFrame(
        year = res_year,
        time_spawn = res_t_spawn,
        spawners = res_spawners,
        eggs_spawned = res_eggs,
        recruits = res_recruits,
        recruits_per_spawner = res_rps
    )
end

"""
    fit_stock_recruitment(spawners, recruits; model_type=:both)

Fit classical fisheries stock-recruitment relationships to empirical `(spawners, recruits)`
observations using analytical least-squares transformations.

# Formulations
1. **Beverton-Holt Model**:
   ``R(S) = \\frac{\\alpha S}{1 + \\beta S}``
   Linearized via reciprocal transformation: ``\\frac{S}{R} = a + b S``,
   where ``\\alpha = 1/a`` and ``\\beta = b/a``.
   - Asymptotic maximum recruitment: ``R_{\\max} = \\alpha / \\beta``.
   - Half-saturation spawner abundance: ``S_{50} = 1 / \\beta``.

2. **Ricker Model**:
   ``R(S) = \\alpha S \\exp(-\\beta S)``
   Linearized via logarithmic transformation: ``\\ln\\left(\\frac{R}{S}\\right) = u - v S``,
   where ``\\alpha = \\exp(u)`` and ``\\beta = v``.
   - Spawner abundance producing peak recruitment: ``S_{\\max} = 1 / \\beta``.
   - Peak recruitment: ``R_{\\max} = \\frac{\\alpha}{\\beta e}``.

# Arguments
- `spawners::AbstractVector{<:Real}`: Vector of spawner proxies (biomass in kg or 10³ eggs).
- `recruits::AbstractVector{<:Real}`: Vector of recruit counts.
- `model_type::Symbol`: `:beverton_holt`, `:ricker`, or `:both` (default: `:both`).

# Returns
NamedTuple containing fitted parameters, goodness-of-fit metrics (``R^2``, AIC),
predictor functions, and optimal model selection.
"""
function fit_stock_recruitment(
    spawners::AbstractVector{<:Real},
    recruits::AbstractVector{<:Real};
    model_type::Symbol = :both
)
    valid_idx = findall(
        i -> spawners[i] > 0.0 && recruits[i] > 0.0,
        eachindex(spawners, recruits)
    )
    n = length(valid_idx)

    if n < 3
        s_mean = isempty(spawners) ? 1.0 : mean(spawners)
        r_mean = isempty(recruits) ? 1.0 : mean(recruits)
        fallback_alpha = (s_mean > 0.0) ? (r_mean / s_mean) : 1.0

        bh_pred = s -> fallback_alpha * s
        rk_pred = s -> fallback_alpha * s

        return (
            n_obs = n,
            is_fitted = false,
            beverton_holt = (
                alpha = fallback_alpha, beta = 0.0, R_max = Inf, S_50 = Inf,
                R2 = 0.0, aic = 0.0, predict = bh_pred
            ),
            ricker = (
                alpha = fallback_alpha, beta = 0.0, S_max = Inf, R_max = Inf,
                R2 = 0.0, aic = 0.0, predict = rk_pred
            ),
            best_model = :none
        )
    end

    s_sub = Float64.(spawners[valid_idx])
    r_sub = Float64.(recruits[valid_idx])
    r_mean = mean(r_sub)
    ss_tot = sum((r - r_mean)^2 for r in r_sub)
    ss_tot = (ss_tot > 1e-12) ? ss_tot : 1.0

    # 1. Beverton-Holt fit: S/R = a + b S
    y_bh = s_sub ./ r_sub
    s_bar = mean(s_sub)
    y_bh_bar = mean(y_bh)
    cov_s_ybh = sum((s_sub[i] - s_bar) * (y_bh[i] - y_bh_bar) for i in 1:n)
    var_s = sum((s - s_bar)^2 for s in s_sub)
    var_s = (var_s > 1e-12) ? var_s : 1.0

    b_bh = cov_s_ybh / var_s
    a_bh = y_bh_bar - b_bh * s_bar

    alpha_bh = (a_bh > 1e-6) ? (1.0 / a_bh) : (r_mean / s_bar)
    beta_bh = (a_bh > 1e-6 && b_bh > 0.0) ? (b_bh / a_bh) : 0.0

    bh_pred = s -> (alpha_bh * s) / (1.0 + beta_bh * s)
    ss_res_bh = sum((r_sub[i] - bh_pred(s_sub[i]))^2 for i in 1:n)
    r2_bh = (ss_tot > 0.0) ? max(0.0, 1.0 - (ss_res_bh / ss_tot)) : 0.0
    aic_bh = n * log(max(1e-12, ss_res_bh / n)) + 4.0
    r_max_bh = (beta_bh > 0.0) ? (alpha_bh / beta_bh) : Inf
    s_50_bh = (beta_bh > 0.0) ? (1.0 / beta_bh) : Inf

    # 2. Ricker fit: ln(R/S) = u - v S
    y_rk = log.(r_sub ./ s_sub)
    y_rk_bar = mean(y_rk)
    cov_s_yrk = sum((s_sub[i] - s_bar) * (y_rk[i] - y_rk_bar) for i in 1:n)
    v_rk = -cov_s_yrk / var_s
    u_rk = y_rk_bar + v_rk * s_bar

    alpha_rk = exp(u_rk)
    beta_rk = v_rk

    rk_pred = s -> alpha_rk * s * exp(-beta_rk * s)
    ss_res_rk = sum((r_sub[i] - rk_pred(s_sub[i]))^2 for i in 1:n)
    r2_rk = (ss_tot > 0.0) ? max(0.0, 1.0 - (ss_res_rk / ss_tot)) : 0.0
    aic_rk = n * log(max(1e-12, ss_res_rk / n)) + 4.0
    s_max_rk = (beta_rk > 0.0) ? (1.0 / beta_rk) : Inf
    r_max_rk = (beta_rk > 0.0) ? (alpha_rk / (beta_rk * exp(1.0))) : Inf

    best_m = if model_type === :beverton_holt
        :beverton_holt
    elseif model_type === :ricker
        :ricker
    else
        (aic_bh <= aic_rk) ? :beverton_holt : :ricker
    end

    return (
        n_obs = n,
        is_fitted = true,
        beverton_holt = (
            alpha = alpha_bh,
            beta = beta_bh,
            R_max = r_max_bh,
            S_50 = s_50_bh,
            R2 = r2_bh,
            aic = aic_bh,
            predict = bh_pred
        ),
        ricker = (
            alpha = alpha_rk,
            beta = beta_rk,
            S_max = s_max_rk,
            R_max = r_max_rk,
            R2 = r2_rk,
            aic = aic_rk,
            predict = rk_pred
        ),
        best_model = best_m
    )
end

"""
    summarize_stock_recruitment(sr_df; fits=nothing)

Produce a formatted textual summary of stock-recruitment dynamics and parameter estimates.
"""
function summarize_stock_recruitment(sr_df::DataFrame; fits = nothing)
    n = nrow(sr_df)
    if n == 0
        return "No stock-recruitment records available for analysis."
    end

    actual_fits = (fits === nothing) ?
        fit_stock_recruitment(sr_df.spawners, sr_df.recruits) : fits

    buf = IOBuffer()
    println(buf, "="^70)
    println(buf, "       ORGANISM STOCK-RECRUITMENT ANALYSIS REPORT       ")
    println(buf, "="^70)
    println(buf, " Observations (cohorts/clutches) : ", n)
    println(buf, " Mean Spawner Index              : ", round(mean(sr_df.spawners), digits = 2))
    println(buf, " Mean Recruitment                : ", round(mean(sr_df.recruits), digits = 2), " recruits")
    println(buf, " Mean Recruits / Spawner         : ", round(mean(sr_df.recruits_per_spawner), digits = 3))
    println(buf, "-"^70)

    if actual_fits.is_fitted
        bh = actual_fits.beverton_holt
        rk = actual_fits.ricker
        println(buf, " [Beverton-Holt Model: R = α·S / (1 + β·S)]")
        println(buf, "   - Alpha (initial slope α)   : ", round(bh.alpha, digits = 4))
        println(buf, "   - Beta  (capacity param β)  : ", round(bh.beta, digits = 6))
        println(buf, "   - Asymptotic Max Recruits   : ", isfinite(bh.R_max) ? round(bh.R_max, digits = 1) : "Inf")
        println(buf, "   - Half-Saturation Stock S50 : ", isfinite(bh.S_50) ? round(bh.S_50, digits = 2) : "Inf")
        println(buf, "   - Coefficient of Determination R²: ", round(bh.R2, digits = 3))
        println(buf, "   - AIC Score                 : ", round(bh.aic, digits = 2))
        println(buf, "-"^70)

        println(buf, " [Ricker Model: R = α·S·exp(-β·S)]")
        println(buf, "   - Alpha (density-indep rate): ", round(rk.alpha, digits = 4))
        println(buf, "   - Beta  (overcompensation)  : ", round(rk.beta, digits = 6))
        println(buf, "   - Peak Spawner Stock S_max  : ", isfinite(rk.S_max) ? round(rk.S_max, digits = 2) : "Inf")
        println(buf, "   - Maximum Recruitment R_max : ", isfinite(rk.R_max) ? round(rk.R_max, digits = 1) : "Inf")
        println(buf, "   - Coefficient of Determination R²: ", round(rk.R2, digits = 3))
        println(buf, "   - AIC Score                 : ", round(rk.aic, digits = 2))
        println(buf, "-"^70)
        println(buf, " Preferred Model by AIC        : ", uppercase(string(actual_fits.best_model)))
    else
        println(buf, " Insufficient sample size for parametric non-linear curve fitting (N < 3).")
    end
    println(buf, "="^70)

    return String(take!(buf))
end

# ==============================================================================
# Simulation Results File Persistence & Annual Production Extraction
# ==============================================================================

"""
    write_dataframe_csv(filepath::String, df::DataFrame)

Write a `DataFrame` to `filepath` in standard comma-separated values (CSV) format
without external dependencies.
"""
function write_dataframe_csv(filepath::String, df::DataFrame)
    open(filepath, "w") do io
        col_names = names(df)
        println(io, join(col_names, ","))
        for row in eachrow(df)
            vals = [string(row[c]) for c in col_names]
            println(io, join(vals, ","))
        end
    end
    return filepath
end

"""
    read_dataframe_csv(filepath::String)

Read a numeric CSV file into a `DataFrame` without external dependencies.
"""
function read_dataframe_csv(filepath::String)
    lines = readlines(filepath)
    filter!(!isempty, lines)
    if isempty(lines)
        return DataFrame()
    end
    header = [String(strip(h)) for h in split(lines[1], ',')]
    data = Dict{String, Vector{Any}}()
    for h in header
        data[h] = Any[]
    end
    for line in lines[2:end]
        parts = split(line, ',')
        for (i, h) in enumerate(header)
            raw = (i <= length(parts)) ? strip(parts[i]) : ""
            val = tryparse(Int, raw)
            if val === nothing
                val = tryparse(Float64, raw)
            end
            push!(data[h], val === nothing ? 0.0 : val)
        end
    end
    df = DataFrame()
    for h in header
        col = data[h]
        if all(x -> x isa Int, col)
            df[!, Symbol(h)] = Int.(col)
        else
            df[!, Symbol(h)] = Float64.(col)
        end
    end
    return df
end

"""
    save_simulation_results(filepath::String, agent_df::DataFrame, model_df::DataFrame;
                            dt_year::Float64 = 365.0,
                            frequency_days::Float64 = dt_year,
                            snapshot_day::Union{Nothing, Float64} = 274.0,
                            meta::Dict = Dict{Symbol, Any}())

Save annual simulation output tables to disk in CSV format for post-hoc macro-ecological
analyses, sampling the population at user-specified output frequency in days (`frequency_days`,
default: 365.0 days = 1 year) and day-of-year snapshot (`snapshot_day`, default: 274.0 = ~1 Sep).

# Arguments
- `filepath::String`: Target file path (typically ending in `.csv`).
- `agent_df::DataFrame`: Agent time-series table.
- `model_df::DataFrame`: Model-level time-series table.
- `dt_year::Float64`: Days per calendar year (default: 365.0).
- `frequency_days::Float64`: Sampling interval in days (default: 365.0 = 1 year).
- `snapshot_day::Union{Nothing, Float64}`: Day of calendar year (1-365) to sample annual survey
  snapshots (default: 274.0, corresponding to ~1 September).
- `meta::Dict`: Optional metadata dictionary.

# Returns
`String` path of saved file.
"""
function save_simulation_results(
    filepath::String,
    agent_df::DataFrame,
    model_df::DataFrame;
    dt_year::Float64 = 365.0,
    frequency_days::Float64 = dt_year,
    snapshot_day::Union{Nothing, Float64} = 274.0,
    meta::Dict = Dict{Symbol, Any}()
)
    ann_df = extract_annual_production(
        agent_df,
        model_df;
        dt_year = dt_year,
        frequency_days = frequency_days,
        snapshot_day = snapshot_day
    )
    write_dataframe_csv(filepath, ann_df)
    return filepath
end

"""
    load_simulation_results(filepath::String)

Load previously saved simulation results from `filepath` in CSV format.

# Returns
`DataFrame` containing annual production timeseries.
"""
function load_simulation_results(filepath::String)
    if !isfile(filepath)
        error("Simulation results file not found: $(filepath)")
    end
    return read_dataframe_csv(filepath)
end

"""
    ANNUAL_PRODUCTION_SCHEMA

Authoritative column order and element type of the `extract_annual_production`
output. Both the empty and the populated path assemble their `DataFrame` from
this, so a run too short to produce a sample still has the same schema as a full
one. Previously the empty path returned 14 columns and the populated path 17, so
`.day` failed only on the runs you were already debugging.
"""
const ANNUAL_PRODUCTION_SCHEMA = (
    # `sample_index`, not `year`: it counts snapshots, so it reads 1, 2, 3...
    # regardless of the calendar date a run starts on. The calendar position of
    # a snapshot is `day` (day of year) plus `time` (elapsed days).
    (:sample_index, Int),
    (:day, Float64),
    (:time, Float64),
    (:total_numbers, Float64),
    (:total_biomass_kg, Float64),
    (:legal_numbers, Float64),
    (:mature_females, Float64),
    (:mature_males, Float64),
    (:fishable_male_biomass_kg, Float64),
    (:mature_male_biomass_kg, Float64),
    (:mature_female_biomass_kg, Float64),
    (:total_egg_production, Float64),
    (:annual_egg_production, Float64),
    (:catch_total, Float64),
    (:catch_legal, Float64),
    (:surplus_production_total, Float64),
    (:surplus_production_legal, Float64)
)

"""
    empty_annual_production() → DataFrame

Zero-row `DataFrame` carrying the full annual production schema.
"""
function empty_annual_production()
    return DataFrame(
        [name => Vector{T}() for (name, T) in ANNUAL_PRODUCTION_SCHEMA]
    )
end

"""
    annual_production_targets(t_max::Real; dt_year = 365.0,
                              frequency_days = dt_year,
                              snapshot_day = 274.0) → Vector{Float64}

Elapsed times at which [`extract_annual_production`](@ref) takes its snapshots.

When `frequency_days` equals `dt_year` the result is one time per year at
`snapshot_day`. When `frequency_days` is sub-annual the result starts one
interval in and steps by `frequency_days`. The regime depends only on
`frequency_days`, so an explicit `snapshot_day` always takes effect.

Shared with the live-snapshot progress reporting in the CLI script, so the
printed progress lines land on the same instants as the data rows.
"""
function annual_production_targets(
    t_max::Real;
    dt_year::Real = 365.0,
    frequency_days::Real = dt_year,
    snapshot_day::Union{Nothing, Real} = 274.0
)
    freq = (frequency_days > 0.0) ? Float64(frequency_days) : Float64(dt_year)
    is_annual = isapprox(freq, dt_year; rtol = 0.0, atol = 1e-9)
    first_day = if is_annual
        (snapshot_day === nothing) ? Float64(dt_year) : Float64(snapshot_day)
    else
        freq
    end

    targets = Float64[]
    t_curr = first_day
    while t_curr <= t_max
        push!(targets, t_curr)
        t_curr += freq
    end
    return targets
end

"""
    extract_annual_production(agent_df::DataFrame, model_df::DataFrame;
                              dt_year::Float64 = 365.0,
                              frequency_days::Float64 = dt_year,
                              snapshot_day::Union{Nothing, Float64} = 274.0)

Aggregate simulation output into abundance, demographic components, commercial catches,
and realized surplus production metrics sampled at specified intervals.

Sampling times are governed by `frequency_days` (default: 365.0 days = 1 year) and
`snapshot_day` (default: 274.0 = ~1 September). When `frequency_days` equals
`dt_year`, one snapshot is taken per year at day `snapshot_day`
(t = (yr - 1) * dt_year + snapshot_day), and the `day` column is that same
`snapshot_day` for every row. When `frequency_days` is sub-annual, snapshots are
taken every `frequency_days` days starting one interval in, and `day` is the
genuine day of year for each sample. The choice of regime depends only on
`frequency_days`, so passing `snapshot_day` explicitly always takes effect.

# Arguments
- `agent_df::DataFrame`: Agent time-series metrics.
- `model_df::DataFrame`: Model time-series metrics.
- `dt_year::Float64`: Days per annual time step (default: 365.0).
- `frequency_days::Float64`: Output sampling frequency in days (default: 365.0 = 1 year).
- `snapshot_day::Union{Nothing, Float64}`: Day of calendar year (1-365) to sample snapshots
  (default: 274.0 = ~1 September).

# Returns
`DataFrame` with columns:
- `sample_index::Int`: Snapshot counter (1, 2, ...). Not a calendar year: it counts
  samples, so a run starting on any date reads the same.
- `day::Float64`: Calendar day of year at each snapshot.
- `time::Float64`: Elapsed days at snapshot.
- `total_numbers::Float64`: Total living organism abundance (N_t).
- `total_biomass_kg::Float64`: Total wet mass of the population in kg.
- `legal_numbers::Float64`: Legal-sized adult male abundance (N_{legal, t}).
- `mature_females::Float64`: Total number of morphometrically mature females.
- `mature_males::Float64`: Total number of morphometrically mature males.
- `fishable_male_biomass_kg::Float64`: Total wet mass of legal fishable males in kg.
- `mature_male_biomass_kg::Float64`: Total wet mass of mature males in kg.
- `mature_female_biomass_kg::Float64`: Total wet mass of mature females (SSB_F) in kg.
- `total_egg_production::Float64`: Cumulative egg production across the stock.
- `annual_egg_production::Float64`: Egg production during the interval.
- `catch_total::Float64`: Harvest removal during interval in numbers (C_t).
- `catch_legal::Float64`: Commercial catch of legal males during interval.
  Identical to `catch_total` in this model, because the fishery removes an animal
  only when it passes the legal-size test.
- `surplus_production_total::Float64`: Realized production (N_{t+1} - N_t + C_t).
- `surplus_production_legal::Float64`: Realized legal production.

The empty and populated paths are both built from `ANNUAL_PRODUCTION_SCHEMA`, so
a short run returns the same columns as a long one.
"""
function extract_annual_production(
    agent_df::DataFrame,
    model_df::DataFrame;
    dt_year::Float64 = 365.0,
    frequency_days::Float64 = dt_year,
    snapshot_day::Union{Nothing, Float64} = 274.0
)
    t_max = maximum(agent_df.time)
    freq = (frequency_days > 0.0) ? frequency_days : dt_year
    t_targets = annual_production_targets(
        t_max;
        dt_year = dt_year,
        frequency_days = frequency_days,
        snapshot_day = snapshot_day
    )
    n_samples = length(t_targets)

    if n_samples < 1
        return empty_annual_production()
    end

    num_col = if :length_stage in propertynames(agent_df)
        :length_stage
    elseif :count_stage in propertynames(agent_df)
        :count_stage
    elseif :total_organisms in propertynames(agent_df)
        :total_organisms
    elseif :total_numbers in propertynames(agent_df)
        :total_numbers
    else
        first(names(agent_df))
    end

    tot_mass_col = if :sum_mass in propertynames(agent_df)
        :sum_mass
    elseif :sum_weight in propertynames(agent_df)
        :sum_weight
    elseif :total_biomass_kg in propertynames(agent_df)
        :total_biomass_kg
    else
        nothing
    end

    # No fallback to an adult-male count here: the legal-male count is defined by
    # the configured legal size, and substituting a size-threshold count for it
    # would report a number that ignores the fishery regulation.
    legal_col = if :count_is_legal_male in propertynames(agent_df)
        :count_is_legal_male
    elseif :legal_numbers in propertynames(agent_df)
        :legal_numbers
    else
        nothing
    end

    mat_f_col = if :count_is_mature_female in propertynames(agent_df)
        :count_is_mature_female
    elseif :count_is_adult_female in propertynames(agent_df)
        :count_is_adult_female
    elseif :mature_females in propertynames(agent_df)
        :mature_females
    elseif :n_adult_females in propertynames(agent_df)
        :n_adult_females
    else
        nothing
    end

    mat_m_col = if :count_is_mature_male in propertynames(agent_df)
        :count_is_mature_male
    elseif :count_is_adult_male in propertynames(agent_df)
        :count_is_adult_male
    elseif :mature_males in propertynames(agent_df)
        :mature_males
    elseif :n_adult_males in propertynames(agent_df)
        :n_adult_males
    else
        nothing
    end

    leg_mass_col = if :sum_mass_if_legal_male in propertynames(agent_df)
        :sum_mass_if_legal_male
    elseif :fishable_male_biomass_kg in propertynames(agent_df)
        :fishable_male_biomass_kg
    elseif :legal_biomass_kg in propertynames(agent_df)
        :legal_biomass_kg
    else
        nothing
    end

    mat_m_mass_col = if :sum_mass_if_mature_male in propertynames(agent_df)
        :sum_mass_if_mature_male
    elseif :mature_male_biomass_kg in propertynames(agent_df)
        :mature_male_biomass_kg
    else
        nothing
    end

    mat_f_mass_col = if :sum_mass_if_mature_female in propertynames(agent_df)
        :sum_mass_if_mature_female
    elseif :mature_female_biomass_kg in propertynames(agent_df)
        :mature_female_biomass_kg
    elseif :female_ssb_kg in propertynames(agent_df)
        :female_ssb_kg
    else
        nothing
    end

    total_n_vec = Float64[]
    total_m_vec = Float64[]
    legal_n_vec = Float64[]
    mat_f_n_vec = Float64[]
    mat_m_n_vec = Float64[]
    leg_mass_vec = Float64[]
    mat_m_mass_vec = Float64[]
    mat_f_mass_vec = Float64[]
    time_vec = Float64[]
    year_vec = Int[]

    for k in 1:n_samples
        t_target = t_targets[k]
        idx = argmin(abs.(agent_df.time .- t_target))
        push!(year_vec, k)
        push!(time_vec, agent_df.time[idx])
        push!(total_n_vec, Float64(agent_df[idx, num_col]))

        tm_val = if tot_mass_col in (:sum_mass, :sum_weight)
            Float64(agent_df[idx, tot_mass_col]) / 1000.0
        elseif tot_mass_col !== nothing
            Float64(agent_df[idx, tot_mass_col])
        else
            0.0
        end
        push!(total_m_vec, tm_val)

        l_val = (legal_col !== nothing) ? Float64(agent_df[idx, legal_col]) : 0.0
        push!(legal_n_vec, l_val)

        mf_val = (mat_f_col !== nothing) ? Float64(agent_df[idx, mat_f_col]) : 0.0
        push!(mat_f_n_vec, mf_val)

        mm_val = (mat_m_col !== nothing) ? Float64(agent_df[idx, mat_m_col]) : 0.0
        push!(mat_m_n_vec, mm_val)

        lm_val = if leg_mass_col == :sum_mass_if_legal_male
            Float64(agent_df[idx, leg_mass_col]) / 1000.0
        elseif leg_mass_col !== nothing
            Float64(agent_df[idx, leg_mass_col])
        else
            0.0
        end
        push!(leg_mass_vec, lm_val)

        mmm_val = if mat_m_mass_col == :sum_mass_if_mature_male
            Float64(agent_df[idx, mat_m_mass_col]) / 1000.0
        elseif mat_m_mass_col !== nothing
            Float64(agent_df[idx, mat_m_mass_col])
        else
            0.0
        end
        push!(mat_m_mass_vec, mmm_val)

        mfm_val = if mat_f_mass_col == :sum_mass_if_mature_female
            Float64(agent_df[idx, mat_f_mass_col]) / 1000.0
        elseif mat_f_mass_col !== nothing
            Float64(agent_df[idx, mat_f_mass_col])
        else
            0.0
        end
        push!(mat_f_mass_vec, mfm_val)
    end

    catch_vec = zeros(Float64, n_samples)
    egg_prod_cum_vec = zeros(Float64, n_samples)
    egg_prod_ann_vec = zeros(Float64, n_samples)

    # 1. Handle Egg Production (Cumulative and Periodical/Annual)
    if :total_eggs_spawned in propertynames(model_df) || :total_egg_production in propertynames(model_df)
        cum_col = if :total_eggs_spawned in propertynames(model_df)
            :total_eggs_spawned
        else
            :total_egg_production
        end

        for k in 1:n_samples
            t_target = t_targets[k]
            idx = argmin(abs.(model_df.time .- t_target))
            egg_prod_cum_vec[k] = Float64(model_df[idx, cum_col])
        end
        egg_prod_ann_vec[1] = egg_prod_cum_vec[1]
        for k in 2:n_samples
            egg_prod_ann_vec[k] = max(0.0, egg_prod_cum_vec[k] - egg_prod_cum_vec[k - 1])
        end
    end

    # 2. Handle Catch (Periodical)
    if :total_catch_numbers in propertynames(model_df)
        cum_catches = Float64[]
        for k in 1:n_samples
            t_target = t_targets[k]
            idx = argmin(abs.(model_df.time .- t_target))
            push!(cum_catches, Float64(model_df.total_catch_numbers[idx]))
        end
        catch_vec[1] = cum_catches[1]
        for k in 2:n_samples
            catch_vec[k] = max(0.0, cum_catches[k] - cum_catches[k - 1])
        end
    end

    sp_total = zeros(Float64, n_samples)
    sp_legal = zeros(Float64, n_samples)
    for k in 1:(n_samples - 1)
        sp_total[k] = total_n_vec[k + 1] - total_n_vec[k] + catch_vec[k]
        sp_legal[k] = legal_n_vec[k + 1] - legal_n_vec[k] + catch_vec[k]
    end
    sp_total[n_samples] = sp_total[max(1, n_samples - 1)]
    sp_legal[n_samples] = sp_legal[max(1, n_samples - 1)]

    # Genuine day of year, so this is correct for annual and sub-annual sampling
    # alike. With an annual frequency it is constant at `snapshot_day`.
    day_vec = [mod(t, dt_year) == 0.0 ? dt_year : mod(t, dt_year) for t in time_vec]
    sample_index_vec = Int.(year_vec)

    cols = (
        sample_index = sample_index_vec,
        day = day_vec,
        time = time_vec,
        total_numbers = total_n_vec,
        total_biomass_kg = total_m_vec,
        legal_numbers = legal_n_vec,
        mature_females = mat_f_n_vec,
        mature_males = mat_m_n_vec,
        fishable_male_biomass_kg = leg_mass_vec,
        mature_male_biomass_kg = mat_m_mass_vec,
        mature_female_biomass_kg = mat_f_mass_vec,
        total_egg_production = egg_prod_cum_vec,
        annual_egg_production = egg_prod_ann_vec,
        catch_total = catch_vec,
        # The same series as `catch_total`, and that is not a shortcut. `agent_step!`
        # removes an animal through the fishery only when `is_legal_male` passes, so
        # every animal this model harvests is a legal male and the two catch series
        # cannot differ. They are reported separately because stock assessment
        # convention separates them, and a caller that assumes they are independent
        # would double-count the removals. If a configuration ever admits sublegal
        # harvest, this stops being true and the two columns must be computed
        # separately.
        catch_legal = catch_vec,
        surplus_production_total = sp_total,
        surplus_production_legal = sp_legal
    )

    df = DataFrame([name => cols[name] for (name, _) in ANNUAL_PRODUCTION_SCHEMA])

    # Propagate per-stage counts from the agent observations (count_stage_i)
    # into the annual table as stage_i columns, so extract_annual_stage_matrix
    # can use the real stage structure instead of reconstructing it (M12).
    for st in 0:30
        src_col = Symbol("count_stage_$(st)")
        dst_col = Symbol("stage_$(st + 1)")
        if src_col in propertynames(agent_df)
            stage_vec = Float64[]
            for k in 1:n_samples
                idx = argmin(abs.(agent_df.time .- t_targets[k]))
                push!(stage_vec, Float64(agent_df[idx, src_col]))
            end
            df[!, dst_col] = stage_vec
        end
    end

    return df
end

# ==============================================================================
# Optimization Utilities (Optim.jl)
# ==============================================================================

"""
    OBJECTIVE_PENALTY

Value [`safe_objective`](@ref) returns for a trial that threw or produced a
non-finite result. Chosen far above any real objective so a diverged trial always
loses the comparison. It doubles as the sentinel for "the search never found
anything but failures": a real optimum is never this large, so `fmin` at or above
it means every trial was penalised.
"""
const OBJECTIVE_PENALTY = 1e300

"""
    safe_objective(f; label = "objective", failures = nothing)

Wrap an objective so that `NaN` and `Inf` results are reported as a large finite
penalty. Optimizers need a totally ordered objective to compare candidates, and
a diverging population-trajectory simulation must lose the comparison rather
than poison the search with a non-orderable value.

An exception thrown by `f` is also mapped to the penalty, but it is recorded and
warned about once. The bare `catch` this replaces made a genuine bug in an
objective indistinguishable from an ordinary diverging trial: the search would
proceed, fail to converge anywhere, and report only that every starting point
failed.

Pass a `Vector` as `failures` to collect the exceptions for reporting by the
caller — `optimize_model` uses that to name the real cause when nothing converges.
"""
function safe_objective(f; label::AbstractString = "objective", failures = nothing)
    reported = Ref(false)
    return function (x...)
        value = try
            Float64(f(x...))
        catch err
            failures !== nothing && push!(failures, err)
            if !reported[]
                reported[] = true
                @warn "objective threw; treating this trial as a large penalty" label =
                    label parameters = x exception = err
            end
            NaN
        end
        return (isfinite(value) ? value : OBJECTIVE_PENALTY)
    end
end

"""
    _attempt_optimize(build) → NamedTuple

Run one optimizer invocation, converting a thrown exception into `ok = false`
rather than losing it. Callers aggregate the failures so the diagnostic names the
actual exception instead of reporting a bare "did not converge".
"""
function _attempt_optimize(build)
    try
        return (ok = true, result = build(), err = nothing)
    catch err
        return (ok = false, result = nothing, err = err)
    end
end

"""
    _failure_summary(failures) → String

One line describing why an optimization did not produce a usable result.
"""
function _failure_summary(failures)
    isempty(failures) && return "no optimizer invocation was attempted"
    msgs = unique(sprint(showerror, err) for err in failures)
    return "$(length(failures)) failed call(s): " * join(msgs, " | ")
end

"""
    optimize_model(f, lower, upper, starts; method=NelderMead(), kwargs...)

Minimize `f` over the box `[lower, upper]` using Optim.jl, restarting from every
point in `starts` and keeping the best solution.

Population-dynamics objectives are cheap but ragged and multi-modal: the
recursively simulated trajectories have kinks at the population floors and many
competing local minima. A single local descent therefore lands on an arbitrary
basin, which is why the fitting routines supply a coarse lattice of starting
points and let Optim refine from each one.

# Arguments
- `f`: objective called as `f(x::Vector{Float64})`.
- `lower`, `upper`: box bounds, one entry per parameter.
- `starts`: starting points, each a vector of the same length.
- `method`: any Optim method; defaults to derivative-free `NelderMead`, which
  suits the non-smooth residuals better than a gradient method.

# Returns
`(xmin, fmin)`: best parameters found and the objective value there.
"""
function optimize_model(
    f,
    lower::AbstractVector{<:Real},
    upper::AbstractVector{<:Real},
    starts::Vector{<:AbstractVector};
    method = NelderMead(),
    x_abstol::Float64 = 1e-10,
    f_reltol::Float64 = 1e-14,
    kwargs...
)
    objective_failures = Any[]
    objective = safe_objective(
        f;
        label = "optimize_model",
        failures = objective_failures
    )
    box = Fminbox(method)
    # Optim v2 takes convergence tolerances through an explicit Options object.
    options = Optim.Options(; x_abstol = x_abstol, f_reltol = f_reltol)
    best_x = collect(Float64, first(starts))
    best_f = Inf
    solved = 0
    call_failures = Any[]
    for start in starts
        x0 = clamp.(collect(Float64, start), lower, upper)
        attempt = _attempt_optimize() do
            optimize(objective, lower, upper, x0, box, options; kwargs...)
        end
        if !attempt.ok
            push!(call_failures, attempt.err)
            continue
        end
        # `minimum` and `minimizer` are pure accessors on a result Optim already
        # produced, so a failure here is a real defect rather than a diverging
        # trial and is allowed to propagate.
        value = Float64(Optim.minimum(attempt.result))
        if isfinite(value) && value < best_f
            candidate = collect(Float64, Optim.minimizer(attempt.result))
            best_f = value
            best_x = length(candidate) == length(best_x) ? candidate : best_x
            solved += 1
        end
    end
    if !isempty(call_failures)
        @warn "optimize_model: some starting points failed to optimize" failures =
            length(call_failures) starts = length(starts) detail = _failure_summary(
            call_failures
        )
    end
    if solved == 0
        # Name the real cause. When no call threw, the objective itself did, and
        # the recorded objective exception is the useful diagnostic.
        cause = isempty(call_failures) ?
                _failure_summary(objective_failures) : _failure_summary(call_failures)
        error(
            "optimize_model: no starting point produced a usable result. " *
            cause *
            " Check that the objective is finite over the box " *
            "[$(lower), $(upper)] and that the method $(nameof(typeof(method))) " *
            "is appropriate for $(length(lower)) parameter(s)."
        )
    end
    if best_f >= OBJECTIVE_PENALTY
        # Every trial was penalised, so the optimizer "converged" to a constant
        # and would otherwise return a meaningless parameter vector with a huge but
        # finite objective. Report it as the failure it is.
        cause = _failure_summary(objective_failures)
        error(
            "optimize_model: every trial was replaced by the divergence penalty, " *
            "so no solution was found. " *
            (isempty(objective_failures) ?
            "The objective returned a non-finite value at every point tried." :
            cause) *
            " Check that the objective is finite over the box " *
            "[$(lower), $(upper)]."
        )
    end
    return (best_x, best_f)
end

"""
    optimize_blocks(build_objective, blocks, x0, block_bounds; passes=3, kwargs...)

Coordinate-descent driver over groups of parameters.

High-dimensional biological models are usually over-parameterized to be
identified jointly, so optimizing every coefficient at once is both slow and
unstable. Instead the coefficients are partitioned into biologically meaningful
blocks (early stages, males, females, fecundity), each block is optimized with
`optimize_model`, and the sweeps repeat until the objective stops improving.
This keeps every sub-problem low-dimensional while still using all the data.

# Arguments
- `build_objective`: `f(x) -> value` over the full concatenated parameter vector.
- `blocks`: vector of index vectors, one per block.
- `x0`: initial full parameter vector.
- `block_bounds`: vector of `(lower, upper)` pairs, one per block.
- `passes`: maximum number of sweeps.

# Returns
`(x, f)`: refined parameters and objective value.
"""
function optimize_blocks(
    build_objective,
    blocks::Vector{<:AbstractVector{<:Integer}},
    x0::Vector{Float64},
    block_bounds::AbstractVector{<:Tuple};
    passes::Int = 3,
    tol::Float64 = 1e-9,
    kwargs...
)
    objective_failures = Any[]
    objective = safe_objective(
        build_objective;
        label = "optimize_blocks",
        failures = objective_failures
    )
    x = copy(x0)
    best_f = objective(x)
    options = Optim.Options(; x_abstol = 1e-9, f_reltol = 1e-13)

    call_failures = Any[]
    for _ in 1:passes
        improved = false
        for (block, (lo, hi)) in zip(blocks, block_bounds)
            isempty(block) && continue
            init_u = [
                begin
                    span = hi[k] - lo[k]
                    eps_k = max(1e-6, 1e-4 * span)
                    min(max(x[b_idx], lo[k] + eps_k), hi[k] - eps_k)
                end for (k, b_idx) in enumerate(block)
            ]
            attempt = _attempt_optimize() do
                optimize(
                    u -> begin
                        trial = copy(x)
                        trial[block] .= u
                        objective(trial)
                    end,
                    lo,
                    hi,
                    init_u,
                    Fminbox(NelderMead()),
                    options;
                    kwargs...
                )
            end
            if !attempt.ok
                push!(call_failures, attempt.err)
                continue
            end
            # Accessors on a result Optim already produced: a failure is a real
            # defect, not a diverging trial.
            value = Float64(Optim.minimum(attempt.result))
            if isfinite(value) && value < best_f - tol * (abs(best_f) + tol)
                candidate = collect(Float64, Optim.minimizer(attempt.result))
                if length(candidate) == length(block)
                    x[block] .= [min(max(candidate[k], lo[k]), hi[k]) for k in 1:length(candidate)]
                    best_f = value
                    improved = true
                end
            end
        end
        improved || break
    end
    if !isempty(call_failures)
        # Block optimization is best-effort by design: a block that will not
        # optimize is reported, but the remaining blocks still run.
        @warn "optimize_blocks: some block optimizations failed" failures =
            length(call_failures) detail = _failure_summary(call_failures)
    end
    if best_f >= OBJECTIVE_PENALTY
        @warn "optimize_blocks: every trial was replaced by the divergence penalty, " *
              "so the result is the starting point" detail = _failure_summary(objective_failures)
    end
    return (x, best_f)
end

"""
    logistic_unit(u)

Map an unconstrained real to the open interval `(0, 1)`.
"""
logistic_unit(u::Real) = 1.0 / (1.0 + exp(-clamp(Float64(u), -60.0, 60.0)))

"""
    logit(p)

Inverse of `logistic_unit`, mapping a probability in `(0, 1)` to the real line.
"""
logit(p::Real) = begin
    q = clamp(Float64(p), 1e-12, 1.0 - 1e-12)
    log(q / (1.0 - q))
end

# ==============================================================================
# Logistic (Schaefer) Surplus Production Model Fitting
# ==============================================================================

"""
    fit_schaefer(time, numbers, catches; label="Population")

Fit the discrete-time Schaefer logistic surplus production model to abundance
and catch time series:

```math
N_{t+1} = N_t + r N_t \\left(1 - \\frac{N_t}{K}\\right) - C_t
```
Surplus production:
```math
P_t = N_{t+1} - N_t + C_t = r N_t - \\left(\\frac{r}{K}\\right) N_t^2
```

# Arguments
- `time::AbstractVector`: Time steps (years).
- `numbers::AbstractVector`: Abundance in numbers (N_t).
- `catches::AbstractVector`: Catch removals in numbers (C_t).
- `label::String`: Label identifier for the population component.

# Returns
NamedTuple containing estimated parameters (r, K, MSY, N_msy, U_msy),
predicted trajectory, R², RMSE, and AIC.
"""
function fit_schaefer(
    time::AbstractVector,
    numbers::AbstractVector,
    catches::AbstractVector;
    label::String = "Population"
)
    n = length(numbers)
    if n < 3
        return (
            r = 0.0,
            K = 0.0,
            MSY = 0.0,
            N_msy = 0.0,
            U_msy = 0.0,
            predicted = Float64.(numbers),
            R2 = 0.0,
            rmse = 0.0,
            aic = Inf,
            label = label,
            is_fitted = false
        )
    end

    n_vals = Float64.(numbers)
    c_vals = Float64.(catches)

    # 1. Analytical process-error regression: P_t = a1 * N_t - a2 * N_t^2.
    #    This gives a data-driven anchor for the optimizer below.
    p_obs = [n_vals[t + 1] - n_vals[t] + c_vals[t] for t in 1:(n - 1)]
    n_pre = n_vals[1:(n - 1)]

    s11 = sum(n_pre .^ 2)
    s12 = -sum(n_pre .^ 3)
    s22 = sum(n_pre .^ 4)
    det = s11 * s22 - s12^2

    y1 = sum(n_pre .* p_obs)
    y2 = -sum((n_pre .^ 2) .* p_obs)

    r_init = 0.35
    k_init = max(maximum(n_vals) * 1.2, 50.0)

    if det > 1e-9
        a1 = (s22 * y1 - s12 * y2) / det
        a2 = (s11 * y2 - s12 * y1) / det
        if a1 > 0.01 && a2 > 1e-12
            r_init = a1
            k_init = a1 / a2
        end
    end

    # 2. Forward-simulate the logistic trajectory for a parameter pair.
    function simulate_trajectory(r_val::Float64, k_val::Float64)
        pred = zeros(Float64, n)
        pred[1] = n_vals[1]
        for t in 1:(n - 1)
            pop_curr = max(1e-6, pred[t])
            growth = r_val * pop_curr * (1.0 - pop_curr / k_val)
            pred[t + 1] = max(0.0, pop_curr + growth - c_vals[t])
        end
        return pred
    end

    # Residuals are measured on relative deviations so that the objective is not
    # dominated by the largest year, which otherwise lets a few high-abundance
    # points dictate the entire fit.
    n_scale = max(mean(abs, n_vals), 1e-6)
    function residual_sum(r_val::Float64, k_val::Float64)
        pred = simulate_trajectory(r_val, k_val)
        return sum(((n_vals .- pred) ./ n_scale) .^ 2)
    end

    # 3. Optimize (r, K) with Optim.jl. The coarse lattice only supplies starting
    #    points; the optimizer does the refinement that a bare grid search could
    #    never provide.
    r_ref = clamp(r_init, 1e-3, 10.0)
    k_ref = clamp(k_init, max(1.0, maximum(n_vals) * 0.05), 1e9)
    starts = Vector{Vector{Float64}}()
    for r_scale in (0.25, 0.5, 1.0, 2.0, 4.0)
        for k_scale in (0.25, 0.5, 1.0, 2.0, 4.0)
            push!(starts, [
                clamp(r_ref * r_scale, 1e-4, 20.0),
                clamp(k_ref * k_scale, 1e-2, 1e10),
            ])
        end
    end

    best_params, f_opt = optimize_model(
        u -> residual_sum(u[1], u[2]),
        [1e-6, 1e-3],          # r must be positive, K must exceed zero
        [20.0, 1e12],
        starts,
    )
    best_r = best_params[1]
    best_k = best_params[2]
    best_pred = simulate_trajectory(best_r, best_k)
    best_rss = sum((n_vals .- best_pred) .^ 2)

    mean_obs = mean(n_vals)
    tss = sum((n_vals .- mean_obs) .^ 2)
    r2 = (tss > 0.0) ? 1.0 - (best_rss / tss) : 0.0
    rmse = sqrt(best_rss / n)
    aic = n * log(max(1e-12, best_rss / n)) + 2.0 * 2.0

    msy = (best_r * best_k) / 4.0
    n_msy = best_k / 2.0
    u_msy = best_r / 2.0

    return (
        r = best_r,
        K = best_k,
        MSY = msy,
        N_msy = n_msy,
        U_msy = u_msy,
        predicted = best_pred,
        R2 = r2,
        rmse = rmse,
        aic = aic,
        label = label,
        is_fitted = true
    )
end

# ==============================================================================
# Deriso-Schnute Delay-Difference Dynamic Model Fitting
# ==============================================================================

"""
    stage_at_maturity_fraction(params, fraction = 0.5; sex = female)

The first developmental stage whose representative size is at or above the size
carrying the requested maturity fraction, from `SpeciesParams.maturity_l50_*` and
the matching slope.

This is the defensible recruitment index: abundance in the last stage before the
animals are counted as spawners. It is derived rather than declared because the
answer moves whenever a stage-size table or an ogive parameter moves — a stage
number written into the configuration goes stale silently, which is how the
recruitment counter came to be keyed to `benthic_anchor_stage`, a size-model
threshold with no biological meaning.

There is no stage that sits *exactly* at 50%: the L50 falls between two stages,
so this returns the first stage at or above the threshold, and the fraction it
actually corresponds to is reported by the caller.
"""
function stage_at_maturity_fraction(params::SpeciesParams, fraction::Real = 0.5;
    sex::Symbol = :female)
    l50 = sex === :male ? params.maturity_l50_male : params.maturity_l50_female
    slope = sex === :male ? params.maturity_slope_male : params.maturity_slope_female
    stages = sex === :male ? params.stage_size_range_stage_male :
             params.stage_size_range_stage_female
    lower = sex === :male ? params.stage_size_range_lower_male :
             params.stage_size_range_lower_female
    upper = sex === :male ? params.stage_size_range_upper_male :
             params.stage_size_range_upper_female
    # Size carrying `fraction` under the logistic ogive.
    lq = log(fraction / (1 - fraction))
    target = exp(log(l50) + lq / slope)
    for (i, s) in enumerate(stages)
        mean_size = (lower[i] + upper[i]) / 2
        mean_size >= target && return Int(s)
    end
    return Int(stages[end])
end
"""
    fit_delay_difference(time, numbers, catches; lag=1, label="Population")

Fit the Deriso-Schnute delay-difference model in numbers:

```math
N_{t+1} = s (N_t - C_t) + R_{t+1}
```
where s ∈ (0, 1) is annual adult survival, and recruitment follows Beverton-Holt:
```math
R_{t+1} = \\frac{\\alpha N_{t+1-lag}}{1 + \\beta N_{t+1-lag}}
```

# Arguments
- `time::AbstractVector`: Time steps (years).
- `numbers::AbstractVector`: Abundance in numbers (N_t).
- `catches::AbstractVector`: Catch removals (C_t).
- `lag::Int`: Spawning-to-recruitment delay in years (default: 1).
- `label::String`: Population label.

# Returns
NamedTuple containing survival s, natural mortality M, recruitment parameters,
predicted trajectory, R², RMSE, and AIC.
"""

function fit_delay_difference(
    time::AbstractVector,
    numbers::AbstractVector,
    catches::AbstractVector;
    lag::Int = 1,
    label::String = "Population"
)
    n = length(numbers)
    if n < (lag + 2)
        return (
            s = 0.80,
            M = -log(0.80),
            alpha = 0.20,
            beta = 0.001,
            lag = lag,
            predicted = Float64.(numbers),
            R2 = 0.0,
            rmse = 0.0,
            aic = Inf,
            label = label,
            is_fitted = false
        )
    end

    n_vals = Float64.(numbers)
    c_vals = Float64.(catches)

    fit_idx = collect((lag + 1):n)
    # Reference scale for the density-dependence parameter. Beverton-Holt
    # recruitment `alpha*S / (1 + beta*S)` only expresses meaningful saturation
    # when `beta` is of order `1/S`, so `beta` is carried in relative units
    # (`beta_effective = b_rel / abundance_scale`) and optimized on that scale.
    # Optimizing a fixed absolute `beta` grid instead forces the fit to
    # `beta = 0` for any abundance series in the thousands or millions, which
    # silently removes density dependence from the model.
    abundance_scale = max(mean(abs, n_vals[fit_idx]), 1.0)

    function simulate_dd(s_val::Float64, alpha_val::Float64, b_rel::Float64)
        beta_eff = b_rel / abundance_scale
        pred = zeros(Float64, n)
        for i in 1:min(n, lag + 1)
            pred[i] = n_vals[i]
        end
        for t in (lag + 1):(n - 1)
            spawner_idx = max(1, t - lag + 1)
            s_stock = max(1e-6, pred[spawner_idx])
            recruits = (alpha_val * s_stock) / (1.0 + beta_eff * s_stock)
            escapement = max(0.0, pred[t] - c_vals[t])
            pred[t + 1] = max(0.0, s_val * escapement + recruits)
        end
        return pred
    end

    n_scale = abundance_scale
    function residual_sum(s_val::Float64, alpha_val::Float64, b_rel::Float64)
        pred = simulate_dd(s_val, alpha_val, b_rel)
        return sum(((n_vals[fit_idx] .- pred[fit_idx]) ./ n_scale) .^ 2)
    end

    # 3. Optimize the three parameters with Optim.jl. Survival is bounded to
    #    (0, 1); alpha is bounded above by a generous multiple of 1 so that a
    #    high-recruitment series can still be represented; the density
    #    dependence is carried in relative units so that the box is
    #    scale-free and meaningful at any abundance magnitude.
    starts = Vector{Vector{Float64}}()
    for s_try in (0.3, 0.5, 0.7, 0.85, 0.95)
        for a_try in (0.05, 0.15, 0.3, 0.5, 0.8, 1.2)
            for b_try in (0.05, 0.25, 0.5, 1.0, 2.0, 4.0)
                push!(starts, [s_try, a_try, b_try])
            end
        end
    end

    best_params, f_opt = optimize_model(
        u -> residual_sum(u[1], u[2], u[3]),
        [0.0, 0.0, 0.0],        # s, alpha, relative density dependence
        [1.0, 5.0, 1e6],
        starts,
    )
    best_s = best_params[1]
    best_alpha = best_params[2]
    best_b_rel = best_params[3]
    best_beta = best_b_rel / abundance_scale
    best_pred = simulate_dd(best_s, best_alpha, best_b_rel)
    best_rss = sum((n_vals[fit_idx] .- best_pred[fit_idx]) .^ 2)

    n_fit = length(fit_idx)
    mean_obs = mean(n_vals[fit_idx])
    tss = sum((n_vals[fit_idx] .- mean_obs) .^ 2)
    r2 = (tss > 0.0) ? 1.0 - (best_rss / tss) : 0.0
    rmse = sqrt(best_rss / n_fit)
    m_nat = (0.0 < best_s < 1.0) ? -log(best_s) : ((best_s >= 1.0) ? 0.0 : Inf)
    aic = n_fit * log(max(1e-12, best_rss / n_fit)) + 2.0 * 3.0

    return (
        s = best_s,
        M = m_nat,
        alpha = best_alpha,
        beta = best_beta,
        lag = lag,
        predicted = best_pred,
        R2 = r2,
        rmse = rmse,
        aic = aic,
        label = label,
        is_fitted = true
    )
end

# ==============================================================================
# Master Macro-Ecological Population Dynamics Fitting & Reporting
# ==============================================================================

"""
    fit_population_dynamics(source; lag=1)

Fit both the Schaefer logistic model and Deriso-Schnute delay-difference model
to simulation results for:
1. The **whole population** (total numbers across all stages).
2. The **fishable component** (legal-sized adult males).

# Arguments
- `source`: File path (`.csv` or `.jls`), `Tuple{DataFrame, DataFrame}`,
  or `DataFrame` containing annual production metrics.
- `lag::Int`: Recruitment delay in years (default: 1).

# Returns
NamedTuple containing `annual_df`, `whole_population`, `fishable_stock`, and `lag`.
"""
function fit_population_dynamics(source; lag::Int = 1)
    ann_df = if source isa String
        res = load_simulation_results(source)
        if res isa DataFrame
            res
        elseif res isa Dict && haskey(res, :agent_df) && haskey(res, :model_df)
            extract_annual_production(res[:agent_df], res[:model_df])
        else
            error("Unrecognized data format in results file: $(source)")
        end
    elseif source isa Tuple && length(source) == 2 &&
           source[1] isa DataFrame && source[2] isa DataFrame
        extract_annual_production(source[1], source[2])
    elseif source isa DataFrame
        source
    else
        error("Unsupported source type for fit_population_dynamics: $(typeof(source))")
    end

    if nrow(ann_df) < 2
        error("Insufficient time steps in annual production table (nrow < 2).")
    end

    # 1. Whole population fits
    schaefer_tot = fit_schaefer(
        ann_df.sample_index,
        ann_df.total_numbers,
        ann_df.catch_total;
        label = "Whole Population"
    )
    dd_tot = fit_delay_difference(
        ann_df.sample_index,
        ann_df.total_numbers,
        ann_df.catch_total;
        lag = lag,
        label = "Whole Population"
    )

    # 2. Fishable component fits
    #
    # The delay-difference fit is driven by males at `recruitment_stage` — the
    # cohort immediately before maturity onset — rather than by legal-sized males.
    # Legal size is a regulatory threshold reached only after maturity, so a model
    # fitted to it cannot speak about recruitment at all; the fishable stock is what
    # the delay-difference process is meant to explain, and that process acts one
    # year earlier, on the cohort entering it.
    #
    # Falls back to `legal_numbers` when the column is absent, so a frame written by
    # an older build still fits rather than erroring.
    fish_series = if :recruits_male in propertynames(ann_df)
        ann_df.recruits_male
    else
        ann_df.legal_numbers
    end
    fish_label = (:recruits_male in propertynames(ann_df)) ?
                 "Fishable Stock (Males at Recruitment Stage)" : "Fishable Stock (Legal Males)"
    schaefer_fish = fit_schaefer(
        ann_df.sample_index,
        fish_series,
        ann_df.catch_legal;
        label = fish_label
    )
    dd_fish = fit_delay_difference(
        ann_df.sample_index,
        fish_series,
        ann_df.catch_legal;
        lag = lag,
        label = fish_label
    )

    return (
        annual_df = ann_df,
        whole_population = (schaefer = schaefer_tot, delay_difference = dd_tot),
        fishable_stock = (schaefer = schaefer_fish, delay_difference = dd_fish),
        lag = lag
    )
end

"""
    summarize_population_dynamics(macro_fits)

Format a comprehensive comparative ASCII report detailing parameter estimates,
goodness of fit, and management reference points for both whole population and
fishable stock components under Schaefer and Delay-Difference models.
"""
function summarize_population_dynamics(macro_fits)
    ann = macro_fits.annual_df
    wp = macro_fits.whole_population
    fs = macro_fits.fishable_stock

    buf = IOBuffer()
    println(buf, "="^75)
    println(buf, "    ORGANISM MACRO-ECOLOGICAL DYNAMIC MODEL ESTIMATION REPORT    ")
    println(buf, "="^75)
    println(buf, " Simulation Horizon (Years) : ", nrow(ann))
    println(buf, " Delay-Difference Lag (k)   : ", macro_fits.lag, " year(s)")
    println(buf, "="^75)

    for (stock_name, stock_data) in [("WHOLE POPULATION", wp), ("FISHABLE STOCK (LEGAL MALES)", fs)]
        sch = stock_data.schaefer
        dd = stock_data.delay_difference

        println(buf, "\n >> [$(stock_name)]")
        println(buf, "-"^75)
        println(buf, " A. Schaefer Logistic Surplus Production Model [N_{t+1} = N_t + rN_t(1 - N_t/K) - C_t]:")
        println(buf, "      Intrinsic Growth Rate (r)      : ", round(sch.r, digits = 4), " yr⁻¹")
        println(buf, "      Carrying Capacity (K)          : ", round(sch.K, digits = 1), " organisms")
        println(buf, "      Max Sustainable Yield (MSY)    : ", round(sch.MSY, digits = 1), " organisms/yr")
        println(buf, "      Abundance at MSY (N_msy)       : ", round(sch.N_msy, digits = 1), " organisms")
        println(buf, "      Exploitation Rate at MSY (U_msy): ", round(sch.U_msy, digits = 4), " yr⁻¹")
        println(buf, "      Goodness of Fit R²             : ", round(sch.R2, digits = 3))
        println(buf, "      Root Mean Squared Error (RMSE) : ", round(sch.rmse, digits = 2))
        println(buf, "      AIC Score                      : ", round(sch.aic, digits = 2))

        println(buf, "\n B. Deriso-Schnute Delay-Difference Model [N_{t+1} = s(N_t - C_t) + αN_{t-k}/(1 + βN_{t-k})]:")
        println(buf, "      Annual Adult Survival (s)      : ", round(dd.s, digits = 4))
        println(buf, "      Instantaneous Natural Mort (M) : ", round(dd.M, digits = 4), " yr⁻¹")
        println(buf, "      Recruitment Scaling (α)        : ", round(dd.alpha, digits = 4))
        println(buf, "      Density Dependence (β)         : ", round(dd.beta, digits = 6))
        println(buf, "      Goodness of Fit R²             : ", round(dd.R2, digits = 3))
        println(buf, "      Root Mean Squared Error (RMSE) : ", round(dd.rmse, digits = 2))
        println(buf, "      AIC Score                      : ", round(dd.aic, digits = 2))

        pref = (sch.aic <= dd.aic) ? "SCHAEFER LOGISTIC" : "DELAY-DIFFERENCE"
        println(buf, "      --> Preferred Model by AIC     : ", pref)
        println(buf, "-"^75)
    end
    println(buf, "="^75)

    return String(take!(buf))
end

# ==============================================================================
# Retrospective Pattern Analysis & Mohn's Rho Diagnostics
# ==============================================================================

"""
    mohn_rho(terminal_peeled::AbstractVector{<:Real}, terminal_base::AbstractVector{<:Real})

Calculate Mohn's rho (ρ), the canonical fisheries diagnostic metric for quantifying
retrospective bias resulting from sequential additions or deletions of data years.

# Mathematical Formulation
```math
\\rho = \\frac{1}{P} \\sum_{p=1}^P \\frac{\\hat{X}_{T-p, p} - \\hat{X}_{T-p, 0}}{\\hat{X}_{T-p, 0}}
```
where:
- ``P`` is the number of retrospective peels.
- ``\\hat{X}_{T-p, p}`` is the estimate at terminal year ``T-p`` from the model fitted
  to data truncated at year ``T-p`` (peel ``p``).
- ``\\hat{X}_{T-p, 0}`` is the estimate at year ``T-p`` from the base model fitted to
  the full unpeeled time series (peel 0).

# References
- Mohn, R. (1999). The retrospective problem in sequential population analysis:
  An investigation using cod fishery and simulated data. ICES J. Mar. Sci., 56(4): 473-488.
- Hurtado-Ferro, F., et al. (2015). Looking in the rear-view mirror: bias and retrospective
  patterns in integrated stock assessment models. ICES J. Mar. Sci., 72(1): 99-110.

# Arguments
- `terminal_peeled::AbstractVector`: Terminal year estimates from each retrospective peel.
- `terminal_base::AbstractVector`: Corresponding estimates at those years from base model.

# Returns
`Float64`: Mohn's rho value. Values in `[-0.20, +0.20]` indicate acceptable retrospective consistency.
"""
function mohn_rho(
    terminal_peeled::AbstractVector{<:Real},
    terminal_base::AbstractVector{<:Real}
)
    if length(terminal_peeled) != length(terminal_base)
        error("Vectors must have identical length for Mohn's rho calculation.")
    end
    n = length(terminal_peeled)
    if n == 0
        return 0.0
    end

    total_rel_diff = 0.0
    valid_count = 0
    for i in 1:n
        b_val = Float64(terminal_base[i])
        p_val = Float64(terminal_peeled[i])
        if abs(b_val) > 1e-6
            total_rel_diff += (p_val - b_val) / b_val
            valid_count += 1
        end
    end

    return (valid_count > 0) ? (total_rel_diff / valid_count) : 0.0
end

"""
    retrospective_analysis(source; n_peels=5, model_type=:schaefer, lag=1)

Perform sequential retrospective reanalyses by peeling back the most recent years of data,
refitting macro-ecological population dynamics models to each truncated time series, and
computing Mohn's rho to quantify retrospective bias for both Whole Population and Fishable Stock.

# Arguments
- `source`: Simulation results file path (`.csv`), `DataFrame`, or `Tuple{DataFrame, DataFrame}`.
- `n_peels::Int`: Desired number of annual peels (default: 5, clamped to available data).
- `model_type::Symbol`: Model structure to evaluate (`:schaefer` [default], `:delay_difference`, or `:both`).
- `lag::Int`: Recruitment lag for delay-difference model in years (default: 1).

# Returns
NamedTuple containing:
- `annual_df`: Reference annual production table.
- `n_peels`: Realized number of peels.
- `model_type`: Model type evaluated.
- `whole_population`: Detailed peel trajectories, terminal estimates, relative errors, and Mohn's rho.
- `fishable_stock`: Corresponding peel metrics for commercial legal males.
"""
function retrospective_analysis(
    source;
    n_peels::Int = 5,
    model_type::Symbol = :schaefer,
    lag::Int = 1
)

    if model_type === :both
        @warn "retrospective_analysis: model_type = :both is not supported; using :schaefer"
        model_type = :schaefer
    end

    ann_df = if source isa String
        res = load_simulation_results(source)
        (res isa DataFrame) ? res : extract_annual_production(res[:agent_df], res[:model_df])
    elseif source isa Tuple && length(source) == 2 &&
           source[1] isa DataFrame && source[2] isa DataFrame
        extract_annual_production(source[1], source[2])
    elseif source isa DataFrame
        source
    else
        error("Unsupported source type for retrospective_analysis: $(typeof(source))")
    end

    total_years = nrow(ann_df)
    if total_years < 4
        error("Retrospective analysis requires at least 4 years of annual data (got $(total_years)).")
    end

    max_peels = total_years - 3
    if n_peels < 1 || n_peels > max_peels
        error("Requested n_peels ($(n_peels)) must be between 1 and $(max_peels).")
    end
    realized_peels = n_peels

    # 1. Base run (Peel 0: full dataset)
    base_macro = fit_population_dynamics(ann_df; lag = lag)

    # 2. Containers for retrospective peels
    peel_indices = collect(1:realized_peels)
    terminal_samples = Int[]

    whole_peel_fits = []
    whole_term_peeled = Float64[]
    whole_term_base = Float64[]
    whole_rel_errors = Float64[]
    whole_params = []

    fish_peel_fits = []
    fish_term_peeled = Float64[]
    fish_term_base = Float64[]
    fish_rel_errors = Float64[]
    fish_params = []

    for p in peel_indices
        t_peel = total_years - p
        push!(terminal_samples, ann_df.sample_index[t_peel])
        sub_df = ann_df[1:t_peel, :]

        # Fit peeled data
        p_macro = fit_population_dynamics(sub_df; lag = lag)

        # Whole Population metrics
        w_fit = (model_type === :delay_difference) ?
                p_macro.whole_population.delay_difference :
                p_macro.whole_population.schaefer
        w_base_fit = (model_type === :delay_difference) ?
                     base_macro.whole_population.delay_difference :
                     base_macro.whole_population.schaefer

        w_p_term = w_fit.predicted[t_peel]
        w_b_term = w_base_fit.predicted[t_peel]
        w_re = (abs(w_b_term) > 1e-6) ? ((w_p_term - w_b_term) / w_b_term) : 0.0

        push!(whole_peel_fits, w_fit)
        push!(whole_term_peeled, w_p_term)
        push!(whole_term_base, w_b_term)
        push!(whole_rel_errors, w_re)
        push!(whole_params, (
            r = (model_type === :delay_difference) ? w_fit.alpha : w_fit.r,
            K = (model_type === :delay_difference) ? w_fit.s : w_fit.K
        ))

        # Fishable Stock metrics
        f_fit = (model_type === :delay_difference) ?
                p_macro.fishable_stock.delay_difference :
                p_macro.fishable_stock.schaefer
        f_base_fit = (model_type === :delay_difference) ?
                     base_macro.fishable_stock.delay_difference :
                     base_macro.fishable_stock.schaefer

        f_p_term = f_fit.predicted[t_peel]
        f_b_term = f_base_fit.predicted[t_peel]
        f_re = (abs(f_b_term) > 1e-6) ? ((f_p_term - f_b_term) / f_b_term) : 0.0

        push!(fish_peel_fits, f_fit)
        push!(fish_term_peeled, f_p_term)
        push!(fish_term_base, f_b_term)
        push!(fish_rel_errors, f_re)
        push!(fish_params, (
            r = (model_type === :delay_difference) ? f_fit.alpha : f_fit.r,
            K = (model_type === :delay_difference) ? f_fit.s : f_fit.K
        ))
    end

    rho_whole = mohn_rho(whole_term_peeled, whole_term_base)
    rho_fishable = mohn_rho(fish_term_peeled, fish_term_base)

    base_whole_fit = (model_type === :delay_difference) ?
                     base_macro.whole_population.delay_difference :
                     base_macro.whole_population.schaefer
    base_fish_fit = (model_type === :delay_difference) ?
                    base_macro.fishable_stock.delay_difference :
                    base_macro.fishable_stock.schaefer

    return (
        annual_df = ann_df,
        n_peels = realized_peels,
        model_type = model_type,
        peel_indices = peel_indices,
        terminal_samples = terminal_samples,
        whole_population = (
            base_fit = base_whole_fit,
            peel_fits = whole_peel_fits,
            terminal_peeled = whole_term_peeled,
            terminal_base = whole_term_base,
            rel_errors = whole_rel_errors,
            params = whole_params,
            mohn_rho = rho_whole
        ),
        fishable_stock = (
            base_fit = base_fish_fit,
            peel_fits = fish_peel_fits,
            terminal_peeled = fish_term_peeled,
            terminal_base = fish_term_base,
            rel_errors = fish_rel_errors,
            params = fish_params,
            mohn_rho = rho_fishable
        )
    )
end

"""
    summarize_retrospective(retro_result)

Format a diagnostic ASCII table and report detailing retrospective peels, terminal revisions,
relative percentage errors, and Mohn's rho values against standard fisheries benchmarks.
"""
function summarize_retrospective(retro_result)
    ann = retro_result.annual_df
    wp = retro_result.whole_population
    fs = retro_result.fishable_stock
    peels = retro_result.peel_indices
    samples = retro_result.terminal_samples
    m_name = (retro_result.model_type === :delay_difference) ?
             "Deriso-Schnute Delay-Difference" : "Schaefer Logistic Surplus Production"

    buf = IOBuffer()
    println(buf, "="^80)
    println(buf, "       ORGANISM RETROSPECTIVE PATTERN & MOHN'S RHO DIAGNOSTIC REPORT       ")
    println(buf, "="^80)
    println(buf, " Assessment Model Evaluated : ", m_name)
    println(buf, " Total Time Series Horizon  : ", nrow(ann), " samples")
    println(buf, " Number of Retrospective Peels: ", retro_result.n_peels)
    println(buf, "="^80)

    for (label, stock) in [("WHOLE POPULATION", wp), ("FISHABLE STOCK (LEGAL MALES)", fs)]
        println(buf, "\n >> [$(label)]")
        println(buf, " Peel | Terminal Sample | Peeled Terminal (N) | Base Reference (N) | Relative Bias (%)")
        println(buf, "------+------------------+---------------------+--------------------+------------------")
        for i in eachindex(peels)
            p_idx = peels[i]
            yr = samples[i]
            p_val = stock.terminal_peeled[i]
            b_val = stock.terminal_base[i]
            re_pct = stock.rel_errors[i] * 100.0
            re_sign = (re_pct >= 0 ? "+" : "")
            re_str = re_sign * string(round(re_pct, digits = 2)) * "%"
            p_str = string(round(p_val, digits = 2))
            b_str = string(round(b_val, digits = 2))
            println(buf, "  -", p_idx, "  |     Sample ", lpad(string(yr), 2),
                    "   |       ", lpad(p_str, 8),
                    "      |      ", lpad(b_str, 8),
                    "      |     ", lpad(re_str, 8))
        end
        println(buf, "------+------------------+---------------------+--------------------+------------------")

        rho = stock.mohn_rho
        status = if abs(rho) <= 0.20
            "ACCEPTABLE (Within standard benchmark range [-0.20, +0.20])"
        elseif rho > 0.20
            "SUBSTANTIAL POSITIVE BIAS (Retrospective overestimation > +0.20)"
        else
            "SUBSTANTIAL NEGATIVE BIAS (Retrospective underestimation < -0.20)"
        end

        rho_sign = (rho >= 0 ? "+" : "")
        rho_str = rho_sign * string(round(rho, digits = 4))
        println(buf, " Mohn's Rho (ρ): ", rho_str, " --> Diagnostic Status: ", status)
    end
    println(buf, "\n" * "="^80)
    return String(take!(buf))
end

"""
    extract_annual_stage_matrix(source; stages=build_stages())

Extract or reconstruct an S x T stage abundance matrix from simulation results.
Supports:
- `DataFrame`: Annual production table with `:total_numbers` and `:legal_numbers`.
- `String`: File path to CSV simulation output.
- `StandardABM`: Running simulation model.

# Arguments
- `source`: Data source.
- `stages::Vector{StageDefinition}`: Life-history stage definitions.

# Returns
Tuple `(stage_matrix, catch_matrix, years)`:
- `stage_matrix::Matrix{Float64}`: S x T matrix of stage abundances.
- `catch_matrix::Matrix{Float64}`: S x T matrix of stage catch removals.
- `years::Vector{Int}`: Evaluated simulation years.
"""
function extract_annual_stage_matrix(
    source;
    stages::Vector{StageDefinition} = build_stages()
)
    n_stages = length(stages)
    legal_indices = [st.index for st in stages if st.is_legal_male]
    immature_indices = [st.index for st in stages if !st.is_legal_male]

    ann_df = if source isa AbstractString
        load_simulation_results(String(source))
    elseif source isa DataFrame
        source
    elseif hasmethod(allagents, Tuple{typeof(source)})
        organisms = collect(allagents(source))
        sizes = [c.size for c in organisms]
        sexes = [c.sex for c in organisms]
        counts, _ = derive_stage_proportions_kde(sizes, sexes; stages=stages)
        stage_mat = reshape(counts, n_stages, 1)
        catch_mat = zeros(Float64, n_stages, 1)
        return (stage_mat, catch_mat, [1])
    else
        error("Unsupported source type for extract_annual_stage_matrix: $(typeof(source))")
    end

    n_years = nrow(ann_df)
    if n_years < 2
        error("Annual production table must contain at least 2 years of observations.")
    end

    years = collect(1:n_years)
    stage_matrix = zeros(Float64, n_stages, n_years)
    catch_matrix = zeros(Float64, n_stages, n_years)

    # Check if explicit stage columns exist (e.g. stage_1, stage_2, ...)
    stage_cols = [Symbol("stage_$(i)") for i in 1:n_stages]
    has_explicit_stages = all(c -> c in propertynames(ann_df), stage_cols)

    if has_explicit_stages
        for yr in 1:n_years
            for i in 1:n_stages
                stage_matrix[i, yr] = Float64(ann_df[yr, stage_cols[i]])
            end
        end
    else
        # Reconstruct stages from total_numbers and legal_numbers. No stage columns
        # were supplied, so the non-legal abundance is spread uniformly across the
        # immature (non-legal) stages. This is a neutral fallback with no magic
        # constants; prefer explicit stage_* columns (or per-stage count
        # observations from the simulation) when available. (M12)
        decay_weights = zeros(Float64, n_stages)
        for st in stages
            if !st.is_legal_male
                decay_weights[st.index] = 1.0
            end
        end
        sum_decay = sum(decay_weights)
        if sum_decay > 0.0
            decay_weights ./= sum_decay
        end

        for yr in 1:n_years
            tot_n = Float64(ann_df.total_numbers[yr])
            leg_n = Float64(ann_df.legal_numbers[yr])
            non_leg_n = max(0.0, tot_n - leg_n)

            for idx in immature_indices
                stage_matrix[idx, yr] = non_leg_n * decay_weights[idx]
            end

            if !isempty(legal_indices)
                leg_split = leg_n / length(legal_indices)
                for idx in legal_indices
                    stage_matrix[idx, yr] = leg_split
                end
            end

            c_leg = (:catch_legal in propertynames(ann_df)) ? Float64(ann_df.catch_legal[yr]) : 0.0
            if !isempty(legal_indices)
                c_split = c_leg / length(legal_indices)
                for idx in legal_indices
                    catch_matrix[idx, yr] = c_split
                end
            end
        end
    end

    return (stage_matrix, catch_matrix, years)
end

"""
    fit_generalized_leslie_model(stage_matrix, catch_matrix=zeros(size(stage_matrix));
                                 stages=build_stages(),
                                 years=1:size(stage_matrix, 2))

Fit a generalized Leslie (Lefkovitch) stage-structured transition matrix model to
multi-year stage abundance trajectories:
```math
\\mathbf{n}_{t+1} = \\mathbf{A} \\mathbf{n}_t - \\mathbf{C}_t
```

# Arguments
- `stage_matrix::Matrix{Float64}`: S x T matrix of stage abundances.
- `catch_matrix::Matrix{Float64}`: S x T matrix of stage catch removals.
- `stages::Vector{StageDefinition}`: Life-history stage definitions.
- `years::AbstractVector{Int}`: Time indices.

# Returns
`LeslieMatrixFit` containing estimated transition matrix and demographic invariants.
"""
function fit_generalized_leslie_model(
    stage_matrix::Matrix{Float64},
    catch_matrix::Matrix{Float64} = zeros(size(stage_matrix));
    stages::Vector{StageDefinition} = build_stages(),
    years::AbstractVector{Int} = 1:size(stage_matrix, 2)
)
    S, T = size(stage_matrix)
    if S != length(stages)
        error("Stage dimension mismatch: matrix has $(S) rows, but $(length(stages)) definitions provided.")
    end
    if T < 2
        error("Transition matrix estimation requires at least T >= 2 years of observations.")
    end

    A = zeros(Float64, S, S)
    male_indices = [st.index for st in stages if st.sex === male]
    female_indices = [st.index for st in stages if st.sex === female]
    repro_indices = [st.index for st in stages if st.is_reproductive]
    legal_indices = [st.index for st in stages if st.is_legal_male]

    # ------------------------------------------------------------------
    # Parameterization
    # ------------------------------------------------------------------
    # Rather than hardcoding one survival per stage (which cannot be identified
    # from a short series and is not a fit at all), the matrix is generated from a
    # small, biologically structured set of coefficients that is then estimated
    # against the observed trajectories:
    #   1 s1            survival of the pooled stage-4 cohort
    #   2 gamma1         stage-4 transition rate
    #   3 s2            survival of the pooled stage-5 cohort
    #   4 gamma2         stage-5 transition rate
    #   5 split          male share of the stage-5 transition
    #   6 s_male_legal   survival of legal-size males
    #   7 s_male_sub     survival of sublegal males
    #   8 gamma_m_base   base male transition rate
    #   9 gamma_m_decay  per-stage decay of the male transition rate
    #  10 pterm_m_base  base male maturation probability
    #  11 pterm_m_exp   maturation exponent
    #  12 s_female      female survival
    #  13 gamma_f_base  base female transition rate
    #  14 gamma_f_decay per-stage decay of the female transition rate
    #  15 pterm_f_base  base female maturation probability
    #  16 pterm_f_exp   maturation exponent
    #  17 fec_base      recruits per reproductive female
    #  18 fec_size_exp  size scaling of the fecundity term
    #  19 fec_size_ref  reference size for the fecundity term
    m6_idx = isempty(male_indices) ? 3 : first(male_indices)
    f6_idx = isempty(female_indices) ? 10 : first(female_indices)

    lp = LeslieParams()
    base_stage = Float64(lp.first_sex_specific_stage)

    """
        build_leslie_matrix(p) -> Matrix

    Assemble the S x S projection matrix from the 19-vector `p`. Recruitment is
    placed in the reproductive columns (M[1, idx] = fec_base * f_scale), so those
    columns sum to `sf + fec_base * f_scale`, which can exceed one.
    """
    function build_leslie_matrix(p::AbstractVector{<:Real})
        s1, gamma1, s2, gamma2, split = p[1], p[2], p[3], p[4], p[5]
        s_ml, s_ms = p[6], p[7]
        gmb, gmd, ptmb, ptme = p[8], p[9], p[10], p[11]
        sf, gfb, gfd, ptfb, ptfe = p[12], p[13], p[14], p[15], p[16]
        fec_base, fec_exp, fec_ref = p[17], p[18], p[19]

        M = zeros(Float64, S, S)

        # Pooled early stages
        M[2, 1] = s1 * gamma1
        M[1, 1] = s1 * (1.0 - gamma1)
        M[m6_idx, 2] = split * s2 * gamma2
        M[f6_idx, 2] = (1.0 - split) * s2 * gamma2
        M[2, 2] = s2 * (1.0 - gamma2)

        # Male stages
        for idx in male_indices
            st = stages[idx]
            is_term = (idx == last(male_indices))
            s_m = st.is_legal_male ? s_ml : s_ms
            gamma_m = is_term ? 0.0 :
                clamp(gmb - gmd * (st.stage - base_stage), 0.0, 1.0)
            p_term = (st.stage <= base_stage) ? 0.0 :
                clamp(ptmb * (st.stage - base_stage)^ptme, 0.0, 0.99)
            if is_term
                M[idx, idx] = s_m
            else
                nxt = idx + 1
                M[nxt, idx] = s_m * gamma_m * (1.0 - p_term)
                M[idx, idx] = s_m * (1.0 - gamma_m + gamma_m * p_term)
            end
        end

        # Female stages
        for idx in female_indices
            st = stages[idx]
            is_term = (idx == last(female_indices))
            gamma_f = is_term ? 0.0 :
                clamp(gfb - gfd * (st.stage - base_stage), 0.0, 1.0)
            p_term = (st.stage <= base_stage) ? 0.0 :
                clamp(ptfb * (st.stage - base_stage)^ptfe, 0.0, 0.99)
            if is_term
                M[idx, idx] = sf
            else
                nxt = idx + 1
                M[nxt, idx] = sf * gamma_f * (1.0 - p_term)
                M[idx, idx] = sf * (1.0 - gamma_f + gamma_f * p_term)
            end
        end

        # Recruitment from reproductive females into the first stage
        for idx in repro_indices
            st = stages[idx]
            f_scale = (fec_ref > 0.0) ? (st.size_mean / fec_ref)^fec_exp : 0.0
            M[1, idx] = fec_base * f_scale
        end

        return M
    end

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    # Observed process error: the year-to-year transition the data actually
    # support. Fitting this rather than a single recursive trajectory avoids the
    # error compounding that makes recursive objectives unstable for short
    # series, while a light trajectory term keeps the projected dynamics honest.
    X = stage_matrix[:, 1:(T-1)]
    Y = stage_matrix[:, 2:T] .+ catch_matrix[:, 1:(T-1)]
    obs_scale = max(mean(abs, stage_matrix), 1e-6)

    function process_error(p::AbstractVector{<:Real})
        M = build_leslie_matrix(p)
        total = 0.0
        for t in 1:(T - 1)
            x = X[:, t]
            total_obs = max(sum(x), 1e-9)
            predicted = M * x
            # Compare on a relative scale so the fit is not dominated by the
            # stages that carry most of the biomass.
            total += sum(((predicted .- Y[:, t]) ./ (obs_scale * sqrt(total_obs))).^2)
        end
        return total
    end

    function trajectory_error(p::AbstractVector{<:Real})
        M = build_leslie_matrix(p)
        pred = zeros(Float64, S, T)
        pred[:, 1] = stage_matrix[:, 1]
        for t in 1:(T - 1)
            pred[:, t + 1] = max.(0.0, M * pred[:, t] .- catch_matrix[:, t])
        end
        return sum(((pred .- stage_matrix) ./ obs_scale) .^ 2)
    end

    # The process error dominates; the trajectory term regularizes toward
    # multi-year consistency without letting recursive noise drive the fit.
    objective(p) = process_error(p) +
                   lp.trajectory_error_weight * trajectory_error(p)

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------
    # Starting values come from the configured Leslie parameters so the fit
    # inherits documented, species-specific priors instead of arbitrary numbers.
    p0 = [
        lp.survival_stage1, lp.gamma_stage1,
        lp.survival_stage2, lp.gamma_stage2, lp.male_split_fraction,
        lp.survival_male_legal, lp.survival_male_sublegal,
        lp.gamma_male_base, lp.gamma_male_decay,
        lp.p_term_male_base, lp.p_term_male_exponent,
        lp.survival_female, lp.gamma_female_base, lp.gamma_female_decay,
        lp.p_term_female_base, lp.p_term_female_exponent,
        lp.fecundity_base, lp.fecundity_size_exponent, lp.fecundity_size_reference,
    ]

    lower = [
        1e-3, 1e-3,
        1e-3, 1e-3, 0.0,
        1e-3, 1e-3,
        1e-3, 0.0, 0.0, 0.1,
        1e-3, 1e-3, 0.0, 0.0, 0.1,
        0.0, 0.1, 1.0,
    ]
    upper = [
        1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0,
        1.0, 1.0, 1.0, 4.0,
        1.0, 1.0, 1.0, 1.0, 4.0,
        1e4, 6.0, 200.0,
    ]

    # Coordinate descent over biologically coherent blocks. Optimizing all 19
    # coefficients jointly is neither identifiable from a short series nor
    # tractable for a derivative-free method, but each block is small and uses
    # every year of data.
    blocks = [
        [1, 2, 3, 4, 5],                     # pooled early stages
        [6, 7, 8, 9, 10, 11],                # male transitions
        [12, 13, 14, 15, 16],                # female transitions
        [17, 18, 19],                        # fecundity
    ]
    block_bounds = [
        (lower[blocks[1]], upper[blocks[1]]),
        (lower[blocks[2]], upper[blocks[2]]),
        (lower[blocks[3]], upper[blocks[3]]),
        (lower[blocks[4]], upper[blocks[4]]),
    ]

    p_hat, obj_hat = optimize_blocks(objective, blocks, p0, block_bounds; passes = 4)
    A = build_leslie_matrix(p_hat)

    # Eigenvalue and Eigenvector Decomposition
    eig_right = eigen(A)
    eigenvalues = eig_right.values
    moduli = abs.(eigenvalues)
    max_idx = argmax(moduli)

    lambda_1 = real(eigenvalues[max_idx])
    lambda_1 = max(lambda_1, 1.0e-6)
    r_growth = log(lambda_1)

    # Right Eigenvector -> Stable Stage Distribution (SSD)
    w_raw = abs.(real.(eig_right.vectors[:, max_idx]))
    sum_w = sum(w_raw)
    stable_stage_dist = (sum_w > 0.0) ? (w_raw ./ sum_w) : fill(1.0 / S, S)

    # Left Eigenvector -> Reproductive Values (RV)
    eig_left = eigen(copy(A'))
    moduli_left = abs.(eig_left.values)
    max_idx_left = argmax(moduli_left)
    v_raw = abs.(real.(eig_left.vectors[:, max_idx_left]))
    ref_v = max(v_raw[1], 1.0e-6)
    reproductive_vals = v_raw ./ ref_v

    # Damping Ratio: lambda_1 / |lambda_2|
    sorted_moduli = sort(moduli, rev = true)
    lambda_2 = (length(sorted_moduli) >= 2) ? sorted_moduli[2] : 0.0
    damping_ratio = (lambda_2 > 1.0e-6) ? (lambda_2 / lambda_1) : 0.0

    # Elasticity Matrix
    v_dot_w = dot(reproductive_vals, stable_stage_dist)
    v_dot_w = max(v_dot_w, 1.0e-6)
    elasticity_mat = zeros(Float64, S, S)
    for i in 1:S
        for j in 1:S
            elasticity_mat[i, j] = (A[i, j] / lambda_1) *
                                   (reproductive_vals[i] * stable_stage_dist[j]) / v_dot_w
        end
    end
    sum_e = sum(elasticity_mat)
    if sum_e > 0.0
        elasticity_mat ./= sum_e
    end

    # Dynamic Trajectory Projection
    predicted_stages = zeros(Float64, S, T)
    predicted_stages[:, 1] = stage_matrix[:, 1]

    for t in 1:(T-1)
        next_n = A * predicted_stages[:, t] .- catch_matrix[:, t]
        predicted_stages[:, t + 1] = max.(0.0, next_n)
    end

    legal_obs = zeros(Float64, T)
    legal_pred = zeros(Float64, T)
    biomass_obs = zeros(Float64, T)
    biomass_pred = zeros(Float64, T)

    for t in 1:T
        for idx in legal_indices
            st = stages[idx]
            wt_kg = st.mean_mass / 1000.0
            legal_obs[t] += stage_matrix[idx, t]
            legal_pred[t] += predicted_stages[idx, t]
            biomass_obs[t] += stage_matrix[idx, t] * wt_kg
            biomass_pred[t] += predicted_stages[idx, t] * wt_kg
        end
    end

    mean_obs = mean(legal_obs)
    ss_tot = sum((legal_obs .- mean_obs).^2)
    ss_res = sum((legal_obs .- legal_pred).^2)
    r2 = (ss_tot > 0.0) ? max(0.0, 1.0 - ss_res / ss_tot) : 0.0
    rmse = sqrt(ss_res / T)
    # 19 estimated coefficients; the penalty must reflect the parameters that
    # were actually fitted or AIC comparisons across models are meaningless.
    k_params = length(p_hat)
    aic = (ss_res > 0.0) ? (T * log(ss_res / T) + 2.0 * k_params) : 0.0

    return LeslieMatrixFit(
        stages,
        A,
        eigenvalues,
        lambda_1,
        r_growth,
        stable_stage_dist,
        reproductive_vals,
        damping_ratio,
        elasticity_mat,
        stage_matrix,
        predicted_stages,
        legal_obs,
        legal_pred,
        biomass_obs,
        biomass_pred,
        r2,
        rmse,
        aic,
        k_params,
        collect(Int, years)
    )
end

"""
    summarize_leslie_model(fit::LeslieMatrixFit)

Format a comprehensive ASCII diagnostic report summarizing transition matrix estimation,
eigenvalues, stable stage distribution, reproductive values, and elasticity.

# Returns
`String` containing the formatted report.
"""
function summarize_leslie_model(fit::LeslieMatrixFit)
    buf = IOBuffer()
    println(buf, "="^85)
    println(buf, "    ORGANISM GENERALIZED LESLIE (STAGE-TRANSITION) DEMOGRAPHIC REPORT   ")
    println(buf, "="^85)
    println(buf, " Evaluated Years               : ", length(fit.years),
            " (Years ", first(fit.years), " to ", last(fit.years), ")")
    println(buf, " Discrete Life Stages          : ", length(fit.stages))
    println(buf, "-"^85)
    println(buf, " [EIGENVALUE & ASYMPTOTIC DEMOGRAPHIC METRICS]")
    println(buf, "   - Dominant Eigenvalue (λ₁): ", lpad(string(round(fit.dominant_eigenvalue, digits = 4)), 8),
            "  (Asymptotic Annual Population Multiplier)")
    println(buf, "   - Intrinsic Growth Rate (r): ", lpad(string(round(fit.intrinsic_growth_rate, digits = 4)), 8),
            " yr⁻¹  (r = ln(λ₁))")
    println(buf, "   - Asymptotic Trajectory   : ",
            fit.dominant_eigenvalue > 1.005 ? "GROWING (λ₁ > 1.0)" :
            (fit.dominant_eigenvalue < 0.995 ? "DECLINING (λ₁ < 1.0)" : "STATIONARY (λ₁ ≈ 1.0)"))
    println(buf, "   - Damping Ratio (ρ)       : ", lpad(string(round(fit.damping_ratio, digits = 3)), 8),
            "  (Rate of convergence to SSD)")
    println(buf, "   - Fishable Stock Fit R²   : ", lpad(string(round(fit.r2, digits = 3)), 8))
    println(buf, "   - Fishable Stock RMSE     : ", lpad(string(round(fit.rmse, digits = 2)), 8), " organisms")
    println(buf, "   - Model AIC Score         : ", lpad(string(round(fit.aic, digits = 2)), 8))
    println(buf, "-"^85)
    println(buf, " [STAGE STRUCTURE, STABLE DISTRIBUTION & REPRODUCTIVE VALUES]")
    println(buf, " Idx | Stage Name          | Size Mean | Mean Mass (g) | SSD (w₁) | Repro Val (v₁) | Legal | Repro")
    println(buf, "-----+---------------------+-----------+---------------+----------+----------------+-------+------")
    for st in fit.stages
        i = st.index
        w_str = lpad(string(round(fit.stable_stage_distribution[i] * 100.0, digits = 2)) * "%", 8)
        v_str = lpad(string(round(fit.reproductive_values[i], digits = 2)), 14)
        sz_str = lpad(string(round(st.size_mean, digits = 1)), 9)
        mass_str = lpad(string(round(st.mean_mass, digits = 1)), 13)
        leg_str = st.is_legal_male ? " YES " : "  -  "
        rep_str = st.is_reproductive ? " YES " : "  -  "
        println(buf, "  ", lpad(string(i), 2), " | ",
                rpad(st.name, 19), " | ",
                sz_str, " | ",
                mass_str, " | ",
                w_str, " | ",
                v_str, " | ",
                leg_str, " | ",
                rep_str)
    end
    println(buf, "="^85)
    return String(take!(buf))
end

"""
    compare_macro_models(annual_data, leslie_fit; lag=1)

Compare legal fishable stock biomass dynamics across three macro-ecological modeling
paradigms: Schaefer surplus production, Deriso-Schnute delay-difference, and
generalized Leslie stage transition matrix.

# Arguments
- `annual_data`: `DataFrame` or file path with annual simulation metrics.
- `leslie_fit::LeslieMatrixFit`: Fitted generalized Leslie model.
- `lag::Int`: Delay-difference recruitment lag in years.

# Returns
`MacroModelComparison` containing comparative metrics, plus `label_*` legend
entries that state each model's key estimated parameters *with units*.

# Note on units
The three fits are performed in **number**, not biomass:
`fit_population_dynamics` passes `total_numbers` / `legal_numbers` and
`catch_total` / `catch_legal` to `fit_schaefer` and `fit_delay_difference`, and
`fit_generalized_leslie_model` works on a per-stage count matrix. So `K` is a
carrying capacity in individuals, `r` and `M` are annual rates, and `MSY` is
individuals per year.

The `biomass_*_kg` fields are a post-hoc rescale of those predicted counts by a
single pooled mean legal mass (`mean_legal_wt_kg`), applied identically to the
observed series. The biomass curves are therefore proportional to the number
curves: the kg axis is a presentation choice and adds no information, which is
why the legend labels carry number units beside it.
"""
function compare_macro_models(
    annual_data,
    leslie_fit::LeslieMatrixFit;
    lag::Int = 1
)
    fits = fit_population_dynamics(annual_data; lag = lag)
    years = leslie_fit.years
    T = length(years)

    # Fall back to the configured allometric mass of a mean-size legal male
    # rather than a bare literal, so the number conversions use are traceable.
    mean_legal_wt_kg = begin
        leg_stages = filter(s -> s.is_legal_male, leslie_fit.stages)
        if !isempty(leg_stages)
            mean([s.mean_mass for s in leg_stages]) / 1000.0
        else
            sp = SpeciesParams()
            calculate_mass(sp, 100.0, :male; mat = 1) / 1000.0
        end
    end

    b_obs = leslie_fit.legal_biomass_obs_kg
    b_leslie = leslie_fit.legal_biomass_pred_kg

    # Schaefer trajectory for fishable stock
    sch_fit = fits.fishable_stock.schaefer
    n_sch = sch_fit.predicted
    if length(n_sch) < T
        n_sch = vcat(n_sch, fill(last(n_sch), T - length(n_sch)))
    elseif length(n_sch) > T
        n_sch = n_sch[1:T]
    end
    b_sch = [n * mean_legal_wt_kg for n in n_sch]

    # Delay-Difference trajectory for fishable stock
    dd_fit = fits.fishable_stock.delay_difference
    n_dd = dd_fit.predicted
    if length(n_dd) < T
        n_dd = vcat(n_dd, fill(last(n_dd), T - length(n_dd)))
    elseif length(n_dd) > T
        n_dd = n_dd[1:T]
    end
    b_dd = [n * mean_legal_wt_kg for n in n_dd]

    # Compute variability metrics
    mu_obs = mean(b_obs)
    sd_obs = (T > 1) ? std(b_obs) : 0.0
    cv_obs = (mu_obs > 0.0) ? (sd_obs / mu_obs) : 0.0

    mu_sch = mean(b_sch)
    sd_sch = (T > 1) ? std(b_sch) : 0.0
    cv_sch = (mu_sch > 0.0) ? (sd_sch / mu_sch) : 0.0

    mu_dd = mean(b_dd)
    sd_dd = (T > 1) ? std(b_dd) : 0.0
    cv_dd = (mu_dd > 0.0) ? (sd_dd / mu_dd) : 0.0

    mu_leslie = mean(b_leslie)
    sd_les = (T > 1) ? std(b_leslie) : 0.0
    cv_les = (mu_leslie > 0.0) ? (sd_les / mu_leslie) : 0.0

    delta_obs = (T >= 2) ? [b_obs[t + 1] - b_obs[t] for t in 1:(T - 1)] : [0.0]
    delta_sch = (T >= 2) ? [b_sch[t + 1] - b_sch[t] for t in 1:(T - 1)] : [0.0]
    delta_dd = (T >= 2) ? [b_dd[t + 1] - b_dd[t] for t in 1:(T - 1)] : [0.0]
    delta_les = (T >= 2) ? [b_leslie[t + 1] - b_leslie[t] for t in 1:(T - 1)] : [0.0]

    var_delta_obs = (length(delta_obs) > 1) ? var(delta_obs) : 0.0
    var_delta_sch = (length(delta_sch) > 1) ? var(delta_sch) : 0.0
    var_delta_dd = (length(delta_dd) > 1) ? var(delta_dd) : 0.0
    var_delta_les = (length(delta_les) > 1) ? var(delta_les) : 0.0

    ss_tot = sum((b_obs .- mu_obs).^2)

    ss_res_sch = sum((b_obs .- b_sch).^2)
    r2_sch = (ss_tot > 0.0) ? max(0.0, 1.0 - ss_res_sch / ss_tot) : 0.0
    rmse_sch = sqrt(ss_res_sch / T)
    aic_sch = (ss_res_sch > 0.0) ? (T * log(ss_res_sch / T) + 2.0 * 2) : 0.0

    ss_res_dd = sum((b_obs .- b_dd).^2)
    r2_dd = (ss_tot > 0.0) ? max(0.0, 1.0 - ss_res_dd / ss_tot) : 0.0
    rmse_dd = sqrt(ss_res_dd / T)
    aic_dd = (ss_res_dd > 0.0) ? (T * log(ss_res_dd / T) + 2.0 * 3) : 0.0

    ss_res_les = sum((b_obs .- b_leslie).^2)
    r2_les = (ss_tot > 0.0) ? max(0.0, 1.0 - ss_res_les / ss_tot) : 0.0
    rmse_les = sqrt(ss_res_les / T)
    # The Leslie penalty must be the number of coefficients the fit estimated. It
    # used to be the literal 4, which understated it by 15 parameters and made the
    # most flexible of the three models look favoured by AIC for having fitted less.
    aic_les = (ss_res_les > 0.0) ?
              (T * log(ss_res_les / T) + 2.0 * leslie_fit.n_free_parameters) : 0.0

    aics = Dict(:schaefer => aic_sch, :delay_difference => aic_dd, :generalized_leslie => aic_les)
    preferred = argmin(aics)

    # Legend entries carrying the key estimated parameters, formatted to three
    # significant figures so a run whose estimates barely move does not produce a
    # legend that changes on every re-fit.
    #
    # The fits themselves are in NUMBER: `fit_population_dynamics` passes
    # `total_numbers` / `legal_numbers` and `catch_total` / `catch_legal`, so K is
    # a carrying capacity in individuals and MSY is individuals per year. The
    # biomass series are a post-hoc rescale of the predicted numbers by a single
    # pooled mean legal mass, applied identically to the observed series — so the
    # biomass curves are proportional to the number curves and carry no extra
    # information. The labels below state the units, because the panel's y-axis is
    # in kg and a bare "K=..." beside it would be read as a biomass.
    g(x) = @sprintf("%.3g", x)
    label_observed = "Stage-mean legal biomass (model basis)"
    label_schaefer = sch_fit.is_fitted ?
        "Schaefer: K=$(g(sch_fit.K)) ind, r=$(g(sch_fit.r))/yr, MSY=$(g(sch_fit.MSY)) ind/yr" :
        "Schaefer (not fitted)"
    label_delay_diff = dd_fit.is_fitted ?
        "Delay-Diff: s=$(g(dd_fit.s)), M=$(g(dd_fit.M))/yr, α=$(g(dd_fit.alpha))" :
        "Delay-Diff (not fitted)"
    label_leslie = "Leslie: λ=$(g(leslie_fit.dominant_eigenvalue)), r=$(g(leslie_fit.intrinsic_growth_rate))/yr"

    return MacroModelComparison(
        years,
        b_obs,
        b_sch,
        b_dd,
        b_leslie,
        cv_obs,
        cv_sch,
        cv_dd,
        cv_les,
        sd_obs,
        sd_sch,
        sd_dd,
        sd_les,
        var_delta_obs,
        var_delta_sch,
        var_delta_dd,
        var_delta_les,
        r2_sch,
        r2_dd,
        r2_les,
        rmse_sch,
        rmse_dd,
        rmse_les,
        aic_sch,
        aic_dd,
        aic_les,
        preferred,
        label_observed,
        label_schaefer,
        label_delay_diff,
        label_leslie,
        leslie_fit.legal_abundance_obs,
        n_sch,
        n_dd,
        leslie_fit.legal_abundance_pred,
        mean_legal_wt_kg
    )

end

"""
    summarize_macro_comparison(comp::MacroModelComparison)

Format an ASCII summary table comparing legal fishable biomass variations, variability,
and goodness of fit across Schaefer, Delay-Difference, and Generalized Leslie models.

# Returns
`String` containing the comparative report.
"""
function summarize_macro_comparison(comp::MacroModelComparison)
    buf = IOBuffer()
    println(buf, "="^85)
    println(buf, "    THREE-WAY MACRO-ECOLOGICAL MODEL COMPARISON: FISHABLE COMPONENT      ")
    println(buf, "="^85)
    println(buf, " Simulation Horizon (Years)    : ", length(comp.years))
    println(buf, " Preferred Model by AIC        : ", uppercase(string(comp.preferred_model)))
    println(buf, "-"^85)
    println(buf, " Metric                      | Observed   | Schaefer   | Delay-Diff | Leslie Matrix")
    println(buf, "-----------------------------+------------+------------+------------+--------------")
    println(buf, " Mean Biomass (kg)           | ",
            lpad(string(round(mean(comp.biomass_obs_kg), digits = 2)), 10), " | ",
            lpad(string(round(mean(comp.biomass_schaefer_kg), digits = 2)), 10), " | ",
            lpad(string(round(mean(comp.biomass_delay_diff_kg), digits = 2)), 10), " | ",
            lpad(string(round(mean(comp.biomass_leslie_kg), digits = 2)), 12))
    println(buf, " Standard Deviation (kg)     | ",
            lpad(string(round(comp.sd_obs, digits = 2)), 10), " | ",
            lpad(string(round(comp.sd_schaefer, digits = 2)), 10), " | ",
            lpad(string(round(comp.sd_delay_diff, digits = 2)), 10), " | ",
            lpad(string(round(comp.sd_leslie, digits = 2)), 12))
    println(buf, " Coeff of Variation (CV)     | ",
            lpad(string(round(comp.cv_obs, digits = 3)), 10), " | ",
            lpad(string(round(comp.cv_schaefer, digits = 3)), 10), " | ",
            lpad(string(round(comp.cv_delay_diff, digits = 3)), 10), " | ",
            lpad(string(round(comp.cv_leslie, digits = 3)), 12))
    println(buf, " Interannual Step Var Var(ΔB)| ",
            lpad(string(round(comp.var_delta_obs, digits = 3)), 10), " | ",
            lpad(string(round(comp.var_delta_schaefer, digits = 3)), 10), " | ",
            lpad(string(round(comp.var_delta_delay_diff, digits = 3)), 10), " | ",
            lpad(string(round(comp.var_delta_leslie, digits = 3)), 12))
    println(buf, " Goodness of Fit R²          |    N/A     | ",
            lpad(string(round(comp.r2_schaefer, digits = 3)), 10), " | ",
            lpad(string(round(comp.r2_delay_diff, digits = 3)), 10), " | ",
            lpad(string(round(comp.r2_leslie, digits = 3)), 12))
    println(buf, " RMSE (kg)                   |    N/A     | ",
            lpad(string(round(comp.rmse_schaefer, digits = 2)), 10), " | ",
            lpad(string(round(comp.rmse_delay_diff, digits = 2)), 10), " | ",
            lpad(string(round(comp.rmse_leslie, digits = 2)), 12))
    println(buf, " AIC Score                   |    N/A     | ",
            lpad(string(round(comp.aic_schaefer, digits = 2)), 10), " | ",
            lpad(string(round(comp.aic_delay_diff, digits = 2)), 10), " | ",
            lpad(string(round(comp.aic_leslie, digits = 2)), 12))
    println(buf, "="^85)
    return String(take!(buf))
end

# ==============================================================================
# Simulation Multi-Attribute Criteria Evaluation (Orchestrator Support)
# ==============================================================================

"""
    evaluate_population_criteria(model, agent_df;   
                                min_population = nothing,
                                max_population = nothing,
                                max_size_drift::Float64 = 0.20,
                                max_maturity_rmse::Float64 = 0.35,
                                max_growth_rmse::Float64 = 15.0,
                                window_steps::Int = 10)

Evaluate a completed simulation run against multi-attribute biological and demographic
criteria to support systematic parameter orchestration and automated model calibration.

# Evaluated Criteria:
1. **No Extinction (`is_persistent`)**: Living population ``N(t) \\ge N_{\\min}``.
   The default floor is `CriteriaParams.min_population` (1, i.e. "not extinct").
   This is deliberately *not* derived from the carrying capacity: the population
   regulates around a fraction of K, so a K-derived floor reported healthy
   self-regulated runs as extinct. The bound and the model were both a function
   of the same K, so the criterion was self-referential.
2. **No Population Explosion (`is_bounded`)**: Living population
   ``N(t) \\le N_{\\max}``. The default ceiling is
   `CriteriaParams.max_population_multiple` times the carrying capacity. This one
   legitimately refers to K — "did not explode past the scale it was configured
   for" is a statement about K — and a multiple rather than K itself because a
   transient overshoot is normal. `peak_population` in the result reports the
   observed maximum so overshoot is visible rather than implied.
3. **Steady Size Structure (`is_size_steady`)**: Relative drift in mean carapace width over
   the final evaluation window does not exceed `max_size_drift`.
4. **Maturity Ogive Fidelity (`is_maturity_realistic`)**: Empirical cross-sectional
   maturity schedule matches the cumulative theoretical curve within
   `max_maturity_rmse`. The empirical ogive estimates the probability of having
   *ever* matured, so it is compared against the cumulative curve over the stages
   an individual of that size has passed, not against the per-ecdysis probability.
5. **Growth Schedule Fidelity (`is_growth_realistic`)**: Observed mean carapace width per
   stage matches reference empirical medians from `growth_schedule()` within `max_growth_rmse`.

# Arguments
- `model`: Agent-based model instance after simulation.
- `agent_df::DataFrame`: Time-series table collected during simulation.
- `min_population::Union{Nothing, Int}`: Minimum living organism threshold.
  `nothing` uses `model.params.criteria.min_population`.
- `max_population::Union{Nothing, Int}`: Maximum living organism ceiling.
  `nothing` uses `max_population_multiple * benthic_carrying_capacity`.
- `max_size_drift::Float64`: Maximum allowable fractional shift in mean size over recent steps.
- `max_maturity_rmse::Float64`: Maximum allowable RMSE against the cumulative maturity curve.
- `max_growth_rmse::Float64`: Maximum allowable RMSE against empirical stage size schedule (mm).
- `window_steps::Int`: Number of recent time steps used to assess size stability.

# Returns
`NamedTuple` with detailed diagnostic metrics, component pass/fail booleans,
composite loss score, and `passed_all::Bool`.
"""
function evaluate_population_criteria(
    model,
    agent_df::DataFrame;
    min_population::Union{Nothing, Int} = nothing,
    max_population::Union{Nothing, Int} = nothing,
    max_size_drift::Union{Nothing, Float64} = nothing,
    max_maturity_rmse::Union{Nothing, Float64} = nothing,
    max_growth_rmse::Union{Nothing, Float64} = nothing,
    window_steps::Union{Nothing, Int} = nothing,
    maturity_ogive_min_count::Union{Nothing, Int} = nothing
)
    # `params` lives in the model's abm properties, not as a struct field, so
    # neither `hasfield` nor `hasproperty` finds it. And `IBMParameters` forwards
    # unknown field reads to its nested structs via `getproperty` without
    # extending `propertynames`, so `hasproperty` is also false for
    # `benthic_carrying_capacity`. Both facts previously sent this code down its
    # fallback branch, which silently substituted a hardcoded carrying capacity
    # for the configured one.
    props = try
        abmproperties(model)
    catch
        nothing
    end
    p = (props !== nothing && haskey(props, :params)) ? props[:params] : nothing
    carrying_capacity = if p !== nothing && hasfield(typeof(p), :environment)
        Float64(getfield(p, :environment).benthic_carrying_capacity)
    else
        300.0
    end
    crit_cfg = if p !== nothing && hasfield(typeof(p), :criteria)
        getfield(p, :criteria)
    else
        CriteriaParams()
    end

    # Thresholds, windows and weights all come from [criteria] unless the caller
    # overrides them. They were literals, so tuning the composite loss required
    # editing the source.
    drift_limit = max_size_drift === nothing ? crit_cfg.max_size_drift : max_size_drift
    maturity_limit = max_maturity_rmse === nothing ? crit_cfg.max_maturity_rmse :
                     max_maturity_rmse
    growth_limit = max_growth_rmse === nothing ? crit_cfg.max_growth_rmse :
                   max_growth_rmse
    size_window = window_steps === nothing ? crit_cfg.window_steps : window_steps
    ogive_min_count = maturity_ogive_min_count === nothing ?
                      crit_cfg.maturity_ogive_min_count : maturity_ogive_min_count

    # Floor is an absolute count, independent of K: the criterion asks whether the
    # run went extinct. Ceiling is a multiple of K, because "did not explode past
    # the scale it was configured for" is a statement about K.
    n_min = (min_population !== nothing) ? min_population : crit_cfg.min_population
    n_max = (max_population !== nothing) ? max_population :
            round(Int, crit_cfg.max_population_multiple * carrying_capacity)

    # 1. Population bounds
    living_organisms = collect(allagents(model))
    n_living = length(living_organisms)

    is_persistent = n_living >= n_min
    is_bounded = n_living <= n_max

    # Observed peak, so a transient overshoot is reported rather than hidden behind
    # a pass/fail on the final value alone.
    peak_population = n_living
    if nrow(agent_df) > 0
        for candidate in (:length_stage, :count_stage, :total_organisms, :total_numbers)
            if candidate in propertynames(agent_df)
                peak_population = max(peak_population, round(Int, maximum(agent_df[!, candidate])))
                break
            end
        end
    end

    if n_living == 0
        return (
            passed_all = false,
            is_persistent = false,
            is_bounded = true,
            is_size_steady = false,
            is_maturity_realistic = false,
            is_growth_realistic = false,
            n_living = 0,
            peak_population = peak_population,
            n_min = n_min,
            n_max = n_max,
            size_drift = 1.0,
            maturity_rmse = 1.0,
            # Reported as a value, not as a verdict: a run with no survivors has
            # no growth error to measure, and this must stay strictly above
            # `crit_cfg.max_growth_rmse` so the growth criterion reads false.
            growth_rmse = growth_limit,
            # The extinction penalty dominates the empty-population loss, with the
            # size-drift and maturity terms at their saturated values and nothing for
            # growth. Every term here is a term the populated path can also
            # contribute, so the two scores are on the same scale.
            composite_score = crit_cfg.extinction_penalty +
                              crit_cfg.size_drift_weight +
                              crit_cfg.maturity_rmse_weight,
            # Same field name as the populated path below; this one used to report
            # `mean_cw` while the other reported `mean_size`, so the two returns
            # did not have the same shape.
            mean_size = 0.0,
            stage_counts = Dict{Int, Int}()
        )
    end

    # 2. Steady size structure (drift over recent steps)
    size_col = nothing
    for candidate in [:size_safe_mean, :mean_size]
        if candidate in propertynames(agent_df)
            size_col = candidate
            break
        end
    end

    size_drift = 0.0
    if size_col !== nothing && nrow(agent_df) >= 2
        w = min(size_window, nrow(agent_df))
        recent_sizes = agent_df[(end - w + 1):end, size_col]
        mean_recent = mean(recent_sizes)
        if mean_recent > 0.0
            size_drift = abs(recent_sizes[end] - recent_sizes[1]) / mean_recent
        end
    end
    is_size_steady = size_drift <= drift_limit

    # 3. Maturity ogive fidelity.
    #    The empirical curve is cross-sectional (has ever matured) and `maturity`
    #    is a per-ecdysis probability, so they are compared against the cumulative
    #    curve over the stages an individual of that size has already passed, not
    #    against the raw per-ecdysis probability. Bins with a handful of
    #    individuals are excluded because their proportions are noise.
    size_bins = crit_cfg.ogive_bin_start:crit_cfg.ogive_bin_step:crit_cfg.ogive_bin_stop
    om = compute_empirical_ogive(model, male, size_bins; min_count = ogive_min_count)
    of = compute_empirical_ogive(model, female, size_bins; min_count = ogive_min_count)

    maturity_sq_errors = Float64[]
    append!(maturity_sq_errors, maturity_ogive_residuals(om, male))
    append!(maturity_sq_errors, maturity_ogive_residuals(of, female))

    maturity_rmse = isempty(maturity_sq_errors) ? 0.0 : sqrt(mean(maturity_sq_errors))
    is_maturity_realistic = maturity_rmse <= maturity_limit

    # 4. Growth schedule fidelity
    ref_schedule = growth_schedule()
    stage_obs_m = Dict{Int, Vector{Float64}}()
    stage_obs_f = Dict{Int, Vector{Float64}}()
    stage_counts = Dict{Int, Int}()

    for org in living_organisms
        stg = org.stage
        stage_counts[stg] = get(stage_counts, stg, 0) + 1
        if org.sex == male
            if !haskey(stage_obs_m, stg)
                stage_obs_m[stg] = Float64[]
            end
            push!(stage_obs_m[stg], org.size)
        else
            if !haskey(stage_obs_f, stg)
                stage_obs_f[stg] = Float64[]
            end
            push!(stage_obs_f[stg], org.size)
        end
    end

    growth_sq_errors = Float64[]
    for (stg, sizes) in stage_obs_m
        if 1 <= stg <= nrow(ref_schedule)
            ref_med = ref_schedule.size_male_med[stg]
            push!(growth_sq_errors, (mean(sizes) - ref_med)^2)
        end
    end
    for (stg, sizes) in stage_obs_f
        if 1 <= stg <= nrow(ref_schedule)
            ref_med = ref_schedule.size_female_med[stg]
            push!(growth_sq_errors, (mean(sizes) - ref_med)^2)
        end
    end

    growth_rmse = isempty(growth_sq_errors) ? 0.0 : sqrt(mean(growth_sq_errors))
    is_growth_realistic = growth_rmse <= growth_limit

    # 5. Composite criteria status and loss
    passed_all = is_persistent && is_bounded && is_size_steady &&
                 is_maturity_realistic && is_growth_realistic

    composite_score = (is_persistent ? 0.0 : crit_cfg.extinction_penalty) +
                      (is_bounded ? 0.0 : crit_cfg.explosion_penalty) +
                      crit_cfg.size_drift_weight * size_drift +
                      crit_cfg.maturity_rmse_weight * maturity_rmse +
                      crit_cfg.growth_rmse_weight * growth_rmse

    mean_size_all = mean([org.size for org in living_organisms])

    return (
        passed_all = passed_all,
        is_persistent = is_persistent,
        is_bounded = is_bounded,
        is_size_steady = is_size_steady,
        is_maturity_realistic = is_maturity_realistic,
        is_growth_realistic = is_growth_realistic,
        n_living = n_living,
        peak_population = peak_population,
        n_min = n_min,
        n_max = n_max,
        size_drift = size_drift,
        maturity_rmse = maturity_rmse,
        growth_rmse = growth_rmse,
        composite_score = composite_score,
        mean_size = mean_size_all,
        stage_counts = stage_counts
    )
end

# -----------------------------------------------------------------------------
# Empirical Biological Model Calibration, KDE, and Trait Sampling
# -----------------------------------------------------------------------------

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
with safety bounds preventing zero bandwidth when variance is tiny.

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
    derive_stage_proportions_kde(sizes, sexes=nothing; stages=build_stages())

Derive the proportion and count of individuals in each discrete stage
from continuous size observations using empirical growth-stage intervals.
"""
function derive_stage_proportions_kde(
    sizes::AbstractVector{<:Real},
    sexes = nothing;
    stages::Vector{StageDefinition} = build_stages(),
    rng::AbstractRNG = Random.default_rng()
)
    n_orgs = length(sizes)
    n_stages = length(stages)
    counts = zeros(Float64, n_stages)
    if n_orgs == 0
        return (counts, zeros(Float64, n_stages))
    end
    # The pooled-sex blocks come first, so the sizes below which sexes cannot be
    # distinguished are the upper edge of the last pooled block and the lower edge
    # of the first sex-specific one. They used to be the literals 22.0 and 14.5,
    # which is the same pair of numbers but stated twice: editing the stage table
    # moved the threshold without moving this code.
    pooled = filter(st -> st.sex === undifferentiated, stages)
    sexed = filter(st -> st.sex !== undifferentiated, stages)
    pooled_upper = isempty(pooled) ? 0.0 : last(last(pooled).size_range)
    sexed_lower = isempty(sexed) ? Inf : first(first(sexed).size_range)
    # Largest stage in the table, used as the upper catch-all for sizes beyond it.
    terminal_stage = isempty(stages) ? 0 : maximum(st.stage for st in stages)
    unknown_sex_fraction = 0.50

    has_sex = (sexes !== nothing && length(sexes) == n_orgs)
    for i in 1:n_orgs
        sz = Float64(sizes[i])
        if sz <= 0.0
            continue
        end
        if sz < sexed_lower
            # Too small to sex: assign to a pooled block by range, falling back to
            # the nearest pooled mean when the size sits outside every block.
            if !isempty(pooled)
                block = findfirst(st -> first(st.size_range) <= sz < last(st.size_range), pooled)
                if block === nothing
                    block = argmin([abs(sz - st.size_mean) for st in pooled])
                end
                counts[pooled[block].index] += 1.0
            end
        else
            is_fem = if has_sex
                sx = sexes[i]
                (sx == 1 || sx == :female || Symbol(sx) == :female)
            else
                rand(rng) < unknown_sex_fraction
            end
            target_sex = is_fem ? female : male
            matched = false
            for st in stages
                if st.sex === target_sex
                    b_min, b_max = st.size_range
                    if b_min <= sz < b_max || (sz >= b_min && st.stage == terminal_stage)
                        counts[st.index] += 1.0
                        matched = true
                        break
                    end
                end
            end
            if !matched
                candidates = filter(s -> s.sex === target_sex, stages)
                if !isempty(candidates)
                    dists = [abs(sz - s.size_mean) for s in candidates]
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

"""
    _fit_ols(x::Vector{Float64}, y::Vector{Float64})

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
    _fit_irls_logistic(cw_vec::Vector{Float64}, temp_vec::Vector{Float64}, y_vec::Vector{Float64})

Internal iteratively reweighted least squares (IRLS) logistic regression for
maturity ogive estimation with thermal sensitivity.
"""
function _fit_irls_logistic(
    cw_vec::Vector{Float64},
    temp_vec::Vector{Float64},
    y_vec::Vector{Float64};
    temp_reference::Float64 = 4.0
)
    n = length(y_vec)
    # Below this the temperature term is not identified and dropping it is more
    # stable than fitting a coefficient on noise.
    temp_variability_threshold::Float64 = 0.1
    has_temp_var = std(temp_vec) > temp_variability_threshold
    # Temperature enters centred on its reference, so the intercept means
    # maturity at that reference rather than at zero degrees. The reference is a
    # species parameter (`maturity_temp_reference`); it was the literal 4.0 here,
    # stated a second time.
    X = has_temp_var ?
        hcat(ones(n), log.(cw_vec), temp_vec .- temp_reference) :
        hcat(ones(n), log.(cw_vec))
    p_dim = size(X, 2)
    theta = zeros(p_dim)
    # Starting values: a size far above the smallest observed, with a moderate
    # slope, so the first iteration lands near a low, flat probability.
    theta[1] = -20.0
    theta[2] = 5.0
    if p_dim == 3
        theta[3] = -0.05
    end
    max_iterations = 30
    weight_floor = 1e-6
    ridge = 1e-4
    tolerance = 1e-5
    damping = 0.5
    min_slope = 0.5
    for _ in 1:max_iterations
        eta = X * theta
        mu = _logistic_safe.(eta)
        w = max.(mu .* (1.0 .- mu), weight_floor)
        z = eta .+ (y_vec .- mu) ./ w
        H = (X' .* w') * X + ridge * I
        g = (X' .* w') * z
        theta_next = H \ g
        if norm(theta_next - theta) < tolerance
            theta = theta_next
            break
        end
        theta = damping .* theta .+ damping .* theta_next
    end
slope = theta[2]
l50 = exp(-theta[1] / slope)
    temp_coef = (p_dim == 3) ? theta[3] : -0.05
    mean_y = mean(y_vec)
    p_mean = min(max(mean_y, 1e-6), 1.0 - 1e-6)
    ll_null = sum(y_vec .* log(p_mean) .+ (1.0 .- y_vec) .* log(1.0 - p_mean))
    p_pred = min.(max.(_logistic_safe.(X * theta), 1e-6), 1.0 - 1e-6)
    ll_fit = sum(y_vec .* log.(p_pred) .+ (1.0 .- y_vec) .* log.(1.0 .- p_pred))
    pseudo_r2 = (abs(ll_null) > 1e-6) ? max(0.0, 1.0 - ll_fit / ll_null) : 0.0
    return (l50, slope, temp_coef, pseudo_r2)
end

"""
    generate_mock_survey_data(; n_samples=500, rng=Random.default_rng(), behavior=default_species_behavior())

Generate a synthetic research survey `DataFrame` representing observational
field records for model calibration.
"""
function generate_mock_survey_data(;
    n_samples::Int = 500,
    rng::AbstractRNG = Random.default_rng(),
    behavior::SpeciesBehavior = default_species_behavior()
)
    sizes = zeros(Float64, n_samples)
    sexes = zeros(Int, n_samples)
    mats = zeros(Int, n_samples)
    masses = zeros(Float64, n_samples)
    temps = zeros(Float64, n_samples)
    fecs = zeros(Float64, n_samples)
    stages_arr = zeros(Int, n_samples)
    shell_ages = zeros(Float64, n_samples)
    stage_weights = [0.18, 0.16, 0.15, 0.14, 0.12, 0.10, 0.08, 0.05, 0.02]
    stage_choices = 4:12
    for i in 1:n_samples
        cum_r = rand(rng)
        acc = 0.0
        chosen_stage = 4
        for (idx, w) in enumerate(stage_weights)
            acc += w
            if cum_r <= acc || idx == length(stage_weights)
                chosen_stage = stage_choices[idx]
                break
            end
        end
        s = rand(rng) < 0.50 ? 1 : 0
        t = rand(rng, truncated(Normal(3.5, 1.5), 0.5, 8.0))
        sz = get_stage_size(behavior, chosen_stage, s; type = :random, rng = rng)
        p_mat = maturity(behavior, s, log(sz); temp = t, rng = rng)
        mat = (rand(rng) < p_mat) ? 1 : 0
        age = (mat == 0) ? rand(rng, Uniform(0.2, 1.0)) : rand(rng, Uniform(1.0, 5.0))
        wt = calculate_mass(behavior, sz, s; mat = mat, stochastic = true, rng = rng)
        fec = 0.0
        if s == 1 && mat == 1
            fec = Float64(fecundity(
                behavior, sz;
                is_primiparous = (age <= 2.0),
                shell_age = age,
                rng = rng
            ))
        end
        sizes[i] = sz
        sexes[i] = s
        mats[i] = mat
        masses[i] = wt
        temps[i] = t
        fecs[i] = fec
        stages_arr[i] = chosen_stage
        shell_ages[i] = age
    end
    return DataFrame(
        size = sizes,
        sex = sexes,
        maturity = mats,
        mass = masses,
        temp_c = temps,
        fecundity = fecs,
        stage = stages_arr,
        shell_age = shell_ages
    )
end

"""
    fit_empirical_biology_model(data::DataFrame; params::SpeciesParams=SpeciesParams())

Calibrate an `EmpiricalBiologicalModel` from empirical observational or research
survey data.

Every fallback used when a stratum is too sparse to fit is read from `params`,
so a species change carries through the calibration instead of leaving the
declared constants behind as a second, silent set of species values.
"""
function fit_empirical_biology_model(
    data::DataFrame;
    params::SpeciesParams = SpeciesParams()
)
    required_cols = [:size, :sex, :maturity, :mass, :temp_c, :fecundity]
    for col in required_cols
        if !(col in propertynames(data))
            error("Data must contain required column: $(col)")
        end
    end
    n_records = nrow(data)
    if n_records < 10
        error("At least 10 observations required to fit empirical model, got $(n_records)")
    end
    mass_params = Dict{Symbol, Tuple{Float64, Float64, Float64}}()
    maturity_params = Dict{Symbol, Tuple{Float64, Float64, Float64}}()
    fecundity_params = Dict{Symbol, Tuple{Float64, Float64, Float64}}()
    stage_modals = Dict{Int, Tuple{Float64, Float64}}()
    r2_scores = Dict{Symbol, Float64}()

    # Minimum records in a stratum before its regression is trusted, and the R²
    # reported when it is not fitted at all. These are diagnostics, not estimates,
    # so they are stated once here rather than at each use.
    min_stratum_records = 5
    min_maturity_records = 10
    min_stage_records = 3
    mass_fallback_r2 = 0.95
    maturity_fallback_r2 = 0.85
    fecundity_fallback_r2 = 0.80

    # The stage table and the size lower bounds are read through a behaviour built
    # from `params`, so a stage missing from the data is filled in from the same
    # configuration the simulation itself uses rather than from the defaults.
    behavior = SpeciesBehaviorImpl(params)
    stages_for_modals = build_stages(behavior)
    llbm_of(stg) = stage_size_lower_bound(behavior, Float64(stg), :male)

    # 1. Allometric Mass Parameters
    strata_definitions = [
        (
            :male_imm,
            (data.sex .== 0) .&& (data.maturity .== 0),
            (
                params.mass_alpha_male_immature,
                params.mass_beta_male_immature,
                params.mass_sigma
            )
        ),
        (
            :male_mat,
            (data.sex .== 0) .&& (data.maturity .== 1),
            (params.mass_alpha_male_mature, params.mass_beta_male_mature, params.mass_sigma)
        ),
        (
            :female_imm,
            (data.sex .== 1) .&& (data.maturity .== 0),
            (
                params.mass_alpha_female_immature,
                params.mass_beta_female_immature,
                params.mass_sigma
            )
        ),
        (
            :female_mat,
            (data.sex .== 1) .&& (data.maturity .== 1),
            (
                params.mass_alpha_female_mature,
                params.mass_beta_female_mature,
                params.mass_sigma
            )
        )
    ]
    for (strat_sym, mask, fallback) in strata_definitions
        sub_sz = data.size[mask]
        sub_wt = data.mass[mask]
        valid = (sub_sz .> 0.0) .&& (sub_wt .> 0.0)
        if count(valid) >= min_stratum_records
            x = log.(Float64.(sub_sz[valid]))
            y = log.(Float64.(sub_wt[valid]))
            intercept, slope, sigma, r2 = _fit_ols(x, y)
            mass_params[strat_sym] = (exp(intercept), slope, sigma)
            r2_scores[strat_sym] = r2
        else
            mass_params[strat_sym] = fallback
            r2_scores[strat_sym] = mass_fallback_r2
        end
    end

    # 2. Logistic Maturity Parameters
    for (sex_sym, s_code, fallback) in [
        (
            :male,
            0,
            (
                params.maturity_l50_male,
                params.maturity_slope_male,
                -0.05
            )
        ),
        (
            :female,
            1,
            (
                params.maturity_l50_female,
                params.maturity_slope_female,
                -0.05
            )
        )
    ]
        mask = (data.sex .== s_code) .&& (data.size .>= params.maturity_size_gate)
        sub_sz = data.size[mask]
        sub_temp = data.temp_c[mask]
        sub_mat = Float64.(data.maturity[mask])
        if count(mask) >= min_maturity_records && length(unique(sub_mat)) > 1
            l50, slope, temp_coef, pseudo_r2 = _fit_irls_logistic(
                Float64.(sub_sz),
                Float64.(sub_temp),
                sub_mat;
                temp_reference = params.maturity_temp_reference
            )
            maturity_params[sex_sym] = (l50, slope, temp_coef)
            r2_scores[Symbol("maturity_", sex_sym)] = pseudo_r2
        else
            maturity_params[sex_sym] = fallback
            r2_scores[Symbol("maturity_", sex_sym)] = maturity_fallback_r2
        end
    end

    # 3. Fecundity Parameters
    # Parity is taken from shell age where the data carry it. Where they do not,
    # the split is made on size at the reference that separates the two forms,
    # which is a weaker criterion and is used only because nothing better exists
    # in the columns available.
    has_shell_age = :shell_age in propertynames(data)
    primiparous_shell_age = params.primiparous_shell_age_max
    parity_split_size = params.fecundity_parity_size_split
    fem_mat_mask = (data.sex .== 1) .&& (data.maturity .== 1) .&& (data.fecundity .> 0.0)
    for (par_sym, par_mask, fallback) in [
        (
            :primiparous,
            has_shell_age ?
                (fem_mat_mask .&& (data.shell_age .<= primiparous_shell_age)) :
                (fem_mat_mask .&& (data.size .< parity_split_size)),
            (
                params.fecundity_primiparous_intercept,
                params.fecundity_primiparous_slope,
                params.fecundity_primiparous_err
            )
        ),
        (
            :multiparous,
            has_shell_age ?
                (fem_mat_mask .&& (data.shell_age .> primiparous_shell_age)) :
                (fem_mat_mask .&& (data.size .>= parity_split_size)),
            (
                params.fecundity_multiparous_intercept,
                params.fecundity_multiparous_slope,
                params.fecundity_multiparous_err
            )
        )
    ]
        sub_sz = data.size[par_mask]
        sub_fec = data.fecundity[par_mask]
        valid = (sub_sz .> 0.0) .&& (sub_fec .> 0.0)
        if count(valid) >= min_stratum_records
            x = log10.(Float64.(sub_sz[valid]))
            y = log10.(Float64.(sub_fec[valid]))
            intercept, slope, sigma, r2 = _fit_ols(x, y)
            fecundity_params[par_sym] = (intercept, slope, sigma)
            r2_scores[par_sym] = r2
        else
            fecundity_params[par_sym] = fallback
            r2_scores[par_sym] = fecundity_fallback_r2
        end
    end

    # 4. Stage Modal Dimensions
    stage_col = (:stage in propertynames(data)) ? data.stage : nothing
    # Stages the survey is expected to cover. The span comes from the stage table
    # so it follows a species change; the endpoints used to be the literals 4 and
    # 12, which are the pooled recruit bin and the largest male stage for the
    # default table and nothing else.
    stage_span = isempty(stages_for_modals) ? (0:-1) :
                 (minimum(st.stage for st in stages_for_modals):
                  maximum(st.stage for st in stages_for_modals))
    for stg in stage_span
        if stage_col !== nothing && count(stage_col .== stg) >= min_stage_records
            szs_stg = Float64.(data.size[stage_col .== stg])
            # A single observation has no spread; floor the SD so downstream draws
            # from this stage are not degenerate.
            stage_modals[stg] = (mean(szs_stg), max(0.5, std(szs_stg)))
        else
            # From the lower bounds of the stage and the one after it, so an
            # unobserved stage is represented by the middle of its own interval.
            lo = llbm_of(stg)
            hi = llbm_of(stg + 1)
            mid_sz = 0.5 * (lo + hi)
            # Floor the SD on an interval no narrower than this, so a species
            # whose stages are close together still produces a usable spread.
            min_modal_sd = 0.8
            sd_sz = max(min_modal_sd, 0.5 * (hi - lo) / 3.0)
            stage_modals[stg] = (mid_sz, sd_sz)
        end
    end

    # 5. Trait Covariance Matrix
    trait_names = [:size, :mass, :maturity, :fecundity]
    trait_matrix = zeros(Float64, n_records, length(trait_names))
    trait_matrix[:, 1] = Float64.(data.size)
    trait_matrix[:, 2] = Float64.(data.mass)
    trait_matrix[:, 3] = Float64.(data.maturity)
    trait_matrix[:, 4] = Float64.(data.fecundity)
    raw_cov = cov(trait_matrix)
    # The ridge keeps the covariance positive definite when a trait is constant in
    # the sample, which would otherwise make a multivariate draw fail.
    trait_covariance_ridge = 1e-4
    sym_cov = Matrix(Hermitian(raw_cov + trait_covariance_ridge * I))

    return EmpiricalBiologicalModel(
        mass_params,
        maturity_params,
        fecundity_params,
        stage_modals,
        trait_names,
        sym_cov,
        r2_scores,
        n_records
    )
end

"""
    estimate_demographics(model::EmpiricalBiologicalModel; size, sex, temp=4.0, shell_age=1.0, params=SpeciesParams(), rng=Random.default_rng())

Predict demographic traits for an individual from an empirical model.

The maturity size gate, its temperature response, and the fecundity senescence
schedule come from `params` rather than from literals, so a calibrated model and
the species configuration describe the same animal.
"""
function estimate_demographics(
    model::EmpiricalBiologicalModel;
    size::Real,
    sex,
    temp::Real = 4.0,
    shell_age::Real = 1.0,
    params::SpeciesParams = SpeciesParams(),
    rng::AbstractRNG = Random.default_rng()
)
    if size <= 0.0
        return (
            maturity_prob = 0.0,
            is_mature = 0,
            mass = 0.0,
            fecundity = 0.0,
            estimated_stage = 4
        )
    end
    is_fem = (sex == 1 || sex == :female || Symbol(sex) == :female)
    sex_sym = is_fem ? :female : :male
    size_gate = params.maturity_size_gate
    temp_ref = params.maturity_temp_reference
    p_mat = if size < size_gate
        0.0
    else
        l50, slope, t_coef = model.maturity_params[sex_sym]
        # The fitted temperature coefficient enters the logit directly. The L50
        # is the fitted value at the reference temperature; no additional shift
        # is applied, to avoid double-counting the temperature response.
        eff_l50 = max(size_gate, l50)
        lin_pred = slope * (log(size) - log(eff_l50)) + t_coef * (temp - temp_ref)
        _logistic_safe(lin_pred)
    end
    mat_val = (rand(rng) < p_mat) ? 1 : 0
    strat = if !is_fem
        (mat_val == 1) ? :male_mat : :male_imm
    else
        (mat_val == 1) ? :female_mat : :female_imm
    end
    alpha, beta_val, sigma = model.mass_params[strat]
    base_mass = alpha * (Float64(size)^beta_val)
    mass_val = base_mass * exp(rand(rng, Normal(0.0, sigma)) - 0.5 * sigma^2)

    fecundity_val = 0.0
    if is_fem && mat_val == 1
        par_sym = (shell_age <= params.primiparous_shell_age_max) ? :primiparous :
                  :multiparous
        a, b, sig = model.fecundity_params[par_sym]
        log10_fec = a + b * log10(size) + rand(rng, Normal(0.0, sig))
        senescence = (par_sym === :multiparous &&
                      shell_age > params.fecundity_senescence_threshold) ?
                      max(
                          params.fecundity_senescence_floor,
                          1.0 - params.fecundity_senescence_rate *
                          (shell_age - params.fecundity_senescence_threshold)
                      ) : 1.0
        fecundity_val = max(0.0, round(10.0^log10_fec * senescence))
    end

    best_stage = 4
    min_dist = Inf
    for (stg, (m, _)) in model.stage_modals
        d = abs(size - m)
        if d < min_dist
            min_dist = d
            best_stage = stg
        end
    end

    return (
        maturity_prob = p_mat,
        is_mature = mat_val,
        mass = mass_val,
        fecundity = fecundity_val,
        estimated_stage = best_stage,
    )
end

function sample_organism_traits(
    model::EmpiricalBiologicalModel;
    size::Real,
    sex,
    temp::Real = 4.0,
    shell_age::Real = 1.0,
    params::SpeciesParams = SpeciesParams(),
    rng::AbstractRNG = Random.default_rng()
)
    return estimate_demographics(
        model;
        size = size,
        sex = sex,
        temp = temp,
        shell_age = shell_age,
        params = params,
        rng = rng
    )
end

"""
    summarize_empirical_model(model::EmpiricalBiologicalModel; io::IO=stdout)

Print a comprehensive summary of an `EmpiricalBiologicalModel`.
"""
function summarize_empirical_model(
    model::EmpiricalBiologicalModel;
    io::IO = stdout
)
    println(io, "="^78)
    println(io, "  Empirical Biological Model Summary (N = $(model.sample_size) records)")
    println(io, "="^78)
    println(io, "\n--- Allometric Mass Parameters (W = α * Size^β) ---")
    println(io, rpad("Stratum", 14), rpad("α", 12), rpad("β", 10), rpad("σ", 10), "R²")
    println(io, "-"^56)
    for strat in [:male_imm, :male_mat, :female_imm, :female_mat]
        if haskey(model.mass_params, strat)
            a, b, s = model.mass_params[strat]
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
    println(io, "\n--- Brood Fecundity Scaling (log10(Eggs) = a + b * log10(Size)) ---")
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
    println(io, "\n--- Stage Modal Dimensions ---")
    println(io, rpad("Stage", 10), rpad("Mean Size (mm)", 16), "SD Size (mm)")
    println(io, "-"^38)
    for stg in sort(collect(keys(model.stage_modals)))
        m_sz, sd_sz = model.stage_modals[stg]
        println(
            io,
            rpad(string(stg), 10),
            rpad(string(round(m_sz, digits = 2)), 16),
            string(round(sd_sz, digits = 2))
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

