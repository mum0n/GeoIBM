"""
Data aggregation and statistical analysis tools for snow crab IBM outputs.

Provides functions to compute empirical maturity ogives, population snapshots,
and growth metrics across instars and life stages.
"""

using DataFrames
using LinearAlgebra
using Statistics

"""
    compute_empirical_ogive(df::DataFrame, target_sex::CrabSex, bins::AbstractRange)

Calculate the empirical proportion of morphometrically mature adult crabs within
discrete carapace width bins for a specified sex.

# Arguments
- `df::DataFrame`: Agent data table containing `:size_mm`, `:sex`, and `:stage` columns.
- `target_sex::CrabSex`: Sex to evaluate (`male` or `female`).
- `bins::AbstractRange`: Carapace width bin boundaries (e.g. `20.0:5.0:120.0`).

# Returns
Tuple `(bin_centers, prop_mature)`:
- `bin_centers::Vector{Float64}`: Midpoint carapace width of each evaluated bin.
- `prop_mature::Vector{Float64}`: Proportion of adult crabs in each evaluated bin.
"""
function compute_empirical_ogive(
    df::DataFrame,
    target_sex::CrabSex,
    bins::AbstractRange
)
    sex_df = filter(row -> row.sex == target_sex, df)
    bin_centers = Float64[]
    prop_mature = Float64[]

    n_bins = length(bins) - 1
    for k in 1:n_bins
        b_min = bins[k]
        b_max = bins[k + 1]

        in_bin = filter(row -> b_min <= row.size_mm < b_max, sex_df)
        if nrow(in_bin) > 0
            push!(bin_centers, (b_min + b_max) / 2.0)
            n_adults = count(s -> s == adult, in_bin.stage)
            push!(prop_mature, n_adults / nrow(in_bin))
        end
    end

    return (bin_centers, prop_mature)
end

"""
    compute_empirical_ogive(model, target_sex, bins)

Calculate the empirical proportion of mature crabs in `model` for `target_sex`.
Accepts `target_sex` as a `CrabSex` enum (`male`, `female`) or integer (0 = male, 1 = female).
"""
function compute_empirical_ogive(
    model,
    target_sex::Union{CrabSex, Integer},
    bins::AbstractRange
)
    crabs = collect(allagents(model))
    if isempty(crabs)
        return (Float64[], Float64[])
    end

    df = DataFrame(
        size_mm = [c.size_mm for c in crabs],
        sex = [c.sex for c in crabs],
        stage = [c.stage for c in crabs]
    )

    actual_sex = if target_sex isa Integer
        (target_sex == 0) ? male : female
    else
        target_sex
    end

    return compute_empirical_ogive(df, actual_sex, bins)
end


"""
    summarize_population(model)

Compute a statistical snapshot of the current living population in `model`,
including demographic stage distributions and stock biomass indicators.

# Returns
NamedTuple with fields:
- `total_crabs::Int`: Count of alive crabs.
- `total_biomass_kg::Float64`: Total wet biomass of living crabs in kilograms.
- `female_ssb_kg::Float64`: Spawning stock biomass of mature females (kg).
- `male_mature_biomass_kg::Float64`: Biomass of mature males (kg).
- `total_ssb_kg::Float64`: Total spawning stock biomass (mature males + females) in kg.
- `legal_biomass_kg::Float64`: Commercially fishable biomass of legal males in kg.
- `mean_weight_g::Float64`: Mean body weight across the population in grams.
- `n_males::Int`: Total males.
- `n_females::Int`: Total females.
- `n_adults::Int`: Total mature adults.
- `n_adult_males::Int`: Total adult males.
- `n_adult_females::Int`: Total adult females.
- `n_juveniles::Int`: Total benthic juveniles.
- `n_larvae::Int`: Total planktonic larvae (zoea + megalopa).
- `mean_cw::Float64`: Mean carapace width (mm).
- `cw_by_instar::Dict{Int, Float64}`: Mean carapace width mapped by instar.
- `count_by_instar::Dict{Int, Int}`: Crab count mapped by instar.
"""
function summarize_population(model)
    crabs = collect(allagents(model))
    total = length(crabs)

    props = abmproperties(model)
    tot_spawned = get(props, :total_eggs_spawned, 0)
    tot_settled = get(props, :total_recruits_settled, 0)

    if total == 0
        return (
            total_crabs = 0,
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
            mean_weight_g = 0.0,
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
            mean_cw = 0.0,
            cw_by_instar = Dict{Int, Float64}(),
            count_by_instar = Dict{Int, Int}()
        )
    end

    n_m = count(c -> c.sex === male, crabs)
    n_f = count(c -> c.sex === female, crabs)
    n_ad = count(c -> c.stage === adult, crabs)
    n_adm = count(c -> c.stage === adult && c.sex === male, crabs)
    n_adf = count(c -> c.stage === adult && c.sex === female, crabs)
    n_ovig = count(c -> c.stage === adult && c.sex === female && c.eggs_count > 0, crabs)
    tot_eggs = sum(c.eggs_count for c in crabs; init = 0)
    n_juv = count(c -> c.stage === juvenile, crabs)
    n_larv = count(c -> c.stage === zoea || c.stage === megalopa, crabs)

    mean_size = mean(c.size_mm for c in crabs)
    mean_wt = mean(c.weight_g for c in crabs)

    # Stock biomass metrics (converted from grams to kilograms)
    tot_biomass = sum(c.weight_g for c in crabs) / 1000.0
    f_ssb = sum((c.weight_g for c in crabs if c.stage === adult && c.sex === female);
                init = 0.0) / 1000.0
    m_mat_b = sum((c.weight_g for c in crabs if c.stage === adult && c.sex === male);
                  init = 0.0) / 1000.0
    tot_ssb = f_ssb + m_mat_b

    leg_b = sum(
        (c.weight_g for c in crabs if c.stage === adult && c.sex === male &&
         c.size_mm >= model.params.legal_size_male);
        init = 0.0
    ) / 1000.0

    instars_present = unique(c.instar for c in crabs)
    cw_by_instar = Dict{Int, Float64}()
    count_by_instar = Dict{Int, Int}()
    for inst in sort(instars_present)
        inst_crabs = filter(c -> c.instar == inst, crabs)
        cw_by_instar[inst] = mean(c.size_mm for c in inst_crabs)
        count_by_instar[inst] = length(inst_crabs)
    end

    return (
        total_crabs = total,
        total_biomass_kg = tot_biomass,
        female_ssb_kg = f_ssb,
        male_mature_biomass_kg = m_mat_b,
        total_ssb_kg = tot_ssb,
        legal_biomass_kg = leg_b,
        total_mature_females = n_adf,
        total_mature_males = n_adm,
        total_fishable_male_mass_kg = leg_b,
        total_mature_male_mass_kg = m_mat_b,
        total_mature_female_mass_kg = f_ssb,
        total_egg_production = tot_spawned,
        mean_weight_g = mean_wt,
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
        mean_cw = mean_size,
        cw_by_instar = cw_by_instar,
        count_by_instar = count_by_instar
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
- `mother_weight_g`: Maternal body weight in grams.
- `recruits`: Settled Instar 4 benthic recruits from the clutch.
- `recruits_per_spawner`: Ratio of recruits produced per unit spawner.
- `density_feedback`: Benthic adult compensatory density feedback factor (1 + N/K).
"""
function extract_stock_recruitment(model)
    props = abmproperties(model)
    if haskey(props, :clutch_recruitment_records)
        raw_df = props[:clutch_recruitment_records]
        if nrow(raw_df) > 0
            df = DataFrame(
                time = raw_df.hatch_time,
                spawners = raw_df.eggs_count ./ 1000.0,
                eggs_count = raw_df.eggs_count,
                mother_weight_g = raw_df.mother_weight_g,
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
        eggs_count = Int[],
        mother_weight_g = Float64[],
        recruits = Int[],
        density_feedback = Float64[],
        recruits_per_spawner = Float64[]
    )
end

"""
    extract_stock_recruitment(agent_df::DataFrame, model_df::DataFrame; lag_days=365.0, window_days=365.0)

Derive population-level annual stock-recruitment timeseries from simulation dataframes.

For each annual cohort window `[t, t + window_days]`, maternal spawner proxy (eggs spawned)
is paired with the resulting benthic recruitment settling in the subsequent lagged window
`[t + lag_days, t + lag_days + window_days]`.

# Arguments
- `agent_df::DataFrame`: Time-stepped agent data collected during simulation.
- `model_df::DataFrame`: Time-stepped model-level data (`:total_eggs_spawned`, `:total_recruits_settled`).
- `lag_days::Real`: Egg incubation plus larval development lag (default: 365.0 days).
- `window_days::Real`: Annual aggregation window (default: 365.0 days).

# Returns
`DataFrame` with columns:
- `year::Int`: Spawning cohort index (1, 2, ...).
- `time_spawn::Float64`: Spawning cohort start time.
- `spawners::Float64`: Thousands of eggs spawned in the cohort window.
- `eggs_spawned::Int`: Total eggs extruded in the cohort window.
- `recruits::Int`: Net new Instar 4 recruits settled during the lagged window.
- `recruits_per_spawner::Float64`: Ratio of recruits per unit spawner.
"""
function extract_stock_recruitment(
    agent_df::DataFrame,
    model_df::DataFrame;
    lag_days::Real = 365.0,
    window_days::Real = 365.0
)
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
    n_cohorts = floor(Int, (t_max - lag_days) / window_days)

    res_year = Int[]
    res_t_spawn = Float64[]
    res_spawners = Float64[]
    res_eggs = Int[]
    res_recruits = Int[]
    res_rps = Float64[]

    for y in 1:n_cohorts
        t_spawn_start = (y - 1) * window_days
        t_spawn_end = y * window_days
        t_rec_start = t_spawn_start + lag_days
        t_rec_end = t_spawn_end + lag_days

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
    r2_bh = clamp(1.0 - (ss_res_bh / ss_tot), 0.0, 1.0)
    aic_bh = n * log(max(1e-12, ss_res_bh / n)) + 4.0
    r_max_bh = (beta_bh > 0.0) ? (alpha_bh / beta_bh) : Inf
    s_50_bh = (beta_bh > 0.0) ? (1.0 / beta_bh) : Inf

    # 2. Ricker fit: ln(R/S) = u - v S
    y_rk = log.(r_sub ./ s_sub)
    y_rk_bar = mean(y_rk)
    cov_s_yrk = sum((s_sub[i] - s_bar) * (y_rk[i] - y_rk_bar) for i in 1:n)
    v_rk = -cov_s_yrk / var_s
    u_rk = y_rk_bar + v_rk * s_bar

    alpha_rk = exp(clamp(u_rk, -20.0, 20.0))
    beta_rk = max(0.0, v_rk)

    rk_pred = s -> alpha_rk * s * exp(-beta_rk * s)
    ss_res_rk = sum((r_sub[i] - rk_pred(s_sub[i]))^2 for i in 1:n)
    r2_rk = clamp(1.0 - (ss_res_rk / ss_tot), 0.0, 1.0)
    aic_rk = n * log(max(1e-12, ss_res_rk / n)) + 4.0
    s_max_rk = (beta_rk > 0.0) ? (1.0 / beta_rk) : Inf
    r_max_rk = (beta_rk > 0.0) ? (alpha_rk / (beta_rk * exp(1.0))) : Inf

    best_m = (aic_bh <= aic_rk) ? :beverton_holt : :ricker

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
    println(buf, "       SNOW CRAB STOCK-RECRUITMENT ANALYSIS REPORT       ")
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
    data = Dict{String, Vector{Float64}}()
    for h in header
        data[h] = Float64[]
    end
    for line in lines[2:end]
        parts = split(line, ',')
        for (i, h) in enumerate(header)
            val = (i <= length(parts)) ? tryparse(Float64, strip(parts[i])) : nothing
            push!(data[h], val === nothing ? 0.0 : val)
        end
    end
    df = DataFrame()
    for h in header
        df[!, Symbol(h)] = data[h]
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
    extract_annual_production(agent_df::DataFrame, model_df::DataFrame;
                              dt_year::Float64 = 365.0,
                              frequency_days::Float64 = dt_year,
                              snapshot_day::Union{Nothing, Float64} = 274.0)

Aggregate simulation output into abundance, demographic components, commercial catches,
and realized surplus production metrics sampled at specified intervals.

Sampling times are governed by `frequency_days` (default: 365.0 days = 1 year) and
`snapshot_day` (default: 274.0 = ~1 September). When `frequency_days` is annual (365 days),
outputs occur at day `snapshot_day` of each year (t = (yr - 1) * 365 + snapshot_day).
When sub-annual or arbitrary frequencies are specified, outputs are recorded at intervals
of `frequency_days` starting from `snapshot_day` (or `frequency_days` if `snapshot_day`
is not customized).

# Arguments
- `agent_df::DataFrame`: Agent time-series metrics.
- `model_df::DataFrame`: Model time-series metrics.
- `dt_year::Float64`: Days per annual time step (default: 365.0).
- `frequency_days::Float64`: Output sampling frequency in days (default: 365.0 = 1 year).
- `snapshot_day::Union{Nothing, Float64}`: Day of calendar year (1-365) to sample snapshots
  (default: 274.0 = ~1 September).

# Returns
`DataFrame` with columns:
- `year::Int`: Sample period index (1, 2, ...).
- `time::Float64`: Elapsed days at snapshot.
- `total_numbers::Float64`: Total living crab abundance (N_t).
- `legal_numbers::Float64`: Legal-sized adult male abundance (N_{legal, t}).
- `mature_females::Float64`: Total number of morphometrically mature females.
- `mature_males::Float64`: Total number of morphometrically mature males.
- `fishable_male_biomass_kg::Float64`: Total wet mass of legal fishable males in kg.
- `mature_male_biomass_kg::Float64`: Total wet mass of mature males in kg.
- `mature_female_biomass_kg::Float64`: Total wet mass of mature females (SSB_F) in kg.
- `total_egg_production::Float64`: Cumulative egg production across the stock.
- `catch_total::Float64`: Harvest removal during interval in numbers (C_t).
- `catch_legal::Float64`: Commercial catch of legal males during interval.
- `surplus_production_total::Float64`: Realized production (N_{t+1} - N_t + C_t).
- `surplus_production_legal::Float64`: Realized legal production.
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

    first_day = if snapshot_day !== nothing
        (snapshot_day == 274.0 && freq < 274.0) ? freq : snapshot_day
    else
        freq
    end

    t_targets = Float64[]
    t_curr = first_day
    while t_curr <= t_max
        push!(t_targets, t_curr)
        t_curr += freq
    end
    n_samples = length(t_targets)

    if n_samples < 1
        return DataFrame(
            year = Int[],
            time = Float64[],
            total_numbers = Float64[],
            legal_numbers = Float64[],
            mature_females = Float64[],
            mature_males = Float64[],
            fishable_male_biomass_kg = Float64[],
            mature_male_biomass_kg = Float64[],
            mature_female_biomass_kg = Float64[],
            total_egg_production = Float64[],
            catch_total = Float64[],
            catch_legal = Float64[],
            surplus_production_total = Float64[],
            surplus_production_legal = Float64[]
        )
    end

    num_col = if :length_stage in propertynames(agent_df)
        :length_stage
    elseif :count_stage in propertynames(agent_df)
        :count_stage
    elseif :total_crabs in propertynames(agent_df)
        :total_crabs
    elseif :total_numbers in propertynames(agent_df)
        :total_numbers
    else
        first(names(agent_df))
    end

    legal_col = if :count_is_legal_male in propertynames(agent_df)
        :count_is_legal_male
    elseif :legal_numbers in propertynames(agent_df)
        :legal_numbers
    elseif :count_is_adult_male in propertynames(agent_df)
        :count_is_adult_male
    else
        nothing
    end

    mat_f_col = if :count_is_adult_female in propertynames(agent_df)
        :count_is_adult_female
    elseif :mature_females in propertynames(agent_df)
        :mature_females
    elseif :n_adult_females in propertynames(agent_df)
        :n_adult_females
    else
        nothing
    end

    mat_m_col = if :count_is_adult_male in propertynames(agent_df)
        :count_is_adult_male
    elseif :mature_males in propertynames(agent_df)
        :mature_males
    elseif :n_adult_males in propertynames(agent_df)
        :n_adult_males
    else
        nothing
    end

    leg_mass_col = if :sum_weight_if_legal_male in propertynames(agent_df)
        :sum_weight_if_legal_male
    elseif :fishable_male_biomass_kg in propertynames(agent_df)
        :fishable_male_biomass_kg
    elseif :legal_biomass_kg in propertynames(agent_df)
        :legal_biomass_kg
    else
        nothing
    end

    mat_m_mass_col = if :sum_weight_if_adult_male in propertynames(agent_df)
        :sum_weight_if_adult_male
    elseif :mature_male_biomass_kg in propertynames(agent_df)
        :mature_male_biomass_kg
    else
        nothing
    end

    mat_f_mass_col = if :sum_weight_if_adult_female in propertynames(agent_df)
        :sum_weight_if_adult_female
    elseif :mature_female_biomass_kg in propertynames(agent_df)
        :mature_female_biomass_kg
    elseif :female_ssb_kg in propertynames(agent_df)
        :female_ssb_kg
    else
        nothing
    end

    total_n_vec = Float64[]
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
        l_val = (legal_col !== nothing) ? Float64(agent_df[idx, legal_col]) : 0.0
        push!(legal_n_vec, l_val)

        mf_val = (mat_f_col !== nothing) ? Float64(agent_df[idx, mat_f_col]) : 0.0
        push!(mat_f_n_vec, mf_val)

        mm_val = (mat_m_col !== nothing) ? Float64(agent_df[idx, mat_m_col]) : 0.0
        push!(mat_m_n_vec, mm_val)

        lm_val = if leg_mass_col === :sum_weight_if_legal_male
            Float64(agent_df[idx, leg_mass_col]) / 1000.0
        elseif leg_mass_col !== nothing
            Float64(agent_df[idx, leg_mass_col])
        else
            0.0
        end
        push!(leg_mass_vec, lm_val)

        mmm_val = if mat_m_mass_col === :sum_weight_if_adult_male
            Float64(agent_df[idx, mat_m_mass_col]) / 1000.0
        elseif mat_m_mass_col !== nothing
            Float64(agent_df[idx, mat_m_mass_col])
        else
            0.0
        end
        push!(mat_m_mass_vec, mmm_val)

        mfm_val = if mat_f_mass_col === :sum_weight_if_adult_female
            Float64(agent_df[idx, mat_f_mass_col]) / 1000.0
        elseif mat_f_mass_col !== nothing
            Float64(agent_df[idx, mat_f_mass_col])
        else
            0.0
        end
        push!(mat_f_mass_vec, mfm_val)
    end

    catch_vec = zeros(Float64, n_samples)
    egg_prod_vec = zeros(Float64, n_samples)
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

    if :total_eggs_spawned in propertynames(model_df)
        for k in 1:n_samples
            t_target = t_targets[k]
            idx = argmin(abs.(model_df.time .- t_target))
            egg_prod_vec[k] = Float64(model_df.total_eggs_spawned[idx])
        end
    elseif :total_egg_production in propertynames(model_df)
        for k in 1:n_samples
            t_target = t_targets[k]
            idx = argmin(abs.(model_df.time .- t_target))
            egg_prod_vec[k] = Float64(model_df.total_egg_production[idx])
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

    return DataFrame(
        year = year_vec,
        time = time_vec,
        total_numbers = total_n_vec,
        legal_numbers = legal_n_vec,
        mature_females = mat_f_n_vec,
        mature_males = mat_m_n_vec,
        fishable_male_biomass_kg = leg_mass_vec,
        mature_male_biomass_kg = mat_m_mass_vec,
        mature_female_biomass_kg = mat_f_mass_vec,
        total_egg_production = egg_prod_vec,
        catch_total = catch_vec,
        catch_legal = catch_vec,
        surplus_production_total = sp_total,
        surplus_production_legal = sp_legal
    )
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

    # 1. Analytical process-error regression: P_t = a1 * N_t - a2 * N_t^2
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
        if a1 > 0.01 && a2 > 1e-8
            r_init = clamp(a1, 0.05, 2.5)
            k_init = clamp(a1 / a2, maximum(n_vals) * 0.5, maximum(n_vals) * 5.0)
        end
    end

    # 2. Refine trajectory fit via grid-coordinate minimization of RSS
    function simulate_trajectory(r_val, k_val)
        pred = zeros(Float64, n)
        pred[1] = n_vals[1]
        for t in 1:(n - 1)
            pop_curr = max(1.0, pred[t])
            growth = r_val * pop_curr * (1.0 - pop_curr / k_val)
            pred[t + 1] = max(0.0, pop_curr + growth - c_vals[t])
        end
        return pred
    end

    best_r = r_init
    best_k = k_init
    best_pred = simulate_trajectory(best_r, best_k)
    best_rss = sum((n_vals .- best_pred) .^ 2)

    # Multi-scale grid search around initial values
    for r_scale in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        for k_scale in [0.7, 0.85, 1.0, 1.15, 1.3, 1.6]
            r_try = clamp(r_init * r_scale, 0.02, 3.0)
            k_try = clamp(k_init * k_scale, maximum(n_vals) * 0.5, maximum(n_vals) * 5.0)
            pred_try = simulate_trajectory(r_try, k_try)
            rss_try = sum((n_vals .- pred_try) .^ 2)
            if rss_try < best_rss
                best_rss = rss_try
                best_r = r_try
                best_k = k_try
                best_pred = pred_try
            end
        end
    end

    mean_obs = mean(n_vals)
    tss = sum((n_vals .- mean_obs) .^ 2)
    r2 = (tss > 0.0) ? max(0.0, 1.0 - (best_rss / tss)) : 0.0
    rmse = sqrt(best_rss / n)
    aic = n * log(max(1e-6, best_rss / n)) + 2.0 * 2.0

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

    function simulate_dd(s_val, alpha_val, beta_val)
        pred = zeros(Float64, n)
        for i in 1:min(n, lag + 1)
            pred[i] = n_vals[i]
        end
        for t in (lag + 1):(n - 1)
            spawner_idx = max(1, t - lag + 1)
            s_stock = max(1.0, pred[spawner_idx])
            recruits = (alpha_val * s_stock) / (1.0 + beta_val * s_stock)
            escapement = max(0.0, pred[t] - c_vals[t])
            pred[t + 1] = max(0.0, s_val * escapement + recruits)
        end
        return pred
    end

    best_s = 0.82
    best_alpha = 0.25
    best_beta = 1.0 / max(10.0, mean(n_vals) * 2.0)

    best_pred = simulate_dd(best_s, best_alpha, best_beta)
    best_rss = sum((n_vals[(lag + 1):end] .- best_pred[(lag + 1):end]) .^ 2)

    # Multi-scale grid search
    s_candidates = [0.60, 0.70, 0.78, 0.82, 0.86, 0.90, 0.94]
    alpha_candidates = [0.05, 0.15, 0.25, 0.40, 0.60, 0.90]
    beta_candidates = [0.0, 0.0005, 0.001, 0.003, 0.008, 0.02]

    for s_try in s_candidates
        for a_try in alpha_candidates
            for b_try in beta_candidates
                pred_try = simulate_dd(s_try, a_try, b_try)
                rss_try = sum((n_vals[(lag + 1):end] .- pred_try[(lag + 1):end]) .^ 2)
                if rss_try < best_rss
                    best_rss = rss_try
                    best_s = s_try
                    best_alpha = a_try
                    best_beta = b_try
                    best_pred = pred_try
                end
            end
        end
    end

    n_fit = length(n_vals[(lag + 1):end])
    mean_obs = mean(n_vals[(lag + 1):end])
    tss = sum((n_vals[(lag + 1):end] .- mean_obs) .^ 2)
    r2 = (tss > 0.0) ? max(0.0, 1.0 - (best_rss / tss)) : 0.0
    rmse = sqrt(best_rss / n_fit)
    m_nat = -log(clamp(best_s, 1e-4, 0.9999))
    aic = n_fit * log(max(1e-6, best_rss / n_fit)) + 2.0 * 3.0

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
1. The **whole population** (total numbers across all instars).
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
        ann_df.year,
        ann_df.total_numbers,
        ann_df.catch_total;
        label = "Whole Population"
    )
    dd_tot = fit_delay_difference(
        ann_df.year,
        ann_df.total_numbers,
        ann_df.catch_total;
        lag = lag,
        label = "Whole Population"
    )

    # 2. Fishable component fits
    schaefer_fish = fit_schaefer(
        ann_df.year,
        ann_df.legal_numbers,
        ann_df.catch_legal;
        label = "Fishable Stock (Legal Males)"
    )
    dd_fish = fit_delay_difference(
        ann_df.year,
        ann_df.legal_numbers,
        ann_df.catch_legal;
        lag = lag,
        label = "Fishable Stock (Legal Males)"
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
    println(buf, "    SNOW CRAB MACRO-ECOLOGICAL DYNAMIC MODEL ESTIMATION REPORT    ")
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
        println(buf, "      Carrying Capacity (K)          : ", round(sch.K, digits = 1), " crabs")
        println(buf, "      Max Sustainable Yield (MSY)    : ", round(sch.MSY, digits = 1), " crabs/yr")
        println(buf, "      Abundance at MSY (N_msy)       : ", round(sch.N_msy, digits = 1), " crabs")
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

    # Ensure each peeled model has at least 3 data points
    realized_peels = clamp(n_peels, 1, total_years - 3)

    # 1. Base run (Peel 0: full dataset)
    base_macro = fit_population_dynamics(ann_df; lag = lag)

    # 2. Containers for retrospective peels
    peel_indices = collect(1:realized_peels)
    terminal_years = Int[]
    
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
        push!(terminal_years, ann_df.year[t_peel])
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
        terminal_years = terminal_years,
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
    years = retro_result.terminal_years
    m_name = (retro_result.model_type === :delay_difference) ?
             "Deriso-Schnute Delay-Difference" : "Schaefer Logistic Surplus Production"

    buf = IOBuffer()
    println(buf, "="^80)
    println(buf, "       SNOW CRAB RETROSPECTIVE PATTERN & MOHN'S RHO DIAGNOSTIC REPORT       ")
    println(buf, "="^80)
    println(buf, " Assessment Model Evaluated : ", m_name)
    println(buf, " Total Time Series Horizon  : ", nrow(ann), " years")
    println(buf, " Number of Retrospective Peels: ", retro_result.n_peels)
    println(buf, "="^80)

    for (label, stock) in [("WHOLE POPULATION", wp), ("FISHABLE STOCK (LEGAL MALES)", fs)]
        println(buf, "\n >> [$(label)]")
        println(buf, " Peel | Terminal Year | Peeled Terminal (N) | Base Reference (N) | Relative Bias (%)")
        println(buf, "------+---------------+---------------------+--------------------+------------------")
        for i in eachindex(peels)
            p_idx = peels[i]
            yr = years[i]
            p_val = stock.terminal_peeled[i]
            b_val = stock.terminal_base[i]
            re_pct = stock.rel_errors[i] * 100.0
            re_sign = (re_pct >= 0 ? "+" : "")
            re_str = re_sign * string(round(re_pct, digits = 2)) * "%"
            p_str = string(round(p_val, digits = 2))
            b_str = string(round(b_val, digits = 2))
            println(buf, "  -", p_idx, "  |     Year ", lpad(string(yr), 2),
                    "   |       ", lpad(p_str, 8),
                    "      |      ", lpad(b_str, 8),
                    "      |     ", lpad(re_str, 8))
        end
        println(buf, "------+---------------+---------------------+--------------------+------------------")
        
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
    extract_annual_stage_matrix(source; stages=build_instar_stages())

Extract or reconstruct an S x T stage abundance matrix from simulation results.
Supports:
- `DataFrame`: Annual production table with `:total_numbers` and `:legal_numbers`.
- `String`: File path to CSV simulation output.
- `StandardABM`: Running simulation model.

# Arguments
- `source`: Data source.
- `stages::Vector{InstarStageDefinition}`: Life-history stage definitions.

# Returns
Tuple `(stage_matrix, catch_matrix, years)`:
- `stage_matrix::Matrix{Float64}`: S x T matrix of stage abundances.
- `catch_matrix::Matrix{Float64}`: S x T matrix of stage catch removals.
- `years::Vector{Int}`: Evaluated simulation years.
"""
function extract_annual_stage_matrix(
    source;
    stages::Vector{InstarStageDefinition} = build_instar_stages()
)
    n_stages = length(stages)
    legal_indices = [st.index for st in stages if st.is_legal_male]
    immature_indices = [st.index for st in stages if !st.is_legal_male]

    ann_df = if source isa AbstractString
        load_simulation_results(String(source))
    elseif source isa DataFrame
        source
    elseif hasmethod(allagents, Tuple{typeof(source)})
        crabs = collect(allagents(source))
        cws = [c.size_mm for c in crabs]
        sexes = [c.sex for c in crabs]
        counts, _ = derive_instar_proportions_kde(cws, sexes; stages=stages)
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
        # Reconstruct stages from total_numbers and legal_numbers
        decay_weights = zeros(Float64, n_stages)
        for st in stages
            if !st.is_legal_male
                decay_weights[st.index] = exp(-0.28 * (st.instar - 4.0))
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
                                 stages=build_instar_stages(),
                                 years=1:size(stage_matrix, 2))

Fit a generalized Leslie (Lefkovitch) stage-structured transition matrix model to
multi-year stage abundance trajectories:
```math
\\mathbf{n}_{t+1} = \\mathbf{A} \\mathbf{n}_t - \\mathbf{C}_t
```

# Arguments
- `stage_matrix::Matrix{Float64}`: S x T matrix of stage abundances.
- `catch_matrix::Matrix{Float64}`: S x T matrix of stage catch removals.
- `stages::Vector{InstarStageDefinition}`: Life-history stage definitions.
- `years::AbstractVector{Int}`: Time indices.

# Returns
`LeslieMatrixFit` containing estimated transition matrix and demographic invariants.
"""
function fit_generalized_leslie_model(
    stage_matrix::Matrix{Float64},
    catch_matrix::Matrix{Float64} = zeros(size(stage_matrix));
    stages::Vector{InstarStageDefinition} = build_instar_stages(),
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

    # Stage 1: Instar 4 (pooled)
    s1 = 0.74
    gamma1 = 0.78
    A[2, 1] = s1 * gamma1
    A[1, 1] = s1 * (1.0 - gamma1)

    # Stage 2: Instar 5 (pooled)
    s2 = 0.76
    gamma2 = 0.75
    m6_idx = isempty(male_indices) ? 3 : first(male_indices)
    f6_idx = isempty(female_indices) ? 10 : first(female_indices)
    A[m6_idx, 2] = 0.50 * s2 * gamma2
    A[f6_idx, 2] = 0.50 * s2 * gamma2
    A[2, 2] = s2 * (1.0 - gamma2)

    # Male stages (Stages 3 to 9)
    for idx in male_indices
        st = stages[idx]
        is_term = (st.instar == 12 || idx == last(male_indices))
        s_m = st.is_legal_male ? 0.78 : 0.80
        gamma_m = is_term ? 0.0 : max(0.15, 0.75 - 0.10 * (st.instar - 6.0))
        p_term = (st.instar <= 6) ? 0.0 : min(0.80, 0.10 * (st.instar - 6.0)^1.3)

        if is_term
            A[idx, idx] = s_m
        else
            next_idx = idx + 1
            A[next_idx, idx] = s_m * gamma_m * (1.0 - p_term)
            A[idx, idx] = s_m * (1.0 - gamma_m + gamma_m * p_term)
        end
    end

    # Female stages (Stages 10 to 15)
    for idx in female_indices
        st = stages[idx]
        is_term = (st.instar == 11 || idx == last(female_indices))
        s_f = 0.80
        gamma_f = is_term ? 0.0 : max(0.10, 0.70 - 0.12 * (st.instar - 6.0))
        p_term_f = (st.instar <= 6) ? 0.0 : min(0.85, 0.15 * (st.instar - 6.0)^1.4)

        if is_term
            A[idx, idx] = s_f
        else
            next_idx = idx + 1
            A[next_idx, idx] = s_f * gamma_f * (1.0 - p_term_f)
            A[idx, idx] = s_f * (1.0 - gamma_f + gamma_f * p_term_f)
        end
    end

    # Fecundity contributions from reproductive mature females to Stage 1
    tot_recruits = sum(stage_matrix[1, 2:T])
    tot_females = sum(stage_matrix[repro_indices, 1:(T-1)])
    fecundity_base = (tot_females > 0.0) ? (tot_recruits / tot_females) : 0.85
    fecundity_base = clamp(fecundity_base, 0.20, 2.50)

    for idx in repro_indices
        st = stages[idx]
        f_scale = (st.cw_mean / 65.0)^2.5
        A[1, idx] = fecundity_base * f_scale
    end

    # Refine matrix transitions via trajectory matching
    Y = stage_matrix[:, 2:T] .+ catch_matrix[:, 1:(T-1)]
    X = stage_matrix[:, 1:(T-1)]
    sum_x = sum(X)
    sum_y = sum(Y)
    scale_factor = (sum_x > 0.0) ? (sum_y / sum_x) : 1.0
    scale_factor = clamp(scale_factor, 0.80, 1.20)

    for j in 1:S
        col_sum = sum(A[i, j] for i in 2:S)
        if col_sum > 0.0
            adj_col = min(0.95, col_sum * (0.85 + 0.15 * scale_factor))
            for i in 2:S
                A[i, j] = (A[i, j] / col_sum) * adj_col
            end
        end
    end

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
    damping_ratio = (lambda_2 > 1.0e-6) ? (lambda_1 / lambda_2) : 10.0

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
            wt_kg = st.mean_weight_g / 1000.0
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
    k_params = 4
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
    println(buf, "    SNOW CRAB GENERALIZED LESLIE (STAGE-TRANSITION) DEMOGRAPHIC REPORT   ")
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
    println(buf, "   - Fishable Stock RMSE     : ", lpad(string(round(fit.rmse, digits = 2)), 8), " crabs")
    println(buf, "   - Model AIC Score         : ", lpad(string(round(fit.aic, digits = 2)), 8))
    println(buf, "-"^85)
    println(buf, " [STAGE STRUCTURE, STABLE DISTRIBUTION & REPRODUCTIVE VALUES]")
    println(buf, " Idx | Stage Name          | CW Mean | Mean Wt (g) | SSD (w₁) | Repro Val (v₁) | Legal | Repro")
    println(buf, "-----+---------------------+---------+-------------+----------+----------------+-------+------")
    for st in fit.stages
        i = st.index
        w_str = lpad(string(round(fit.stable_stage_distribution[i] * 100.0, digits = 2)) * "%", 8)
        v_str = lpad(string(round(fit.reproductive_values[i], digits = 2)), 14)
        cw_str = lpad(string(round(st.cw_mean, digits = 1)), 7)
        wt_str = lpad(string(round(st.mean_weight_g, digits = 1)), 11)
        leg_str = st.is_legal_male ? " YES " : "  -  "
        rep_str = st.is_reproductive ? " YES " : "  -  "
        println(buf, "  ", lpad(string(i), 2), " | ",
                rpad(st.name, 19), " | ",
                cw_str, " | ",
                wt_str, " | ",
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
`MacroModelComparison` containing comparative metrics.
"""
function compare_macro_models(
    annual_data,
    leslie_fit::LeslieMatrixFit;
    lag::Int = 1
)
    fits = fit_population_dynamics(annual_data; lag = lag)
    years = leslie_fit.years
    T = length(years)

    mean_legal_wt_kg = 0.60
    if !isempty(leslie_fit.stages)
        leg_stages = filter(s -> s.is_legal_male, leslie_fit.stages)
        if !isempty(leg_stages)
            mean_legal_wt_kg = mean([s.mean_weight_g for s in leg_stages]) / 1000.0
        end
    end

    b_obs = leslie_fit.legal_biomass_obs_kg
    b_leslie = leslie_fit.legal_biomass_pred_kg

    # Schaefer trajectory for fishable stock
    sch_fit = fits.fishable_stock.schaefer
    n_sch = sch_fit.predicted
    b_sch = [n * mean_legal_wt_kg for n in n_sch]
    if length(b_sch) < T
        b_sch = vcat(b_sch, fill(last(b_sch), T - length(b_sch)))
    elseif length(b_sch) > T
        b_sch = b_sch[1:T]
    end

    # Delay-Difference trajectory for fishable stock
    dd_fit = fits.fishable_stock.delay_difference
    n_dd = dd_fit.predicted
    b_dd = [n * mean_legal_wt_kg for n in n_dd]
    if length(b_dd) < T
        b_dd = vcat(b_dd, fill(last(b_dd), T - length(b_dd)))
    elseif length(b_dd) > T
        b_dd = b_dd[1:T]
    end

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
    aic_les = (ss_res_les > 0.0) ? (T * log(ss_res_les / T) + 2.0 * 4) : 0.0

    aics = Dict(:schaefer => aic_sch, :delay_difference => aic_dd, :leslie => aic_les)
    preferred = argmin(aics)

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
        preferred
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
2. **No Population Explosion (`is_bounded`)**: Living population ``N(t) \\le N_{\\max}``.
3. **Steady Size Structure (`is_size_steady`)**: Relative drift in mean carapace width over
   the final evaluation window does not exceed `max_size_drift`.
4. **Maturity Ogive Fidelity (`is_maturity_realistic`)**: Empirical maturity schedule
   matches theoretical logistic curves (`maturity(sex, log(cw))`) within `max_maturity_rmse`.
5. **Growth Schedule Fidelity (`is_growth_realistic`)**: Observed mean carapace width per
   instar matches reference empirical medians from `growth_schedule()` within `max_growth_rmse`.

# Arguments
- `model`: Agent-based model instance after simulation.
- `agent_df::DataFrame`: Time-series table collected during simulation.
- `min_population::Union{Nothing, Int}`: Minimum living crab threshold (defaults to 10% of K, min 15).
- `max_population::Union{Nothing, Int}`: Maximum living crab ceiling (defaults to 300% of K).
- `max_size_drift::Float64`: Maximum allowable fractional shift in mean CW over recent steps.
- `max_maturity_rmse::Float64`: Maximum allowable RMSE against theoretical maturity ogive.
- `max_growth_rmse::Float64`: Maximum allowable RMSE against empirical instar size schedule (mm).
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
    max_size_drift::Float64 = 0.20,
    max_maturity_rmse::Float64 = 0.35,
    max_growth_rmse::Float64 = 15.0,
    window_steps::Int = 10
)
    k_cap = (hasproperty(model, :params) && hasproperty(model.params, :benthic_carrying_capacity)) ?
        Float64(model.params.benthic_carrying_capacity) : 300.0

    n_min = (min_population !== nothing) ? min_population : max(15, round(Int, 0.10 * k_cap))
    n_max = (max_population !== nothing) ? max_population : round(Int, 3.0 * k_cap)

    # 1. Population bounds
    living_crabs = collect(allagents(model))
    n_living = length(living_crabs)

    is_persistent = n_living >= n_min
    is_bounded = n_living <= n_max

    if n_living == 0
        return (
            passed_all = false,
            is_persistent = false,
            is_bounded = true,
            is_size_steady = false,
            is_maturity_realistic = false,
            is_growth_realistic = false,
            n_living = 0,
            size_drift = 1.0,
            maturity_rmse = 1.0,
            growth_rmse = 100.0,
            composite_score = 1000.0,
            mean_cw = 0.0,
            instar_counts = Dict{Int, Int}()
        )
    end

    # 2. Steady size structure (drift over recent steps)
    size_col = nothing
    for candidate in [:size_mm_safe_mean, :mean_size_mm, :size_mm_mean, :mean_cw]
        if candidate in propertynames(agent_df)
            size_col = candidate
            break
        end
    end

    size_drift = 0.0
    if size_col !== nothing && nrow(agent_df) >= 2
        w = min(window_steps, nrow(agent_df))
        recent_sizes = agent_df[(end - w + 1):end, size_col]
        mean_recent = mean(recent_sizes)
        if mean_recent > 0.0
            size_drift = abs(recent_sizes[end] - recent_sizes[1]) / mean_recent
        end
    end
    is_size_steady = size_drift <= max_size_drift

    # 3. Maturity ogive fidelity
    cw_bins = 20.0:10.0:140.0
    cm_m, pm_m = compute_empirical_ogive(model, male, cw_bins)
    cm_f, pm_f = compute_empirical_ogive(model, female, cw_bins)

    mat_sq_errors = Float64[]
    for i in eachindex(cm_m)
        p_theory = maturity(0, log(cm_m[i]))
        push!(mat_sq_errors, (pm_m[i] - p_theory)^2)
    end
    for i in eachindex(cm_f)
        p_theory = maturity(1, log(cm_f[i]))
        push!(mat_sq_errors, (pm_f[i] - p_theory)^2)
    end

    maturity_rmse = isempty(mat_sq_errors) ? 0.0 : sqrt(mean(mat_sq_errors))
    is_maturity_realistic = maturity_rmse <= max_maturity_rmse

    # 4. Growth schedule fidelity
    ref_schedule = growth_schedule()
    instar_obs_m = Dict{Int, Vector{Float64}}()
    instar_obs_f = Dict{Int, Vector{Float64}}()
    instar_counts = Dict{Int, Int}()

    for c in living_crabs
        inst = c.instar
        instar_counts[inst] = get(instar_counts, inst, 0) + 1
        if c.sex == male
            if !haskey(instar_obs_m, inst)
                instar_obs_m[inst] = Float64[]
            end
            push!(instar_obs_m[inst], c.size_mm)
        else
            if !haskey(instar_obs_f, inst)
                instar_obs_f[inst] = Float64[]
            end
            push!(instar_obs_f[inst], c.size_mm)
        end
    end

    growth_sq_errors = Float64[]
    for (inst, sizes) in instar_obs_m
        if 1 <= inst <= nrow(ref_schedule)
            ref_med = ref_schedule.cwm_med[inst]
            push!(growth_sq_errors, (mean(sizes) - ref_med)^2)
        end
    end
    for (inst, sizes) in instar_obs_f
        if 1 <= inst <= nrow(ref_schedule)
            ref_med = ref_schedule.cwf_med[inst]
            push!(growth_sq_errors, (mean(sizes) - ref_med)^2)
        end
    end

    growth_rmse = isempty(growth_sq_errors) ? 0.0 : sqrt(mean(growth_sq_errors))
    is_growth_realistic = growth_rmse <= max_growth_rmse

    # 5. Composite criteria status and loss
    passed_all = is_persistent && is_bounded && is_size_steady &&
                 is_maturity_realistic && is_growth_realistic

    composite_score = (is_persistent ? 0.0 : 150.0) +
                      (is_bounded ? 0.0 : 150.0) +
                      25.0 * size_drift +
                      15.0 * maturity_rmse +
                      1.0 * growth_rmse

    mean_cw_all = mean([c.size_mm for c in living_crabs])

    return (
        passed_all = passed_all,
        is_persistent = is_persistent,
        is_bounded = is_bounded,
        is_size_steady = is_size_steady,
        is_maturity_realistic = is_maturity_realistic,
        is_growth_realistic = is_growth_realistic,
        n_living = n_living,
        size_drift = size_drift,
        maturity_rmse = maturity_rmse,
        growth_rmse = growth_rmse,
        composite_score = composite_score,
        mean_cw = mean_cw_all,
        instar_counts = instar_counts
    )
end


