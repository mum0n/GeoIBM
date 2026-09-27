"""
    src/gui.jl

Interactive Graphical User Interface (GUI) and real-time visualization
dashboard for the organism Individual-Based Model (IBM).

Provides three interlinked dashboard interfaces:
1. Parameter inputs/controls, real-time spatial environmental maps,
   and dynamic population time-series trajectories.
2. Demographic size structures (length frequencies by sex, instar stages)
   and real-time fitting of macro-ecological dynamics models
   (Schaefer logistic surplus production, Deriso-Schnute delay-difference,
   and Generalized Leslie/Lefkovitch transition matrix).
3. Realized process functional relationships (empirical maturity ogives,
   fishery selectivity, bioenergetic thermal response, and stock-recruitment)
   plotted against theoretical and empirical drivers.

Built upon `GLMakie` and `Agents.jl` reactive Observables.
"""

import GLMakie
import GLMakie: Figure, Axis, SliderGrid, Toggle, Button, Menu, Label, GridLayout
import GLMakie: AxisAspect, Point2f, Observable, on, lift, @lift
using DataFrames
using Distributions
using LinearAlgebra
using Printf
using Random
using Statistics
using Agents

"""
    create_empty_annual_data()

Initialize an empty DataFrame schema for annual demographic and fishery metrics
used in macro-model fitting and biomass dynamics tracking.

Built from `ANNUAL_PRODUCTION_SCHEMA` so the GUI's annual frame carries exactly
the columns `extract_annual_production` produces, plus the two GUI-only tracking
columns. The two were separate hand-written lists, so the GUI frame was missing
`day`, `total_biomass_kg` and `annual_egg_production` and could not be fed to the
macro-model fitters without a missing-column failure.
"""
function create_empty_annual_data()
    df = empty_annual_production()
    df.recruits = Float64[]
    df.cumulative_catch = Float64[]
    # Realised mean wet weight per component, in kg. Recorded so the
    # number-to-biomass conversion applied to the macro-model trajectories is
    # carried with the data rather than being implicit in the plotting code.
    df.mean_weight_legal_kg = Float64[]
    df.mean_weight_male_kg = Float64[]
    df.mean_weight_female_kg = Float64[]
    df.mean_weight_total_kg = Float64[]
    return df
end

# ------------------------------------------------------------------------------
# Data Structure for GUI State Management
# ------------------------------------------------------------------------------

"""
    IBMGuiState

Mutable container maintaining active simulation objects, reactive Makie
observables, time-series history buffers, and concurrent background tasks.

# Mathematical Formulation
At each simulation step ``t \\to t + \\Delta t``:
- Spatial coordinates ``(x_i, y_i) \\in \\Omega`` and physical traits
  (carapace width ``L_i``, wet weight ``W_i``, developmental stage ``S_i``)
  are streamed to OpenGL particle buffers.
- Population abundance ``N(t)`` and fishable biomass ``B_{\\text{legal}}(t)``
  are appended to temporal histories.
- Realized empirical maturity rates ``\\hat{P}(L) = \\frac{N_{\\text{adult}, L}}{N_{\\text{total}, L}}``
  are dynamically derived across discrete carapace width bins.
- At annual intervals (day 274 = September 1 snapshot), macro-ecological
  models are recalibrated against empirical simulated production.
"""
mutable struct IBMGuiState
    model::StandardABM
    is_running::Observable{Bool}
    sim_speed_days::Observable{Float64}
    current_time::Observable{Float64}
    status_text::Observable{String}

    # Spatial map observables
    obs_env_field::Observable{Matrix{Float64}}
    obs_agent_x::Observable{Vector{Float32}}
    obs_agent_y::Observable{Vector{Float32}}
    obs_agent_sizes::Observable{Vector{Float32}}
    obs_agent_colors::Observable{Vector{String}}

    # Time series history buffers & observables
    t_history::Vector{Float64}
    n_total_history::Vector{Float64}
    n_females_history::Vector{Float64}
    n_males_history::Vector{Float64}
    n_legal_history::Vector{Float64}
    biomass_legal_history::Vector{Float64}
    catch_history::Vector{Float64}
    # Realised mean wet weight (kg) of each live component, used to convert the
    # number-based macro-model fits to biomass for the dashboard figure.
    mean_weight_legal_history::Vector{Float64}
    mean_weight_male_history::Vector{Float64}
    mean_weight_female_history::Vector{Float64}
    mean_weight_total_history::Vector{Float64}
    catch_at_year_start::Float64
    current_year_idx::Int

    obs_ts_total::Observable{Vector{Point2f}}
    obs_ts_females::Observable{Vector{Point2f}}
    obs_ts_males::Observable{Vector{Point2f}}
    obs_ts_legal::Observable{Vector{Point2f}}
    obs_ts_biomass::Observable{Vector{Point2f}}
    obs_ts_catch::Observable{Vector{Point2f}}

    # Size structure & demographic observables
    obs_cw_male::Observable{Vector{Float32}}
    obs_cw_female::Observable{Vector{Float32}}
    obs_stage_counts::Observable{Vector{Point2f}}
    obs_stage_counts_male::Observable{Vector{Point2f}}
    obs_stage_counts_female::Observable{Vector{Point2f}}

    # Realized functional relationships observables
    obs_ogive_male_pts::Observable{Vector{Point2f}}
    obs_ogive_female_pts::Observable{Vector{Point2f}}
    obs_selectivity_pts::Observable{Vector{Point2f}}
    obs_sr_pts::Observable{Vector{Point2f}}

    # Macro-model calibration state
    annual_data::DataFrame
    macro_obs_biomass::Observable{Vector{Point2f}}
    macro_sch_biomass::Observable{Vector{Point2f}}
    macro_dd_biomass::Observable{Vector{Point2f}}
    macro_les_biomass::Observable{Vector{Point2f}}
    macro_metrics_text::Observable{String}
    # Legend entries carrying the key estimated parameters of each macro model,
    # rebuilt on every re-fit so the legend tracks the current calibration.
    macro_label_observed::Observable{String}
    macro_label_sch::Observable{String}
    macro_label_dd::Observable{String}
    macro_label_les::Observable{String}
    macro_conversion_text::Observable{String}

    # Snapshot state cache
    snapshots::Dict{Int, Any}
    snapshot_labels::Observable{Vector{String}}

    # Asynchronous task handle
    sim_task::Union{Nothing, Task}
end

# ------------------------------------------------------------------------------
# Core Stepping & Analytical Update Functions
# ------------------------------------------------------------------------------

"""
    step_gui_simulation!(state::IBMGuiState, days::Float64)

Advance the underlying agent-based model by `days` days and refresh all
reactive GUI observables across the three dashboard interfaces.

# Arguments
- `state::IBMGuiState`: Active GUI state container.
- `days::Float64`: Elapsed simulation time to step forward.

# Returns
`Nothing`.
"""
function step_gui_simulation!(state::IBMGuiState, days::Float64)
    model = state.model
    dt = abmproperties(model)[:dt]
    n_steps = max(1, round(Int, days / dt))

    # Advance model
    prev_year = floor(Int, get(abmproperties(model), :current_time, 0.0) / 365.0)
    
    if days > 0.0
        Agents.step!(model, n_steps)
    end

    curr_t = Float64(abmproperties(model)[:current_time])
    state.current_time[] = curr_t
    
    curr_year = floor(Int, curr_t / 365.0)
    if curr_year > prev_year
        state.snapshots[curr_year] = (
            model = deepcopy(model),
            t_history = copy(state.t_history),
            n_total_history = copy(state.n_total_history),
            n_females_history = copy(state.n_females_history),
            n_males_history = copy(state.n_males_history),
            n_legal_history = copy(state.n_legal_history),
            biomass_legal_history = copy(state.biomass_legal_history),
            catch_history = copy(state.catch_history),
            annual_data = copy(state.annual_data)
        )
        labels = String[]
        for y in sort(collect(keys(state.snapshots)))
            push!(labels, "Year $y")
        end
        state.snapshot_labels[] = labels
    end

    # 1. Update Spatial Map Observables
    env_field = copy(model.temperature_field)
    state.obs_env_field[] = env_field

    living = collect(allagents(model))
    n_living = length(living)

    xs = Float32[Float32(a.pos[1]) for a in living]
    ys = Float32[Float32(a.pos[2]) for a in living]
    szs = Float32[Float32(max(4.0, a.size * 0.15)) for a in living]
    cols = String[organism_color(a, state.model.params) for a in living]

    state.obs_agent_x[] = xs
    state.obs_agent_y[] = ys
    state.obs_agent_sizes[] = szs
    state.obs_agent_colors[] = cols

    # 2. Update Demographic Counts & Timeseries Buffers
    params = abmproperties(model)[:params]
    legal_sz = params.fishery.legal_size_male
    adult_sz = params.organism.adult_size_threshold
    n_fem = count(is_mature_female, living)
    n_mal = count(is_mature_male, living)
    n_leg = count(o -> is_legal_male(o, legal_sz, adult_sz), living)

    # Compute legal fishable biomass in kg
    b_leg_kg = sum(
        mass_if_legal_male(o, legal_sz, adult_sz) for o in living;
        init = 0.0
    ) / 1000.0
    cum_catch = Float64(abmproperties(model)[:total_catch_biomass_kg])

    # Mean wet weight of each live component, in kg, measured from the individuals
    # actually present rather than from the stage-mean allometry. The macro-model
    # fits are in number, so a count trajectory is converted to biomass by
    # multiplying by the mean weight of the component it describes; using the
    # realised mean rather than a fitted constant keeps the plotted conversion
    # consistent with the measured biomass on the same axes.
    mean_weight_kg = function (orgs)
        isempty(orgs) ? 0.0 : sum(o -> o.mass, orgs; init = 0.0) / (1000.0 * length(orgs))
    end
    legal_orgs = [o for o in living if is_legal_male(o, legal_sz, adult_sz)]
    male_orgs = [o for o in living if is_mature_male(o)]
    female_orgs = [o for o in living if is_mature_female(o)]
    mw_leg_kg = mean_weight_kg(legal_orgs)
    mw_mal_kg = mean_weight_kg(male_orgs)
    mw_fem_kg = mean_weight_kg(female_orgs)
    mw_tot_kg = mean_weight_kg(living)

    push!(state.t_history, curr_t)
    push!(state.n_total_history, Float64(n_living))
    push!(state.n_females_history, Float64(n_fem))
    push!(state.n_males_history, Float64(n_mal))
    push!(state.n_legal_history, Float64(n_leg))
    push!(state.biomass_legal_history, b_leg_kg)
    push!(state.mean_weight_legal_history, mw_leg_kg)
    push!(state.mean_weight_male_history, mw_mal_kg)
    push!(state.mean_weight_female_history, mw_fem_kg)
    push!(state.mean_weight_total_history, mw_tot_kg)

    year_idx = floor(Int, curr_t / 365.0) + 1
    if year_idx > state.current_year_idx
        state.current_year_idx = year_idx
        state.catch_at_year_start = cum_catch
    end
    annual_catch = cum_catch - state.catch_at_year_start

    push!(state.catch_history, annual_catch)

    # Update timeseries line observables
    pts_tot = [Point2f(state.t_history[i], state.n_total_history[i])
               for i in eachindex(state.t_history)]
    pts_fem = [Point2f(state.t_history[i], state.n_females_history[i])
               for i in eachindex(state.t_history)]
    pts_mal = [Point2f(state.t_history[i], state.n_males_history[i])
               for i in eachindex(state.t_history)]
    pts_leg = [Point2f(state.t_history[i], state.n_legal_history[i])
               for i in eachindex(state.t_history)]
    pts_bio = [Point2f(state.t_history[i], state.biomass_legal_history[i])
               for i in eachindex(state.t_history)]
    pts_cat = [Point2f(state.t_history[i], state.catch_history[i])
               for i in eachindex(state.t_history)]

    state.obs_ts_total[] = pts_tot
    state.obs_ts_females[] = pts_fem
    state.obs_ts_males[] = pts_mal
    state.obs_ts_legal[] = pts_leg
    state.obs_ts_biomass[] = pts_bio
    state.obs_ts_catch[] = pts_cat

    # 3. Update Size Structure & Stage Demographics
    cw_m = Float32[Float32(a.size) for a in living if a.sex == male]
    cw_f = Float32[Float32(a.size) for a in living if a.sex == female]
    state.obs_cw_male[] = isempty(cw_m) ? Float32[0.0] : cw_m
    state.obs_cw_female[] = isempty(cw_f) ? Float32[0.0] : cw_f

    # Instar stage distribution by sex (Instars 4 to 14)
    stg_counts = Point2f[]
    stg_males = Point2f[]
    stg_females = Point2f[]
    for stg in 4:14
        cnt_m = count(a -> a.stage == stg && a.sex == male, living)
        cnt_f = count(a -> a.stage == stg && a.sex == female, living)
        push!(stg_counts, Point2f(Float32(stg), Float32(cnt_m + cnt_f)))
        push!(stg_males, Point2f(Float32(stg), Float32(cnt_m)))
        push!(stg_females, Point2f(Float32(stg), Float32(cnt_f)))
    end
    state.obs_stage_counts[] = stg_counts
    state.obs_stage_counts_male[] = stg_males
    state.obs_stage_counts_female[] = stg_females

    # 4. Update Realized Functional Relationships (Ogives, Selectivity, SR)
    update_functional_observables!(state, living)

    # 5. Check and Record Annual Snapshots for Macro-Model Fitting
    check_and_update_macro_models!(state, curr_t, n_living, n_leg, n_fem,
                                   n_mal, b_leg_kg, cum_catch)

    # Status summary update
    yr_str = @sprintf("%.2f", curr_t / 365.0)
    month_idx = floor(Int, (curr_t % 365.0) / 30.416) + 1
    state.status_text[] = "Day: $(round(Int, curr_t)) | Month: $(month_idx) | Year: $(yr_str)"

    return nothing
end

"""
    update_functional_observables!(state::IBMGuiState, living::Vector{Organism})

Compute empirical maturity fractions and selectivity distributions from living
agents and stream points to interface 3 observables.
"""
function update_functional_observables!(state::IBMGuiState,
                                       living::Vector{Organism})
    if isempty(living)
        return nothing
    end

    bins = 20.0:10.0:150.0
    n_bins = length(bins) - 1

    # Empirical maturity ogive for males
    m_pts = Point2f[]
    f_pts = Point2f[]
    sel_pts = Point2f[]

    legal_sz = abmproperties(state.model)[:params].fishery.legal_size_male

    for k in 1:n_bins
        b_low = bins[k]
        b_high = bins[k + 1]
        mid = Float32((b_low + b_high) / 2.0)

        # Males
        m_in_bin = filter(a -> a.sex == male && b_low <= a.size < b_high, living)
        if !isempty(m_in_bin)
            p_mat = count(a -> a.is_mature, m_in_bin) / length(m_in_bin)
            push!(m_pts, Point2f(mid, Float32(p_mat)))

            # Realized selectivity fraction: legal males in bin
            p_sel = count(a -> a.size >= legal_sz, m_in_bin) / length(m_in_bin)
            push!(sel_pts, Point2f(mid, Float32(p_sel)))
        end

        # Females
        f_in_bin = filter(a -> a.sex == female && b_low <= a.size < b_high, living)
        if !isempty(f_in_bin)
            p_mat_f = count(a -> a.is_mature, f_in_bin) / length(f_in_bin)
            push!(f_pts, Point2f(mid, Float32(p_mat_f)))
        end
    end

    state.obs_ogive_male_pts[] = m_pts
    state.obs_ogive_female_pts[] = f_pts
    state.obs_selectivity_pts[] = sel_pts

    # Realized Stock-Recruitment Points:
    # 1. From explicit clutch recruitment records (if populated)
    # 2. From annual cohort lag history (eggs spawned in yr t -> recruits settled in yr t+1)
    sr_pts = Point2f[]
    clutch_df = abmproperties(state.model)[:clutch_recruitment_records]
    if !isempty(clutch_df)
        for row in eachrow(clutch_df)
            push!(sr_pts, Point2f(Float32(row.egg_count), Float32(row.recruits_settled)))
        end
    end

    ann_df = state.annual_data
    if nrow(ann_df) >= 2 && (:recruits in propertynames(ann_df))
        for yr in 1:(nrow(ann_df) - 1)
            s_eggs = Float32(ann_df.total_egg_production[yr])
            r_rec = Float32(ann_df.recruits[yr + 1])
            if s_eggs > 0.0f0 || r_rec > 0.0f0
                push!(sr_pts, Point2f(s_eggs, r_rec))
            end
        end
    end
    state.obs_sr_pts[] = sr_pts

    return nothing
end

"""
    check_and_update_macro_models!(state, curr_t, n_tot, n_leg, n_fem,
                                   n_mal, b_leg_kg, cum_catch)

Detect annual snapshot intervals (Day 274 ~ September 1), register annual
demographic observations, and recalibrate the three macro-ecological models.
"""
function check_and_update_macro_models!(
    state::IBMGuiState,
    curr_t::Float64,
    n_tot::Int,
    n_leg::Int,
    n_fem::Int,
    n_mal::Int,
    b_leg_kg::Float64,
    cum_catch::Float64
)
    # Annual snapshot day comes from the configuration, not a literal: a run
    # configured for a different survey day must record on that day.
    snapshot_day = Float64(state.model.params.output.snapshot_day)
    year_idx = floor(Int, curr_t / 365.0) + 1
    target_snapshot_day = (year_idx - 1) * 365.0 + snapshot_day

    # Record annual production row if within step window of snapshot
    # and not already recorded for this year
    already_recorded = !isempty(state.annual_data) &&
                       any(state.annual_data.sample_index .== year_idx)

    if curr_t >= target_snapshot_day && !already_recorded
        # Compute interval catch for biomass (for records if needed)
        prev_catch_kg = isempty(state.annual_data) ? 0.0 :
                     last(state.annual_data.cumulative_catch)
        int_catch_kg = max(0.0, cum_catch - prev_catch_kg)

        # Compute interval catch in numbers (CRITICAL for Macro-models)
        cum_catch_num = Float64(abmproperties(state.model)[:total_catch_numbers])
        prev_catch_num = isempty(state.annual_data) ? 0.0 :
                         sum(state.annual_data.catch_total; init = 0.0)
        int_catch_num = max(0.0, cum_catch_num - prev_catch_num)

        # Compute interval recruits settled (CRITICAL for Stock-Recruitment)
        cum_rec = Float64(abmproperties(state.model)[:total_recruits_settled])
        prev_rec = isempty(state.annual_data) ? 0.0 :
                   sum(state.annual_data.recruits; init = 0.0)
        int_rec = max(0.0, cum_rec - prev_rec)

        # Live biomass comes from the same summarizer the CLI report uses, so the
        # GUI and the saved output cannot disagree.
        summary = summarize_population(state.model)

        new_row = DataFrame(
            sample_index = [year_idx],
            day = [Float64(mod(curr_t, 365.0) == 0.0 ? 365.0 : mod(curr_t, 365.0))],
            time = [curr_t],
            total_numbers = [Float64(n_tot)],
            total_biomass_kg = [Float64(summary.total_biomass_kg)],
            legal_numbers = [Float64(n_leg)],
            mature_females = [Float64(n_fem)],
            mature_males = [Float64(n_mal)],
            fishable_male_biomass_kg = [b_leg_kg],
            mature_male_biomass_kg = [b_leg_kg],
            mature_female_biomass_kg = [b_leg_kg * 0.7],
            total_egg_production = [Float64(abmproperties(state.model)[:annual_eggs_spawned])],
            annual_egg_production = [Float64(abmproperties(state.model)[:annual_eggs_spawned])],
            recruits = [int_rec],
            catch_total = [int_catch_num],
            cumulative_catch = [cum_catch],
            catch_legal = [int_catch_num],
            surplus_production_total = [0.0],
            surplus_production_legal = [0.0],
            # Realised mean wet weight per component, so the number-to-biomass
            # conversion applied to the macro-model trajectories is recorded with
            # the data rather than being implicit in the plotting code. Taken from
            # the history the update step has just appended, so there is one
            # computation of these rather than two that could disagree.
            mean_weight_legal_kg = [last(state.mean_weight_legal_history)],
            mean_weight_male_kg = [last(state.mean_weight_male_history)],
            mean_weight_female_kg = [last(state.mean_weight_female_history)],
            mean_weight_total_kg = [last(state.mean_weight_total_history)]
        )
        append!(state.annual_data, new_row)

        # Trigger macro-model recalibration if we have >= 4 annual periods
        if nrow(state.annual_data) >= 4
            recalibrate_macro_models!(state)
        end
    end

    return nothing
end

"""
    recalibrate_macro_models!(state::IBMGuiState)

Fit Schaefer logistic surplus production, Deriso-Schnute delay-difference,
and Generalized Leslie matrix models against the accumulated annual time series.
"""
function recalibrate_macro_models!(state::IBMGuiState)
    ann_df = state.annual_data
    T = nrow(ann_df)
    if T < 4
        return nothing
    end

    try
        # Extract stage transition matrices and fit Leslie model
        stage_mat, catch_mat, yrs = extract_annual_stage_matrix(ann_df)
        leslie_fit = fit_generalized_leslie_model(stage_mat, catch_mat; years = yrs)
        macro_comp = compare_macro_models(ann_df, leslie_fit; lag = 1)

        # Update macro-model trajectory observables.
        #
        # The observed series is the *measured* annual legal-biomass sum, not
        # `macro_comp.biomass_obs_kg`. The latter is what the macro models are
        # calibrated in: legal abundance times the stage-*mean* allometric mass,
        # which runs about 12% below the true sum of individual masses. Plotting
        # it as "Annual Survey Biomass" put two different quantities on the same
        # axes, so the survey markers could never sit on the fishable-biomass line
        # even though both are legal biomass. x alignment was never the problem.
        measured = if :fishable_male_biomass_kg in propertynames(ann_df)
            Float64.(ann_df.fishable_male_biomass_kg)
        else
            Float64.(macro_comp.biomass_obs_kg)
        end
        obs_pts = [Point2f(ann_df.time[i], measured[i]) for i in 1:T]
        # The macro-model fits are in NUMBER, not biomass: `fit_population_dynamics`
        # passes `legal_numbers` and `catch_legal`, and the Leslie fit works on a
        # per-stage count matrix. `compare_macro_models` also returns a `biomass_*`
        # series, but that one is rescaled by a single pooled *stage-mean* legal
        # mass, which sits below the real mean weight of the legal males present
        # (and below the measured sum, because a stage mean ignores the within-stage
        # size spread). For this figure the conversion is done here instead:
        # numbers x the realised mean weight of the component, per snapshot.
        mw_col = if :mean_weight_legal_kg in propertynames(ann_df)
            Float64.(ann_df.mean_weight_legal_kg)
        else
            fill(macro_comp.mean_component_weight_kg, T)
        end
        to_biomass(nums) = [Float64(nums[i]) * mw_col[i] for i in 1:T]

        sch_kg = to_biomass(macro_comp.numbers_schaefer)
        dd_kg = to_biomass(macro_comp.numbers_delay_diff)
        les_kg = to_biomass(macro_comp.numbers_leslie)

        sch_pts = [Point2f(ann_df.time[i], sch_kg[i]) for i in 1:T]
        dd_pts  = [Point2f(ann_df.time[i], dd_kg[i]) for i in 1:T]
        les_pts = [Point2f(ann_df.time[i], les_kg[i]) for i in 1:T]

        state.macro_obs_biomass[] = obs_pts
        state.macro_sch_biomass[] = sch_pts
        state.macro_dd_biomass[]  = dd_pts
        state.macro_les_biomass[] = les_pts

        # Legend entries carry the key estimated parameters, rebuilt on each
        # re-fit so they track the current calibration.
        state.macro_label_observed[] = "Annual Surveyed Biomass"
        state.macro_label_sch[] = macro_comp.label_schaefer
        state.macro_label_dd[] = macro_comp.label_delay_diff
        state.macro_label_les[] = macro_comp.label_leslie
        # The conversion actually applied, so a reader can see what "biomass" means
        # on this panel.
        mw_now = mw_col[end]
        state.macro_conversion_text[] = "kg = numbers × $(round(mw_now, digits=4)) kg mean legal-male weight"

        pref = uppercase(string(macro_comp.preferred_model))
        r2_sch = round(macro_comp.r2_schaefer, digits = 2)
        r2_dd  = round(macro_comp.r2_delay_diff, digits = 2)
        r2_les = round(macro_comp.r2_leslie, digits = 2)

        state.macro_metrics_text[] = "Preferred: $(pref) | " *
            "Schaefer R²: $(r2_sch) | Delay-Diff R²: $(r2_dd) | " *
            "Leslie Matrix R²: $(r2_les)"
    catch err
        # Gracefully handle non-invertible matrix edge cases during early burn-in
        @warn "Macro-model calibration deferred: $(err)"
    end

    return nothing
end

"""
    with_overrides(::Type{T}, base::T; kwargs...) where {T}

Return a copy of the immutable parameter struct `base` with the supplied keyword
arguments applied. Fields absent from `kwargs` are carried over unchanged, so
callers never need to restate the full field list.
"""
function with_overrides(::Type{T}, base::T; kwargs...) where {T}
    values = Any[get(kwargs, field, getfield(base, field)) for field in fieldnames(T)]
    return T(values...)
end

"""
    reconstruct_params(p::IBMParameters; kwargs...)

Reconstruct an immutable `IBMParameters` hierarchy with modified parameters.
Any keyword matching a field of any nested parameter struct is applied. Derived
from `parameter_structs()`, so a newly added parameter struct is carried through
without editing this function.
"""
function reconstruct_params(p::IBMParameters; kwargs...)
    # Derived from `parameter_structs()` rather than hand-listing the fields: the
    # hand-written version silently dropped any struct added later, which is how
    # the GUI's Reset button lost a parameter without any test noticing.
    nested = Dict{Symbol, Any}()
    for (field, struct_type, _) in parameter_structs()
        nested[field] = with_overrides(struct_type, getfield(p, field); kwargs...)
    end
    return IBMParameters(; nested...)
end

# ------------------------------------------------------------------------------
# GUI Construction and View Layouts
# ------------------------------------------------------------------------------

"""
    launch_organism_gui(; display_window = true, kwargs...)

Build the interactive GUI from `default_ibm_configuration()` overridden by
`kwargs`. Keyword names must match the declared parameter names exactly, such as
`n_agents`, `grid`, `dt`, and `seed`.
"""
function launch_organism_gui(; display_window::Bool = true, kwargs...)
    config = default_ibm_configuration()
    for (key, value) in kwargs
        config[Symbol(key)] = value
    end
    sync_grid_dimensions!(config; explicit = Set(Symbol.(keys(kwargs))))
    return launch_organism_gui(config; display_window = display_window)
end

"""
    launch_organism_gui(config::Dict{Symbol, Any}; display_window = true)

Build and display the interactive, three-dashboard GUI for `organism_ibm.jl`
using the provided configuration dictionary.

# Arguments
- `config::Dict{Symbol, Any}`: Configuration dictionary containing all simulation parameters.
- `display_window::Bool`: Set to `true` to render native OpenGL window.

# Returns
`Tuple{Figure, IBMGuiState}`: Makie figure handle and GUI state container.
"""
function launch_organism_gui(config::Dict{Symbol, Any}; display_window::Bool = true)
    # Activate GLMakie for high-performance interactive OpenGL rendering
    GLMakie.activate!()

    # 1. Initialize Simulation Model
    params = IBMParameters(config)

    model = initialize_model(
        n_agents = config[:n_agents],
        grid = config[:grid],
        dt = config[:dt],
        params = params,
        seed = config[:seed]
    )

    living_init = collect(allagents(model))
    xs_init = Float32[Float32(a.pos[1]) for a in living_init]
    ys_init = Float32[Float32(a.pos[2]) for a in living_init]
    sz_init = Float32[Float32(max(4.0, a.size * 0.15)) for a in living_init]
    col_init = String[organism_color(a, params) for a in living_init]

    # 2. Construct IBMGuiState Container
    state = IBMGuiState(
        model,
        Observable(false),               # is_running
        Observable(5.0),                 # sim_speed_days
        Observable(0.0),                 # current_time
        Observable("Initialized | Ready to run"),
        Observable(copy(model.temperature_field)),
        Observable(xs_init),
        Observable(ys_init),
        Observable(sz_init),
        Observable(col_init),
        Float64[0.0],                    # t_history
        Float64[length(living_init)],    # n_total_history
        Float64[0.0],                    # n_females_history
        Float64[0.0],                    # n_males_history
        Float64[0.0],                    # n_legal_history
        Float64[0.0],                    # biomass_legal_history
        Float64[0.0],                    # catch_history
        Float64[0.0],                    # mean_weight_legal_history
        Float64[0.0],                    # mean_weight_male_history
        Float64[0.0],                    # mean_weight_female_history
        Float64[0.0],                    # mean_weight_total_history
        0.0,                             # catch_at_year_start
        1,                               # current_year_idx
        Observable(Point2f[(0.0, length(living_init))]),
        Observable(Point2f[(0.0, 0.0)]),
        Observable(Point2f[(0.0, 0.0)]),
        Observable(Point2f[(0.0, 0.0)]),
        Observable(Point2f[(0.0, 0.0)]),
        Observable(Point2f[(0.0, 0.0)]),
        Observable(Float32[Float32(a.size) for a in living_init if a.sex == male]),
        Observable(Float32[Float32(a.size) for a in living_init if a.sex == female]),
        Observable(Point2f[]),           # obs_stage_counts
        Observable(Point2f[]),           # obs_stage_counts_male
        Observable(Point2f[]),           # obs_stage_counts_female
        Observable(Point2f[]),           # obs_ogive_male_pts
        Observable(Point2f[]),           # obs_ogive_female_pts
        Observable(Point2f[]),           # obs_selectivity_pts
        Observable(Point2f[]),           # obs_sr_pts
        create_empty_annual_data(),      # annual_data
        Observable(Point2f[]),
        Observable(Point2f[]),
        Observable(Point2f[]),
        Observable(Point2f[]),
        Observable("Macro models awaiting annual data (>= 4 years)"),
        Observable("Annual Surveyed Biomass"),   # macro_label_observed
        Observable("Schaefer"),                  # macro_label_sch
        Observable("Delay-Difference"),          # macro_label_dd
        Observable("Leslie Matrix"),             # macro_label_les
        Observable(""),                          # macro_conversion_text
        Dict{Int, Any}(),                # snapshots
        Observable(String["Year 0"]),    # snapshot_labels
        nothing
    )
    
    # Save year 0 initial state
    state.snapshots[0] = (
        model = deepcopy(model),
        t_history = copy(state.t_history),
        n_total_history = copy(state.n_total_history),
        n_females_history = copy(state.n_females_history),
        n_males_history = copy(state.n_males_history),
        n_legal_history = copy(state.n_legal_history),
        biomass_legal_history = copy(state.biomass_legal_history),
        catch_history = copy(state.catch_history),
        annual_data = copy(state.annual_data)
    )

    # 3. Create Main Makie Figure Window
    fig = Figure(
        size = (1750, 1000),
        title = "Organism IBM: Real-Time Ecosystem & Macro-Model Dashboard"
    )

    # Left Column: Unified Controls & Information Panel
    left_grid = fig[1, 1] = GridLayout(tellheight = false, width = 340)

    # 1. Summary Information (Time, etc.)
    Label(left_grid[1, 1], state.status_text, fontsize = 16, halign = :center, font = :bold)

    # 2. Transport Controls
    transport_grid = left_grid[2, 1] = GridLayout()
    btn_run = Button(transport_grid[1, 1], label = @lift($(state.is_running) ? "Pause" : "Run"))
    btn_step = Button(transport_grid[1, 2], label = "Step")
    Label(transport_grid[1, 3], "Total Sim Time (days):")
    step_tb = Textbox(transport_grid[1, 4], stored_string = "10", validator = Float64, width = 60)
    
    Label(transport_grid[2, 1:2], "Sim speed (d):", halign=:right)
    speed_tb = Textbox(transport_grid[2, 3:4], stored_string = "5.0", validator = Float64, width = 60)
    on(speed_tb.stored_string) do s; state.sim_speed_days[] = parse(Float64, s); end

    btn_reset = Button(transport_grid[3, 1:2], label = "Reset")
    btn_refit = Button(transport_grid[3, 3:4], label = "Refit Macro")

    Label(transport_grid[4, 1:2], "Restore Year:", halign=:right)
    snap_menu = Menu(transport_grid[4, 3:4], options = state.snapshot_labels, width = 90)

    on(snap_menu.selection) do sel
        if sel !== nothing
            m = match(r"Year (\d+)", sel)
            if m !== nothing
                y = parse(Int, m.captures[1])
                if haskey(state.snapshots, y)
                    snap = state.snapshots[y]
                    state.model = deepcopy(snap.model)
                    state.t_history = copy(snap.t_history)
                    state.n_total_history = copy(snap.n_total_history)
                    state.n_females_history = copy(snap.n_females_history)
                    state.n_males_history = copy(snap.n_males_history)
                    state.n_legal_history = copy(snap.n_legal_history)
                    state.biomass_legal_history = copy(snap.biomass_legal_history)
                    state.catch_history = copy(snap.catch_history)
                    state.annual_data = copy(snap.annual_data)
                    state.current_year_idx = y
                    state.current_time[] = Float64(abmproperties(state.model)[:current_time])
                    step_gui_simulation!(state, 0.0)
                end
            end
        end
    end

    # 3. Parameter Steering
    Label(left_grid[3, 1], "Parameter Steering", font = :bold, fontsize = 16)
    param_grid = left_grid[4, 1] = GridLayout(tellwidth = false)

    labels = ["Base Temp (°C)", "Temp Amp (°C)", "Food Max", "Carrying Cap", "Habitat Qual", "Habitat Qual SD", "Legal Size", "Harvest Prob", "Harvest Start (d)", "Harvest End (d)"]
    start_vals = [
        params.environment.temp_base,
        params.environment.temp_amplitude,
        params.environment.food_max,
        params.environment.benthic_carrying_capacity,
        params.environment.habitat_quality_mean,
        params.environment.habitat_quality_sd,
        params.fishery.legal_size_male,
        params.fishery.fishing_probability_daily,
        params.fishery.fishing_season_start,
        params.fishery.fishing_season_end
    ]

    tboxes = Textbox[]
    for i in 1:10
        Label(param_grid[i, 1], labels[i], halign = :left)
        tb = Textbox(param_grid[i, 2], stored_string = string(start_vals[i]), validator = Float64)
        push!(tboxes, tb)
    end

    on(tboxes[1].stored_string) do s; abmproperties(state.model)[:params] = reconstruct_params(abmproperties(state.model)[:params]; temp_base = parse(Float64, s)); end
    on(tboxes[2].stored_string) do s; abmproperties(state.model)[:params] = reconstruct_params(abmproperties(state.model)[:params]; temp_amplitude = parse(Float64, s)); end
    on(tboxes[3].stored_string) do s; abmproperties(state.model)[:params] = reconstruct_params(abmproperties(state.model)[:params]; food_max = parse(Float64, s)); end
    on(tboxes[4].stored_string) do s; abmproperties(state.model)[:params] = reconstruct_params(abmproperties(state.model)[:params]; benthic_carrying_capacity = parse(Float64, s)); end
    on(tboxes[5].stored_string) do s; abmproperties(state.model)[:params] = reconstruct_params(abmproperties(state.model)[:params]; habitat_quality_mean = clamp(parse(Float64, s), 0.0, 1.0)); end
    on(tboxes[6].stored_string) do s; abmproperties(state.model)[:params] = reconstruct_params(abmproperties(state.model)[:params]; habitat_quality_sd = parse(Float64, s)); end
    on(tboxes[7].stored_string) do s; abmproperties(state.model)[:params] = reconstruct_params(abmproperties(state.model)[:params]; legal_size_male = parse(Float64, s)); end
    on(tboxes[8].stored_string) do s; abmproperties(state.model)[:params] = reconstruct_params(abmproperties(state.model)[:params]; fishing_probability_daily = parse(Float64, s)); end
    on(tboxes[9].stored_string) do s; abmproperties(state.model)[:params] = reconstruct_params(abmproperties(state.model)[:params]; fishing_season_start = parse(Float64, s)); end
    on(tboxes[10].stored_string) do s; abmproperties(state.model)[:params] = reconstruct_params(abmproperties(state.model)[:params]; fishing_season_end = parse(Float64, s)); end

    # 4. Process Feature Switches (Toggles)
    tog_grid = left_grid[5, 1] = GridLayout(tellwidth = false)
    Label(tog_grid[1, 1:4], "Biological Process Toggles", font = :bold)

    tog_energetics = Toggle(tog_grid[2, 1], active = params.toggles.use_energetics)
    Label(tog_grid[2, 2], "Energetics", fontsize = 12)
    tog_thermal = Toggle(tog_grid[2, 3], active = params.toggles.use_thermal_stress)
    Label(tog_grid[2, 4], "Thermal", fontsize = 12)

    tog_cannibalism = Toggle(tog_grid[3, 1], active = params.toggles.use_cannibalism)
    Label(tog_grid[3, 2], "Cannibalism", fontsize = 12)
    tog_senescence = Toggle(tog_grid[3, 3], active = params.toggles.use_senescence)
    Label(tog_grid[3, 4], "Senescence", fontsize = 12)

    lift(tog_energetics.active) do a
        abmproperties(state.model)[:params] = reconstruct_params(
            abmproperties(state.model)[:params]; use_energetics = a
        )
    end
    lift(tog_thermal.active) do a
        abmproperties(state.model)[:params] = reconstruct_params(
            abmproperties(state.model)[:params]; use_thermal_stress = a
        )
    end
    lift(tog_cannibalism.active) do a
        abmproperties(state.model)[:params] = reconstruct_params(
            abmproperties(state.model)[:params]; use_cannibalism = a
        )
    end
    lift(tog_senescence.active) do a
        abmproperties(state.model)[:params] = reconstruct_params(
            abmproperties(state.model)[:params]; use_senescence = a
        )
    end

    # Right Content Area: Holds the Three Interlinked Dashboards
    content_grid = fig[1, 2] = GridLayout()

    # --------------------------------------------------------------------------
    # Col 1: Map, Macro-Models, Stock-Recruitment
    # --------------------------------------------------------------------------
    ax_map = Axis(
        content_grid[1:2, 1],
        title = "Spatial Habitat & Organism Distribution",
        xlabel = "Grid X",
        ylabel = "Grid Y",
        aspect = AxisAspect(1)
    )
    GLMakie.heatmap!(ax_map, state.obs_env_field, colormap = :thermal)
    GLMakie.scatter!(
        ax_map,
        state.obs_agent_x,
        state.obs_agent_y,
        markersize = state.obs_agent_sizes,
        color = state.obs_agent_colors,
        strokewidth = 0.5,
        strokecolor = :black
    )

    # The macro-model fits are in number, so the panel states the conversion it
    # applied to put them in kg. The goodness-of-fit summary was computed on every
    # re-fit but never rendered; it goes in the same title so the reader can see
    # both the conversion and how well the models fit.
    ax_macro = Axis(
        content_grid[3, 1],
        title = lift(
            (conv, met) -> "Biomass, Catch & Macro-Models\n" *
                (isempty(conv) ? "" : conv * "\n") * met,
            state.macro_conversion_text,
            state.macro_metrics_text
        ),
        xlabel = "Simulation Time (days)",
        ylabel = "Biomass (kg)"
    )
    GLMakie.lines!(ax_macro, state.obs_ts_biomass, label = "Legal Fishable Biomass", color = :purple, linewidth = 2.2)
    GLMakie.lines!(ax_macro, state.obs_ts_catch, label = "Annual Catch", color = :darkorange, linewidth = 1.8, linestyle = :dash)
    GLMakie.scatter!(ax_macro, state.macro_obs_biomass, color = :black, markersize = 9, label = state.macro_label_observed[])
    GLMakie.lines!(ax_macro, state.macro_sch_biomass, color = :royalblue, linewidth = 2.0, linestyle = :dash, label = state.macro_label_sch[])
    GLMakie.lines!(ax_macro, state.macro_dd_biomass, color = :seagreen, linewidth = 2.0, linestyle = :dashdot, label = state.macro_label_dd[])
    GLMakie.lines!(ax_macro, state.macro_les_biomass, color = :darkorange, linewidth = 2.2, label = state.macro_label_les[])
    GLMakie.axislegend(ax_macro, position = :lt, labelsize = 10, framevisible = false, backgroundcolor = :transparent, patchsize = (15, 10))

    ax_sr = Axis(
        content_grid[4, 1],
        title = "Stock-Recruitment Relationship",
        xlabel = "Eggs Produced (S)",
        ylabel = "Recruits Settled (R)"
    )
    GLMakie.scatter!(ax_sr, state.obs_sr_pts, color = :seagreen, markersize = 7, label = "Realized Benthic Batches")
    GLMakie.axislegend(ax_sr, position = :lt, labelsize = 10, framevisible = false, backgroundcolor = :transparent, patchsize = (15, 10))

    # --------------------------------------------------------------------------
    # Col 2: Trajectories, Stage Counts, Size Frequency, Ogive
    # --------------------------------------------------------------------------
    ax_ts_n = Axis(
        content_grid[1, 2],
        title = "Abundance Trajectories",
        xlabel = "Time (days)",
        ylabel = "Abundance (count)"
    )
    GLMakie.lines!(ax_ts_n, state.obs_ts_total, label = "Total Organisms", color = :royalblue, linewidth = 2.0)
    GLMakie.lines!(ax_ts_n, state.obs_ts_females, label = "Mature Females", color = :crimson, linewidth = 1.8)
    GLMakie.lines!(ax_ts_n, state.obs_ts_males, label = "Mature Males", color = :seagreen, linewidth = 1.8)
    GLMakie.lines!(ax_ts_n, state.obs_ts_legal, label = "Legal Fishable Males", color = :purple, linewidth = 2.0)
    GLMakie.axislegend(ax_ts_n, position = :lt, labelsize = 10, framevisible = false, backgroundcolor = :transparent, patchsize = (15, 10))

    ax_stage = Axis(
        content_grid[2, 2],
        title = "Stage Abundance Structure by Sex",
        xlabel = "Stage",
        ylabel = "Count",
        xticks = 4:14
    )
    obs_stage_m_shifted = lift(state.obs_stage_counts_male) do pts
        [Point2f(p[1] - 0.15f0, p[2]) for p in pts]
    end
    obs_stage_f_shifted = lift(state.obs_stage_counts_female) do pts
        [Point2f(p[1] + 0.15f0, p[2]) for p in pts]
    end
    obs_stage_m_segs = lift(obs_stage_m_shifted) do pts
        segs = Point2f[]
        for p in pts
            push!(segs, Point2f(p[1], 0.0f0))
            push!(segs, p)
        end
        return segs
    end
    obs_stage_f_segs = lift(obs_stage_f_shifted) do pts
        segs = Point2f[]
        for p in pts
            push!(segs, Point2f(p[1], 0.0f0))
            push!(segs, p)
        end
        return segs
    end
    GLMakie.linesegments!(ax_stage, obs_stage_m_segs, color = :royalblue, linewidth = 1.8)
    GLMakie.linesegments!(ax_stage, obs_stage_f_segs, color = :crimson, linewidth = 1.8)
    GLMakie.scatter!(ax_stage, obs_stage_m_shifted, color = :royalblue, markersize = 10, label = "Males")
    GLMakie.scatter!(ax_stage, obs_stage_f_shifted, color = :crimson, markersize = 10, label = "Females")
    GLMakie.axislegend(ax_stage, position = :rt, labelsize = 10, framevisible = false, backgroundcolor = :transparent, patchsize = (15, 10))

    ax_size = Axis(
        content_grid[3, 2],
        title = "Size Frequency by Sex",
        xlabel = "Size (mm)",
        ylabel = "Density"
    )
    xlims!(ax_size, 0, 165)
    GLMakie.density!(ax_size, state.obs_cw_male, color = (:royalblue, 0.45), strokecolor = :royalblue, strokewidth = 1.5, label = "Males")
    GLMakie.density!(ax_size, state.obs_cw_female, color = (:crimson, 0.45), strokecolor = :crimson, strokewidth = 1.5, label = "Females")
    GLMakie.axislegend(ax_size, position = :rt, labelsize = 10, framevisible = false, backgroundcolor = :transparent, patchsize = (15, 10))

    ax_ogive = Axis(
        content_grid[4, 2],
        title = "Maturity Ogive & Fishery Selectivity",
        xlabel = "Size (mm)",
        ylabel = "P(Mature) / Selectivity"
    )
    xlims!(ax_ogive, 0, 165)
    cw_ref = 20.0:1.0:140.0
    GLMakie.lines!(ax_ogive, cw_ref, [maturity(0, log(c)) for c in cw_ref], color = :black, linewidth = 2.0, label = "Theoretical Male")
    GLMakie.lines!(ax_ogive, cw_ref, [maturity(1, log(c)) for c in cw_ref], color = :grey40, linewidth = 2.0, linestyle = :dash, label = "Theoretical Female")
    GLMakie.lines!(ax_ogive, cw_ref, [selectivity(c; s50 = params.fishery.legal_size_male) for c in cw_ref], color = :purple, linewidth = 2.0, label = "Theoretical S(L)")
    GLMakie.scatter!(ax_ogive, state.obs_ogive_male_pts, color = :royalblue, markersize = 8, label = "Realized Males")
    GLMakie.scatter!(ax_ogive, state.obs_ogive_female_pts, color = :crimson, markersize = 8, label = "Realized Females")
    GLMakie.axislegend(ax_ogive, position = :lt, labelsize = 10, framevisible = false, backgroundcolor = :transparent, patchsize = (15, 10))

    on(state.current_time) do _
        try
            autolimits!(ax_ts_n)
            autolimits!(ax_stage)
            autolimits!(ax_macro)
            autolimits!(ax_size)
            xlims!(ax_size, 0, 165)
            autolimits!(ax_ogive)
            xlims!(ax_ogive, 0, 165)
            autolimits!(ax_sr)
        catch
        end
    end

    # --------------------------------------------------------------------------
    # Transport Button Callbacks
    # --------------------------------------------------------------------------
    on(btn_run.clicks) do _
        state.is_running[] = !state.is_running[]
    end

    on(btn_step.clicks) do _
        val_str = step_tb.stored_string[]
        step_days = parse(Float64, val_str)
        step_gui_simulation!(state, step_days)
    end

    on(btn_reset.clicks) do _
        state.is_running[] = false
        # Rebuild from the *current* parameters rather than constructing a fresh
        # set, so the species, leslie, phenology, mortality and organism values
        # loaded from the configuration file survive a reset. Constructing
        # IBMParameters from the ten text boxes alone silently reverted everything
        # else to its struct default.
        p = reconstruct_params(
            abmproperties(state.model)[:params];
            temp_base = parse(Float64, tboxes[1].stored_string[]),
            temp_amplitude = parse(Float64, tboxes[2].stored_string[]),
            food_max = parse(Float64, tboxes[3].stored_string[]),
            benthic_carrying_capacity = parse(Float64, tboxes[4].stored_string[]),
            habitat_quality_mean = clamp(parse(Float64, tboxes[5].stored_string[]), 0.0, 1.0),
            habitat_quality_sd = parse(Float64, tboxes[6].stored_string[]),
            legal_size_male = parse(Float64, tboxes[7].stored_string[]),
            fishing_probability_daily = parse(Float64, tboxes[8].stored_string[]),
            fishing_season_start = parse(Float64, tboxes[9].stored_string[]),
            fishing_season_end = parse(Float64, tboxes[10].stored_string[])
        )
        state.model = initialize_model(
            n_agents = config[:n_agents],
            grid = config[:grid],
            dt = config[:dt],
            params = p,
            seed = config[:seed]
        )
        empty!(state.t_history)
        empty!(state.n_total_history)
        empty!(state.n_females_history)
        empty!(state.n_males_history)
        empty!(state.n_legal_history)
        empty!(state.biomass_legal_history)
        empty!(state.catch_history)
        state.catch_at_year_start = 0.0
        state.current_year_idx = 1
        state.annual_data = create_empty_annual_data()
        step_gui_simulation!(state, 0.0)
        state.status_text[] = "Model Reset | Day 0"
    end

    on(btn_refit.clicks) do _
        if nrow(state.annual_data) >= 4
            recalibrate_macro_models!(state)
        else
            state.status_text[] = "Refit failed: Need >= 4 annual periods (currently $(nrow(state.annual_data)))"
        end
    end

    # --------------------------------------------------------------------------
    # Asynchronous Stepping Loop
    # --------------------------------------------------------------------------
    state.sim_task = @async while true
        if state.is_running[]
            step_gui_simulation!(state, state.sim_speed_days[])
            sleep(0.02) # Yield to OpenGL UI event loop
        else
            sleep(0.08)
        end
    end

    # Initial paint
    step_gui_simulation!(state, 0.0)

    if display_window
        display(fig)
    end

    return (fig, state)
end
