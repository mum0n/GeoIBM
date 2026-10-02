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
# Plot-scale conventions
# ------------------------------------------------------------------------------

"""
    COUNT_PLOT_FLOOR

Smallest count any log-scaled dashboard axis will draw.

`log10(0)` is `-Inf`, which no axis can place, and zero is a real value here
rather than a missing one: a stage with no survivors, a clutch that has not been
laid yet, a cohort that has already individuated. Flooring the count before the
log keeps the transform finite and keeps the axis monotonic in the count.

One individual is the floor, so a plotted value of `0` reads as "at most one",
which is the honest statement about a sub-unit count. It is deliberately not
`0.1`: that would imply the model resolves fractions of an animal, and it puts a
spurious decade between "extinct" and "present".
"""
const COUNT_PLOT_FLOOR = 1.0

"""
    log10_count(n::Real)

`log10` of a count, floored at [`COUNT_PLOT_FLOOR`](@ref) so that an empty stage
plots at a finite position instead of `-Inf`.

Applied at *draw* time rather than at record time. The history buffers hold raw
counts, so a run can still be exported, refitted or re-plotted on a linear axis
without having lost the numbers.
"""
log10_count(n::Real) = log10(max(n, COUNT_PLOT_FLOOR))

"""
    GUI_EARLY_STAGES

The aggregate, pre-individuation stages drawn on the stage-abundance figure.

These are `AggregateGroup` cohorts, not `Organism` agents, and `AggregateGroup`
carries no `sex` field — sex is not discriminated below the individuation
boundary (`OrganismParams.juvenile_max_stage`). They are therefore drawn as a
single grey line each rather than as a male/female pair.
"""
const GUI_EARLY_STAGES = 0:3

# Presentation of the five undifferentiated series on the stage-abundance figure.
# Shade runs dark to light with developmental stage, so the ordering survives a
# greyscale print, and the dash patterns separate lines that would otherwise be
# indistinguishable in the legend.
const GUI_EARLY_LABELS = ("Eggs (brooded)", "Stage 0", "Stage 1", "Stage 2", "Stage 3")
const GUI_EARLY_COLOURS = (:grey10, :grey25, :grey40, :grey55, :grey70)
const GUI_EARLY_DASHES = (:solid, :dash, :dot, :dashdot, :dashdotdot)

@assert length(GUI_EARLY_LABELS) == length(GUI_EARLY_STAGES) + 1
@assert length(GUI_EARLY_COLOURS) == length(GUI_EARLY_LABELS)
@assert length(GUI_EARLY_DASHES) == length(GUI_EARLY_LABELS)

"""
    RESCAN_IMMUTABLE

Parameters that describe the *shape* of a run rather than its dynamics, and so
cannot be changed on a live model: the grid extent, agent count and time step are
baked into the `StandardABM` when it is constructed, and `seed` only has meaning
at construction. Applying a new `grid_nx` to a running model would leave every
agent's position outside the new space.

[`rescan_config!`](@ref) reports these as requiring a restart instead of applying
them.
"""
const RESCAN_IMMUTABLE = (:grid_nx, :grid_ny, :grid_size, :n_agents, :dt, :seed)

"""
    flatten_toml(cfg, prefix = "")

Flatten a parsed TOML table to `"section.key" => value` pairs.

The configuration is nested (`[environment]` then `temp_base`) while the parameter
structs are flat fields on a named struct, so every comparison between the file
and the model works on the flattened form.
"""
function flatten_toml(cfg::AbstractDict, prefix = "")
    out = Dict{String, Any}()
    for (k, v) in cfg
        key = isempty(prefix) ? String(k) : string(prefix, ".", k)
        if v isa AbstractDict
            merge!(out, flatten_toml(v, key))
        else
            out[key] = v
        end
    end
    return out
end

"""
    config_diff(previous, current)

Values that differ between two configuration tables, as `key => (old, new)` pairs
sorted by key.

Both arguments may be nested (as parsed from TOML) or already flattened;
`flatten_toml` is applied to each, so passing a nested table by mistake compares
like with like instead of reporting every nested table as a single changed key.

Keys present in `previous` but absent from `current` are reported with a `nothing`
new value, so a parameter deleted from the file shows up in the report rather than
being silently ignored.
"""
function config_diff(previous::AbstractDict, current::AbstractDict)
    prev = flatten_toml(previous)
    cur = flatten_toml(current)
    out = Pair{String, Any}[]
    for k in sort!(collect(keys(cur)))
        new = cur[k]
        if !haskey(prev, k)
            push!(out, k => (nothing, new))
        elseif !isequal(prev[k], new)
            push!(out, k => (prev[k], new))
        end
    end
    for k in sort!(collect(keys(prev)))
        haskey(cur, k) || push!(out, k => (prev[k], nothing))
    end
    return out
end

"""
    rescan_config!(state::Any) -> String

Re-read `state.config_path`, apply only what changed since the last scan, and
update the snapshot. Returns a one-line report for the status bar, or `""` when
there was nothing to do.

The diff is the point, and it is not an optimisation. `configs/snowcrab.toml` is
complete — it declares every parameter — so re-parsing it and rebuilding the
parameter tree from the result would re-apply all 260 values annually and silently
discard anything the user had changed in the parameter-steering panel since the
last scan. Comparing against the previous scan and touching only the fields that
genuinely moved keeps those live edits. The panel's edits are not written back to
the file, so the two are independent by design: the file is the scenario, the
panel is the perturbation.

Each changed key is applied as a leaf keyword via `reconstruct_params`, starting
from the *live* parameters rather than from a fresh struct.

Keys in [`RESCAN_IMMUTABLE`](@ref) are reported and skipped, as are keys deleted
from the file: for a deletion there is no value to restore, and the panel may have
changed it in the meantime. The snapshot records what the file now says rather than
what was applied, so those keys are not re-reported on every subsequent scan.
"""
# `state` is unannotated: `IBMGuiState` is declared further down this file, so the
# type is not yet in scope to annotate a signature here.
function rescan_config!(state)
    path = state.config_path
    (isempty(path) || !isfile(path)) && return ""

    current = flatten_toml(TOML.parsefile(path))
    diff = config_diff(state.config_snapshot, current)
    isempty(diff) && return ""

    applied = String[]
    skipped = String[]
    # `reconstruct_params` takes *leaf* field names, not whole sections: it passes
    # every keyword to every parameter struct and keeps the ones that are fields of
    # that struct. Handing it `environment => new_struct` would be silently ignored,
    # because `environment` is not a field of `EnvironmentParams`. That is safe to
    # rely on here only because all 260 leaf names are unique across sections, so a
    # name always identifies exactly one parameter; `test/runtests.jl` asserts it.
    kwargs_list = Pair{Symbol, Any}[]
    for (key, (old, new)) in diff
        parts = split(key, '.')
        if length(parts) < 2 || new === nothing || Symbol(parts[2]) in RESCAN_IMMUTABLE
            push!(skipped, key)
            continue
        end
        push!(kwargs_list, Symbol(parts[2]) => new)
        push!(applied, "$key: $old -> $new")
    end

    if !isempty(kwargs_list)
        p = abmproperties(state.model)[:params]
        abmproperties(state.model)[:params] = reconstruct_params(p; kwargs_list...)
    end

    state.config_snapshot = current

    msg = "Config: $(length(applied)) applied"
    isempty(skipped) || (msg *= ", restart-only: " * join(skipped, ", "))
    return msg
end

"""
    GUI_SIZE_BIN_WIDTH_MM

Width of one size-frequency bin.

Five millimetres is about one instar for the stages the fishery cares about, which
is the resolution the size structure is read at; finer bins mostly add sampling
noise to the tail, where a handful of large males dominate.
"""
const GUI_SIZE_BIN_WIDTH_MM = 5.0

"""
    size_frequency(sizes, bin_width)

Histogram density of `sizes` on a regular grid starting at zero.

Returns `(centres, density)` where `density[i]` is the number of individuals in
bin `i` divided by `total * bin_width`, i.e. individuals per millimetre, so the
series integrates to one and the four size-frequency curves are comparable
regardless of how many males and females are alive.

A histogram rather than a kernel density estimate. The panel is drawn on a log10
axis, and a KDE is the wrong shape for that in two ways: it is positive by
construction and so cannot represent a genuinely empty size range, and it leaks
density across an empty bin from its neighbours, which on a log axis reads as a
population where there is none. Bins are half-open `[lo, hi)` so no size is
double-counted at a boundary.
"""
function size_frequency(sizes::AbstractVector{<:Real},
    bin_width::Real = GUI_SIZE_BIN_WIDTH_MM)
    width = Float64(bin_width)
    width > 0.0 || error("bin width must be positive, got $width")
    kept = filter(s -> isfinite(s), sizes)
    isempty(kept) && return (Float64[], Float64[])
    upper = max(ceil(maximum(kept) / width), 1.0) * width
    n_bins = Int(round(upper / width))
    counts = zeros(Int, n_bins)
    for s in kept
        b = clamp(floor(Int, s / width) + 1, 1, n_bins)
        counts[b] += 1
    end
    n = length(kept)
    centres = [(i - 0.5) * width for i in 1:n_bins]
    return (centres, counts ./ (n * width))
end

"""
    DENSITY_PLOT_FLOOR

Smallest density the log-scaled size-frequency axis will draw.

Distinct from [`COUNT_PLOT_FLOOR`](@ref): that floor is one individual, which is
meaningful for a count, whereas a density is per-millimetre and genuinely lands
far below one. This is a plotting sentinel only — roughly twelve decades below any
density the model produces — so an empty bin is drawn at the floor instead of
disappearing or dragging the axis to `-Inf`, without implying a density was
measured there.
"""
const DENSITY_PLOT_FLOOR = 1e-12

"""
    log10_density(d::Real)

`log10` of a density, floored at [`DENSITY_PLOT_FLOOR`](@ref).
"""
log10_density(d::Real) = log10(max(d, DENSITY_PLOT_FLOOR))

# ------------------------------------------------------------------------------
# Data Structure for GUI State Management
# ------------------------------------------------------------------------------

"""
    CALENDAR_MONTH_END

Last julian day of each calendar month in a 365-day year, so day 59 is the end of
February. Used to turn the julian day into a month the reader recognises.
"""
const CALENDAR_MONTH_END = (31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365)

"""
    calendar_month(day_of_year::Integer, days_per_year::Real)

Calendar month (1-12) containing the 1-based julian `day_of_year`.

Exact for a 365-day year. In a 366-day year, February is given its extra day by
`days_per_year >= 366`, so every later month boundary stays on its calendar date;
the only cost is that day 366 reports as December rather than as a 13th month.

This replaces `floor((day % 365) / 30.416) + 1`, which put the last day of every
month into the next one.
"""
function calendar_month(day_of_year::Integer, days_per_year::Real = 365.0)
    leap = days_per_year >= 366.0
    acc = 0
    for (m, len) in enumerate((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31))
        acc += (m == 2 && leap) ? 1 : 0
        acc += len
        day_of_year <= acc && return m
    end
    return 12
end

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
# `@kwdef` generates a keyword constructor alongside the positional one. This
# matters here: the state carries 66 fields, and a purely positional call site
# silently desynchronises whenever a field is added, removed or reordered on one
# side only. The failure surfaces as a `MethodError` whose signature dump runs to
# several thousand characters, with no indication of *which* field is wrong. With
# keywords, Julia checks names rather than positions, so the compiler and the test
# suite both catch it immediately.
Base.@kwdef mutable struct IBMGuiState
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
    config_path::String  # Path to the TOML config file for annual rescan
    # Flattened form of the config as last read, so a rescan can tell what the
    # user edited since. See `rescan_config!`.
    config_snapshot::Dict{String, Any}

    obs_ts_total::Observable{Vector{Point2f}}
    obs_ts_females::Observable{Vector{Point2f}}
    obs_ts_males::Observable{Vector{Point2f}}
    obs_ts_legal::Observable{Vector{Point2f}}
    obs_ts_biomass::Observable{Vector{Point2f}}
    obs_ts_catch::Observable{Vector{Point2f}}

    # Pre-individuation abundances for the stage-abundance figure: eggs currently
    # brooded, then one series per `GUI_EARLY_STAGES` aggregate cohort. Raw
    # counts, as above; the log transform happens when the figure is drawn.
    n_egg_history::Vector{Float64}
    n_early_stage_history::Vector{Vector{Float64}}
    # Individuated (sex-discriminated) abundances over the same stages.
    n_early_agents_history::Vector{NTuple{3, Float64}}  # (male, female, undifferentiated)

    # Current (not time-series) abundances for the stage figure, written every
    # step. These are separate observables rather than a `lift` over the history
    # buffers because `lift` over a plain `Vector` *iterates* it, so the closure
    # would receive one buffer element at a time rather than the buffer.
    obs_current_egg::Observable{Float64}
    obs_current_early_stages::Observable{NTuple{4, Float64}}
    obs_current_agents::Observable{NTuple{3, Float64}}

    obs_ts_egg::Observable{Vector{Point2f}}
    # One observable per `GUI_EARLY_STAGES` value, held as a vector of observables
    # rather than one observable wrapping a vector of vectors: `lines!` takes a
    # single point vector, so a nested `Observable{Vector{Vector{Point2f}}}` has
    # to be unwrapped with `[]` at every call site and silently fails if skipped.
    obs_ts_early_stage::Vector{Observable{Vector{Point2f}}}
    obs_ts_early_agents::Observable{Vector{Point2f}}
    obs_ts_early_agents_f::Observable{Vector{Point2f}}
    obs_ts_early_agents_u::Observable{Vector{Point2f}}


    # Size structure & demographic observables. `*_mature_*` are the subset of
    # each sex that the model itself flags as mature, so the panel shows maturity
    # as a proportion of each sex's own size structure rather than against an
    # independently chosen threshold.
    obs_cw_male::Observable{Vector{Float32}}
    obs_cw_female::Observable{Vector{Float32}}
    obs_cw_mature_male::Observable{Vector{Float32}}
    obs_cw_mature_female::Observable{Vector{Float32}}

    # Realized functional relationships observables
    obs_ogive_male_pts::Observable{Vector{Point2f}}
    obs_ogive_female_pts::Observable{Vector{Point2f}}
    obs_selectivity_pts::Observable{Vector{Point2f}}
    obs_sr_pts::Observable{Vector{Point2f}}
    # Raw (untransformed) spawner/recruit pairs, in the order plotted. The panel
    # draws log10 of both axes, but `fit_stock_recruitment` is fitted in linear
    # space, so the untransformed pairs have to survive the transform.
    sr_raw::Vector{Tuple{Float64, Float64}}
    # Ricker fit `R = a·S·exp(-b·S)`, drawn in log10 and annotated with its
    # parameters. Empty when there are too few points to fit.
    obs_sr_fit_curve::Observable{Vector{Point2f}}
    obs_sr_fit_text::Observable{String}

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
    sim_task::Union{Nothing, Task} = nothing
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
    params = abmproperties(model)[:params]
    # Year length comes from the configuration, not a literal 365. The GUI used to
    # assume a 365-day year here while the rest of the model honoured
    # `SimulationParams.days_per_year`, so a leap-year configuration shifted every
    # GUI year boundary, snapshot and season test relative to the simulation's own.
    dpy = params.simulation.days_per_year

    # Advance model
    prev_year = floor(Int, get(abmproperties(model), :current_time, 0.0) / dpy)

    if days > 0.0
        Agents.step!(model, n_steps)
    end

    curr_t = Float64(abmproperties(model)[:current_time])
    state.current_time[] = curr_t

    curr_year = floor(Int, curr_t / dpy)
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

        # Re-read the configuration once a simulated year, so a scenario can be
        # edited while the run is in progress. Only fields that actually moved in
        # the file are applied, so parameter-steering edits made in the GUI survive
        # the scan. A failure here must not stop the run: a half-typed TOML file is
        # a normal thing to have on disk while someone is editing it.
        if !isempty(state.config_path)
            try
                msg = rescan_config!(state)
                isempty(msg) || (state.status_text[] *= " | $msg")
            catch err
                state.status_text[] *= " | Config rescan failed: " *
                    first(split(sprint(showerror, err), '\n'))
            end
        end
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

    # Record one sample per *advance*, not per call. `launch_organism_gui` ends
    # with an initial `step_gui_simulation!(state, 0.0)` to paint the panel before
    # any stepping, and the snapshot/restore handler also repaints with `0.0`;
    # pushing unconditionally there appended a second point at t = 0, so every
    # trajectory in the dashboard carried a duplicated origin.
    #
    # Decided once, before any buffer is appended. Re-evaluating `t_history[end]`
    # at the later stage/egg block would read the value this block just wrote and
    # see "no advance", silently leaving those buffers one short of the time axis.
    # An empty buffer always records: Reset clears these before repainting, so the
    # post-reset step must not compare against a previous time that no longer exists.
    advanced = isempty(state.t_history) || curr_t != state.t_history[end]
    if advanced
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
    end

    year_idx = floor(Int, curr_t / dpy) + 1
    if year_idx > state.current_year_idx
        state.current_year_idx = year_idx
        state.catch_at_year_start = cum_catch
    end
    annual_catch = cum_catch - state.catch_at_year_start

    # Annual catch is rescaled against `catch_at_year_start` above, which must
    # happen on every call including a repaint; only the *record* is gated on an
    # advance, so this buffer stays the same length as the rest.
    if advanced
        push!(state.catch_history, annual_catch)
    end

    # Update timeseries line observables. The four abundance series are counts and
    # go through `log10_count`; biomass and annual catch are kilograms, not counts,
    # and a log axis would be the wrong scale for them (and would divide by zero
    # on any day the catch is nil).
    pts_tot = [Point2f(state.t_history[i], Float32(log10_count(state.n_total_history[i])))
               for i in eachindex(state.t_history)]
    pts_fem = [Point2f(state.t_history[i], Float32(log10_count(state.n_females_history[i])))
               for i in eachindex(state.t_history)]
    pts_mal = [Point2f(state.t_history[i], Float32(log10_count(state.n_males_history[i])))
               for i in eachindex(state.t_history)]
    pts_leg = [Point2f(state.t_history[i], Float32(log10_count(state.n_legal_history[i])))
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
    # Maturity uses the agent's own flag rather than re-deriving a size threshold,
    # so the mature curve tracks exactly the individuals the rest of the model
    # treats as mature (and which the ogive and macro panels also use).
    cw_mm = Float32[Float32(a.size) for a in living if a.sex == male && a.is_mature]
    cw_mf = Float32[Float32(a.size) for a in living if a.sex == female && a.is_mature]
    state.obs_cw_mature_male[] = isempty(cw_mm) ? Float32[0.0] : cw_mm
    state.obs_cw_mature_female[] = isempty(cw_mf) ? Float32[0.0] : cw_mf

    # Pre-individuation abundances for the stage-abundance figure, in one pass.
    #
    # Eggs are counted from the brooding females' clutches, which is the pool
    # *before* hatching: a clutch becomes a `stage = 0` aggregate cohort only at
    # `org.hatch_day`, so eggs and stage 0 are disjoint pools and both are needed.
    # Summing `egg_count` unconditionally matches `summarize_population_state`,
    # which reports the same quantity as `total_eggs_brooded`; the undifferentiated
    # branch is defensive, since `build_stages` currently emits only male and female
    # agents, but a species that tracked sex later would otherwise be silently
    # dropped from the figure.
    n_egg = 0
    n_agent_male = 0
    n_agent_female = 0
    n_agent_undiff = 0
    for o in living
        if o.sex === male
            n_agent_male += 1
        elseif o.sex === female
            n_agent_female += 1
        else
            n_agent_undiff += 1
        end
        n_egg += o.egg_count
    end

    # Aggregate cohort abundances for `GUI_EARLY_STAGES`. `AggregateGroup` has no
    # `sex` field, so these stages are undifferentiated by construction.
    early_counts = zeros(Float64, length(GUI_EARLY_STAGES))
    agg = get(abmproperties(model), :aggregate_populations, nothing)
    if agg !== nothing
        for g in agg.groups
            i = findfirst(==(g.stage), GUI_EARLY_STAGES)
            i === nothing || (early_counts[i] += g.abundance)
        end
    end

    # Same `advanced` flag as the demographic buffers above: these are read back
    # with `eachindex(t_history)`, so a repaint that appended without advancing
    # would leave them one longer than the time axis they are plotted against.
    if advanced
        push!(state.n_egg_history, Float64(n_egg))
        for i in eachindex(state.n_early_stage_history)
            push!(state.n_early_stage_history[i], early_counts[i])
        end
        push!(state.n_early_agents_history,
            (Float64(n_agent_male), Float64(n_agent_female), Float64(n_agent_undiff)))
    end

    # Current values are recomputed on every call, including a repaint, so the
    # stage figure reflects the model after a snapshot restore as well as after a
    # step.
    state.obs_current_egg[] = Float64(n_egg)
    state.obs_current_early_stages[] =
        (early_counts[1], early_counts[2], early_counts[3], early_counts[4])
    state.obs_current_agents[] =
        (Float64(n_agent_male), Float64(n_agent_female), Float64(n_agent_undiff))

    # The figure is drawn on a log10 count axis; the buffers above keep raw counts.
    t_hist = state.t_history
    state.obs_ts_egg[] =
        [Point2f(t_hist[i], Float32(log10_count(state.n_egg_history[i])))
         for i in eachindex(t_hist)]
    for k in eachindex(state.obs_ts_early_stage)
        buf = state.n_early_stage_history[k]
        state.obs_ts_early_stage[k][] =
            [Point2f(t_hist[i], Float32(log10_count(buf[i]))) for i in eachindex(t_hist)]
    end
    state.obs_ts_early_agents[] =
        [Point2f(t_hist[i], Float32(log10_count(state.n_early_agents_history[i][1])))
         for i in eachindex(t_hist)]
    state.obs_ts_early_agents_f[] =
        [Point2f(t_hist[i], Float32(log10_count(state.n_early_agents_history[i][2])))
         for i in eachindex(t_hist)]
    state.obs_ts_early_agents_u[] =
        [Point2f(t_hist[i], Float32(log10_count(state.n_early_agents_history[i][3])))
         for i in eachindex(t_hist)]

    # 4. Update Realized Functional Relationships (Ogives, Selectivity, SR)
    update_functional_observables!(state, living)

    # 5. Check and Record Annual Snapshots for Macro-Model Fitting
    check_and_update_macro_models!(state, curr_t, n_living, n_leg, n_fem,
                                   n_mal, b_leg_kg, cum_catch)

    # Status summary. Four distinct quantities that used to be conflated under
    # "Day": the year, the month, the julian day *within* the year, and the total
    # elapsed model time. Year is an integer year count, day is the 1-based day of
    # the year (so it wraps at the year boundary instead of growing without bound),
    # and the elapsed total is reported under its own name rather than as "Day" —
    # "Day: 730" in the second year of a run is not a day of anything.
    year_n = floor(Int, curr_t / dpy)
    day_of_year = round(Int, min(mod(curr_t, dpy) + 1.0, dpy))
    month_n = calendar_month(day_of_year, dpy)
    state.status_text[] = string("Year: ", year_n,
        "   Month: ", month_n,
        "   Day: ", day_of_year,
        "   Duration: ", round(Int, curr_t), " d")

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
    #
    # Both axes are counts spanning several decades (eggs produced vs. recruits that
    # actually settled), so both are plotted as log10 and floored: a year with no
    # recruitment is a real point at the origin of the count scale, not a gap, and
    # `log10(0)` is `-Inf`.
    sr_pts = Point2f[]
    sr_raw = Tuple{Float64, Float64}[]
    clutch_df = abmproperties(state.model)[:clutch_recruitment_records]
    if !isempty(clutch_df)
        for row in eachrow(clutch_df)
            s_raw = Float64(row.egg_count)
            r_raw = Float64(row.recruits_settled)
            push!(sr_raw, (s_raw, r_raw))
            push!(sr_pts, Point2f(Float32(log10_count(s_raw)),
                Float32(log10_count(r_raw))))
        end
    end

    ann_df = state.annual_data
    if nrow(ann_df) >= 2 && (:recruits in propertynames(ann_df))
        for yr in 1:(nrow(ann_df) - 1)
            s_raw = Float64(ann_df.total_egg_production[yr])
            r_raw = Float64(ann_df.recruits[yr + 1])
            push!(sr_raw, (s_raw, r_raw))
            push!(sr_pts, Point2f(Float32(log10_count(s_raw)),
                Float32(log10_count(r_raw))))
        end
    end
    state.obs_sr_pts[] = sr_pts
    state.sr_raw = sr_raw

    update_sr_fit!(state)

    return nothing
end

"""
    update_sr_fit!(state::IBMGuiState)

Fit a Ricker stock-recruitment curve to `state.sr_raw` and stream it to the
panel's curve and annotation observables.

`R = a*S*exp(-b*S)`, fitted by `fit_stock_recruitment`, which linearises it as
`ln(R/S) = u - b*S`. That fit is in linear space even though the panel draws
log10 axes, which is why it is done on `state.sr_raw` rather than on the plotted
points.

The curve is evaluated on a spawner grid spanning the observed range and then
logged for display. Parameters are rounded to three significant digits: `a` is a
recruits-per-spawner ratio of order one while `b` is of order `1/S` and often
many decades smaller, so fixed decimals would print one of them as `0.000`.

Nothing is drawn until `fit_stock_recruitment` reports it actually fitted — it
returns a through-origin fallback below three usable points, and plotting that
fallback would put a confident-looking curve through two numbers.
"""
function update_sr_fit!(state::IBMGuiState)
    spawners = [s for (s, _) in state.sr_raw]
    recruits = [r for (_, r) in state.sr_raw]
    fit = fit_stock_recruitment(spawners, recruits)
    rk = fit.ricker

    if !fit.is_fitted
        state.obs_sr_fit_curve[] = Point2f[]
        state.obs_sr_fit_text[] = ""
        return nothing
    end

    s_lo, s_hi = extrema(spawners)
    # Extend a little past the observed range so the curve's turnover is visible
    # rather than clipped at the last point.
    span = max(s_hi - s_lo, s_hi * 0.05, eps(Float64))
    grid_s = range(max(s_lo - 0.1 * span, 0.0), s_hi + 0.5 * span; length = 128)
    curve = [Point2f(Float32(log10_count(s)), Float32(log10_count(rk.predict(s))))
             for s in grid_s if s > 0.0 && rk.predict(s) > 0.0]
    state.obs_sr_fit_curve[] = curve

    a_r = round(rk.alpha; sigdigits = 3)
    b_r = round(rk.beta; sigdigits = 3)
    state.obs_sr_fit_text[] = string(
        "Ricker:  R = ", a_r, " S exp(-", b_r, " S)\n",
        "n = ", fit.n_obs,
        isfinite(rk.S_max) ? "   Sₘₐₓ = " * string(round(rk.S_max; sigdigits = 3)) : "",
        "   R² = ", round(rk.R2; digits = 3)
    )
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
    dpy = state.model.params.simulation.days_per_year
    year_idx = floor(Int, curr_t / dpy) + 1
    target_snapshot_day = (year_idx - 1) * dpy + snapshot_day

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

        # Julian day within the year: 1-based, so the last day of a 365-day year
        # reads as 365 rather than 0. `floor(curr_t) - (year_idx - 1) * dpy` is
        # exact at the year boundary, where `mod` would return 0.
        julian_day = Float64(clamp(
            floor(Int, curr_t) - (year_idx - 1) * round(Int, dpy), 1, round(Int, dpy)
        ))

        new_row = DataFrame(
            sample_index = [year_idx],
            day = [julian_day],
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
function launch_organism_gui(; display_window::Bool = true,
    config_path::AbstractString = "", kwargs...)
    config = default_ibm_configuration()
    # `config_path` selects which file the annual rescan re-reads; it is not a
    # model parameter, so it is taken out here rather than being merged into the
    # configuration where it would become an unrecognised key.
    for (key, value) in kwargs
        config[Symbol(key)] = value
    end
    sync_grid_dimensions!(config; explicit = Set(Symbol.(keys(kwargs))))
    return launch_organism_gui(config; display_window = display_window,
        config_path = String(config_path))
end

"""
    launch_organism_gui(config::Dict{Symbol, Any}; display_window = true,
                         config_path = "")

Build and display the interactive, three-dashboard GUI for `organism_ibm.jl`
using the provided configuration dictionary.

# Arguments
- `config::Dict{Symbol, Any}`: Configuration dictionary containing all simulation parameters.
- `display_window::Bool`: Set to `true` to render native OpenGL window.
- `config_path::String`: Path to the TOML config file. If provided, the GUI will
  rescan this file every simulation year to pick up user edits.

# Returns
`Tuple{Figure, IBMGuiState}`: Makie figure handle and GUI state container.
"""
function launch_organism_gui(config::Dict{Symbol, Any}; display_window::Bool = true,
                              config_path::String = "")
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
    #
    # Keyword form throughout: see the `@kwdef` note on the struct. Every field is
    # named, so the compiler rejects a misspelled or missing one at the call site
    # instead of the failure surfacing as an unreadable signature mismatch.
    state = IBMGuiState(
        model = model,
        is_running = Observable(false),
        sim_speed_days = Observable(5.0),
        current_time = Observable(0.0),
        status_text = Observable("Initialized | Ready to run"),
        obs_env_field = Observable(copy(model.temperature_field)),
        obs_agent_x = Observable(xs_init),
        obs_agent_y = Observable(ys_init),
        obs_agent_sizes = Observable(sz_init),
        obs_agent_colors = Observable(col_init),
        t_history = Float64[0.0],
        n_total_history = Float64[length(living_init)],
        n_females_history = Float64[0.0],
        n_males_history = Float64[0.0],
        n_legal_history = Float64[0.0],
        biomass_legal_history = Float64[0.0],
        catch_history = Float64[0.0],
        mean_weight_legal_history = Float64[0.0],
        mean_weight_male_history = Float64[0.0],
        mean_weight_female_history = Float64[0.0],
        mean_weight_total_history = Float64[0.0],
        catch_at_year_start = 0.0,
        current_year_idx = 1,
        config_path = config_path,
        # Seeded from the file so the first scan reports only what changes after
        # launch, rather than treating every declared parameter as an edit.
        config_snapshot = isempty(config_path) || !isfile(config_path) ?
            Dict{String, Any}() : flatten_toml(TOML.parsefile(config_path)),
        obs_ts_total = Observable(Point2f[(0.0, log10_count(length(living_init)))]),
        obs_ts_females = Observable(Point2f[(0.0, log10_count(0.0))]),
        obs_ts_males = Observable(Point2f[(0.0, log10_count(0.0))]),
        obs_ts_legal = Observable(Point2f[(0.0, log10_count(0.0))]),
        obs_ts_biomass = Observable(Point2f[(0.0, 0.0)]),
        obs_ts_catch = Observable(Point2f[(0.0, 0.0)]),
        n_egg_history = Float64[0.0],
        n_early_stage_history = [Float64[0.0] for _ in GUI_EARLY_STAGES],
        n_early_agents_history = NTuple{3, Float64}[(0.0, 0.0, 0.0)],
        obs_current_egg = Observable(0.0),
        obs_current_early_stages = Observable((0.0, 0.0, 0.0, 0.0)),
        obs_current_agents = Observable((0.0, 0.0, 0.0)),
        obs_ts_egg = Observable(Point2f[(0.0, log10_count(0.0))]),
        obs_ts_early_stage = [Observable(Point2f[(0.0, log10_count(0.0))])
                               for _ in GUI_EARLY_STAGES],
        obs_ts_early_agents = Observable(Point2f[(0.0, log10_count(0.0))]),
        obs_ts_early_agents_f = Observable(Point2f[(0.0, log10_count(0.0))]),
        obs_ts_early_agents_u = Observable(Point2f[(0.0, log10_count(0.0))]),
        obs_cw_male = Observable(Float32[Float32(a.size) for a in living_init if a.sex == male]),
        obs_cw_female = Observable(Float32[Float32(a.size) for a in living_init if a.sex == female]),
        obs_cw_mature_male = Observable(Float32[Float32(a.size)
                                               for a in living_init if a.sex == male && a.is_mature]),
        obs_cw_mature_female = Observable(Float32[Float32(a.size)
                                                 for a in living_init if a.sex == female && a.is_mature]),
        obs_ogive_male_pts = Observable(Point2f[]),
        obs_ogive_female_pts = Observable(Point2f[]),
        obs_selectivity_pts = Observable(Point2f[]),
        obs_sr_pts = Observable(Point2f[]),
        sr_raw = Tuple{Float64, Float64}[],
        obs_sr_fit_curve = Observable(Point2f[]),
        obs_sr_fit_text = Observable(""),
        annual_data = create_empty_annual_data(),
        macro_obs_biomass = Observable(Point2f[]),
        macro_sch_biomass = Observable(Point2f[]),
        macro_dd_biomass = Observable(Point2f[]),
        macro_les_biomass = Observable(Point2f[]),
        macro_metrics_text = Observable("Macro models awaiting annual data (>= 4 years)"),
        macro_label_observed = Observable("Annual Surveyed Biomass"),
        macro_label_sch = Observable("Schaefer"),
        macro_label_dd = Observable("Delay-Difference"),
        macro_label_les = Observable("Leslie Matrix"),
        macro_conversion_text = Observable(""),
        snapshots = Dict{Int, Any}(),
        snapshot_labels = Observable(String["Year 0"]),
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

    # 1. Summary Information (calendar position and elapsed model time)
    Label(left_grid[1, 1], state.status_text, fontsize = 15, halign = :center, font = :bold)

    # 2. Transport Controls
    #
    # Run / Step / Reset / Refit Macro sit together in one block directly under the
    # time readout. They were previously split across two rows and interleaved with
    # the two timeboxes, so "Reset" sat a full row below "Step" and the controls a
    # user reaches for most often were not adjacent.
    transport_grid = left_grid[2, 1] = GridLayout()
    Label(transport_grid[1, 1], "Transport", font = :bold, fontsize = 13)
    btn_run = Button(transport_grid[2, 1:2], label = @lift($(state.is_running) ? "Pause" : "Run"))
    btn_step = Button(transport_grid[2, 3:4], label = "Step")
    btn_reset = Button(transport_grid[3, 1:2], label = "Reset")
    btn_refit = Button(transport_grid[3, 3:4], label = "Refit Macro")

    # "Step by" is how far one press of Step advances the model, in model days.
    # It was labelled "Total Sim Time", which describes the run rather than the
    # control and disagreed with what the button does.
    Label(transport_grid[4, 1:2], "Step by (days):", halign = :right)
    step_tb = Textbox(transport_grid[4, 3:4], stored_string = "10", validator = Float64, width = 60)

    # "Update interval" is how many model days one animation frame advances while
    # running, so it is a per-frame step, not a speed in wall-clock units — the
    # frame also sleeps a fixed 0.02 s regardless of this value.
    Label(transport_grid[5, 1:2], "Update interval (days):", halign = :right)
    speed_tb = Textbox(transport_grid[5, 3:4], stored_string = "5.0", validator = Float64, width = 60)
    on(speed_tb.stored_string) do s; state.sim_speed_days[] = parse(Float64, s); end

    Label(transport_grid[6, 1:2], "Restore Year:", halign = :right)
    snap_menu = Menu(transport_grid[6, 3:4], options = state.snapshot_labels, width = 90)

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

    # One table drives the widgets, the live-edit handler and the Reset handler.
    # Previously each of the three was a separate list indexed positionally, and
    # dropping `Legal Size` from the GUI shifted every later live handler onto the
    # wrong field while the Reset handler was left reading a box that no longer
    # existed — a silent parameter swap rather than an error. Deriving all three
    # from one table means a parameter is added, removed or reordered in one place.
    #
    # `Legal Size` is deliberately absent: it is a management threshold, not a
    # tunable, and the size-frequency panel now draws it as a reference line so
    # the fishery cut is visible against the population it selects. It remains
    # settable in the configuration file.
    #
    # `clamp01` marks the two habitat-quality fields, which are probabilities:
    # clamping there is what stops a typo producing a nonsense distribution
    # instead of an error.
    steering_controls = [
        (label = "Base Temp (°C)", kw = :temp_base, start = params.environment.temp_base,
            parser = Float64, validator = Float64, clamp01 = false),
        (label = "Temp Amp (°C)", kw = :temp_amplitude,
            start = params.environment.temp_amplitude,
            parser = Float64, validator = Float64, clamp01 = false),
        (label = "Food Max", kw = :food_max, start = params.environment.food_max,
            parser = Float64, validator = Float64, clamp01 = false),
        (label = "Food Inputs", kw = :food_replenishment_rate,
            start = params.environment.food_replenishment_rate,
            parser = Float64, validator = Float64, clamp01 = false),
        (label = "Carrying Cap", kw = :benthic_carrying_capacity,
            start = params.environment.benthic_carrying_capacity,
            parser = Float64, validator = Float64, clamp01 = false),
        (label = "Habitat Qual", kw = :habitat_quality_mean,
            start = params.environment.habitat_quality_mean,
            parser = Float64, validator = Float64, clamp01 = true),
        (label = "Habitat Qual SD", kw = :habitat_quality_sd,
            start = params.environment.habitat_quality_sd,
            parser = Float64, validator = Float64, clamp01 = true),
        (label = "HSI Peaks", kw = :habitat_peaks, start = params.environment.habitat_peaks,
            parser = Int, validator = Int, clamp01 = false),
        (label = "Harvest Prob", kw = :fishing_probability_daily,
            start = params.fishery.fishing_probability_daily,
            parser = Float64, validator = Float64, clamp01 = false),
        (label = "Harvest Start (d)", kw = :fishing_season_start,
            start = params.fishery.fishing_season_start,
            parser = Float64, validator = Float64, clamp01 = false),
        (label = "Harvest End (d)", kw = :fishing_season_end,
            start = params.fishery.fishing_season_end,
            parser = Float64, validator = Float64, clamp01 = false)
    ]

    tboxes = Textbox[]
    for (i, c) in enumerate(steering_controls)
        Label(param_grid[i, 1], c.label, halign = :left)
        push!(tboxes,
            Textbox(param_grid[i, 2], stored_string = string(c.start),
                validator = c.validator))
    end

    # Read every control into the kwargs `reconstruct_params` expects, so the live
    # handler and the Reset handler cannot disagree about which box is which field.
    function steering_overrides()
        return Pair{Symbol, Any}[
            c.kw => (c.clamp01 ? clamp(c.parser(tb.stored_string[]), 0.0, 1.0) :
                c.parser(tb.stored_string[]))
            for (c, tb) in zip(steering_controls, tboxes)
        ]
    end

    for tb in tboxes
        on(tb.stored_string) do _
            abmproperties(state.model)[:params] = reconstruct_params(
                abmproperties(state.model)[:params]; steering_overrides()...
            )
        end
    end

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

    # Right Content Area: Single grid with 12 rows × 2 cols (implicit).
    # LCM(3,4)=12 gives equal height within each column:
    # Left col (3 plots): each gets 4 rows. Right col (4 plots): each gets 3 rows.
    # Default equal row heights give exactly the desired proportions:
    # Left: 3 plots × 4 rows = 12 rows → each plot 4/12 = 1/3 height
    # Right: 4 plots × 3 rows = 12 rows → each plot 3/12 = 1/4 height
    content_grid = fig[1, 2] = GridLayout()

    # --------------------------------------------------------------------------
    # Left Column: Map (rows 1–4), Macro (rows 5–8), SR (rows 9–12)
    # --------------------------------------------------------------------------
    ax_map = Axis(
        content_grid[1:4, 1],
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
    # applied to put them in kg. That conversion line, and the goodness-of-fit
    # summary computed on every re-fit, were rendered into the title as a subtitle.
    # They were removed: the title then wrapped to two or three lines, the numbers
    # competed with the panel they annotated, and the conversion is a property of
    # the fit rather than of the figure. Both are still computed and still carried
    # in `state.macro_conversion_text` / `state.macro_metrics_text`, so they are
    # available without being drawn.
    ax_macro = Axis(
        content_grid[5:8, 1],
        title = "Biomass, Catch & Macro-Models",
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

    # The stock-recruit panel is wrapped in its own one-cell GridLayout so the fit
    # annotation can be laid over the axis in the same cell. It is a `Label`, not a
    # `text!` plot: this Makie version rejects `relative` on `Makie.Text`
    # outright, and `text!` has no in-place text update, so a reactive `text!`
    # would have to be re-created on every observation and leak a plot each time.
    sr_cell = content_grid[9:12, 1] = GridLayout()
    ax_sr = Axis(
        sr_cell[1, 1],
        title = "Stock-Recruitment Relationship",
        xlabel = "log10(Eggs Produced)",
        ylabel = "log10(Recruits Settled)"
    )
    GLMakie.scatter!(ax_sr, state.obs_sr_pts, color = :seagreen, markersize = 7, label = "Realized Benthic Batches")
    GLMakie.lines!(ax_sr, state.obs_sr_fit_curve, color = :crimson, linewidth = 2.2,
        label = "Ricker fit")
    # Parameters go in the top-left corner as figure text rather than only in the
    # legend: they are the point of the panel, and a legend entry cannot carry
    # `S_max` and `R²` legibly beside the observed points. Empty until the fit
    # has three usable points.
    Label(sr_cell[1, 1], state.obs_sr_fit_text;
        halign = :left, valign = :top, fontsize = 11, color = :black)
    GLMakie.axislegend(ax_sr, position = :lb, labelsize = 10, framevisible = false, backgroundcolor = :transparent, patchsize = (15, 10))

    # Right Column: Abundance (rows 1–3), Stage (rows 4–6), Size (rows 7–9), Ogive (rows 10–12)
    # --------------------------------------------------------------------------
    ax_ts_n = Axis(
        content_grid[1:3, 2],
        title = "Abundance Trajectories",
        xlabel = "Time (days)",
        ylabel = "log10(count)"
    )
    GLMakie.lines!(ax_ts_n, state.obs_ts_total, label = "Total Organisms", color = :royalblue, linewidth = 2.0)
    GLMakie.lines!(ax_ts_n, state.obs_ts_females, label = "Mature Females", color = :crimson, linewidth = 1.8)
    GLMakie.lines!(ax_ts_n, state.obs_ts_males, label = "Mature Males", color = :seagreen, linewidth = 1.8)
    GLMakie.lines!(ax_ts_n, state.obs_ts_legal, label = "Legal Fishable Males", color = :purple, linewidth = 2.0)
    GLMakie.axislegend(ax_ts_n, position = :lt, labelsize = 10, framevisible = false, backgroundcolor = :transparent, patchsize = (15, 10))

    # Stage abundance by stage. Current abundances of eggs, stages 0-3, and
    # individuated agents. `AggregateGroup` has no `sex`, so the early stages are
    # single grey lines. Individuated agents use the male/female pair.
    ax_stage = Axis(
        content_grid[4:6, 2],
        title = "Stage Abundance",
        xlabel = "Stage",
        ylabel = "log10(number)",
        xticks = ([0, 1, 2, 3, 4, 5], ["Eggs", "0", "1", "2", "3", "Agents"])
    )
    # Current early-stage abundances: eggs (idx 0), stages 0-3 (idx 1-4)
    obs_stage_early = lift(state.obs_current_egg, state.obs_current_early_stages) do n_egg, stages
        vals = Float32[log10_count(n_egg)]
        for s in stages
            push!(vals, log10_count(s))
        end
        x_vals = Float32[0.0, 1.0, 2.0, 3.0, 4.0]
        Point2f[Point2f(x_vals[i], vals[i]) for i in eachindex(vals)]
    end
    # Current agent abundances by sex (stage 5 = Agents)
    obs_stage_agents = lift(state.obs_current_agents) do mfu
        m, f, u = mfu
        x_val = 5.0f0
        [Point2f(x_val, Float32(log10_count(m))),
         Point2f(x_val, Float32(log10_count(f))),
         Point2f(x_val, Float32(log10_count(u)))]
    end

    GLMakie.stairs!(ax_stage, obs_stage_early,
        label = "Early Stages", color = :grey30, linewidth = 2.2, linestyle = :solid)
    GLMakie.scatter!(ax_stage, obs_stage_early,
        color = :grey30, markersize = 10, label = "")
    GLMakie.scatter!(ax_stage, lift(p -> [p[1]], obs_stage_agents),
        color = :royalblue, markersize = 10, label = "Agents: Males")
    GLMakie.scatter!(ax_stage, lift(p -> [p[2]], obs_stage_agents),
        color = :crimson, markersize = 10, label = "Agents: Females")
    GLMakie.scatter!(ax_stage, lift(p -> [p[3]], obs_stage_agents),
        color = :grey50, markersize = 10, label = "Agents: Undiff.")
    GLMakie.axislegend(ax_stage, position = :lt, labelsize = 9, framevisible = false, backgroundcolor = :transparent, patchsize = (15, 10))

    # Four density curves: each sex, and the mature subset of each sex.
    # The y axis is linear density so the total area under each curve integrates
    # to one and the four curves are directly comparable. Colours follow the
    # dashboard's existing convention (male blue, female crimson) with maturity
    # as a heavier line over the corresponding sex, so the two are read as a pair.
    ax_size = Axis(
        content_grid[7:9, 2],
        title = "Size Frequency by Sex and Maturity",
        xlabel = "Size (mm)",
        ylabel = "density"
    )
    xlims!(ax_size, 0, 165)
    obs_size_freq_m = lift(state.obs_cw_male) do v
        c, d = size_frequency(v)
        [Point2f(Float32(c[i]), Float32(d[i])) for i in eachindex(c)]
    end
    obs_size_freq_f = lift(state.obs_cw_female) do v
        c, d = size_frequency(v)
        [Point2f(Float32(c[i]), Float32(d[i])) for i in eachindex(c)]
    end
    obs_size_freq_mm = lift(state.obs_cw_mature_male) do v
        c, d = size_frequency(v)
        [Point2f(Float32(c[i]), Float32(d[i])) for i in eachindex(c)]
    end
    obs_size_freq_mf = lift(state.obs_cw_mature_female) do v
        c, d = size_frequency(v)
        [Point2f(Float32(c[i]), Float32(d[i])) for i in eachindex(c)]
    end
    GLMakie.lines!(ax_size, obs_size_freq_m, color = (:royalblue, 0.45), linewidth = 1.5, linestyle = :dash, label = "Males")
    GLMakie.lines!(ax_size, obs_size_freq_f, color = (:crimson, 0.45), linewidth = 1.5, linestyle = :dash, label = "Females")
    GLMakie.stairs!(ax_size, obs_size_freq_mm, color = :royalblue, linewidth = 2.2, label = "Mature Males")
    GLMakie.stairs!(ax_size, obs_size_freq_mf, color = :crimson, linewidth = 2.2, label = "Mature Females")
    GLMakie.vlines!(ax_size, params.fishery.legal_size_male, 0.0f0, 1.0f0,
        color = :purple, linewidth = 1.5, linestyle = :dot, label = "Legal size (male)")
    GLMakie.axislegend(ax_size, position = :lt, labelsize = 9, framevisible = false, backgroundcolor = :transparent, patchsize = (15, 10))

    ax_ogive = Axis(
        content_grid[10:12, 2],
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
            abmproperties(state.model)[:params]; steering_overrides()...
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
        empty!(state.n_egg_history)
        for series in state.n_early_stage_history
            empty!(series)
        end
        empty!(state.n_early_agents_history)
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
