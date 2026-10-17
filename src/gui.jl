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
import GLMakie: Point2f, Observable, on, lift, @lift
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
    df.recruits_male = Float64[]
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

The aggregate cohort stages tracked as a time series on the stage figure.

These are `AggregateGroup` cohorts, not `Organism` agents, and `AggregateGroup`
carries no `sex` field, so they are drawn as a single undifferentiated series.

Starts at `FIRST_COHORT_STAGE`, not 0: stage 0 is the brooded egg, which is
carried by a female rather than being an `AggregateGroup`, so including it left a
series that was permanently zero and made the panel look as though the pelagic
stages were never populated.
"""
const GUI_EARLY_STAGES = FIRST_COHORT_STAGE:3

# Early stage labels, line palettes, and dash patterns are retrieved dynamically
# from `OrganismParams.stage_labels` via `cohort_stage_label` to ensure exact
# correspondence with active TOML parameter files. Undifferentiated aggregate cohorts
# are visualized with balanced sex allocation.

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
const RESCAN_IMMUTABLE = (:grid_nx, :grid_ny, :grid_size, :dt, :seed)

"""
    CONTROL_PANEL_WIDTH

Width of the control column, in pixels. Named because the figure's three columns
are sized from it — see the `fig.layout.colsizes` assignment in
`launch_organism_gui`.

Sized to the panel's own content. The widest row is a steering label beside its
text box ("Restore Year:" against a 90 px menu), which needs a little over 340 px;
at 340 the column overflowed and clipped its left-hand labels, since a grid wider
than its slot spills symmetrically about its centre.
"""
const CONTROL_PANEL_WIDTH = 400

"""
    DASHBOARD_SIZE

Window size in pixels.

920 tall, which is 80% of the 1150 the balanced layout needed. The width is not a
free choice: every plotting panel is the same width, so the window is exactly
`CONTROL_PANEL_WIDTH + 2 x (height / 2) + 2 x COLUMN_GAP` =
400 + 2 x 460 + 40 = 1360. Leaving the previous 1680 would have put 320 px of empty
margin down the right-hand side.

The height is what sets the panel width: the plot columns are half the height wide
so the square map fills its column, and the four panels stacked down the right take
three rows each.
"""
const DASHBOARD_SIZE = (1360, 920)

"""
    SEX_DENSITY_ALPHA

Fill opacity of every kernel density curve in the size-frequency panel.

One value for all four series. The mature subsets are the same kind of thing as the
whole-sex curves and overlay them, so they share their transparency; when the mature
curves were opaque they read as a different kind of quantity rather than as a subset.
They stay distinguishable by hue — slate grey and dark orange against the whole-sex
blue and red — and by the heavier stroke.
"""
const SEX_DENSITY_ALPHA = 0.45

"""
    SIZE_DENSITY_BANDWIDTH_MM

Kernel bandwidth for the size-frequency panel, in millimetres.

One value for all four series. A per-series bandwidth is chosen from each sample's
own spread and size, so the mature series — a few tens of individuals in a narrow
size range — would be smoothed far more finely than the whole-sex series and drawn
as a spike beside it. A common bandwidth puts all four on the same footing, which is
what makes them comparable.
"""
const SIZE_DENSITY_BANDWIDTH_MM = 3.0

"""
    kde_density(sizes, grid; bandwidth = SIZE_DENSITY_BANDWIDTH_MM, scale = :probability)

Gaussian kernel density of `sizes` over `grid`.

`scale = :probability` returns a density integrating to one, so curves from groups
of very different sizes are directly comparable on a linear axis — each is read as
a shape. `scale = :count` returns individuals per millimetre, whose area is the
group size, so a subset reads as lying inside its parent.

Non-finite sizes are dropped rather than contributing a spike, and an empty sample
returns all zeros — no placeholder value is substituted, since a single fake
observation at 0 mm would draw a phantom cohort at the origin.
"""
function kde_density(sizes, grid;
    bandwidth = SIZE_DENSITY_BANDWIDTH_MM, scale::Symbol = :probability)
    scale in (:probability, :count) ||
        error("scale must be :probability or :count, got $scale")
    vals = [Float64(s) for s in sizes if isfinite(s)]
    n = length(vals)
    out = zeros(Float64, length(grid))
    n == 0 && return out
    h = Float64(bandwidth)
    h > 0 || error("bandwidth must be positive, got $h")
    k = 1 / (h * sqrt(2 * pi))
    for s in vals
        @inbounds for (i, x) in enumerate(grid)
            z = (x - s) / h
            out[i] += k * exp(-0.5 * z * z)
        end
    end
    # No extra scaling by `n` here: the loop already accumulates one kernel per
    # observation, so the integral is already the group size and each peak is `n`
    # times a single observation's peak. Multiplying again made it `n²`.
return scale === :probability ? out ./ n : out
end

"""
    PELAGIC_STAGE_LABELS

Names of the pelagic cohort stages, in order: zoea1, zoea2 and the semibenthic
megalopa. Supplied by `OrganismParams.stage_labels` in the configuration file for
the figures; kept here for the tests and for anything that needs the names without
a parameter object.
"""
const PELAGIC_STAGE_LABELS = ("z1", "z2", "m")

# `STAGE_UNSEXED_COLOUR` used to sit here, a grey for the stages that carry no
# sex. Every stage from the egg to the first individuated one is now drawn as a
# 50/50 male/female pair, so nothing on the stage figure is unsexed and the
# constant has no remaining use.

"""
    cohort_stage_label(stage, params)

Plot name for a cohort stage, from `OrganismParams.stage_labels` in the
configuration file.

`stage_labels` is in stage order — element `i` is model stage `i - 1` — so the
names live in the TOML rather than being derived here, which is what keeps the
figure from disagreeing with the file. A stage past the end of the declared list,
or a species that declares none, falls back to the derived name: a benthic instar
number counting from one at `larval_stage_count`, which is the convention the
default list encodes.
"""
function cohort_stage_label(stage::Integer, params, larval_count::Integer)
    labels = params.organism.stage_labels
    idx = stage + 1
    if 1 <= idx <= length(labels)
        return String(labels[idx])
    end
    return stage < larval_count ? string(stage) : string(stage - larval_count + 1)
end

"""
    stem!(ax, xs, ys; kwargs...)

A vertical stem from the axis up to a marker at each point — a lollipop glyph.

Used for the stage figure, where the quantity is a sum of discrete individuals
rather than an estimate. An earlier version drew a Poisson interval around each
point; that implied a precision a head count does not have, so the interval is gone
and the stem now carries the meaning: height, not precision.

The segments are drawn with `linesegments!` from explicit endpoints rather than
with `vlines!`. `vlines!` given an observable height vector drew every stem at full
axis height regardless of the value, which put a line through all eighteen stages
whatever their counts were. Explicit from/to points cannot drift that way: each
segment ends at its own point.

`linesegments!` takes one point vector and reads consecutive pairs as segments, so
the from/to points are interleaved here rather than passed as two vectors — that
two-vector form is not a conversion it accepts.
"""
function stem!(ax, xs::AbstractVector{<:Real}, ys; color = :black, markersize = 7,
    linewidth = 2.0, label = nothing)
    xs32 = Float32.(xs)
    if ys isa Observable
        segs = lift(ys) do v
            pts = Point2f[]
            for i in eachindex(xs32)
                push!(pts, Point2f(xs32[i], 0.0f0))
                push!(pts, Point2f(xs32[i], Float32(v[i])))
            end
            pts
        end
        tops = lift(ys) do v
            [Point2f(xs32[i], Float32(v[i])) for i in eachindex(xs32)]
        end
        linesegments!(ax, segs; color = color, linewidth = linewidth, label = label)
        scatter!(ax, tops; color = color, markersize = markersize)
    else
        pts = Point2f[]
        tops = Point2f[]
        for i in eachindex(xs32)
            push!(pts, Point2f(xs32[i], 0.0f0))
            push!(pts, Point2f(xs32[i], Float32(ys[i])))
            push!(tops, Point2f(xs32[i], Float32(ys[i])))
        end
        linesegments!(ax, pts; color = color, linewidth = linewidth, label = label)
        scatter!(ax, tops; color = color, markersize = markersize)
    end
end

# ------------------------------------------------------------------------------
# Shared Dashboard Panel Setup Kernels (Common across GUI and Notebooks)
# ------------------------------------------------------------------------------

"""
    setup_spatial_panel!(ax, env_field, xs, ys, sizes, colors)

Populate spatial thermal environment heatmap with `:thermal` colormap and
superimposed demographic agent particles. Accepts either Observables or Arrays.
"""
function setup_spatial_panel!(
    ax,
    env_field,
    xs,
    ys,
    sizes,
    colors;
    colormap = :thermal,
    colorrange = nothing
)
    cr = if colorrange !== nothing
        colorrange
    elseif env_field isa AbstractArray
        valid_vals = filter(!isnan, env_field)
        if isempty(valid_vals)
            (0.0, 1.0)
        else
            ext = extrema(valid_vals)
            ext[1] == ext[2] ? (ext[1] - 1.0, ext[2] + 1.0) : ext
        end
    else
        nothing
    end

    hm = if cr !== nothing
        heatmap!(ax, env_field; colormap = colormap, colorrange = cr)
    else
        heatmap!(ax, env_field; colormap = colormap)
    end

    scatter!(
        ax, xs, ys;
        markersize = sizes,
        color = colors,
        strokewidth = 0.5,
        strokecolor = :black
    )
    return hm
end


"""
    setup_abundance_panel!(ax, ts_total, ts_females, ts_males, ts_legal)

Populate log10 abundance trajectories for Total, Mature Females, Mature Males,
and Legal Fishable Males. Accepts either Observables or Arrays.
"""
function setup_abundance_panel!(ax, ts_total, ts_females, ts_males, ts_legal)
    lines!(ax, ts_total, label = "Total Organisms", color = :royalblue, linewidth = 2.0)
    lines!(ax, ts_females, label = "Mature Females", color = :crimson, linewidth = 1.8)
    lines!(ax, ts_males, label = "Mature Males", color = :seagreen, linewidth = 1.8)
    lines!(ax, ts_legal, label = "Legal Fishable Males", color = :purple, linewidth = 2.0)
    axislegend(ax, position = :lt, labelsize = 10, framevisible = false,
               backgroundcolor = :transparent, patchsize = (15, 10))
end

"""
    setup_stages_panel!(ax, half_w, base_agg, base_agent, n_male_st, n_fem_st,
                        egg_y, agg_y, male_y, female_y)

Populate grouped stage abundance forest marks (`stem!`), splitting brooded egg
and early aggregate cohort stages 50/50 and individuated stages by sex.
"""
function setup_stages_panel!(ax, half_w, base_agg, base_agent, n_male_st, n_fem_st,
                             egg_y, agg_y, male_y, female_y)
    stem!(ax, [1.0 - half_w], egg_y; color = :royalblue, label = "Males")
    stem!(ax, [1.0 + half_w], egg_y; color = :crimson, label = "Females")
    stem!(ax, base_agg .- half_w, agg_y; color = :royalblue, label = nothing)
    stem!(ax, base_agg .+ half_w, agg_y; color = :crimson, label = nothing)
    stem!(ax, base_agent[1:n_male_st] .- half_w, male_y; color = :royalblue, label = nothing)
    stem!(ax, base_agent[1:n_fem_st] .+ half_w, female_y; color = :crimson, label = nothing)
    axislegend(ax, position = :rt, labelsize = 8, framevisible = false,
               backgroundcolor = :transparent, patchsize = (15, 10))
end

"""
    setup_size_density_panel!(ax, imm_m, mat_m, imm_f, mat_f, legal_size; peak = nothing)

Populate normalized size probability densities for immature/mature males and females
alongside the legal size reference line.
"""
function setup_size_density_panel!(ax, imm_m, mat_m, imm_f, mat_f, legal_size; peak = nothing)
    lines!(ax, imm_m; color = :royalblue, linewidth = 1.5, linestyle = :dash, label = "Immature Males")
    lines!(ax, mat_m; color = :slategrey, linewidth = 2.2, label = "Mature Males")
    lines!(ax, imm_f; color = :crimson, linewidth = 1.5, linestyle = :dash, label = "Immature Females")
    lines!(ax, mat_f; color = :darkorange, linewidth = 2.2, label = "Mature Females")
    
    if peak !== nothing
        vlines!(ax, legal_size, 0.0f0, peak;
                color = :purple, linewidth = 1.5, linestyle = :dot, label = "Legal size (male)")
    else
        vlines!(ax, [legal_size];
                color = :purple, linewidth = 1.5, linestyle = :dot, label = "Legal size (male)")
    end
    axislegend(ax, position = :lt, labelsize = 9, framevisible = false,
               backgroundcolor = :transparent, patchsize = (15, 10))
end

"""
    setup_ogive_panel!(ax, legal_size, realized_males, realized_females)

Populate theoretical maturity curves, selectivity retention, and realized binned observations.
"""
function setup_ogive_panel!(ax, legal_size, realized_males, realized_females)
    cw_ref = 20.0:1.0:140.0
    lines!(ax, cw_ref, [maturity(0, log(c)) for c in cw_ref],
           color = :black, linewidth = 2.0, label = "Theoretical Male")
    lines!(ax, cw_ref, [maturity(1, log(c)) for c in cw_ref],
           color = :grey40, linewidth = 2.0, linestyle = :dash, label = "Theoretical Female")
    lines!(ax, cw_ref, [selectivity(c; s50 = legal_size) for c in cw_ref],
           color = :purple, linewidth = 2.0, label = "Theoretical S(L)")
    
    scatter!(ax, realized_males, color = :royalblue, markersize = 8, label = "Realized Males")
    scatter!(ax, realized_females, color = :crimson, markersize = 8, label = "Realized Females")
    axislegend(ax, position = :lt, labelsize = 10, framevisible = false,
               backgroundcolor = :transparent, patchsize = (15, 10))
end

"""
    setup_macro_panel!(ax, ts_biomass, ts_catch, obs_pts, sch_pts, dd_pts, les_pts;
                       label_obs = "Annual Surveyed Biomass", label_sch = "Schaefer Fit",
                       label_dd = "Delay-Diff Fit", label_les = "Leslie Fit")

Populate biomass and catch trajectories alongside macro-ecological surplus production,
delay-difference, and Leslie model fits.
"""
function setup_macro_panel!(ax, ts_biomass, ts_catch, obs_pts, sch_pts, dd_pts, les_pts;
                            label_obs = "Annual Surveyed Biomass",
                            label_sch = "Schaefer Fit",
                            label_dd = "Delay-Diff Fit",
                            label_les = "Leslie Fit")
    lines!(ax, ts_biomass, label = "Legal Fishable Biomass", color = :purple, linewidth = 2.2)
    lines!(ax, ts_catch, label = "Annual Catch", color = :darkorange, linewidth = 1.8, linestyle = :dash)
    scatter!(ax, obs_pts, color = :black, markersize = 9, label = label_obs)
    lines!(ax, sch_pts, color = :royalblue, linewidth = 2.0, linestyle = :dash, label = label_sch)
    lines!(ax, dd_pts, color = :seagreen, linewidth = 2.0, linestyle = :dashdot, label = label_dd)
    lines!(ax, les_pts, color = :darkorange, linewidth = 2.2, label = label_les)
    axislegend(ax, position = :lt, labelsize = 10, framevisible = false,
               backgroundcolor = :transparent, patchsize = (15, 10))
end

"""
    setup_stock_recruitment_panel!(ax, sr_pts, ricker_pts, bh_pts)

Populate realized log10 spawner-recruit points and fitted Ricker and Beverton-Holt curves.
"""
function setup_stock_recruitment_panel!(ax, sr_pts, ricker_pts, bh_pts)
    scatter!(ax, sr_pts, color = :seagreen, markersize = 7,
             label = "Realized (males at recruitment stage)")
    lines!(ax, ricker_pts, color = :crimson, linewidth = 2.2, label = "Ricker fit")
    lines!(ax, bh_pts, color = :royalblue, linewidth = 2.2, linestyle = :dash,
           label = "Beverton–Holt fit")
    axislegend(ax, position = :lb, labelsize = 9, framevisible = false,
               backgroundcolor = :transparent, patchsize = (15, 10))
end

"""
    COLUMN_GAP

Space reserved between the figure's columns, in pixels. Makie inserts its own gap
as well; this is subtracted so the two plotting columns still fit inside the window
once the figure's default gap is accounted for.
"""
const COLUMN_GAP = 20

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
    obs_bg_layer::Observable{Symbol}
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
        obs_current_agents::Observable{NTuple{3, Float64}}

    # Per-stage abundance for the stage figure, as three parallel series rather than
    # one series per stage. Aggregate cohorts are `AggregateGroup`, which carries no
    # `sex`, so stages below the individuation boundary are a single undifferentiated
    # series; agent stages are split by sex.
    #
    # Indexed by stage through the offsets below: `obs_stage_agg[k]` is stage `k - 1`;
    # `obs_stage_male[k]` is stage `juvenile_max_stage - 1 + k`.
    obs_stage_agg::Observable{Vector{Float64}}
    obs_stage_male::Observable{Vector{Float64}}
    obs_stage_female::Observable{Vector{Float64}}

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
    # The complement of each mature series, split by the agent's own `is_mature`
    # flag. The size panel plots immature and mature side by side rather than the
    # whole sex, because a whole-sex curve mixes the two populations and its shape
    # then depends on how many of each are alive rather than on size alone.
    obs_cw_immature_male::Observable{Vector{Float32}}
    obs_cw_immature_female::Observable{Vector{Float32}}

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
    obs_sr_fit_curve_bh::Observable{Vector{Point2f}}
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
    current_layer = state.obs_bg_layer[]
    mprops = abmproperties(model)
    env_field = if current_layer === :habitat_quality && haskey(mprops, :habitat_quality_field)
        copy(model.habitat_quality_field)
    else
        copy(model.temperature_field)
    end
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
    # An empty series stays empty. It used to be replaced by the single value 0.0 so
    # that the plot had something to draw, but that 0.0 is a size like any other and
    # put a phantom cohort of one animal at the origin of the size axis.
    state.obs_cw_male[] = cw_m
    state.obs_cw_female[] = cw_f
    # Maturity uses the agent's own flag rather than re-deriving a size threshold,
    # so the mature curve tracks exactly the individuals the rest of the model
    # treats as mature (and which the ogive and macro panels also use).
    cw_mm = Float32[Float32(a.size) for a in living if a.sex == male && a.is_mature]
    cw_mf = Float32[Float32(a.size) for a in living if a.sex == female && a.is_mature]
    state.obs_cw_mature_male[] = cw_mm
    state.obs_cw_mature_female[] = cw_mf
    cw_im = Float32[Float32(a.size) for a in living if a.sex == male && !a.is_mature]
    cw_if = Float32[Float32(a.size) for a in living if a.sex == female && !a.is_mature]
    state.obs_cw_immature_male[] = cw_im
    state.obs_cw_immature_female[] = cw_if

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
    # Per-stage tallies for the stage figure. The agent stages run from
    # `juvenile_max_stage` (the individuation boundary) up to each sex's own
    # terminal stage — 6..16 for males and 6..14 for females under the snow crab
    # configuration. The species stage tables stop at 12 and 11; those bound the
    # Leslie projection, not the agents, which molt on past them, so the figure is
    # sized from `max_stage_*` instead or the largest cohorts would fall off the end.
    stage_base = params.organism.juvenile_max_stage
    stage_agg = zeros(Float64, length(state.obs_stage_agg[]))
    stage_male = zeros(Float64, length(state.obs_stage_male[]))
    stage_female = zeros(Float64, length(state.obs_stage_female[]))
    for o in living
        k = o.stage - stage_base + 1
        if o.sex === male
            n_agent_male += 1
            if 1 <= k <= length(stage_male)
                stage_male[k] += 1.0
            end
        elseif o.sex === female
            n_agent_female += 1
            if 1 <= k <= length(stage_female)
                stage_female[k] += 1.0
            end
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
            # Full per-stage tally for the stage figure. Indexed by stage, not
            # stage + 1: stage 0 is the brooded egg, which is carried by the female
            # rather than being a cohort, so the cohorts start at stage 1.
            k = g.stage
            if 1 <= k <= length(stage_agg)
                stage_agg[k] += g.abundance
            end
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
    state.obs_current_agents[] =
        (Float64(n_agent_male), Float64(n_agent_female), Float64(n_agent_undiff))
    state.obs_stage_agg[] = stage_agg
    state.obs_stage_male[] = stage_male
    state.obs_stage_female[] = stage_female

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

    # Realized Stock-Recruitment Points, for the **male** component.
    #
    # Spawners are annual egg production *halved*. The panel follows one sex —
    # recruitment at the stage just before maturity onset, for males — and only half
    # of a clutch becomes male, so the denominator has to be the eggs that could
    # produce those males. Using the whole clutch would halve every ratio and bias
    # both the Ricker and the Beverton-Holt parameters by the same factor.
    #
    # `annual_egg_production`, not `total_egg_production`. The latter is named
    # "total" and described as cumulative, and a stock-recruit fit interprets the
    # spawner value as the production *in that year* — pairing a running total
    # against an annual recruit count. The GUI writes an annual value into both
    # columns, so the two only agree by accident; the correctly named one is used.
    #
    # Both axes are counts spanning several decades, so both are plotted as log10
    # and floored: a year with no recruitment is a real point at the origin of the
    # count scale, not a gap, and `log10(0)` is `-Inf`.
    sr_pts = Point2f[]
    sr_raw = Tuple{Float64, Float64}[]
    ann_df = state.annual_data
    eggs_col = (:annual_egg_production in propertynames(ann_df)) ?
               :annual_egg_production : :total_egg_production
    recruits_col = (:recruits_male in propertynames(ann_df)) ?
                   :recruits_male : :recruits
    if nrow(ann_df) >= 2 && (recruits_col in propertynames(ann_df)) &&
       (eggs_col in propertynames(ann_df))
        for yr in 1:(nrow(ann_df) - 1)
            s_raw = Float64(ann_df[!, eggs_col][yr]) / 2
            r_raw = Float64(ann_df[!, recruits_col][yr + 1])
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
    bh = fit.beverton_holt

    if !fit.is_fitted
        state.obs_sr_fit_curve[] = Point2f[]
        state.obs_sr_fit_curve_bh[] = Point2f[]
        state.obs_sr_fit_text[] = ""
        return nothing
    end

    s_lo, s_hi = extrema(spawners)
    # Extend a little past the observed range so each curve's turnover is visible
    # rather than clipped at the last point.
    span = max(s_hi - s_lo, s_hi * 0.05, eps(Float64))
    grid_s = range(max(s_lo - 0.1 * span, 0.0), s_hi + 0.5 * span; length = 128)
    ricker = [Point2f(Float32(log10_count(s)), Float32(log10_count(rk.predict(s))))
              for s in grid_s if s > 0.0 && rk.predict(s) > 0.0]
    beverton = [Point2f(Float32(log10_count(s)), Float32(log10_count(bh.predict(s))))
                for s in grid_s if s > 0.0 && bh.predict(s) > 0.0]
    state.obs_sr_fit_curve[] = ricker
    state.obs_sr_fit_curve_bh[] = beverton

    # Both forms, with the better-supported one named. They answer the same
    # question — recruits per spawner — and routinely disagree about it: Ricker
    # has a stock-dependent, overcompensating form and Beverton-Holt saturates
    # smoothly, so on a short unexploited series the ranking is a genuine model
    # choice rather than a formality, and AIC is what settles it.
    a_r = round(rk.alpha; sigdigits = 3)
    b_r = round(rk.beta; sigdigits = 3)
    a_b = round(bh.alpha; sigdigits = 3)
    b_b = round(bh.beta; sigdigits = 8)
    lines = String[
        string("Ricker       R = ", a_r, "·S·exp(-", b_r, "·S)"),
        string("             R² = ", round(rk.R2; digits = 3),
            "   AIC = ", round(rk.aic; digits = 1)),
        string("Beverton–Holt R = ", a_b, "·S / (1 + ", b_b, "·S)"),
        string("             R² = ", round(bh.R2; digits = 3),
            "   AIC = ", round(bh.aic; digits = 1)),
    ]
    push!(lines, string("n = ", fit.n_obs,
        "   better: ", fit.best_model === :ricker ? "Ricker" : "Beverton–Holt"))
    state.obs_sr_fit_text[] = join(lines, '\n')
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

        # Recruitment: the cohort immediately before maturity onset, counted by sex.
    #
    # Sex-specific, so it is counted among individuated `Organism` agents — an
    # `AggregateGroup` carries no `sex` field, and the old counter ran off the
    # aggregate transition into `benthic_anchor_stage`, a *size-model* threshold
    # ("first stage with sex-specific bounds"). That made "recruits settled" mean
    # "cohort members that advanced into the stage where the size formula changes",
    # which is neither settlement nor anything a reader would call recruitment.
    #
    # `recruits` is the female count, which is the classical stock-recruit pairing:
    # recruits per spawner. `recruits_male` is the same stage for males, used by the
    # fishable-component delay-difference fit, since the fishable stock is male.
    rec_stage = abmproperties(state.model)[:params].species.recruitment_stage
    rec_female = 0
    rec_male = 0
    for a in allagents(state.model)
        a.stage == rec_stage || continue
        a.sex === female && (rec_female += 1)
        a.sex === male && (rec_male += 1)
    end

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
            recruits = [Float64(rec_female)],
            recruits_male = [Float64(rec_male)],
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
`grid`, `dt`, and `seed`.
"""
function launch_organism_gui(; display_window::Bool = true,
    config_path::AbstractString = "", kwargs...)
    # A `config_path` that names a readable file is the base configuration, not
    # merely the file the annual rescan re-reads. It used to start from
    # `default_ibm_configuration()` unconditionally, so pointing the dashboard at a
    # TOML loaded nothing from it — the initial abundance distribution, the stage
    # schedules and every other declared value were silently ignored, and the panel
    # ran on struct defaults instead.
    config = if isempty(config_path) || !isfile(config_path)
        default_ibm_configuration()
    else
        load_ibm_configuration(String(config_path))
    end
    # `n_agents` is not a configuration key — it is the total the seeding vectors
    # sum to — so it is kept out of `config` and passed straight through to
    # `initialize_model`. It cannot be `pop!`ed: `kwargs` is an immutable
    # `Base.Pairs`, so it is skipped in the merge instead.
    n_agents = get(kwargs, :n_agents, nothing)
    for (key, value) in kwargs
        key === :n_agents && continue
        config[Symbol(key)] = value
    end
    sync_grid_dimensions!(config; explicit = Set(Symbol.(keys(kwargs))))
    return launch_organism_gui(config; display_window = display_window,
        config_path = String(config_path), n_agents = n_agents)
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
    config_path::String = "", n_agents::Union{Int, Nothing} = nothing)
    # Activate GLMakie for high-performance interactive OpenGL rendering
    GLMakie.activate!()

    # 1. Initialize Simulation Model
    params = IBMParameters(config)

    model = initialize_model(
        grid = config[:grid],
        dt = config[:dt],
        params = params,
        seed = config[:seed],
        # The initial abundance is three vectors in the configuration. `n_agents` is
        # the sum of the two sex vectors when it is not given, and an override when it
        # is; it is taken here rather than merged into `config`, where it would be an
        # unrecognised key now that it is not a parameter of its own.
        n_agents = n_agents
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
        is_running = Observable(true),
        sim_speed_days = Observable(5.0),
        current_time = Observable(0.0),
        status_text = Observable("Simulation Active | Running"),
        obs_env_field = Observable(copy(model.temperature_field)),
        obs_bg_layer = Observable(:temperature),
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
                obs_current_agents = Observable((0.0, 0.0, 0.0)),
        # Stage 0 is the brooded egg, which is not a cohort, so the aggregate series
        # covers stages 1 to `juvenile_max_stage - 1` and is indexed by stage.
        obs_stage_agg = Observable(zeros(Float64, params.organism.juvenile_max_stage - 1)),
        obs_stage_male = Observable(zeros(Float64,
            params.organism.max_stage_male - params.organism.juvenile_max_stage + 1)),
        obs_stage_female = Observable(zeros(Float64,
            params.organism.max_stage_female - params.organism.juvenile_max_stage + 1)),
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
        obs_cw_immature_male = Observable(Float32[Float32(a.size)
                                          for a in living_init if a.sex == male && !a.is_mature]),
        obs_cw_immature_female = Observable(Float32[Float32(a.size)
                                            for a in living_init if a.sex == female && !a.is_mature]),
        obs_ogive_male_pts = Observable(Point2f[]),
        obs_ogive_female_pts = Observable(Point2f[]),
        obs_selectivity_pts = Observable(Point2f[]),
        obs_sr_pts = Observable(Point2f[]),
        sr_raw = Tuple{Float64, Float64}[],
        obs_sr_fit_curve = Observable(Point2f[]),
        obs_sr_fit_curve_bh = Observable(Point2f[]),
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
    fig = Figure(size = DASHBOARD_SIZE)

    # No figure-level title. The status line is the first thing in the left grid and
    # sits flush against the top of the window; a scene-drawn window title occupied
    # the same strip and clipped the top of the date/time readout. Every panel
    # already carries its own title, so the window title identified nothing that the
    # panels do not.
    #
    # The window title is 1680 x 920 — 80% of the 2100 x 1150 the balanced
    # three-column layout needed. The height is the binding dimension: the left
    # column is half the height wide so its square map fills the column, and the
    # four panels stacked down the right each take a quarter of the height.

    # Left Column: Unified Controls & Information Panel
    #
    # The control panel is wrapped in a one-column outer grid whose first row is a
    # spacer. Without it the status readout is the first thing in the figure and is
    # drawn flush against y = 0, where the top of the glyphs is clipped off. The
    # wrapper keeps every row index below unchanged.
    left_panel = fig[1, 1] = GridLayout(width = CONTROL_PANEL_WIDTH, tellheight = false)
    Label(left_panel[1, 1], ""; valign = :top)
    # No `width` here. The inner grid inherits the outer one's width; asserting it
    # a second time made the nested pair offset each other and pushed the panel's
    # left-hand labels off the left edge of the window.
    left_grid = left_panel[2, 1] = GridLayout(tellheight = false)

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

    Label(transport_grid[7, 1:2], "Map Layer:", halign = :right)
    layer_menu = Menu(
        transport_grid[7, 3:4],
        options = ["Temperature", "Habitat Quality (HSI)"],
        default = "Temperature",
        width = 110
    )
    on(layer_menu.selection) do sel
        if sel !== nothing
            state.obs_bg_layer[] = sel == "Habitat Quality (HSI)" ?
                                   :habitat_quality : :temperature
            step_gui_simulation!(state, 0.0)
        end
    end

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
        (label = "Temp Offset (°C)", kw = :temp_offset, start = params.environment.temp_offset,
            parser = Float64, validator = Float64, clamp01 = false),
        (label = "Temp Multiplier", kw = :temp_multiplier,
            start = params.environment.temp_multiplier,
            parser = Float64, validator = Float64, clamp01 = false),
        (label = "Food Max", kw = :food_max, start = params.environment.food_max,
            parser = Float64, validator = Float64, clamp01 = false),
        (label = "Food Inputs", kw = :food_replenishment_rate,
            start = params.environment.food_replenishment_rate,
            parser = Float64, validator = Float64, clamp01 = false),
        (label = "Carrying Cap", kw = :benthic_carrying_capacity,
            start = params.environment.benthic_carrying_capacity,
            parser = Float64, validator = Float64, clamp01 = false),
        (label = "Habitat Offset", kw = :habitat_offset,
            start = params.environment.habitat_offset,
            parser = Float64, validator = Float64, clamp01 = false),
        (label = "Habitat Multiplier", kw = :habitat_multiplier,
            start = params.environment.habitat_multiplier,
            parser = Float64, validator = Float64, clamp01 = false),
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

    # --------------------------------------------------------------------------
    # Plotting area: two sibling column grids, plus explicit figure-level sizing.
    # --------------------------------------------------------------------------
    # The layout is three sibling columns in the figure — controls, left plots,
    # right plots — rather than one content grid holding two columns of panels.
    #
    # That distinction is the whole fix. With a single content grid, its two columns
    # were sized by `Auto()` from their contents, and the contents are lopsided: the
    # four time-series and distribution panels on the right each advertise a wide
    # preferred size, while the three panels on the left are mostly blank. The left
    # column ended up about 190 px against roughly 1200 px on the right, and its
    # tick and axis labels spilled across the parameter-steering panel.
    #
    # Sizing *that* grid's columns could not fix it, for a reason worth recording:
    # the grid is itself placed inside the figure by content-driven sizing, so any
    # width imposed on its columns was re-derived from the surrounding content and
    # the imbalance reappeared outside the grid. Setting column sizes before the axes
    # exist raises a BoundsError because the grid is 1x1 at that point; adding
    # `rowsizes` as well blows the layout up past the figure bounds; and wrapping a
    # column in a nested grid with a fixed width pins the column but collapses its
    # children's plot rectangles to zero width.
    #
    # Sizing the *figure* does stick, because the figure is the outermost layout and
    # nothing sits above it to re-derive its width. `fig.layout.colsizes` is assigned
    # at the end of this function, once all three children exist.

    # Left plots: Map, Macro, SR. 12 rows, sized so the map is square and fills the
    # column and the two panels below it match a right-column panel's height.
    left_content = fig[1, 2] = GridLayout()
    # Right plots: Abundance, Stage, Size, Ogive. 12 rows, 3 per panel -> 1/4 height
    # each. LCM(3, 4) = 12 so both columns divide the same height evenly.
    right_content = fig[1, 3] = GridLayout()

    # --------------------------------------------------------------------------
    # Left Column: Map (rows 1–6), Macro (rows 7–9), SR (rows 10–12)
    #
    # The map takes half the height (6 rows) and the two panels below it take three
    # rows each, matching the three rows a right-column panel gets. So Macro and SR
    # are exactly as tall as the panels to their right, and — since the left column
    # is sized to the map — exactly as wide as it too.
    # --------------------------------------------------------------------------
    ax_map = Axis(
        left_content[1:6, 1],
        title = "Spatial Habitat & Organism Distribution",
        xlabel = "Grid X",
        ylabel = "Grid Y"
    )
    setup_spatial_panel!(
        ax_map, state.obs_env_field, state.obs_agent_x, state.obs_agent_y,
        state.obs_agent_sizes, state.obs_agent_colors
    )

    ax_macro = Axis(
        left_content[7:9, 1],
        title = "Biomass, Catch & Macro-Models",
        xlabel = "Simulation Time (days)",
        ylabel = "Biomass (kg)"
    )
    setup_macro_panel!(
        ax_macro, state.obs_ts_biomass, state.obs_ts_catch, state.macro_obs_biomass,
        state.macro_sch_biomass, state.macro_dd_biomass, state.macro_les_biomass;
        label_obs = state.macro_label_observed[],
        label_sch = state.macro_label_sch[],
        label_dd = state.macro_label_dd[],
        label_les = state.macro_label_les[]
    )

    sr_cell = left_content[10:12, 1] = GridLayout()
    sr_cell.tellheight = true
    sr_cell.tellwidth = true
    ax_sr = Axis(
        sr_cell[1, 1],
        title = "Stock-Recruitment (Male Component)",
        xlabel = "log10(Annual Eggs / 2)",
        ylabel = "log10(Males at Recruitment Stage)"
    )
    setup_stock_recruitment_panel!(
        ax_sr, state.obs_sr_pts, state.obs_sr_fit_curve, state.obs_sr_fit_curve_bh
    )
    GLMakie.Label(sr_cell[1, 1], state.obs_sr_fit_text;
        halign = :left, valign = :top, fontsize = 8, color = :black,
        tellwidth = false, tellheight = false)

    # Right Column: Abundance (rows 1–3), Stage (rows 4–6), Size (rows 7–9), Ogive (rows 10–12)
    # --------------------------------------------------------------------------
    ax_ts_n = Axis(
        right_content[1:3, 1],
        title = "Abundance Trajectories",
        xlabel = "Time (days)",
        ylabel = "log10(count)"
    )
    setup_abundance_panel!(
        ax_ts_n, state.obs_ts_total, state.obs_ts_females, state.obs_ts_males, state.obs_ts_legal
    )

    n_agg = params.organism.juvenile_max_stage - 1
    n_male_st = params.organism.max_stage_male - params.organism.juvenile_max_stage + 1
    n_fem_st = params.organism.max_stage_female - params.organism.juvenile_max_stage + 1
    max_stage = params.organism.max_stage_male

    larval_count = larval_stage_count(SpeciesBehaviorImpl())
    stage_labels = [cohort_stage_label(s, params, larval_count) for s in 0:max_stage]
    first_agent_cat = 2 + n_agg
    n_cats = length(stage_labels)

    ax_stage = Axis(
        right_content[4:6, 1],
        title = "Stage Abundance",
        xlabel = "Stage",
        ylabel = "log10(number)",
        xticks = (collect(1:n_cats), stage_labels),
        xticklabelrotation = pi / 2
    )

    half_w = 0.21
    obs_male_y = lift(state.obs_stage_male) do v
        Float64[log10_count(x) for x in v]
    end
    obs_female_y = lift(state.obs_stage_female) do v
        Float64[log10_count(x) for x in v]
    end
    obs_agg_y = lift(state.obs_stage_agg) do v
        Float64[log10_count(x / 2) for x in v]
    end
    obs_egg_y = lift(state.obs_current_egg) do n
        Float64[log10_count(n / 2)]
    end

    base_agg = Float64.(2:(1 + n_agg))
    base_agent = Float64.(first_agent_cat:(first_agent_cat + max(n_male_st, n_fem_st) - 1))

    setup_stages_panel!(
        ax_stage, half_w, base_agg, base_agent, n_male_st, n_fem_st,
        obs_egg_y, obs_agg_y, obs_male_y, obs_female_y
    )

    ax_size = Axis(
        right_content[7:9, 1],
        title = "Size Frequency by Sex and Maturity",
        xlabel = "Size (mm)",
        ylabel = "Density"
    )
    xlims!(ax_size, 0, 165)
    size_grid = collect(0.0:0.5:165.0)
    kde(obs) = lift(v -> [Point2f(Float32(size_grid[i]),
            Float32(kde_density(v, size_grid; scale = :probability)[i]))
            for i in eachindex(size_grid)], obs)

    obs_immature_m = kde(state.obs_cw_immature_male)
    obs_mature_m = kde(state.obs_cw_mature_male)
    obs_immature_f = kde(state.obs_cw_immature_female)
    obs_mature_f = kde(state.obs_cw_mature_female)

    peak = lift((a, b, c, d) -> maximum(p[2] for p in vcat(a, b, c, d)),
        obs_immature_m, obs_mature_m, obs_immature_f, obs_mature_f)

    setup_size_density_panel!(
        ax_size, obs_immature_m, obs_mature_m, obs_immature_f, obs_mature_f,
        params.fishery.legal_size_male; peak = peak
    )

    ax_ogive = Axis(
        right_content[10:12, 1],
        title = "Maturity Ogive & Fishery Selectivity",
        xlabel = "Size (mm)",
        ylabel = "P(Mature) / Selectivity"
    )
    xlims!(ax_ogive, 0, 165)
    setup_ogive_panel!(
        ax_ogive, params.fishery.legal_size_male,
        state.obs_ogive_male_pts, state.obs_ogive_female_pts
    )

    # The 50/50 split, applied at the one level that sticks.
    #
    # The control panel is fixed because its contents are fixed-width widgets that
    # must not be squeezed; the two plotting columns are given an explicit pixel
    # width so they come out equal.
    #
    # Sizing the figure works where sizing the inner content grid did not, because
    # the figure is the outermost layout and there is nothing above it to re-derive
    # the width from the contents.
    #
    # The widths are fixed rather than relative because `Relative` does not mean
    # "fraction of what is left" in this layout engine. With
    # `[Fixed(340), Relative(1), Relative(1)]` each relative column was measured at
    # about 830 px instead of the 705 px that would fill the window exactly, so the
    # total came to roughly 2000 px in a 1750 px figure: the control panel was pushed
    # off the left edge entirely and the right-hand panels were cut off at the
    # right. Pixel widths are exact and are honest for a window this size.
    #
    # Assigned after all three children exist: assigning more column sizes than the
    # grid currently has raises a BoundsError.
    fig_w, fig_h = GLMakie.size(fig.scene)
    # Both plotting columns are half the figure height wide. That makes the square
    # map fill its cell rather than letterbox, and it makes the right-hand panels
    # the same width as the biomass/macro panel beside them. The two lower left
    # panels each take three of the twelve rows, so they come out the height of a
    # right-hand panel too.
    left_w = round(Int, fig_h / 2)
    right_w = left_w
    # The gap is set explicitly because the width arithmetic above assumes it. Left
    # to its default the three columns plus their gaps overflowed the window, and the
    # layout spilled symmetrically — clipping the control panel's left-hand labels
    # and cutting off the right-hand panels.
    fig.layout.default_colgap = Fixed(COLUMN_GAP)
    fig.layout.colsizes = [Fixed(CONTROL_PANEL_WIDTH), Fixed(left_w), Fixed(right_w)]
    # Each plot grid also needs its own single column sized. Sizing only the figure
    # fixes where a grid sits and how wide its slot is; the grid's own column is
    # still `Auto()`, sized from the axes inside, and for the map that meant a square
    # preferred width — its plot rectangle came out zero pixels wide. Each grid has
    # exactly one column here, so this assigns one size to one column and cannot hit
    # the arity mismatch seen when assigning two sizes to a one-column grid.
    left_content.colsizes = [Fixed(left_w)]
    right_content.colsizes = [Fixed(right_w)]
    # Row heights are left at Makie's default. Stating them made things worse: the
    # cells are sized from the space an axis needs for its title, tick labels and
    # legend, so `Fixed` acts as a floor and each row came out around 92 px instead
    # of the 77 px requested. With twelve rows that pushed the right column to
    # 4 x 267 = 1068 px in a 920 px figure and pushed the bottom panels off the top.
    # The default already divides 6/3/3 as intended, giving the map six rows and the
    # two panels below it three each — the same height as a panel to the right.
    # `sr_cell` is gone: the stock-recruit fit parameters moved to the axis
    # subtitle, so the panel no longer needs a nested overlay grid and now sizes
    # itself like every other panel.

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
