"""
    src/visualization.jl

Visualization and diagnostic plotting tools for the organism
(*Chionoecetes opilio*) individual-based model (IBM).

Provides specialized diagnostic figures for population dynamics, demographic
stage structures, size distributions by sex, allometric growth trajectories,
bioenergetics, spatial distributions over environmental fields, and a unified
executive simulation dashboard, implemented natively with Makie.jl (CairoMakie).
"""

using Printf

using CairoMakie
using DataFrames
using Agents
using Statistics

"""
    plot_population_timeseries(agent_df; model_df=nothing)

Generate a multi-panel timeseries plot showing temporal dynamics of the
simulated population.

# Visualized Metrics
- **Panel 1 (Abundance)**: Mature adults and ovigerous females.
- **Panel 2 (Body Size)**: Mean carapace width (mm) across time.
- **Panel 3 (Recruitment, optional)**: Cumulative benthic recruitment of Stage 4 settlers
  if `model_df` contains `:total_recruits_settled`.

# Arguments
- `agent_df::DataFrame`: Time-stepped agent data collected during simulation.
- `model_df::Union{DataFrame, Nothing}`: Optional time-stepped model-level data.

# Returns
`CairoMakie.Figure`: Combined multi-panel timeseries figure.
"""
function plot_population_timeseries(
    agent_df::DataFrame;
    model_df::Union{DataFrame, Nothing} = nothing
)
    has_recruits = (model_df !== nothing &&
                    :total_recruits_settled in propertynames(model_df))
    h = has_recruits ? 850 : 600
    fig = Figure(size = (850, h))

    ax1 = Axis(
        fig[1, 1],
        title = "Organism Abundance Dynamics",
        xlabel = "Time",
        ylabel = "Count"
    )
    if :count_is_adult_male in propertynames(agent_df)
        lines!(
            ax1, agent_df.time, (agent_df.count_is_adult_male),
            label = "Mature Adults", linewidth = 2.0, color = :seagreen
        )
    end
    if :count_is_ovigerous in propertynames(agent_df)
        lines!(
            ax1, agent_df.time, agent_df.count_is_ovigerous,
            label = "Ovigerous Females", linewidth = 2.0, color = :crimson
        )
    end
    axislegend(ax1, position = :rt)

    mean_size_series = if :safe_mean_size in propertynames(agent_df)
        agent_df.safe_mean_size
    elseif :mean_size in propertynames(agent_df)
        agent_df.mean_size
    else
        zeros(nrow(agent_df))
    end

    ax2 = Axis(
        fig[2, 1],
        title = "Mean Size Progression",
        xlabel = "Time",
        ylabel = "Size (mm)"
    )
    lines!(
        ax2, agent_df.time, mean_size_series,
        label = "Mean Population Size", linewidth = 2.0, color = :darkorange
    )
    axislegend(ax2, position = :rt)

    if has_recruits
        ax3 = Axis(
            fig[3, 1],
            title = "Benthic Recruitment (Stage 4)",
            xlabel = "Time",
            ylabel = "Cumulative Recruits"
        )
        lines!(
            ax3, model_df.time, model_df.total_recruits_settled,
            label = "Settled Recruits", linewidth = 2.0, color = :teal
        )
        axislegend(ax3, position = :rt)
    end

    return fig
end

"""
    plot_stage_distribution(model)

Generate a grouped bar chart displaying the frequency of living organisms across
developmental stages, categorized by sex.

# Biological Significance
Organism developmental progress is marked by discrete developmental stages.
Pre-benthic stages (egg, zoea, megalopa) develop at stage 0, while stages 1 through 14+
represent benthic juveniles and mature adults. This plot visualizes cohort abundance
and recruitment progression across the demographic age-structure.

# Arguments
- `model::ABM`: Agents.jl simulation model containing `Organism` agents.

# Returns
`CairoMakie.Figure`: Grouped bar chart of organism counts per stage by sex.
"""
function plot_stage_distribution(model)
    fig = Figure(size = (750, 450))
    organisms = collect(allagents(model))
    if isempty(organisms)
        Label(fig[1, 1], "No living organisms to display", fontsize = 16, color = :gray)
        return fig
    end

    stage_range = 4:14
    males_by_stage = zeros(Int, length(stage_range))
    females_by_stage = zeros(Int, length(stage_range))

    for c in organisms
        idx = c.stage - first(stage_range) + 1
        if 1 <= idx <= length(stage_range)
            if c.sex === male
                males_by_stage[idx] += 1
            elseif c.sex === female
                females_by_stage[idx] += 1
            end
        end
    end

    stage_coords = Float64.(collect(stage_range))
    ax = Axis(
        fig[1, 1],
        title = "Stage Demographic Structure by Sex",
        xlabel = "Developmental Stage",
        ylabel = "Abundance (Organisms)",
        xticks = collect(stage_range)
    )

    barplot!(
        ax, stage_coords .- 0.18, males_by_stage,
        width = 0.35, color = (:dodgerblue, 0.85), label = "Males"
    )
    barplot!(
        ax, stage_coords .+ 0.18, females_by_stage,
        width = 0.35, color = (:hotpink, 0.85), label = "Females"
    )
    axislegend(ax, position = :rt)

    return fig
end

"""
    plot_size_by_sex(model)

Generate overlaid size-frequency distributions (carapace width in mm)
separated by sex for all living individuals.

# Biological Significance
Illustrates sexual dimorphism in *Chionoecetes opilio*. Females undergo terminal
molt at smaller sizes (typically 50-80 mm size), whereas males can continue to
grow to stages 12-14+ (exceeding 140 mm size).

# Arguments
- `model::ABM`: Simulation model containing `Organism` agents.

# Returns
`CairoMakie.Figure`: Overlaid histogram of carapace widths for males and females.
"""
function plot_size_by_sex(model)
    fig = Figure(size = (750, 450))
    organisms = collect(allagents(model))
    if isempty(organisms)
        Label(fig[1, 1], "No living organisms to display", fontsize = 16, color = :gray)
        return fig
    end

    male_sizes = [Float64(c.size) for c in organisms if c.sex === male]
    female_sizes = [Float64(c.size) for c in organisms if c.sex === female]

    ax = Axis(
        fig[1, 1],
        title = "Carapace Width Distribution by Sex",
        xlabel = "Size (mm)",
        ylabel = "Density"
    )

    if !isempty(male_sizes)
        hist!(
            ax, male_sizes, bins = 25, normalization = :pdf,
            color = (:dodgerblue, 0.55), strokewidth = 1, strokecolor = :white,
            label = "Males (n = $(length(male_sizes)))"
        )
    end

    if !isempty(female_sizes)
        hist!(
            ax, female_sizes, bins = 25, normalization = :pdf,
            color = (:hotpink, 0.55), strokewidth = 1, strokecolor = :white,
            label = "Females (n = $(length(female_sizes)))"
        )
    end

    vlines!(
        ax, [model.params.legal_size_male],
        linestyle = :dash, linewidth = 2.0, color = :darkred,
        label = "Legal Male Size ($(model.params.legal_size_male) mm)"
    )

    axislegend(ax, position = :rt)
    return fig
end

"""
    plot_stage_allometry(model)

Scatter plot of individual organism carapace widths across stages contrasted
with the empirical theoretical allometric mean curves (`llbm` and `llbf`).

# Allometric Reference
- Males:   ``Size = \\exp(1.917564 + 0.298914 \\cdot (stage - 3))``
- Females: ``Size = \\exp(2.198848 + 0.315026 \\cdot (stage - 4))``

# Arguments
- `model::ABM`: Simulation model containing `Organism` agents.

# Returns
`CairoMakie.Figure`: Carapace width vs stage scatter with empirical bounds.
"""
function plot_stage_allometry(model)
    fig = Figure(size = (750, 480))
    organisms = collect(allagents(model))
    if isempty(organisms)
        Label(fig[1, 1], "No living organisms to display", fontsize = 16, color = :gray)
        return fig
    end

    stage_grid = 4:14
    cw_males_theory = [llbm(i) for i in stage_grid]
    cw_females_theory = [llbf(i) for i in stage_grid]

    ax = Axis(
        fig[1, 1],
        title = "Stage Allometric Scaling: Simulation vs Theory",
        xlabel = "Developmental Stage",
        ylabel = "Carapace Width (mm)",
        xticks = collect(stage_grid)
    )

    lines!(
        ax, collect(stage_grid), cw_males_theory,
        label = "Empirical Male Curve (llbm)", linewidth = 2.5, color = :blue
    )
    lines!(
        ax, collect(stage_grid), cw_females_theory,
        label = "Empirical Female Curve (llbf)", linewidth = 2.5,
        color = :magenta, linestyle = :dash
    )

    male_stages = [Float64(c.stage) + randn(Random.Xoshiro(0)) * 0.08 for c in organisms if c.sex === male]
    male_sizes = [Float64(c.size) for c in organisms if c.sex === male]
    if !isempty(male_sizes)
        scatter!(
            ax, male_stages, male_sizes,
            label = "Simulated Males", color = (:dodgerblue, 0.65), markersize = 8
        )
    end

    female_stages = [Float64(c.stage) + randn(Random.Xoshiro(0)) * 0.08 for c in organisms if c.sex === female]
    female_sizes = [Float64(c.size) for c in organisms if c.sex === female]
    if !isempty(female_sizes)
        scatter!(
            ax, female_stages, female_sizes,
            label = "Simulated Females", color = (:hotpink, 0.65), markersize = 8
        )
    end

    axislegend(ax, position = :lt)
    return fig
end

"""
    plot_reproduction_dynamics(agent_df, model_df)

Plot reproductive activity, egg production, and recruitment pulses over time.

# Panels
- **Panel 1**: Temporal trajectory of brooding ovigerous females.
- **Panel 2**: Cumulative eggs spawned into the system.
- **Panel 3**: Cumulative benthic recruits (Stage 4) settled.

# Arguments
- `agent_df::DataFrame`: Agent metrics tracked over simulation.
- `model_df::DataFrame`: Model properties tracked over simulation.

# Returns
`CairoMakie.Figure`: Multi-panel reproduction and recruitment figure.
"""
function plot_reproduction_dynamics(
    agent_df::DataFrame,
    model_df::DataFrame
)
    fig = Figure(size = (850, 850))

    ax1 = Axis(
        fig[1, 1],
        title = "Brooding Ovigerous Females",
        xlabel = "Time",
        ylabel = "Ovigerous Count"
    )
    if :count_is_ovigerous in propertynames(agent_df)
        lines!(
            ax1, agent_df.time, agent_df.count_is_ovigerous,
            label = "Active Ovigerous Females", linewidth = 2.0, color = :crimson
        )
        axislegend(ax1, position = :rt)
    else
        Label(fig[1, 1], "Ovigerous metrics not tracked", color = :gray)
    end

    ax2 = Axis(
        fig[2, 1],
        title = "Cumulative Egg Production",
        xlabel = "Time",
        ylabel = "Eggs (Millions)"
    )
    if :total_eggs_spawned in propertynames(model_df)
        lines!(
            ax2, model_df.time, model_df.total_eggs_spawned ./ 1.0e6,
            label = "Total Spawned Eggs", linewidth = 2.0, color = :purple
        )
        axislegend(ax2, position = :rt)
    else
        Label(fig[2, 1], "Egg metrics not tracked", color = :gray)
    end

    ax3 = Axis(
        fig[3, 1],
        title = "Cumulative Benthic Recruitment (Stage 4)",
        xlabel = "Time",
        ylabel = "Recruits Settled"
    )
    if :total_recruits_settled in propertynames(model_df)
        lines!(
            ax3, model_df.time, model_df.total_recruits_settled,
            label = "Settled Recruits", linewidth = 2.0, color = :teal
        )
        axislegend(ax3, position = :rt)
    else
        Label(fig[3, 1], "Recruitment metrics not tracked", color = :gray)
    end

    return fig
end

"""
    plot_spatial_habitat(model)

Generate a 2D spatial habitat map displaying benthic water temperature with
living organisms superimposed by demographic stage and sex.

# Demarcated Groups
- Juveniles (Stage 4+): Yellow circles
- Adult Males: Dodgerblue squares
- Adult Non-Ovigerous Females: Pink triangles
- Adult Ovigerous Females: Crimson stars

# Arguments
- `model::ABM`: Simulation model containing space and agents.

# Returns
`CairoMakie.Figure`: 2D spatial habitat and distribution figure.
"""
function plot_spatial_habitat(model)
    fig = Figure(size = (800, 650))
    nx, ny = size(model.temperature_field)

    ax = Axis(
        fig[1, 1],
        title = "Spatial Organism Habitat & Distribution",
        xlabel = "X Coordinate (Grid Cells)",
        ylabel = "Y Coordinate (Grid Cells)",
        aspect = AxisAspect(1)
    )

    hm = heatmap!(
        ax, 1:nx, 1:ny, model.temperature_field,
        colormap = :coolwarm
    )
    Colorbar(fig[1, 2], hm, label = "Temp (°C)")

    organisms = collect(allagents(model))
    if !isempty(organisms)
        juv_x = [Float64(c.pos[1]) for c in organisms if !is_adult(c)]
        juv_y = [Float64(c.pos[2]) for c in organisms if !is_adult(c)]

        adm_x = [Float64(c.pos[1]) for c in organisms if is_mature_male(c)]
        adm_y = [Float64(c.pos[2]) for c in organisms if is_mature_male(c)]

        adf_x = [Float64(c.pos[1]) for c in organisms if is_mature_female(c) && !is_ovigerous(c)]
        adf_y = [Float64(c.pos[2]) for c in organisms if is_mature_female(c) && !is_ovigerous(c)]

        ovig_x = [Float64(c.pos[1]) for c in organisms if is_ovigerous(c)]
        ovig_y = [Float64(c.pos[2]) for c in organisms if is_ovigerous(c)]

        if !isempty(juv_x)
            scatter!(
                ax, juv_x, juv_y, label = "Juveniles ($(length(juv_x)))",
                color = :gold, marker = :circle, markersize = 9,
                strokecolor = :black, strokewidth = 0.5
            )
        end
        if !isempty(adm_x)
            scatter!(
                ax, adm_x, adm_y, label = "Adult Males ($(length(adm_x)))",
                color = :dodgerblue, marker = :rect, markersize = 10,
                strokecolor = :black, strokewidth = 0.5
            )
        end
        if !isempty(adf_x)
            scatter!(
                ax, adf_x, adf_y, label = "Adult Females ($(length(adf_x)))",
                color = :hotpink, marker = :utriangle, markersize = 10,
                strokecolor = :black, strokewidth = 0.5
            )
        end
        if !isempty(ovig_x)
            scatter!(
                ax, ovig_x, ovig_y, label = "Ovigerous Females ($(length(ovig_x)))",
                color = :crimson, marker = :star5, markersize = 14,
                strokecolor = :white, strokewidth = 0.8
            )
        end
        axislegend(ax, position = :rt)
    end

    return fig
end

"""
    fmt_param(x) → String

Format an estimated parameter for a legend entry: three significant figures.

Fixed precision keeps a legend from churning between re-fits when the estimates
barely move, and three significant figures keeps entries readable when a parameter
spans orders of magnitude.
"""
fmt_param(x::Real) = @sprintf("%.3g", Float64(x))
fmt_param(x) = string(x)

"""
    plot_maturity_ogives(model; size_bins=20.0:10.0:140.0, ecdyses=1, min_count=1)

Contrast the simulated empirical proportion mature against the theoretical
logistic maturity schedules for both male and female organisms.

# Theoretical Model
```math
P_{\\text{mature}}(Size) = \\frac{1}{1 + \\exp(-k (\\ln Size - \\ln L_{50}))}
```
where `L50` is the median size at maturity and `k` the log-logistic scale.

The two curves measure different quantities and are labelled accordingly. The
per-ecdysis curve is the probability of maturing at a *single ecdysis*. The
simulated points are a *cross-sectional* proportion: every individual of a given
size, pooled regardless of age, so they estimate the probability of having ever
matured. The cumulative curve `1 - (1 - p) ^ k` is the like-for-like
comparison. With `ecdyses = nothing` (the default) `k` is derived per size from
the stage size ranges, which is the same quantity `evaluate_population_criteria`
scores against, so the plot and the criterion cannot disagree.

# Arguments
- `model::ABM`: Simulation model containing `Organism` agents.
- `size_bins::AbstractRange`: Carapace size bin boundaries.
- `ecdyses::Union{Nothing, Int}`: Ecdyses an individual is assumed to have had at
  a given size. `nothing` derives them from the stage model.
- `min_count::Int`: Bins holding fewer individuals are omitted.

# Returns
`CairoMakie.Figure`: Empirical vs theoretical maturity ogive comparison.
"""
function plot_maturity_ogives(
    model;
    size_bins = 20.0:10.0:140.0,
    ecdyses::Union{Nothing, Int} = nothing,
    min_count::Int = 1
)
    fig = Figure(size = (820, 520))
    size_eval = collect(20.0:1.0:140.0)

    # `deterministic = true` returns the expected curve. The default draw re-samples
    # L50 and the slope at every size, which renders the reference as a Monte Carlo
    # wiggle rather than the theoretical ogive.
    male_ecdysis = [maturity(0, log(sz); deterministic = true) for sz in size_eval]
    female_ecdysis = [maturity(1, log(sz); deterministic = true) for sz in size_eval]
    if ecdyses !== nothing
        male_cumulative = [1.0 - (1.0 - p)^ecdyses for p in male_ecdysis]
        female_cumulative = [1.0 - (1.0 - p)^ecdyses for p in female_ecdysis]
    else
        male_cumulative, female_cumulative = cumulative_maturity_curves(size_eval)
    end

    om = compute_empirical_ogive(model, male, size_bins; min_count = min_count)
    of = compute_empirical_ogive(model, female, size_bins; min_count = min_count)

    ax = Axis(
        fig[1, 1],
        title = "Organism Maturity Ogives: Simulation vs Theory",
        subtitle = "cross-sectional bins with n >= $(min_count); marker area ∝ sample size " *
                   "(males n=$(sum(om.counts)), females n=$(sum(of.counts)))",
        xlabel = "Carapace Size (mm)",
        ylabel = "Proportion Mature"
    )

    lines!(
        ax, size_eval, male_ecdysis,
        label = "Male, per ecdysis (theory)", linewidth = 2.5, color = :blue
    )
    lines!(
        ax, size_eval, female_ecdysis,
        label = "Female, per ecdysis (theory)", linewidth = 2.5,
        color = :red, linestyle = :dash
    )
    if ecdyses !== 1
        lines!(
            ax, size_eval, male_cumulative,
            label = "Male, cumulative over stages passed",
            linewidth = 1.8, color = :blue, linestyle = :dot
        )
        lines!(
            ax, size_eval, female_cumulative,
            label = "Female, cumulative over stages passed",
            linewidth = 1.8, color = :red, linestyle = :dot
        )
    end

    scale_markers(c) = 6.0 .+ 14.0 .* sqrt.(c ./ max(maximum(c; init = 1), 1))

    if !isempty(om.bin_centers)
        scatter!(
            ax, om.bin_centers, om.prop_mature,
            label = "Simulated males (n shown by size)",
            color = :dodgerblue, marker = :rect, markersize = scale_markers(om.counts)
        )
    end
    if !isempty(of.bin_centers)
        scatter!(
            ax, of.bin_centers, of.prop_mature,
            label = "Simulated females (n shown by size)",
            color = :crimson, marker = :circle, markersize = scale_markers(of.counts)
        )
    end


    # Top-left: the maturity fraction rises from lower-left to upper-right, so the
    # lower-left is the empty corner and the legend belongs there rather than over
    # the mature end of the curve.
    axislegend(ax, position = :lt)
    return fig
end

"""
    plot_simulation_dashboard(model, agent_df; model_df=nothing)

Generate a unified 6-panel executive summary dashboard combining the core
biological and ecological indicators of the organism simulation.

# Panels Included
1. **Panel A**: Abundance & demographic stage timeseries.
2. **Panel B**: Stage demographic distribution by sex.
3. **Panel C**: Carapace width distribution by sex (dimorphism).
4. **Panel D**: Spatial habitat distribution over bottom temperature.
5. **Panel E**: Stage allometry compared with empirical curves.
6. **Panel F**: Cumulative recruitment and egg production.

# Arguments
- `model::ABM`: Simulation model instance.
- `agent_df::DataFrame`: Time-stepped agent data.
- `model_df::Union{DataFrame, Nothing}`: Optional time-stepped model data.

# Returns
`CairoMakie.Figure`: High-impact 6-panel dashboard figure.
"""
function plot_simulation_dashboard(
    model,
    agent_df::DataFrame;
    model_df::Union{DataFrame, Nothing} = nothing
)
    fig = Figure(size = (1400, 1500))

    # Panel 1: Population abundance timeseries
    ax1 = Axis(
        fig[1, 1],
        title = "A: Population Abundance",
        xlabel = "Time",
        ylabel = "Count"
    )
    if :count_is_adult_male in propertynames(agent_df)
        lines!(
            ax1, agent_df.time, (agent_df.count_is_adult_male),
            label = "Adults", linewidth = 2.0, color = :seagreen
        )
    end
    if :count_is_ovigerous in propertynames(agent_df)
        lines!(
            ax1, agent_df.time, agent_df.count_is_ovigerous,
            label = "Ovigerous", linewidth = 2.0, color = :crimson
        )
    end
    axislegend(ax1, position = :rt)

    # Panel 2: Stage distribution
    organisms = collect(allagents(model))
    ax2 = Axis(
        fig[1, 2],
        title = "B: Stage Structure",
        xlabel = "Developmental Stage",
        ylabel = "Abundance (Organisms)",
        xticks = collect(4:14)
    )
    if !isempty(organisms)
        stage_range = 4:14
        males_by_stage = zeros(Int, length(stage_range))
        females_by_stage = zeros(Int, length(stage_range))
        for c in organisms
            idx = c.stage - first(stage_range) + 1
            if 1 <= idx <= length(stage_range)
                if c.sex === male
                    males_by_stage[idx] += 1
                elseif c.sex === female
                    females_by_stage[idx] += 1
                end
            end
        end
        stage_coords = Float64.(collect(stage_range))
        barplot!(
            ax2, stage_coords .- 0.18, males_by_stage,
            width = 0.35, color = (:dodgerblue, 0.85), label = "Males"
        )
        barplot!(
            ax2, stage_coords .+ 0.18, females_by_stage,
            width = 0.35, color = (:hotpink, 0.85), label = "Females"
        )
        axislegend(ax2, position = :rt)
    end

    # Panel 3: Size distribution by sex
    ax3 = Axis(
        fig[2, 1],
        title = "C: Size Dimorphism",
        xlabel = "Size (mm)",
        ylabel = "Density"
    )
    if !isempty(organisms)
        male_sizes = [Float64(c.size) for c in organisms if c.sex === male]
        female_sizes = [Float64(c.size) for c in organisms if c.sex === female]
        if !isempty(male_sizes)
            hist!(
                ax3, male_sizes, bins = 25, normalization = :pdf,
                color = (:dodgerblue, 0.55), strokewidth = 1, strokecolor = :white,
                label = "Males"
            )
        end
        if !isempty(female_sizes)
            hist!(
                ax3, female_sizes, bins = 25, normalization = :pdf,
                color = (:hotpink, 0.55), strokewidth = 1, strokecolor = :white,
                label = "Females"
            )
        end
        vlines!(
            ax3, [model.params.legal_size_male],
            linestyle = :dash, linewidth = 2.0, color = :darkred,
            label = "Legal Male Size"
        )
        axislegend(ax3, position = :rt)
    end

    # Panel 4: Spatial habitat
    nx, ny = size(model.temperature_field)
    ax4 = Axis(
        fig[2, 2],
        title = "D: Spatial Distribution",
        xlabel = "X Coordinate",
        ylabel = "Y Coordinate",
        aspect = AxisAspect(1)
    )
    hm = heatmap!(ax4, 1:nx, 1:ny, model.temperature_field, colormap = :coolwarm)
    Colorbar(fig[2, 3], hm, label = "Temp (°C)")
    if !isempty(organisms)
        juv_x = [Float64(c.pos[1]) for c in organisms if !is_adult(c)]
        juv_y = [Float64(c.pos[2]) for c in organisms if !is_adult(c)]
        adm_x = [Float64(c.pos[1]) for c in organisms if is_mature_male(c)]
        adm_y = [Float64(c.pos[2]) for c in organisms if is_mature_male(c)]
        if !isempty(juv_x)
            scatter!(ax4, juv_x, juv_y, color = :gold, markersize = 8, label = "Juv")
        end
        if !isempty(adm_x)
            scatter!(ax4, adm_x, adm_y, color = :dodgerblue, marker = :rect,
                     markersize = 9, label = "Adult M")
        end
    end

    # Panel 5: Stage allometry
    ax5 = Axis(
        fig[3, 1],
        title = "E: Allometric Scaling",
        xlabel = "Developmental Stage",
        ylabel = "Carapace Width (mm)",
        xticks = collect(4:14)
    )
    stage_grid = 4:14
    lines!(
        ax5, collect(stage_grid), [llbm(i) for i in stage_grid],
        label = "llbm (Male)", linewidth = 2.5, color = :blue
    )
    lines!(
        ax5, collect(stage_grid), [llbf(i) for i in stage_grid],
        label = "llbf (Female)", linewidth = 2.5, color = :magenta, linestyle = :dash
    )
    if !isempty(organisms)
        male_st = [Float64(c.stage) + randn(Random.Xoshiro(0)) * 0.08 for c in organisms if c.sex === male]
        male_sz = [Float64(c.size) for c in organisms if c.sex === male]
        if !isempty(male_sz)
            scatter!(ax5, male_st, male_sz, color = (:dodgerblue, 0.65), markersize = 7)
        end
    end
    axislegend(ax5, position = :lt)

    # Panel 6: Recruitment or Mean Size
    if model_df !== nothing && :total_recruits_settled in propertynames(model_df)
        ax6 = Axis(
            fig[3, 2],
            title = "F: Benthic Recruitment (Stage 4)",
            xlabel = "Time",
            ylabel = "Cumulative Recruits"
        )
        lines!(
            ax6, model_df.time, model_df.total_recruits_settled,
            label = "Settled Recruits", linewidth = 2.0, color = :teal
        )
        axislegend(ax6, position = :rt)
    else
        ax6 = Axis(
            fig[3, 2],
            title = "F: Mean Body Size",
            xlabel = "Time",
            ylabel = "Size (mm)"
        )
        mean_sz = if :safe_mean_size in propertynames(agent_df)
            agent_df.safe_mean_size
        elseif :mean_size in propertynames(agent_df)
            agent_df.mean_size
        else
            zeros(nrow(agent_df))
        end
        lines!(
            ax6, agent_df.time, mean_sz,
            label = "Mean Size", linewidth = 2.0, color = :orange
        )
        axislegend(ax6, position = :rt)
    end

    return fig
end

"""
    plot_stock_recruitment(sr_data; fits=nothing, title="Organism Stock-Recruitment Relationship")

Visualize the empirical stock-recruitment relationship outcome from a simulation.

Produces a 3-panel diagnostic figure:
- **Panel A (Stock-Recruitment Models)**: Empirical (S, R) scatter points overlaid with
  fitted Beverton-Holt and Ricker non-linear curves, and a reference replacement slope.
- **Panel B (Compensatory Density Dependence)**: Recruits per Spawner (R / S) vs. Spawner
  index, demonstrating compensatory survival decline as maternal spawner abundance increases.
- **Panel C (Cohort Timeseries)**: Paired temporal trajectory of spawner index and lagged
  recruitment counts across cohorts/years.

# Arguments
- `sr_data`: `DataFrame` produced by `extract_stock_recruitment`, or a `StandardABM` model.
- `fits`: Optional pre-computed output from `fit_stock_recruitment`.
- `title::String`: Plot title.

# Returns
`CairoMakie.Figure`: Multi-panel stock-recruitment diagnostic figure.
"""
function plot_stock_recruitment(
    sr_data;
    fits = nothing,
    title::String = "Organism Stock-Recruitment Relationship"
)
    fig = Figure(size = (1500, 480))
    Label(fig[0, :], title, font = :bold, fontsize = 16)

    df = if sr_data isa DataFrame
        sr_data
    else
        extract_stock_recruitment(sr_data)
    end

    n = nrow(df)
    if n == 0 || !(:spawners in propertynames(df)) || !(:recruits in propertynames(df))
        Label(
            fig[1, 1:3],
            "No recruitment events recorded\n(simulation duration < incubation or zero mature females)",
            fontsize = 14,
            color = :gray
        )
        return fig
    end

    s_vals = Float64.(df.spawners)
    r_vals = Float64.(df.recruits)
    rps_vals = (hasproperty(df, :recruits_per_spawner)) ?
        Float64.(df.recruits_per_spawner) :
        [(s > 0.0 ? r / s : 0.0) for (s, r) in zip(s_vals, r_vals)]

    actual_fits = (fits === nothing) ? fit_stock_recruitment(s_vals, r_vals) : fits

    s_max = maximum(s_vals)
    s_grid = range(0.0, max(1.0, s_max * 1.15), length = 150)

    # Panel 1: S vs R
    ax1 = Axis(
        fig[1, 1],
        title = "A: Stock-Recruitment Curves",
        xlabel = "Spawning Stock Index (10³ eggs / biomass)",
        ylabel = "Benthic Recruits (Stage 4 Count)"
    )
    scatter!(
        ax1, s_vals, r_vals,
        label = "Simulation Observations (N=$(n))",
        markersize = 10, strokewidth = 1.0, strokecolor = :royalblue,
        color = (:cornflowerblue, 0.85)
    )

    if actual_fits.is_fitted
        bh = actual_fits.beverton_holt
        rk = actual_fits.ricker
        r_bh = [bh.predict(s) for s in s_grid]
        r_rk = [rk.predict(s) for s in s_grid]

        bh_rmax_lbl = isfinite(bh.R_max) ? string(round(Int, bh.R_max)) : "Inf"
bh_lbl = "Beverton-Holt (R²=$(round(bh.R2, digits=2)), Rmax=$(bh_rmax_lbl))"
        rk_lbl = "Ricker (R²=$(round(rk.R2, digits=2)), Rmax=$(round(Int, rk.R_max)))"

        lines!(ax1, collect(s_grid), r_bh, label = bh_lbl, linewidth = 2.5, color = :seagreen)
        lines!(ax1, collect(s_grid), r_rk, label = rk_lbl, linewidth = 2.5,
               linestyle = :dash, color = :darkorange)
    else
        mean_s = mean(s_vals)
        mean_r = mean(r_vals)
        ref_slope = (mean_s > 0.0) ? (mean_r / mean_s) : 1.0
        lines!(ax1, collect(s_grid), ref_slope .* collect(s_grid),
               label = "Linear Scaling", linewidth = 1.5, linestyle = :dot, color = :gray)
    end
    axislegend(ax1, position = :lt)

    # Panel 2: Compensatory Survival (R / S vs S)
    ax2 = Axis(
        fig[1, 2],
        title = "B: Compensatory Density Dependence",
        xlabel = "Spawning Stock Index (S)",
        ylabel = "Recruits per Spawner (R / S)"
    )
    scatter!(
        ax2, s_vals, rps_vals,
        label = "Observed Survival Ratio",
        markersize = 10, color = (:crimson, 0.85)
    )

    if actual_fits.is_fitted
        bh = actual_fits.beverton_holt
        rk = actual_fits.ricker
        s_pos = filter(s -> s > 0.0, collect(s_grid))
        rps_bh = [bh.predict(s) / s for s in s_pos]
        rps_rk = [rk.predict(s) / s for s in s_pos]

        lines!(ax2, s_pos, rps_bh, label = "BH Expectation", linewidth = 2.0, color = :seagreen)
        lines!(ax2, s_pos, rps_rk, label = "Ricker Expectation", linewidth = 2.0,
               linestyle = :dash, color = :darkorange)
    end
    axislegend(ax2, position = :rt)

    # Panel 3: Cohort Dynamics Timeseries
    x_cohort = if hasproperty(df, :sample_index)
        Float64.(df.sample_index)
    elseif hasproperty(df, :time)
        round.(Float64.(df.time) ./ 365.0, digits = 1)
    else
        Float64.(1:n)
    end

    x_lbl = if hasproperty(df, :sample_index)
        "Cohort (Sample Index)"
    else
        "Time (Years post-hatch)"
    end
    ax3 = Axis(
        fig[1, 3],
        title = "C: Spawner-Recruit Cohort Dynamics",
        xlabel = x_lbl,
        ylabel = "Spawners / Recruits"
    )
    lines!(
        ax3, x_cohort, s_vals,
        label = "Spawning Stock (S)", linewidth = 2.0, color = :royalblue
    )
    scatter!(ax3, x_cohort, s_vals, color = :royalblue, markersize = 8)

    lines!(
        ax3, x_cohort, r_vals,
        label = "Settled Recruits (R)", linewidth = 2.0, linestyle = :dash,
        color = :forestgreen
    )
    scatter!(ax3, x_cohort, r_vals, color = :forestgreen, marker = :rect, markersize = 8)
    axislegend(ax3, position = :rt)

    return fig
end

"""
    plot_stock_recruitment(model, agent_df; model_df=nothing, title=...)

Convenience overload extracting stock-recruitment data from `model` and simulation dataframes
before rendering the diagnostic stock-recruitment figure.
"""
function plot_stock_recruitment(
    model,
    agent_df::DataFrame;
    model_df::Union{DataFrame, Nothing} = nothing,
    title::String = "Organism Stock-Recruitment Relationship"
)
    sr_df = extract_stock_recruitment(model)
    if nrow(sr_df) >= 2
        return plot_stock_recruitment(sr_df; title = title)
    end

    if model_df !== nothing
        sr_ts = extract_stock_recruitment(agent_df, model_df)
        if nrow(sr_ts) >= 1
            return plot_stock_recruitment(sr_ts; title = title)
        end
    end

    return plot_stock_recruitment(sr_df; title = title)
end

"""
    export_all_plots(model, agent_df, output_dir; model_df=nothing, ext="png")

Batch export all individual diagnostic figures, including stock-recruitment relationships,
and the unified simulation dashboard to `output_dir`.

# Returns
`Vector{String}`: Paths to all successfully generated plot files.
"""
function export_all_plots(
    model,
    agent_df::DataFrame,
    output_dir::String;
    model_df::Union{DataFrame, Nothing} = nothing,
    ext::String = "png"
)
    mkpath(output_dir)
    generated_files = String[]

    # Optional panels are best-effort: a short or single-run series cannot
    # support a macro fit, and that is not an error worth aborting the whole
    # export for. But a genuine failure (a plotting bug, a bad path) must not be
    # indistinguishable from "skipped" -- the caller used to receive a shorter
    # list and report success, so record what failed and why.
    function try_export(build, filename)
        try
            path = joinpath(output_dir, filename)
            save(path, build())
            push!(generated_files, path)
            return true
        catch err
            @warn "plot export failed" filename = filename exception = err
            return false
        end
    end

    # 1. Population Dynamics
    fn_dyn = joinpath(output_dir, "population_dynamics.$(ext)")
    p_dyn = plot_population_timeseries(agent_df; model_df = model_df)
    save(fn_dyn, p_dyn)
    push!(generated_files, fn_dyn)

    # 2. Stage Distribution
    fn_inst = joinpath(output_dir, "stage_distribution.$(ext)")
    p_inst = plot_stage_distribution(model)
    save(fn_inst, p_inst)
    push!(generated_files, fn_inst)

    # 3. Size Distribution by Sex
    fn_sex = joinpath(output_dir, "size_distribution_by_sex.$(ext)")
    p_sex = plot_size_by_sex(model)
    save(fn_sex, p_sex)
    push!(generated_files, fn_sex)

    # 4. Stage Allometry
    fn_allo = joinpath(output_dir, "stage_allometry.$(ext)")
    p_allo = plot_stage_allometry(model)
    save(fn_allo, p_allo)
    push!(generated_files, fn_allo)

    # 5. Spatial Habitat
    fn_spat = joinpath(output_dir, "spatial_habitat.$(ext)")
    p_spat = plot_spatial_habitat(model)
    save(fn_spat, p_spat)
    push!(generated_files, fn_spat)

    # 6. Maturity Ogives
    fn_mat = joinpath(output_dir, "maturity_ogive.$(ext)")
    p_mat = plot_maturity_ogives(model)
    save(fn_mat, p_mat)
    push!(generated_files, fn_mat)

    # 7. Reproduction Dynamics
    if model_df !== nothing
        fn_rep = joinpath(output_dir, "reproduction_dynamics.$(ext)")
        p_rep = plot_reproduction_dynamics(agent_df, model_df)
        save(fn_rep, p_rep)
        push!(generated_files, fn_rep)
    end

    # 8. Executive Simulation Dashboard
    fn_dash = joinpath(output_dir, "simulation_dashboard.$(ext)")
    p_dash = plot_simulation_dashboard(model, agent_df; model_df = model_df)
    save(fn_dash, p_dash)
    push!(generated_files, fn_dash)

    # 9. Stock-Recruitment Relationship
    fn_sr = joinpath(output_dir, "stock_recruitment.$(ext)")
    p_sr = plot_stock_recruitment(model, agent_df; model_df = model_df)
    save(fn_sr, p_sr)
    push!(generated_files, fn_sr)

    # 10. Macro-Ecological Population Dynamics Fits (Schaefer & Delay-Difference)
    if model_df !== nothing
        ann_prod = try
            extract_annual_production(agent_df, model_df)
        catch err
            @warn "annual production extraction failed; macro panels skipped" exception =
                err
            nothing
        end

        if ann_prod !== nothing && nrow(ann_prod) >= 3
            try_export("macro_population_dynamics.$(ext)") do
                plot_population_dynamics_fits(fit_population_dynamics(ann_prod))
            end
        end

        # 11. Retrospective Pattern Analysis
        if ann_prod !== nothing && nrow(ann_prod) >= 5
            try_export("macro_retrospective.$(ext)") do
                retro_res = retrospective_analysis(
                    ann_prod;
                    n_peels = min(5, nrow(ann_prod) - 3)
                )
                plot_retrospective_analysis(retro_res)
            end
        end

        # 12. Generalized Leslie Model & 3-Way Macro Comparison
        if ann_prod !== nothing && nrow(ann_prod) >= 4
            try_export("stage_specific_dynamics.$(ext)") do
                stage_mat, catch_mat, yrs = extract_annual_stage_matrix(ann_prod)
                leslie_fit = fit_generalized_leslie_model(
                    stage_mat,
                    catch_mat;
                    years = yrs
                )
                plot_stage_specific_dynamics(leslie_fit)
            end

            try_export("fishable_macro_comparison.$(ext)") do
                stage_mat, catch_mat, yrs = extract_annual_stage_matrix(ann_prod)
                leslie_fit = fit_generalized_leslie_model(
                    stage_mat,
                    catch_mat;
                    years = yrs
                )
                plot_macro_model_comparison(compare_macro_models(ann_prod, leslie_fit))
            end
        end
    end

    # 13. Spatial Ecosystem Snapshot using Agents.jl abmplot
    try_export("abm_spatial_summary.$(ext)") do
        plot_abm_summary(model)
    end

    return generated_files
end

"""
    plot_population_dynamics_fits(macro_fits; title="Organism Macro-Ecological Model Fits")

Render a 4-panel diagnostic figure visualizing post-hoc Schaefer logistic and
Deriso-Schnute delay-difference model fits:
- **Panel A (Whole Population Abundance)**: Observed total numbers vs Schaefer
  and Delay-Difference projected trajectories.
- **Panel B (Whole Population Surplus Production)**: Realized annual production
  vs abundance with fitted Schaefer parabola and N_MSY reference point.
- **Panel C (Fishable Stock Abundance)**: Observed legal-sized males vs Schaefer
  and Delay-Difference projected trajectories.
- **Panel D (Fishable Stock Surplus Production)**: Realized legal production
  vs legal abundance with fitted Schaefer parabola and commercial MSY point.

# Arguments
- `macro_fits`: Output NamedTuple from `fit_population_dynamics`.
- `title::String`: Overall plot title.

# Returns
`CairoMakie.Figure`: Multi-panel figure.
"""
function plot_population_dynamics_fits(
    macro_fits;
    title::String = "Organism Macro-Ecological Dynamic Model Fits"
)
    fig = Figure(size = (1400, 900))
    Label(fig[0, :], title, font = :bold, fontsize = 16)

    ann = macro_fits.annual_df
    wp = macro_fits.whole_population
    fs = macro_fits.fishable_stock

    samples = ann.sample_index
    n_tot_obs = ann.total_numbers
    n_leg_obs = ann.legal_numbers

    # Panel 1: Whole Population Abundance Trajectory
    ax1 = Axis(
        fig[1, 1],
        title = "A: Whole Population Abundance",
        xlabel = "Sample Index",
        ylabel = "Abundance (Total Organisms)"
    )
    scatter!(
        ax1, samples, n_tot_obs, label = "Observed Total Numbers",
        markersize = 10, color = (:dodgerblue, 0.85)
    )
    if wp.schaefer.is_fitted
        lines!(
            ax1, samples, wp.schaefer.predicted,
            label = "Schaefer: K=$(fmt_param(wp.schaefer.K)), r=$(fmt_param(wp.schaefer.r)), MSY=$(fmt_param(wp.schaefer.MSY))  R²=$(round(wp.schaefer.R2, digits=2))",
            linewidth = 2.5, color = :seagreen
        )
    end
    if wp.delay_difference.is_fitted
        lines!(
            ax1, samples, wp.delay_difference.predicted,
            label = "Delay-Diff: s=$(fmt_param(wp.delay_difference.s)), M=$(fmt_param(wp.delay_difference.M)), α=$(fmt_param(wp.delay_difference.alpha))  R²=$(round(wp.delay_difference.R2, digits=2))",
            linewidth = 2.5, linestyle = :dash, color = :darkorange
        )
    end
    axislegend(ax1, position = :rt)

    # Panel 2: Whole Population Surplus Production
    sp_tot = ann.surplus_production_total
    ax2 = Axis(
        fig[1, 2],
        title = "B: Whole Population Surplus Production",
        xlabel = "Abundance (N)",
        ylabel = "Surplus Production (ΔN + C)"
    )
    scatter!(
        ax2, n_tot_obs, sp_tot, label = "Observed Production",
        markersize = 10, color = (:dodgerblue, 0.85)
    )
    if wp.schaefer.is_fitted && wp.schaefer.K > 0.0
        n_grid = range(0.0, wp.schaefer.K * 1.15, length = 150)
        p_curve = [wp.schaefer.r * n * (1.0 - n / wp.schaefer.K) for n in n_grid]
        lines!(
            ax2, collect(n_grid), p_curve,
            label = "Schaefer (MSY=$(round(Int, wp.schaefer.MSY)))",
            linewidth = 2.5, color = :seagreen
        )
        vlines!(
            ax2, [wp.schaefer.N_msy],
            label = "N_MSY ($(round(Int, wp.schaefer.N_msy)))",
            linestyle = :dot, linewidth = 2.0, color = :crimson
        )
    end
    axislegend(ax2, position = :rt)

    # Panel 3: Fishable Component Abundance Trajectory
    ax3 = Axis(
        fig[2, 1],
        title = "C: Fishable Stock Abundance",
        xlabel = "Sample Index",
        ylabel = "Abundance (Legal Males)"
    )
    scatter!(
        ax3, samples, n_leg_obs, label = "Observed Legal Males",
        markersize = 10, color = (:crimson, 0.85)
    )
    if fs.schaefer.is_fitted
        lines!(
            ax3, samples, fs.schaefer.predicted,
            label = "Schaefer: K=$(fmt_param(fs.schaefer.K)), r=$(fmt_param(fs.schaefer.r)), MSY=$(fmt_param(fs.schaefer.MSY))  R²=$(round(fs.schaefer.R2, digits=2))",
            linewidth = 2.5, color = :seagreen
        )
    end
    if fs.delay_difference.is_fitted
        lines!(
            ax3, samples, fs.delay_difference.predicted,
            label = "Delay-Diff: s=$(fmt_param(fs.delay_difference.s)), M=$(fmt_param(fs.delay_difference.M)), α=$(fmt_param(fs.delay_difference.alpha))  R²=$(round(fs.delay_difference.R2, digits=2))",
            linewidth = 2.5, linestyle = :dash, color = :darkorange
        )
    end
    axislegend(ax3, position = :rt)

    # Panel 4: Fishable Component Surplus Production
    sp_leg = ann.surplus_production_legal
    ax4 = Axis(
        fig[2, 2],
        title = "D: Fishable Stock Surplus Production",
        xlabel = "Legal Abundance (N_legal)",
        ylabel = "Surplus Production (ΔN_leg + C_leg)"
    )
    scatter!(
        ax4, n_leg_obs, sp_leg, label = "Observed Legal Production",
        markersize = 10, color = (:crimson, 0.85)
    )
    if fs.schaefer.is_fitted && fs.schaefer.K > 0.0
        n_grid_leg = range(0.0, max(10.0, fs.schaefer.K * 1.15), length = 150)
        p_curve_leg = [fs.schaefer.r * n * (1.0 - n / fs.schaefer.K) for n in n_grid_leg]
        lines!(
            ax4, collect(n_grid_leg), p_curve_leg,
            label = "Schaefer (MSY=$(round(Int, fs.schaefer.MSY)))",
            linewidth = 2.5, color = :seagreen
        )
        vlines!(
            ax4, [fs.schaefer.N_msy],
            label = "N_MSY ($(round(Int, fs.schaefer.N_msy)))",
            linestyle = :dot, linewidth = 2.0, color = :crimson
        )
    end
    axislegend(ax4, position = :rt)

    return fig
end

"""
    plot_retrospective_analysis(retro_result; title="Organism Retrospective Pattern & Mohn's ρ")

Render a 4-panel diagnostic figure visualizing retrospective patterns and Mohn's rho:
- **Panel A (Whole Population Spaghetti Plot)**: Observed total numbers, reference base
  assessment trajectory, and truncated retrospective peel trajectories.
- **Panel B (Fishable Stock Spaghetti Plot)**: Corresponding retrospective trajectories for
  commercial legal adult males.
- **Panel C (Relative Errors & Benchmarks)**: Percentage relative bias per peel for both
  population components, contrasted against standard ±20% benchmark bounds.
- **Panel D (Parameter Stability Across Peels)**: Evolution of carrying capacity K (or
  survival s) as additional data years are added to the assessment.

# Arguments
- `retro_result`: NamedTuple returned by `retrospective_analysis`.
- `title::String`: Top-level figure title.

# Returns
`CairoMakie.Figure`: 4-panel composite diagnostic plot.
"""
function plot_retrospective_analysis(
    retro_result;
    title::String = "Organism Retrospective Pattern & Mohn's ρ"
)
    fig = Figure(size = (1400, 950))
    Label(fig[0, :], title, font = :bold, fontsize = 16)

    ann = retro_result.annual_df
    wp = retro_result.whole_population
    fs = retro_result.fishable_stock
    peels = retro_result.peel_indices
    samples = ann.sample_index
    n_samples = length(samples)

    peel_colors = [:darkorange, :purple, :teal, :magenta, :brown, :deepskyblue, :olive]
    rho_w_str = (wp.mohn_rho >= 0 ? "+" : "") * string(round(wp.mohn_rho, digits = 3))
    rho_f_str = (fs.mohn_rho >= 0 ? "+" : "") * string(round(fs.mohn_rho, digits = 3))

    # Panel 1: Whole Population Retrospective Trajectories
    ax1 = Axis(
        fig[1, 1],
        title = "A: Whole Population (Mohn's ρ = " * rho_w_str * ")",
        xlabel = "Assessment Sample",
        ylabel = "Abundance (Organism Numbers)"
    )
    scatter!(
        ax1, samples, ann.total_numbers,
        label = "Observed Total Numbers", markersize = 8, color = :royalblue
    )
    lines!(
        ax1, samples, wp.base_fit.predicted,
        label = "Base Assessment (Peel 0)", linewidth = 3.0, color = :royalblue
    )
    for i in eachindex(peels)
        p_idx = peels[i]
        c = peel_colors[mod1(i, length(peel_colors))]
        t_peel = n_samples - p_idx
        peel_samples = samples[1:t_peel]
        peel_pred = wp.peel_fits[i].predicted[1:t_peel]
        lines!(
            ax1, peel_samples, peel_pred,
            label = "Peel -$p_idx (to Sample $(peel_samples[end]))",
            linestyle = :dash, linewidth = 1.8, color = c
        )
        scatter!(
            ax1, [peel_samples[end]], [peel_pred[end]],
            markersize = 10, color = c, strokewidth = 1.5, strokecolor = :black
        )
    end
    axislegend(ax1, position = :rt)

    # Panel 2: Fishable Stock Retrospective Trajectories
    ax2 = Axis(
        fig[1, 2],
        title = "B: Fishable Stock (Mohn's ρ = " * rho_f_str * ")",
        xlabel = "Assessment Sample",
        ylabel = "Legal Abundance (Size ≥ legal_size_male)"
    )
    scatter!(
        ax2, samples, ann.legal_numbers,
        label = "Observed Legal Numbers", markersize = 8, color = :seagreen
    )
    lines!(
        ax2, samples, fs.base_fit.predicted,
        label = "Base Assessment (Peel 0)", linewidth = 3.0, color = :seagreen
    )
    for i in eachindex(peels)
        p_idx = peels[i]
        c = peel_colors[mod1(i, length(peel_colors))]
        t_peel = n_samples - p_idx
        peel_samples = samples[1:t_peel]
        peel_pred = fs.peel_fits[i].predicted[1:t_peel]
        lines!(
            ax2, peel_samples, peel_pred,
            label = "Peel -$p_idx (to Sample $(peel_samples[end]))",
            linestyle = :dash, linewidth = 1.8, color = c
        )
        scatter!(
            ax2, [peel_samples[end]], [peel_pred[end]],
            markersize = 10, color = c, strokewidth = 1.5, strokecolor = :black
        )
    end
    axislegend(ax2, position = :rt)

    # Panel 3: Relative Errors by Peel
    term_samples = retro_result.terminal_samples
    wp_re_pct = wp.rel_errors .* 100.0
    fs_re_pct = fs.rel_errors .* 100.0

    ax3 = Axis(
        fig[2, 1],
        title = "C: Relative Bias by Peel vs ±20% Benchmark",
        xlabel = "Terminal Peel Sample",
        ylabel = "Terminal Relative Error (%)"
    )
    hlines!(ax3, [0.0], color = :black, linewidth = 1.0)
    hlines!(
        ax3, [20.0, -20.0],
        label = "±20% Acceptable Bound", linestyle = :dot,
        linewidth = 2.0, color = :crimson
    )
    scatter!(
        ax3, term_samples .- 0.1, wp_re_pct,
        label = "Whole Pop (ρ=" * rho_w_str * ")",
        markersize = 10, color = :royalblue
    )
    lines!(ax3, term_samples .- 0.1, wp_re_pct, linewidth = 1.5, color = :royalblue)
    scatter!(
        ax3, term_samples .+ 0.1, fs_re_pct,
        label = "Fishable (ρ=" * rho_f_str * ")",
        markersize = 10, marker = :diamond, color = :seagreen
    )
    lines!(ax3, term_samples .+ 0.1, fs_re_pct, linewidth = 1.5, color = :seagreen)
    axislegend(ax3, position = :rt)

    # Panel 4: Parameter Stability Across Data Peels
    k_whole = [wp.params[i].K for i in length(peels):-1:1]
    push!(
        k_whole,
        (retro_result.model_type === :delay_difference) ?
        wp.base_fit.s : wp.base_fit.K
    )
    data_samples = [retro_result.terminal_samples[i] for i in length(peels):-1:1]
    push!(data_samples, samples[end])

    param_label = (retro_result.model_type === :delay_difference) ?
                  "Survival s" : "Carrying Capacity K"
    ax4 = Axis(
        fig[2, 2],
        title = "D: Parameter Stability Across Sequential Reanalyses",
        xlabel = "Available Data Horizon (Terminal Sample)",
        ylabel = param_label
    )
    lines!(
        ax4, data_samples, k_whole,
        label = "Whole Pop " * param_label, linewidth = 2.5, color = :royalblue
    )
    scatter!(ax4, data_samples, k_whole, markersize = 9, color = :royalblue)

    k_fish = [fs.params[i].K for i in length(peels):-1:1]
    push!(
        k_fish,
        (retro_result.model_type === :delay_difference) ?
        fs.base_fit.s : fs.base_fit.K
    )
    lines!(
        ax4, data_samples, k_fish,
        label = "Fishable " * param_label, linewidth = 2.0,
        linestyle = :dash, color = :seagreen
    )
    scatter!(ax4, data_samples, k_fish, marker = :diamond, markersize = 9, color = :seagreen)
    axislegend(ax4, position = :rt)

    return fig
end

"""
    plot_stage_specific_dynamics(fit::LeslieMatrixFit; title="Organism Stage-Specific Dynamics & Transition Analysis")

Render a 4-panel diagnostic figure visualizing generalized Leslie stage-transition dynamics:
- **Panel A (Stage Trajectories)**: Historical stage abundance timeseries.
- **Panel B (Stable Stage Distribution)**: Observed average stage structure vs asymptotic right eigenvector.
- **Panel C (Reproductive Value Profile)**: Left eigenvector (v_1) showing relative reproductive value across stages.
- **Panel D (Transition Elasticity Heatmap)**: Proportional elasticity matrix E indicating demographic sensitivity.

# Arguments
- `fit::LeslieMatrixFit`: Fitted stage-transition model.
- `title::String`: Main plot super-title.

# Returns
`CairoMakie.Figure`: 4-panel diagnostic figure.
"""
function plot_stage_specific_dynamics(
    fit::LeslieMatrixFit;
    title::String = "Organism Stage-Specific Dynamics & Transition Analysis"
)
    fig = Figure(size = (1400, 950))
    Label(fig[0, :], title, font = :bold, fontsize = 16)

    stages = fit.stages
    S = length(stages)
    years = fit.years

    # Panel 1: Stage Abundance Trajectories
    ax1 = Axis(
        fig[1, 1],
        title = "A: Stage Abundance Dynamics Across Stages",
        xlabel = "Year",
        ylabel = "Abundance (Numbers)"
    )
    lines!(
        ax1, years, fit.observed_stages[1, :],
        label = "Stage 4 (Recruit)", linewidth = 2.5, color = :seagreen
    )
    scatter!(ax1, years, fit.observed_stages[1, :], markersize = 7, color = :seagreen)

    if S >= 2
        lines!(
            ax1, years, fit.observed_stages[2, :],
            label = "Stage 5 (Juv)", linewidth = 2.0, color = :teal
        )
        scatter!(ax1, years, fit.observed_stages[2, :], marker = :rect,
                 markersize = 6, color = :teal)
    end

    for st in stages
        if st.is_legal_male
            lines!(
                ax1, years, fit.observed_stages[st.index, :],
                label = st.name * " [Legal]", linewidth = 2.2, color = :dodgerblue
            )
            scatter!(ax1, years, fit.observed_stages[st.index, :],
                     marker = :diamond, markersize = 7, color = :dodgerblue)
        end
    end
    for st in stages
        if st.is_reproductive && (st.stage == 9 || st.stage == 11)
            lines!(
                ax1, years, fit.observed_stages[st.index, :],
                label = st.name * " [Repro]", linewidth = 2.0,
                linestyle = :dash, color = :crimson
            )
            scatter!(ax1, years, fit.observed_stages[st.index, :],
                     markersize = 6, color = :crimson)
        end
    end
    axislegend(ax1, position = :rt)

    # Panel 2: Observed Stage Distribution vs Stable Stage Distribution
    obs_mean = [mean(fit.observed_stages[i, :]) for i in 1:S]
    sum_obs = sum(obs_mean)
    obs_prop = (sum_obs > 0.0) ? (obs_mean ./ sum_obs) .* 100.0 : zeros(Float64, S)
    ssd_pct = fit.stable_stage_distribution .* 100.0
    stage_indices = Float64.(collect(1:S))

    ax2 = Axis(
        fig[1, 2],
        title = "B: Observed vs Asymptotic Stable Stage Distribution (SSD)",
        xlabel = "Stage Index (1: Stg 4, 8-9: Legal Males, 13-15: Repro Females)",
        ylabel = "Proportion of Population (%)",
        xticks = collect(1:S)
    )
    barplot!(
        ax2, stage_indices .- 0.18, obs_prop,
        width = 0.35, color = (:royalblue, 0.85), label = "Observed Mean (%)"
    )
    barplot!(
        ax2, stage_indices .+ 0.18, ssd_pct,
        width = 0.35, color = (:darkorange, 0.85), label = "Theoretical SSD (w₁)"
    )
    axislegend(ax2, position = :rt)

    # Panel 3: Stage Reproductive Value
    rv_vals = fit.reproductive_values
    ax3 = Axis(
        fig[2, 1],
        title = "C: Stage Reproductive Value Profile (Fisher's RV)",
        xlabel = "Stage Index",
        ylabel = "Relative Reproductive Value (v₁ / v₁[1])",
        xticks = collect(1:S)
    )
    lines!(ax3, stage_indices, rv_vals, linewidth = 2.5, color = :crimson,
           label = "Reproductive Value")
    scatter!(ax3, stage_indices, rv_vals, markersize = 9, color = :crimson)

    repro_idx = [st.index for st in stages if st.is_reproductive]
    if !isempty(repro_idx)
        scatter!(
            ax3, Float64.(repro_idx), rv_vals[repro_idx],
            label = "Reproductive Females", markersize = 12,
            marker = :star5, color = :gold
        )
    end
    axislegend(ax3, position = :lt)

    # Panel 4: Stage Elasticity Matrix Heatmap
    E_mat = fit.elasticity_matrix
    ax4 = Axis(
        fig[2, 2],
        title = "D: Stage Transition Elasticity Matrix (∑ E_ij = 1.0)",
        xlabel = "From Stage j",
        ylabel = "To Stage i",
        aspect = AxisAspect(1),
        xticks = collect(1:S),
        yticks = collect(1:S)
    )
    hm_e = heatmap!(ax4, 1:S, 1:S, E_mat, colormap = :viridis)
    Colorbar(fig[2, 3], hm_e, label = "Elasticity")

    return fig
end

"""
    plot_macro_model_comparison(comp::MacroModelComparison; title="Three-Way Macro-Ecological Model Comparison: Fishable Stock")

Render a 4-panel comparative figure evaluating legal fishable biomass dynamics across:
- Schaefer Logistic Surplus Production Model
- Deriso-Schnute Delay-Difference Dynamic Model
- Generalized Leslie (Lefkovitch) Stage Transition Matrix Model
against empirical simulation observations.

# Arguments
- `comp::MacroModelComparison`: Comparative summary object.
- `title::String`: Super-title for the figure.

# Returns
`CairoMakie.Figure`: 4-panel comparative figure.
"""
function plot_macro_model_comparison(
    comp::MacroModelComparison;
    title::String = "Three-Way Macro-Ecological Model Comparison: Fishable Stock"
)
    fig = Figure(size = (1400, 950))
    Label(fig[0, :], title, font = :bold, fontsize = 16)

    yrs = comp.years

    # Panel 1: Fishable Biomass Trajectories
    ax1 = Axis(
        fig[1, 1],
        title = "A: Annual Biomass (Survey Period) Trajectories",
        xlabel = "Year",
        ylabel = "Biomass (kg)"
    )
    lines!(
        ax1, yrs, comp.biomass_obs_kg,
        label = comp.label_observed, linewidth = 2.5, color = :black
    )
    scatter!(ax1, yrs, comp.biomass_obs_kg, markersize = 9, color = :black)

    # Legend entries carry the key estimated parameters as well as goodness of fit:
    # identifying the curve is not much use without saying what it estimated.
    lines!(
        ax1, yrs, comp.biomass_schaefer_kg,
        label = comp.label_schaefer * "  R²=" * string(round(comp.r2_schaefer, digits = 2)),
        linewidth = 2.2, linestyle = :dash, color = :royalblue
    )
    lines!(
        ax1, yrs, comp.biomass_delay_diff_kg,
        label = comp.label_delay_diff * "  R²=" * string(round(comp.r2_delay_diff, digits = 2)),
        linewidth = 2.2, linestyle = :dashdot, color = :seagreen
    )
    lines!(
        ax1, yrs, comp.biomass_leslie_kg,
        label = comp.label_leslie * "  R²=" * string(round(comp.r2_leslie, digits = 2)),
        linewidth = 2.2, color = :darkorange
    )
    axislegend(ax1, position = :rt)

    # Panel 2: Model Residuals Across Years
    res_sch = comp.biomass_obs_kg .- comp.biomass_schaefer_kg
    res_dd = comp.biomass_obs_kg .- comp.biomass_delay_diff_kg
    res_les = comp.biomass_obs_kg .- comp.biomass_leslie_kg

    ax2 = Axis(
        fig[1, 2],
        title = "B: Prediction Residuals Across Years",
        xlabel = "Year",
        ylabel = "Residual (Obs - Pred, kg)"
    )
    hlines!(ax2, [0.0], linestyle = :dot, color = :gray, linewidth = 1.5)
    lines!(ax2, yrs, res_sch, label = "Schaefer", linewidth = 2.0, color = :royalblue)
    scatter!(ax2, yrs, res_sch, marker = :rect, markersize = 8, color = :royalblue)
    lines!(ax2, yrs, res_dd, label = "Delay-Diff", linewidth = 2.0, color = :seagreen)
    scatter!(ax2, yrs, res_dd, marker = :diamond, markersize = 8, color = :seagreen)
    lines!(ax2, yrs, res_les, label = "Leslie Matrix", linewidth = 2.0, color = :darkorange)
    scatter!(ax2, yrs, res_les, marker = :circle, markersize = 8, color = :darkorange)
    axislegend(ax2, position = :rt)

    # Panel 3: Biomass Variations & Coefficient of Variation
    cat_names = ["Observed", "Schaefer", "Delay-Diff", "Leslie Matrix"]
    cvs = [comp.cv_obs, comp.cv_schaefer, comp.cv_delay_diff, comp.cv_leslie]
    sds = [comp.sd_obs, comp.sd_schaefer, comp.sd_delay_diff, comp.sd_leslie]
    x_cats = Float64.(1:4)

    ax3 = Axis(
        fig[2, 1],
        title = "C: Fishable Biomass Variability & Dispersion",
        ylabel = "Variation Metric",
        xticks = (1:4, cat_names)
    )
    barplot!(
        ax3, x_cats .- 0.18, cvs,
        width = 0.35, label = "Coeff of Variation (CV = σ / μ)",
        color = (:royalblue, 0.85)
    )
    barplot!(
        ax3, x_cats .+ 0.18, sds,
        width = 0.35, label = "Standard Deviation σ (kg)",
        color = (:mediumpurple, 0.85)
    )
    axislegend(ax3, position = :rt)

    # Panel 4: Model Performance & AIC Scores
    model_labels = ["Schaefer", "Delay-Diff", "Leslie Matrix"]
    r2_vals = [comp.r2_schaefer, comp.r2_delay_diff, comp.r2_leslie]
    rmse_vals = [comp.rmse_schaefer, comp.rmse_delay_diff, comp.rmse_leslie]
    aic_vals = [comp.aic_schaefer, comp.aic_delay_diff, comp.aic_leslie]
    min_aic = minimum(aic_vals)
    delta_aics = aic_vals .- min_aic
    x_models = Float64.(1:3)

    pref_str = uppercase(string(comp.preferred_model))
    ax4 = Axis(
        fig[2, 2],
        title = "D: Model Performance & AIC Ranking (Pref: $(pref_str))",
        ylabel = "Metric Score",
        xticks = (1:3, model_labels)
    )
    barplot!(
        ax4, x_models .- 0.22, r2_vals,
        width = 0.22, label = "Goodness of Fit R²", color = (:seagreen, 0.85)
    )
    barplot!(
        ax4, x_models, rmse_vals,
        width = 0.22, label = "RMSE (kg)", color = (:royalblue, 0.85)
    )
    barplot!(
        ax4, x_models .+ 0.22, delta_aics,
        width = 0.22, label = "ΔAIC (0 = Preferred)", color = (:crimson, 0.85)
    )
    axislegend(ax4, position = :rt)

    return fig
end

# ==============================================================================
# Agents.jl Native Visualizations and Animation Tools
# ==============================================================================

"""
    organism_color(org::Organism, params::IBMParameters)

Determine the discrete visualization color for an individual organism based
on its life history stage, sex, maturity, and reproductive status:
- Larval stages (`zoea`, `megalopa`): Amber (`#E69F00`)
- Benthic immature females: Pink (`#CC79A7`)
- Benthic immature males: Sky blue (`#56B4E9`)
- Adult ovigerous females carrying egg clutches: Crimson (`#D55E00`)
- Adult non-ovigerous females: Purple (`#785EF0`)
- Legal-sized adult males: Dark blue (`#003B73`)
- Sublegal adult males: Royal blue (`#0072B2`)

The legal/sublegal split uses `params.fishery.legal_size_male`, so a reconfigured
fishery recolors the map instead of silently disagreeing with the demographic
counts.

# Arguments
- `org::Organism`: Individual organism agent.
- `params::IBMParameters`: Active parameters supplying the adult size threshold
  and the legal male size.

# Returns
`String`: Hex color code for graphical representation.
"""
function organism_color(org::Organism, params::IBMParameters)
    if org.stage in (0, 1, 2, 3) || org.sex === undifferentiated
        return "#E69F00"
    elseif !is_adult(org)
        return (org.sex === female) ? "#CC79A7" : "#56B4E9"
    else
        if is_ovigerous(org)
            return "#D55E00"
        elseif org.sex === female
            return "#785EF0"
        else
            return is_legal_male(org, params.fishery.legal_size_male,
                                 params.organism.adult_size_threshold) ?
                    "#003B73" : "#0072B2"
        end
    end
end

"""
    organism_size(org::Organism; min_size=4.0, max_size=24.0)

Compute marker display size in points scaled from individual carapace width
(Size in mm):
``\\text{size} = \\min(\\max(Size / 4.0, min\\_size), max\\_size)``

# Arguments
- `org::Organism`: Individual organism agent.
- `min_size::Float64`: Minimum marker size in points (default: 4.0).
- `max_size::Float64`: Maximum marker size in points (default: 24.0).

# Returns
`Float64`: Display marker diameter in points.
"""
function organism_size(
    org::Organism;
    min_size::Float64 = 4.0,
    max_size::Float64 = 24.0
)
    s = org.size / 4.0
    return min(max(s, min_size), max_size)
end

"""
    organism_marker(org::Organism)

Select scatter marker shape based on sex classification:
- Female (`female`): Circle (`:circle`)
- Male (`male`): Square (`:rect`)
- Undifferentiated larvae (`undifferentiated`): Diamond (`:diamond`)

# Arguments
- `org::Organism`: Individual organism agent.

# Returns
`Symbol`: Marker shape symbol for Makie plotting.
"""
function organism_marker(org::Organism)
    if org.sex === female
        return :circle
    elseif org.sex === male
        return :rect
    else
        return :diamond
    end
end

"""
    organism_heatarray(model, field::Symbol=:temperature_field)

Extract a 2D matrix from the ABM spatial environment to serve as a background
heatarray for `Agents.abmplot` or `Agents.abmvideo`.

# Arguments
- `model`: Organism simulation model.
- `field::Symbol`: Environmental field name (`:temperature_field` or
  `:food_availability_field`, default: `:temperature_field`).

# Returns
`Matrix{Float64}`: 2D spatial property array.
"""
function organism_heatarray(model, field::Symbol = :temperature_field)
    if field in (:temperature, :temp, :temperature_field)
        return model.temperature_field
    elseif field in (:food, :food_availability, :food_availability_field)
        return model.food_availability_field
    else
        error("Unknown heatarray field: $(field). Supported: :temperature, :food")
    end
end

"""
    plot_abm_summary(model; heat_field=:temperature, output_path=nothing,
                     figure_size=(800, 800), title="Organism Spatial Ecosystem")

Render a publication-ready spatial ABM summary snapshot of the organism population
using `Agents.abmplot` over the specified environmental landscape (temperature or food).

# Arguments
- `model`: Configured organism simulation model.
- `heat_field::Union{Symbol, String}`: Environmental field (`:temperature` or `:food`).
- `output_path::Union{String, Nothing}`: Optional destination file path to save figure.
- `figure_size::Tuple{Int, Int}`: Figure dimensions in pixels (default: (800, 800)).
- `title::String`: Plot title.

# Returns
`CairoMakie.Figure`: The generated Makie figure object.
"""
function plot_abm_summary(
    model;
    heat_field::Union{Symbol, String} = :temperature,
    output_path::Union{String, Nothing} = nothing,
    figure_size::Tuple{Int, Int} = (800, 800),
    title::String = "Organism Spatial Ecosystem"
)
    field_sym = Symbol(heat_field)
    cbar_label = if field_sym in (:food, :food_availability, :food_availability_field)
        "Benthic Food Density"
    else
        "Bottom Temperature (°C)"
    end

    heat_fn = m -> organism_heatarray(m, field_sym)
    n_organisms = nagents(model)
    sim_time = hasproperty(model, :current_time) ? model.current_time : 0.0

    full_title = "$(title) (Day $(round(sim_time, digits=1)), Living Organisms: $(n_organisms))"

    fig, ax, abmobs = Agents.abmplot(
        model;
        agent_color = org -> organism_color(org, model.params),
        agent_size = organism_size,
        agent_marker = organism_marker,
        heatarray = heat_fn,
        add_colorbar = true,
        colorbar_label = cbar_label,
        figure = (size = figure_size,),
        axis = (title = full_title, titlealign = :left)
    )

    if output_path !== nothing
        mkpath(dirname(abspath(output_path)))
        save(output_path, fig)
    end

    return fig
end

"""
    animate_organism_simulation(model; filepath="organism_animation.mp4",
                                frames=120, dt=2, framerate=15,
                                heat_field=:temperature, figure_size=(700, 700),
                                title="Organism Ecosystem Dynamics")

Record an animated video or GIF of the spatial simulation evolution using
`Agents.abmvideo`, rendering organism movement, growth, development, and reproduction
over the dynamic environmental field.

# Arguments
- `model`: Configured organism model to simulate and record.
- `filepath::String`: Destination path for video file (`.mp4`, `.gif`, or `.webm`).
- `frames::Int`: Number of animation frames to record (default: 120).
- `dt::Int`: Simulation days advanced per animation frame (default: 2).
- `framerate::Int`: Playback framerate in frames per second (default: 15).
- `heat_field::Union{Symbol, String}`: Environmental background field (`:temperature` or `:food`).
- `figure_size::Tuple{Int, Int}`: Resolution of animation frames (default: (700, 700)).
- `title::String`: Base title prefix.

# Returns
`String`: Absolute path to the recorded animation file.
"""
function animate_organism_simulation(
    model;
    filepath = "organism_animation.mp4",
    frames = 120,
    dt = 2,
    framerate = 15,
    heat_field = :temperature,
    figure_size = (700, 700),
    title = "Organism Ecosystem Dynamics"
)
    heat_fn = m -> organism_heatarray(m, Symbol(heat_field))
    mkpath(dirname(abspath(filepath)))
    Agents.abmvideo(
        filepath,
        model;
        frames = frames,
        dt = dt,
        framerate = framerate,
        heatarray = heat_fn,
        figure = (size = figure_size,),
        title = title,
        agent_color = org -> organism_color(org, model.params),
        agent_size = organism_size,
        agent_marker = organism_marker
    )
    return filepath
end

"""
    plot_functional_relationships(params_list::Vector{IBMParameters};
                                  type=:maturity, label_prefix="Set", ...)

Render parameterized process curves across different parameter sets, for comparing
configurations side by side.

# Arguments
- `params_list::Vector{IBMParameters}`: List of parameter profiles.
- `type::Symbol`: `:maturity`, `:fecundity`, `:selectivity`, or `:growth`.
- `label_prefix::String`: Legend prefix for each parameter set.

# Notes
The species-level functions take a `SpeciesParams`, not a whole `IBMParameters`;
this extracts the species section so a caller can pass the parameter object the
model actually holds. It previously passed the whole `IBMParameters`, so every
branch raised a `TypeError`.

# Returns
`CairoMakie.Figure`: Visualized functional relationship curves.
"""
function plot_functional_relationships(
    params_list::Vector{IBMParameters};
    type::Symbol = :maturity,
    label_prefix::String = "Set",
    title::Union{String, Nothing} = nothing,
    xlabel::Union{String, Nothing} = nothing,
    ylabel::Union{String, Nothing} = nothing
)
    fig = Figure(size = (800, 600))

    if type === :maturity
        title_str = title !== nothing ? title : "Maturity Ogives (Functional Space)"
        xlabel_str = xlabel !== nothing ? xlabel : "Size (mm)"
        ylabel_str = ylabel !== nothing ? ylabel : "P(Mature)"
        ax = Axis(fig[1, 1], title = title_str, xlabel = xlabel_str, ylabel = ylabel_str)

        size_vals = exp.(range(log(10.0), log(250.0), length = 100))
        for (i, params) in enumerate(params_list)
            probs = [
                maturity(sz; params = params.species, deterministic = true)
                for sz in size_vals
            ]
            lines!(ax, size_vals, probs, label = "$(label_prefix)$i", linewidth = 2.5)
        end
        axislegend(ax, position = :rb)

    elseif type === :fecundity
        title_str = title !== nothing ? title : "Fecundity Curves (Functional Space)"
        xlabel_str = xlabel !== nothing ? xlabel : "Size (mm)"
        ylabel_str = ylabel !== nothing ? ylabel : "Total Eggs"
        ax = Axis(fig[1, 1], title = title_str, xlabel = xlabel_str, ylabel = ylabel_str)

        size_vals = collect(10.0:5.0:250.0)
        for (i, params) in enumerate(params_list)
            eggs = [fecundity(sz; fecundity_parameterization = "moncton",
                              is_primiparous = true, params = params.species,
                              deterministic = true) for sz in size_vals]
            lines!(ax, size_vals, eggs, label = "$(label_prefix)$i", linewidth = 2.5)
        end
        axislegend(ax, position = :lt)

    elseif type === :selectivity
        title_str = title !== nothing ? title : "Fishery Selectivity (Functional Space)"
        xlabel_str = xlabel !== nothing ? xlabel : "Size (mm)"
        ylabel_str = ylabel !== nothing ? ylabel : "Retention Probability"
        ax = Axis(fig[1, 1], title = title_str, xlabel = xlabel_str, ylabel = ylabel_str)

        size_vals = collect(20.0:2.0:180.0)
        for (i, params) in enumerate(params_list)
            probs = selectivity.(size_vals; params = params.species)
            lines!(ax, size_vals, probs, label = "$(label_prefix)$i", linewidth = 2.5)
        end
        axislegend(ax, position = :rb)

    elseif type === :growth
        title_str = title !== nothing ? title : "Growth Schedules (Functional Space)"
        xlabel_str = xlabel !== nothing ? xlabel : "Stage"
        ylabel_str = ylabel !== nothing ? ylabel : "Carapace Width (mm)"
        ax = Axis(fig[1, 1], title = title_str, xlabel = xlabel_str, ylabel = ylabel_str)

        stage_vals = collect(4:10)
        for (i, params) in enumerate(params_list)
            behavior = SpeciesBehaviorImpl(params.species)
            schedule = growth_schedule(behavior)
            rows = filter(r -> r.stage in stage_vals, eachrow(schedule))
            lines!(
                ax,
                [r.stage for r in rows],
                [r.size_male_med for r in rows],
                label = "$(label_prefix)$i", linewidth = 2.5
            )
        end
        axislegend(ax, position = :lt)

    else
        error("Unsupported functional type: $type. Use :maturity, :fecundity, :selectivity, or :growth.")
    end

    return fig
end
