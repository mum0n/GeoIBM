"""
    src/visualization.jl

Visualization and plotting tools for the snow crab (*Chionoecetes opilio*)
individual-based model (IBM).

Provides specialized diagnostic plots for population dynamics, demographic
stage and instar structures, size distributions by sex, allometric growth
trajectories, bioenergetics, spatial distributions over environmental fields,
and a unified executive simulation dashboard.
"""

using Plots
using DataFrames
using Agents
using Statistics
import CairoMakie


"""
    plot_population_timeseries(agent_df; model_df=nothing)

Generate a multi-panel timeseries plot showing the temporal dynamics of the
simulated snow crab population.

# Visualized Metrics
- **Panel 1 (Abundance)**: Total living crabs, mature adults, and ovigerous females.
- **Panel 2 (Body Size)**: Mean carapace width (mm) across time.
- **Panel 3 (Recruitment, optional)**: Cumulative benthic recruitment of Instar 4 settlers
  if `model_df` contains `:total_recruits_settled`.

# Arguments
- `agent_df::DataFrame`: Time-stepped agent data collected during simulation.
- `model_df::Union{DataFrame, Nothing}`: Optional time-stepped model-level data.

# Returns
`Plots.Plot`: Combined multi-panel timeseries figure.
"""
function plot_population_timeseries(
    agent_df::DataFrame;
    model_df::Union{DataFrame, Nothing} = nothing
)
    p1 = plot(
        agent_df.time,
        agent_df.length_stage,
        xlabel = "Time (Days)",
        ylabel = "Count",
        title = "Snow Crab Abundance Dynamics",
        label = "Total Living Crabs",
        linewidth = 2.0,
        color = :royalblue
    )

    if :count_is_adult in propertynames(agent_df)
        plot!(
            p1,
            agent_df.time,
            agent_df.count_is_adult,
            label = "Mature Adults",
            linewidth = 2.0,
            color = :seagreen
        )
    end

    if :count_is_ovigerous in propertynames(agent_df)
        plot!(
            p1,
            agent_df.time,
            agent_df.count_is_ovigerous,
            label = "Ovigerous Females",
            linewidth = 2.0,
            color = :crimson
        )
    end

    p2 = plot(
        agent_df.time,
        agent_df.safe_mean_size_mm,
        xlabel = "Time (Days)",
        ylabel = "Carapace Width (mm)",
        title = "Mean Carapace Width Progression",
        label = "Mean Population CW",
        linewidth = 2.0,
        color = :darkorange
    )

    if model_df !== nothing && :total_recruits_settled in propertynames(model_df)
        p3 = plot(
            model_df.time,
            model_df.total_recruits_settled,
            xlabel = "Time (Days)",
            ylabel = "Cumulative Recruits",
            title = "Benthic Recruitment (Instar 4)",
            label = "Settled Recruits",
            linewidth = 2.0,
            color = :teal
        )
        return plot(p1, p2, p3, layout = (3, 1), size = (850, 850), margin = 5Plots.mm)
    else
        return plot(p1, p2, layout = (2, 1), size = (850, 600), margin = 5Plots.mm)
    end
end


"""
    plot_instar_distribution(model)

Generate a grouped bar chart displaying the frequency of living crabs across
developmental instars, categorized by sex.

# Biological Significance
Snow crab developmental progress is marked by discrete instars. Instars 1-3
correspond to pelagic larvae, while instars 4-14+ represent benthic juveniles
and mature adults. This plot visualizes cohort abundance and recruitment
progression across the demographic age-structure.

# Arguments
- `model::ABM`: Agents.jl simulation model containing `SnowCrab` agents.

# Returns
`Plots.Plot`: Grouped bar chart of crab counts per instar by sex.
"""
function plot_instar_distribution(model)
    crabs = collect(allagents(model))
    if isempty(crabs)
        return plot(title = "No living crabs to display", size = (700, 450))
    end

    instar_range = 4:14
    males_by_instar = zeros(Int, length(instar_range))
    females_by_instar = zeros(Int, length(instar_range))

    for c in crabs
        idx = c.instar - 3
        if 1 <= idx <= length(instar_range)
            if c.sex === male
                males_by_instar[idx] += 1
            elseif c.sex === female
                females_by_instar[idx] += 1
            end
        end
    end

    instar_coords = collect(instar_range)

    p = bar(
        instar_coords .- 0.18,
        males_by_instar,
        bar_width = 0.35,
        label = "Males",
        color = :dodgerblue,
        alpha = 0.85,
        xlabel = "Developmental Instar",
        ylabel = "Abundance (Crabs)",
        title = "Instar Demographic Structure by Sex",
        legend = :topright,
        size = (750, 450),
        margin = 5Plots.mm
    )

    bar!(
        p,
        instar_coords .+ 0.18,
        females_by_instar,
        bar_width = 0.35,
        label = "Females",
        color = :hotpink,
        alpha = 0.85
    )

    return p
end


"""
    plot_size_by_sex(model)

Generate overlaid size-frequency distributions (carapace width in mm)
separated by sex for all living individuals.

# Biological Significance
Illustrates sexual dimorphism in *Chionoecetes opilio*. Females undergo terminal
molt at smaller sizes (typically 50-80 mm CW), whereas males can continue to
grow to instars 12-14+ (exceeding 140 mm CW).

# Arguments
- `model::ABM`: Simulation model containing `SnowCrab` agents.

# Returns
`Plots.Plot`: Overlaid histogram of carapace widths for males and females.
"""
function plot_size_by_sex(model)
    crabs = collect(allagents(model))
    if isempty(crabs)
        return plot(title = "No living crabs to display", size = (750, 450))
    end

    male_sizes = [c.size_mm for c in crabs if c.sex === male]
    female_sizes = [c.size_mm for c in crabs if c.sex === female]

    p = plot(
        title = "Carapace Width Distribution by Sex",
        xlabel = "Carapace Width (mm)",
        ylabel = "Density",
        size = (750, 450),
        margin = 5Plots.mm
    )

    if !isempty(male_sizes)
        histogram!(
            p,
            male_sizes,
            bins = 25,
            normalize = :pdf,
            alpha = 0.55,
            color = :dodgerblue,
            linecolor = :white,
            label = "Males (n = $(length(male_sizes)))"
        )
    end

    if !isempty(female_sizes)
        histogram!(
            p,
            female_sizes,
            bins = 25,
            normalize = :pdf,
            alpha = 0.55,
            color = :hotpink,
            linecolor = :white,
            label = "Females (n = $(length(female_sizes)))"
        )
    end

    # Vertical line indicating commercial legal size for males
    vline!(
        p,
        [model.params.legal_size_male],
        linestyle = :dash,
        linewidth = 2,
        color = :darkred,
        label = "Legal Male Size ($(model.params.legal_size_male) mm)"
    )

    return p
end


"""
    plot_instar_allometry(model)

Scatter plot of individual crab carapace widths across instars contrasted
with the empirical theoretical allometric mean curves (`llbm` and `llbf`).

# Allometric Reference
- Males:   ``CW = \\exp(1.917564 + 0.298914 \\cdot (instar - 3))``
- Females: ``CW = \\exp(2.198848 + 0.315026 \\cdot (instar - 4))``

# Arguments
- `model::ABM`: Simulation model containing `SnowCrab` agents.

# Returns
`Plots.Plot`: Carapace width vs instar scatter with empirical bounds.
"""
function plot_instar_allometry(model)
    crabs = collect(allagents(model))
    if isempty(crabs)
        return plot(title = "No living crabs to display", size = (750, 480))
    end

    instar_grid = 4:14
    cw_males_theory = [llbm(i) for i in instar_grid]
    cw_females_theory = [llbf(i) for i in instar_grid]

    p = plot(
        instar_grid,
        cw_males_theory,
        label = "Empirical Male Curve (llbm)",
        linewidth = 2.5,
        color = :blue,
        xlabel = "Developmental Instar",
        ylabel = "Carapace Width (mm)",
        title = "Instar Allometric Scaling: Simulation vs Theory",
        legend = :topleft,
        size = (750, 480),
        margin = 5Plots.mm
    )

    plot!(
        p,
        instar_grid,
        cw_females_theory,
        label = "Empirical Female Curve (llbf)",
        linewidth = 2.5,
        color = :magenta,
        linestyle = :dash
    )

    # Plot simulated male agents
    male_instars = [c.instar + randn() * 0.08 for c in crabs if c.sex === male]
    male_sizes = [c.size_mm for c in crabs if c.sex === male]
    if !isempty(male_sizes)
        scatter!(
            p,
            male_instars,
            male_sizes,
            label = "Simulated Males",
            color = :dodgerblue,
            markersize = 4,
            alpha = 0.65
        )
    end

    # Plot simulated female agents
    female_instars = [c.instar + randn() * 0.08 for c in crabs if c.sex === female]
    female_sizes = [c.size_mm for c in crabs if c.sex === female]
    if !isempty(female_sizes)
        scatter!(
            p,
            female_instars,
            female_sizes,
            label = "Simulated Females",
            color = :hotpink,
            markersize = 4,
            alpha = 0.65
        )
    end

    return p
end


"""
    plot_reproduction_dynamics(agent_df, model_df)

Plot reproductive activity, egg production, and recruitment pulses over time.

# Panels
- **Panel 1**: Temporal trajectory of brooding ovigerous females.
- **Panel 2**: Cumulative eggs spawned into the system.
- **Panel 3**: Cumulative benthic recruits (Instar 4) settled.

# Arguments
- `agent_df::DataFrame`: Agent metrics tracked over simulation.
- `model_df::DataFrame`: Model properties tracked over simulation.

# Returns
`Plots.Plot`: Multi-panel reproduction and recruitment figure.
"""
function plot_reproduction_dynamics(
    agent_df::DataFrame,
    model_df::DataFrame
)
    p1 = if :count_is_ovigerous in propertynames(agent_df)
        plot(
            agent_df.time,
            agent_df.count_is_ovigerous,
            xlabel = "Time (Days)",
            ylabel = "Ovigerous Count",
            title = "Brooding Ovigerous Females",
            label = "Active Ovigerous Females",
            linewidth = 2,
            color = :crimson
        )
    else
        plot(title = "Ovigerous metrics not tracked")
    end

    p2 = if :total_eggs_spawned in propertynames(model_df)
        plot(
            model_df.time,
            model_df.total_eggs_spawned ./ 1.0e6,
            xlabel = "Time (Days)",
            ylabel = "Eggs (Millions)",
            title = "Cumulative Egg Production",
            label = "Total Spawned Eggs",
            linewidth = 2,
            color = :purple
        )
    else
        plot(title = "Egg metrics not tracked")
    end

    p3 = if :total_recruits_settled in propertynames(model_df)
        plot(
            model_df.time,
            model_df.total_recruits_settled,
            xlabel = "Time (Days)",
            ylabel = "Recruits Settled",
            title = "Cumulative Benthic Recruitment (Instar 4)",
            label = "Settled Recruits",
            linewidth = 2,
            color = :teal
        )
    else
        plot(title = "Recruitment metrics not tracked")
    end

    return plot(p1, p2, p3, layout = (3, 1), size = (850, 850), margin = 5Plots.mm)
end


"""
    plot_spatial_habitat(model)

Generate a 2D spatial habitat map displaying benthic water temperature with
living crabs superimposed by demographic stage and sex.

# Demarcated Groups
- Juveniles (Instar 4+): Yellow circles
- Adult Males: Dodgerblue squares
- Adult Non-Ovigerous Females: Pink triangles
- Adult Ovigerous Females: Crimson stars

# Arguments
- `model::ABM`: Simulation model containing space and agents.

# Returns
`Plots.Plot`: 2D spatial habitat and distribution figure.
"""
function plot_spatial_habitat(model)
    nx, ny = size(model.temperature_field)

    p = heatmap(
        1:nx,
        1:ny,
        model.temperature_field',
        title = "Spatial Crab Habitat & Distribution",
        xlabel = "X Coordinate (Grid Cells)",
        ylabel = "Y Coordinate (Grid Cells)",
        color = :coolwarm,
        aspect_ratio = :equal,
        colorbar_title = "Temp (°C)",
        size = (800, 650),
        margin = 5Plots.mm
    )

    crabs = collect(allagents(model))
    if isempty(crabs)
        return p
    end

    # Group positions by demographic category
    juv_x = [c.pos[1] for c in crabs if c.stage === juvenile]
    juv_y = [c.pos[2] for c in crabs if c.stage === juvenile]

    adm_x = [c.pos[1] for c in crabs if c.stage === adult && c.sex === male]
    adm_y = [c.pos[2] for c in crabs if c.stage === adult && c.sex === male]

    adf_x = [c.pos[1] for c in crabs if c.stage === adult && c.sex === female &&
             c.eggs_count == 0]
    adf_y = [c.pos[2] for c in crabs if c.stage === adult && c.sex === female &&
             c.eggs_count == 0]

    ovig_x = [c.pos[1] for c in crabs if c.stage === adult && c.sex === female &&
              c.eggs_count > 0]
    ovig_y = [c.pos[2] for c in crabs if c.stage === adult && c.sex === female &&
              c.eggs_count > 0]

    if !isempty(juv_x)
        scatter!(p, juv_x, juv_y, label = "Juveniles ($(length(juv_x)))",
                 color = :gold, markershape = :circle, markersize = 5,
                 markerstrokecolor = :black, markerstrokewidth = 0.5)
    end

    if !isempty(adm_x)
        scatter!(p, adm_x, adm_y, label = "Adult Males ($(length(adm_x)))",
                 color = :dodgerblue, markershape = :square, markersize = 6,
                 markerstrokecolor = :black, markerstrokewidth = 0.5)
    end

    if !isempty(adf_x)
        scatter!(p, adf_x, adf_y, label = "Adult Females ($(length(adf_x)))",
                 color = :hotpink, markershape = :utriangle, markersize = 6,
                 markerstrokecolor = :black, markerstrokewidth = 0.5)
    end

    if !isempty(ovig_x)
        scatter!(p, ovig_x, ovig_y, label = "Ovigerous Females ($(length(ovig_x)))",
                 color = :crimson, markershape = :star5, markersize = 8,
                 markerstrokecolor = :white, markerstrokewidth = 0.8)
    end

    return p
end


"""
    plot_maturity_ogives(model; cw_bins=20.0:10.0:140.0)

Contrast the simulated empirical proportion mature against theoretical
logistic maturity schedules for both male and female snow crabs.

# Theoretical Model
```math
P_{\\text{mature}}(CW) = \\frac{1}{1 + \\exp(-k \\cdot (CW - L_{50}))}
```
where ``L_{50} \\approx 91.8`` mm for males and ``51.6`` mm for females.

# Arguments
- `model::ABM`: Simulation model containing `SnowCrab` agents.
- `cw_bins::AbstractRange`: Carapace width bin boundaries.

# Returns
`Plots.Plot`: Empirical vs theoretical maturity ogive comparison.
"""
function plot_maturity_ogives(model; cw_bins = 20.0:10.0:140.0)
    cw_eval = 20.0:1.0:140.0
    pm_male_theory = [maturity(0, log(cw)) for cw in cw_eval]
    pm_female_theory = [maturity(1, log(cw)) for cw in cw_eval]

    p = plot(
        cw_eval,
        pm_male_theory,
        label = "Theoretical Male (L50=91.8 mm)",
        linewidth = 2.5,
        color = :blue,
        xlabel = "Carapace Width (mm)",
        ylabel = "Proportion Mature",
        title = "Snow Crab Maturity Ogives: Simulation vs Theory",
        legend = :bottomright,
        size = (750, 480),
        margin = 5Plots.mm
    )

    plot!(
        p,
        cw_eval,
        pm_female_theory,
        label = "Theoretical Female (L50=51.6 mm)",
        linewidth = 2.5,
        color = :red,
        linestyle = :dash
    )

    cm_m, pm_m = compute_empirical_ogive(model, male, cw_bins)
    cm_f, pm_f = compute_empirical_ogive(model, female, cw_bins)

    if !isempty(cm_m)
        scatter!(p, cm_m, pm_m, label = "Simulated Males",
                 color = :dodgerblue, markershape = :square, markersize = 6)
    end

    if !isempty(cm_f)
        scatter!(p, cm_f, pm_f, label = "Simulated Females",
                 color = :crimson, markershape = :circle, markersize = 6)
    end

    return p
end


"""
    plot_simulation_dashboard(model, agent_df; model_df=nothing)

Generate a unified 6-panel executive summary dashboard combining the core
biological and ecological indicators of the snow crab simulation.

# Panels Included
1. **Panel A**: Abundance & demographic stage timeseries.
2. **Panel B**: Instar demographic distribution by sex.
3. **Panel C**: Carapace width distribution by sex (dimorphism).
4. **Panel D**: Spatial habitat distribution over bottom temperature.
5. **Panel E**: Instar allometry compared with empirical curves.
6. **Panel F**: Cumulative recruitment and egg production.

# Arguments
- `model::ABM`: Simulation model instance.
- `agent_df::DataFrame`: Time-stepped agent data.
- `model_df::Union{DataFrame, Nothing}`: Optional time-stepped model data.

# Returns
`Plots.Plot`: High-impact 6-panel dashboard figure.
"""
function plot_simulation_dashboard(
    model,
    agent_df::DataFrame;
    model_df::Union{DataFrame, Nothing} = nothing
)
    # Panel 1: Population abundance timeseries
    p1 = plot(
        agent_df.time,
        agent_df.length_stage,
        xlabel = "Time (Days)",
        ylabel = "Count",
        title = "A: Population Abundance",
        label = "Total Crabs",
        linewidth = 2,
        color = :royalblue
    )
    if :count_is_adult in propertynames(agent_df)
        plot!(p1, agent_df.time, agent_df.count_is_adult, label = "Adults",
              linewidth = 2, color = :seagreen)
    end
    if :count_is_ovigerous in propertynames(agent_df)
        plot!(p1, agent_df.time, agent_df.count_is_ovigerous, label = "Ovigerous",
              linewidth = 2, color = :crimson)
    end

    # Panel 2: Instar distribution
    p2 = plot_instar_distribution(model)
    plot!(p2, title = "B: Instar Structure")

    # Panel 3: Size distribution by sex
    p3 = plot_size_by_sex(model)
    plot!(p3, title = "C: Size Dimorphism")

    # Panel 4: Spatial habitat
    p4 = plot_spatial_habitat(model)
    plot!(p4, title = "D: Spatial Distribution")

    # Panel 5: Instar allometry
    p5 = plot_instar_allometry(model)
    plot!(p5, title = "E: Allometric Scaling")

    # Panel 6: Cumulative recruitment
    p6 = if model_df !== nothing && :total_recruits_settled in propertynames(model_df)
        plot(
            model_df.time,
            model_df.total_recruits_settled,
            xlabel = "Time (Days)",
            ylabel = "Count",
            title = "F: Benthic Recruitment (Instar 4)",
            label = "Cumulative Recruits",
            linewidth = 2,
            color = :teal
        )
    else
        plot(
            agent_df.time,
            agent_df.safe_mean_size_mm,
            xlabel = "Time (Days)",
            ylabel = "CW (mm)",
            title = "F: Mean Body Size",
            label = "Mean CW",
            linewidth = 2,
            color = :orange
        )
    end

    p_dash = plot(
        p1, p2, p3, p4, p5, p6,
        layout = (3, 2),
        size = (1400, 1500),
        margin = 6Plots.mm
    )

    return p_dash
end


"""
    plot_stock_recruitment(sr_data; fits=nothing, title="Snow Crab Stock-Recruitment Relationship")

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
`Plots.Plot`: Multi-panel stock-recruitment diagnostic figure.
"""
function plot_stock_recruitment(
    sr_data;
    fits = nothing,
    title::String = "Snow Crab Stock-Recruitment Relationship"
)
    df = if sr_data isa DataFrame
        sr_data
    else
        extract_stock_recruitment(sr_data)
    end

    n = nrow(df)
    if n == 0 || !(:spawners in propertynames(df)) || !(:recruits in propertynames(df))
        p = plot(
            title = title,
            legend = false,
            grid = false,
            framestyle = :none
        )
        annotate!(
            p, 0.5, 0.5,
            text("No recruitment events recorded\n(simulation duration < incubation or zero mature females)", 11, :center, :gray)
        )
        return p
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
    p1 = scatter(
        s_vals,
        r_vals,
        xlabel = "Spawning Stock Index (10³ eggs / biomass)",
        ylabel = "Benthic Recruits (Instar 4 Count)",
        title = "A: Stock-Recruitment Curves",
        label = "Simulation Observations (N=$(n))",
        markersize = 5.0,
        markerstrokewidth = 1.0,
        markerstrokecolor = :royalblue,
        color = :cornflowerblue,
        alpha = 0.85
    )

    if actual_fits.is_fitted
        bh = actual_fits.beverton_holt
        rk = actual_fits.ricker
        r_bh = [bh.predict(s) for s in s_grid]
        r_rk = [rk.predict(s) for s in s_grid]

        bh_lbl = "Beverton-Holt (R²=$(round(bh.R2, digits=2)), Rmax=$(round(Int, bh.R_max)))"
        rk_lbl = "Ricker (R²=$(round(rk.R2, digits=2)), Rmax=$(round(Int, rk.R_max)))"

        plot!(p1, s_grid, r_bh, label = bh_lbl, linewidth = 2.5, color = :seagreen)
        plot!(p1, s_grid, r_rk, label = rk_lbl, linewidth = 2.5, linestyle = :dash, color = :darkorange)
    else
        mean_s = mean(s_vals)
        mean_r = mean(r_vals)
        ref_slope = (mean_s > 0.0) ? (mean_r / mean_s) : 1.0
        plot!(p1, s_grid, ref_slope .* s_grid, label = "Linear Scaling", linewidth = 1.5, linestyle = :dot, color = :gray)
    end

    # Panel 2: Compensatory Survival (R / S vs S)
    p2 = scatter(
        s_vals,
        rps_vals,
        xlabel = "Spawning Stock Index (S)",
        ylabel = "Recruits per Spawner (R / S)",
        title = "B: Compensatory Density Dependence",
        label = "Observed Survival Ratio",
        markersize = 5.0,
        color = :crimson,
        alpha = 0.85
    )

    if actual_fits.is_fitted
        bh = actual_fits.beverton_holt
        rk = actual_fits.ricker
        s_pos = filter(s -> s > 0.0, s_grid)
        rps_bh = [bh.predict(s) / s for s in s_pos]
        rps_rk = [rk.predict(s) / s for s in s_pos]

        plot!(p2, s_pos, rps_bh, label = "BH Expectation", linewidth = 2.0, color = :seagreen)
        plot!(p2, s_pos, rps_rk, label = "Ricker Expectation", linewidth = 2.0, linestyle = :dash, color = :darkorange)
    end

    # Panel 3: Cohort Dynamics Timeseries
    x_cohort = if hasproperty(df, :year)
        df.year
    elseif hasproperty(df, :time)
        round.(df.time ./ 365.0, digits = 1)
    else
        1:n
    end

    x_lbl = hasproperty(df, :year) ? "Cohort (Year)" : "Time (Years post-hatch)"

    p3 = plot(
        x_cohort,
        s_vals,
        xlabel = x_lbl,
        ylabel = "Spawners / Recruits",
        title = "C: Spawner-Recruit Cohort Dynamics",
        label = "Spawning Stock (S)",
        linewidth = 2.0,
        marker = :circle,
        markersize = 4.0,
        color = :royalblue
    )
    plot!(
        p3,
        x_cohort,
        r_vals,
        label = "Settled Recruits (R)",
        linewidth = 2.0,
        linestyle = :dash,
        marker = :square,
        markersize = 4.0,
        color = :forestgreen
    )

    p_combined = plot(
        p1, p2, p3,
        layout = (1, 3),
        size = (1500, 480),
        plot_title = title,
        margin = 5Plots.mm
    )

    return p_combined
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
    title::String = "Snow Crab Stock-Recruitment Relationship"
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

Batch export all 9 individual diagnostic figures, including stock-recruitment relationships,
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

    # 1. Population Dynamics
    fn_dyn = joinpath(output_dir, "population_dynamics.$(ext)")
    p_dyn = plot_population_timeseries(agent_df; model_df = model_df)
    savefig(p_dyn, fn_dyn)
    push!(generated_files, fn_dyn)

    # 2. Instar Distribution
    fn_inst = joinpath(output_dir, "instar_distribution.$(ext)")
    p_inst = plot_instar_distribution(model)
    savefig(p_inst, fn_inst)
    push!(generated_files, fn_inst)

    # 3. Size Distribution by Sex
    fn_sex = joinpath(output_dir, "size_distribution_by_sex.$(ext)")
    p_sex = plot_size_by_sex(model)
    savefig(p_sex, fn_sex)
    push!(generated_files, fn_sex)

    # 4. Instar Allometry
    fn_allo = joinpath(output_dir, "instar_allometry.$(ext)")
    p_allo = plot_instar_allometry(model)
    savefig(p_allo, fn_allo)
    push!(generated_files, fn_allo)

    # 5. Spatial Habitat
    fn_spat = joinpath(output_dir, "spatial_habitat.$(ext)")
    p_spat = plot_spatial_habitat(model)
    savefig(p_spat, fn_spat)
    push!(generated_files, fn_spat)

    # 6. Maturity Ogives
    fn_mat = joinpath(output_dir, "maturity_ogive.$(ext)")
    p_mat = plot_maturity_ogives(model)
    savefig(p_mat, fn_mat)
    push!(generated_files, fn_mat)

    # 7. Reproduction Dynamics
    if model_df !== nothing
        fn_rep = joinpath(output_dir, "reproduction_dynamics.$(ext)")
        p_rep = plot_reproduction_dynamics(agent_df, model_df)
        savefig(p_rep, fn_rep)
        push!(generated_files, fn_rep)
    end

    # 8. Executive Simulation Dashboard
    fn_dash = joinpath(output_dir, "simulation_dashboard.$(ext)")
    p_dash = plot_simulation_dashboard(model, agent_df; model_df = model_df)
    savefig(p_dash, fn_dash)
    push!(generated_files, fn_dash)

    # 9. Stock-Recruitment Relationship
    fn_sr = joinpath(output_dir, "stock_recruitment.$(ext)")
    p_sr = plot_stock_recruitment(model, agent_df; model_df = model_df)
    savefig(p_sr, fn_sr)
    push!(generated_files, fn_sr)

    # 10. Macro-Ecological Population Dynamics Fits (Schaefer & Delay-Difference)
    if model_df !== nothing
        try
            ann_prod = extract_annual_production(agent_df, model_df)
            if nrow(ann_prod) >= 3
                macro_fits = fit_population_dynamics(ann_prod)
                fn_macro = joinpath(output_dir, "macro_population_dynamics.$(ext)")
                p_macro = plot_population_dynamics_fits(macro_fits)
                savefig(p_macro, fn_macro)
                push!(generated_files, fn_macro)
            end
        catch
            # Gracefully bypass if time series is too short for annual aggregation
        end

        # 11. Retrospective Pattern Analysis (when at least 5 annual data points exist)
        try
            ann_prod = extract_annual_production(agent_df, model_df)
            if nrow(ann_prod) >= 5
                retro_res = retrospective_analysis(ann_prod; n_peels = min(5, nrow(ann_prod) - 3))
                fn_retro = joinpath(output_dir, "macro_retrospective.$(ext)")
                p_retro = plot_retrospective_analysis(retro_res)
                savefig(p_retro, fn_retro)
                push!(generated_files, fn_retro)
            end
        catch
            # Gracefully bypass if time series is too short for retrospective analysis
        end

        # 12. Generalized Leslie Model & 3-Way Macro Comparison (when >= 4 annual data points exist)
        try
            ann_prod = extract_annual_production(agent_df, model_df)
            if nrow(ann_prod) >= 4
                stage_mat, catch_mat, yrs = extract_annual_stage_matrix(ann_prod)
                leslie_fit = fit_generalized_leslie_model(stage_mat, catch_mat; years = yrs)
                fn_stages = joinpath(output_dir, "stage_specific_dynamics.$(ext)")
                p_stages = plot_stage_specific_dynamics(leslie_fit)
                savefig(p_stages, fn_stages)
                push!(generated_files, fn_stages)

                macro_comp = compare_macro_models(ann_prod, leslie_fit)
                fn_comp = joinpath(output_dir, "fishable_macro_comparison.$(ext)")
                p_comp = plot_macro_model_comparison(macro_comp)
                savefig(p_comp, fn_comp)
                push!(generated_files, fn_comp)
            end
        catch
            # Gracefully bypass if error in Leslie estimation
        end
    end

    # 13. Spatial Ecosystem Snapshot using Agents.jl abmplot
    try
        fn_abm = joinpath(output_dir, "abm_spatial_summary.$(ext)")
        plot_abm_summary(model; output_path = fn_abm)
        push!(generated_files, fn_abm)
    catch
        # Gracefully bypass if headless environment lacks display context
    end

    return generated_files
end


"""
    plot_population_dynamics_fits(macro_fits; title="Snow Crab Macro-Ecological Model Fits")

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
`Plots.Plot`: Multi-panel figure.
"""
function plot_population_dynamics_fits(
    macro_fits;
    title::String = "Snow Crab Macro-Ecological Dynamic Model Fits"
)
    ann = macro_fits.annual_df
    wp = macro_fits.whole_population
    fs = macro_fits.fishable_stock

    years = ann.year
    n_tot_obs = ann.total_numbers
    n_leg_obs = ann.legal_numbers

    # Panel 1: Whole Population Abundance Trajectory
    p1 = scatter(
        years,
        n_tot_obs,
        label = "Observed Total Numbers",
        xlabel = "Simulation Year",
        ylabel = "Abundance (Total Crabs)",
        title = "A: Whole Population Abundance",
        markersize = 6.0,
        color = :dodgerblue,
        alpha = 0.85
    )
    if wp.schaefer.is_fitted
        plot!(
            p1,
            years,
            wp.schaefer.predicted,
            label = "Schaefer (R²=$(round(wp.schaefer.R2, digits=2)))",
            linewidth = 2.5,
            color = :seagreen
        )
    end
    if wp.delay_difference.is_fitted
        plot!(
            p1,
            years,
            wp.delay_difference.predicted,
            label = "Delay-Diff (R²=$(round(wp.delay_difference.R2, digits=2)))",
            linewidth = 2.5,
            linestyle = :dash,
            color = :darkorange
        )
    end

    # Panel 2: Whole Population Surplus Production
    sp_tot = ann.surplus_production_total
    p2 = scatter(
        n_tot_obs,
        sp_tot,
        label = "Observed Production",
        xlabel = "Abundance (N)",
        ylabel = "Surplus Production (ΔN + C)",
        title = "B: Whole Population Surplus Production",
        markersize = 6.0,
        color = :dodgerblue,
        alpha = 0.85
    )
    if wp.schaefer.is_fitted && wp.schaefer.K > 0.0
        n_grid = range(0.0, wp.schaefer.K * 1.15, length = 150)
        p_curve = [wp.schaefer.r * n * (1.0 - n / wp.schaefer.K) for n in n_grid]
        plot!(
            p2,
            n_grid,
            p_curve,
            label = "Schaefer (MSY=$(round(Int, wp.schaefer.MSY)))",
            linewidth = 2.5,
            color = :seagreen
        )
        vline!(
            p2,
            [wp.schaefer.N_msy],
            label = "N_MSY ($(round(Int, wp.schaefer.N_msy)))",
            linestyle = :dot,
            linewidth = 2.0,
            color = :crimson
        )
    end

    # Panel 3: Fishable Component Abundance Trajectory
    p3 = scatter(
        years,
        n_leg_obs,
        label = "Observed Legal Males",
        xlabel = "Simulation Year",
        ylabel = "Abundance (Legal Males ≥ 95mm)",
        title = "C: Fishable Stock Abundance",
        markersize = 6.0,
        color = :crimson,
        alpha = 0.85
    )
    if fs.schaefer.is_fitted
        plot!(
            p3,
            years,
            fs.schaefer.predicted,
            label = "Schaefer (R²=$(round(fs.schaefer.R2, digits=2)))",
            linewidth = 2.5,
            color = :seagreen
        )
    end
    if fs.delay_difference.is_fitted
        plot!(
            p3,
            years,
            fs.delay_difference.predicted,
            label = "Delay-Diff (R²=$(round(fs.delay_difference.R2, digits=2)))",
            linewidth = 2.5,
            linestyle = :dash,
            color = :darkorange
        )
    end

    # Panel 4: Fishable Component Surplus Production
    sp_leg = ann.surplus_production_legal
    p4 = scatter(
        n_leg_obs,
        sp_leg,
        label = "Observed Legal Production",
        xlabel = "Legal Abundance (N_legal)",
        ylabel = "Surplus Production (ΔN_leg + C_leg)",
        title = "D: Fishable Stock Surplus Production",
        markersize = 6.0,
        color = :crimson,
        alpha = 0.85
    )
    if fs.schaefer.is_fitted && fs.schaefer.K > 0.0
        n_grid_leg = range(0.0, max(10.0, fs.schaefer.K * 1.15), length = 150)
        p_curve_leg = [fs.schaefer.r * n * (1.0 - n / fs.schaefer.K) for n in n_grid_leg]
        plot!(
            p4,
            n_grid_leg,
            p_curve_leg,
            label = "Schaefer (MSY=$(round(Int, fs.schaefer.MSY)))",
            linewidth = 2.5,
            color = :seagreen
        )
        vline!(
            p4,
            [fs.schaefer.N_msy],
            label = "N_MSY ($(round(Int, fs.schaefer.N_msy)))",
            linestyle = :dot,
            linewidth = 2.0,
            color = :crimson
        )
    end

    p_combined = plot(
        p1, p2, p3, p4,
        layout = (2, 2),
        size = (1400, 900),
        plot_title = title,
        margin = 5Plots.mm
    )

    return p_combined
end

# ==============================================================================
# Retrospective Pattern Analysis & Mohn's Rho Diagnostics Plot
# ==============================================================================

"""
    plot_retrospective_analysis(retro_result; title="Snow Crab Retrospective Pattern & Mohn's ρ")

Render a 4-panel diagnostic figure visualizing retrospective patterns and Mohn's rho:
- **Panel A (Whole Population Spaghetti Plot)**: Observed total numbers, reference base
  assessment trajectory, and truncated retrospective peel trajectories terminating at each peel year.
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
`Plots.Plot`: 4-panel composite diagnostic plot.
"""
function plot_retrospective_analysis(
    retro_result;
    title::String = "Snow Crab Retrospective Pattern & Mohn's ρ"
)
    ann = retro_result.annual_df
    wp = retro_result.whole_population
    fs = retro_result.fishable_stock
    peels = retro_result.peel_indices
    years = ann.year
    n_years = length(years)

    peel_colors = [:darkorange, :purple, :teal, :magenta, :brown, :deepskyblue, :olive]

    rho_w_str = (wp.mohn_rho >= 0 ? "+" : "") * string(round(wp.mohn_rho, digits = 3))
    rho_f_str = (fs.mohn_rho >= 0 ? "+" : "") * string(round(fs.mohn_rho, digits = 3))

    # Panel 1: Whole Population Retrospective Trajectories
    p1 = scatter(
        years,
        ann.total_numbers,
        label = "Observed Total Numbers",
        xlabel = "Assessment Year",
        ylabel = "Abundance (Crab Numbers)",
        title = "A: Whole Population (Mohn's ρ = " * rho_w_str * ")",
        markersize = 5.0,
        color = :royalblue,
        alpha = 0.8
    )
    # Base model trajectory
    plot!(
        p1,
        years,
        wp.base_fit.predicted,
        label = "Base Assessment (Peel 0)",
        linewidth = 3.0,
        color = :royalblue
    )
    # Peels
    for i in eachindex(peels)
        p_idx = peels[i]
        c = peel_colors[mod1(i, length(peel_colors))]
        t_peel = n_years - p_idx
        peel_years = years[1:t_peel]
        peel_pred = wp.peel_fits[i].predicted[1:t_peel]
        plot!(
            p1,
            peel_years,
            peel_pred,
            label = "Peel -$p_idx (to Yr $(peel_years[end]))",
            linestyle = :dash,
            linewidth = 1.8,
            color = c
        )
        scatter!(
            p1,
            [peel_years[end]],
            [peel_pred[end]],
            label = "",
            markersize = 6.0,
            color = c,
            markerstrokewidth = 1.5,
            markerstrokecolor = :black
        )
    end

    # Panel 2: Fishable Stock Retrospective Trajectories
    p2 = scatter(
        years,
        ann.legal_numbers,
        label = "Observed Legal Numbers",
        xlabel = "Assessment Year",
        ylabel = "Legal Abundance (CW ≥ 95mm)",
        title = "B: Fishable Stock (Mohn's ρ = " * rho_f_str * ")",
        markersize = 5.0,
        color = :seagreen,
        alpha = 0.8
    )
    # Base model trajectory
    plot!(
        p2,
        years,
        fs.base_fit.predicted,
        label = "Base Assessment (Peel 0)",
        linewidth = 3.0,
        color = :seagreen
    )
    # Peels
    for i in eachindex(peels)
        p_idx = peels[i]
        c = peel_colors[mod1(i, length(peel_colors))]
        t_peel = n_years - p_idx
        peel_years = years[1:t_peel]
        peel_pred = fs.peel_fits[i].predicted[1:t_peel]
        plot!(
            p2,
            peel_years,
            peel_pred,
            label = "Peel -$p_idx (to Yr $(peel_years[end]))",
            linestyle = :dash,
            linewidth = 1.8,
            color = c
        )
        scatter!(
            p2,
            [peel_years[end]],
            [peel_pred[end]],
            label = "",
            markersize = 6.0,
            color = c,
            markerstrokewidth = 1.5,
            markerstrokecolor = :black
        )
    end

    # Panel 3: Relative Errors by Peel
    term_yrs = retro_result.terminal_years
    wp_re_pct = wp.rel_errors .* 100.0
    fs_re_pct = fs.rel_errors .* 100.0

    p3 = plot(
        xlabel = "Terminal Peel Year",
        ylabel = "Terminal Relative Error (%)",
        title = "C: Relative Bias by Peel vs ±20% Benchmark"
    )
    hline!(p3, [0.0], label = "", color = :black, linewidth = 1.0)
    hline!(
        p3,
        [20.0, -20.0],
        label = "±20% Acceptable Bound",
        linestyle = :dot,
        linewidth = 2.0,
        color = :crimson
    )
    scatter!(
        p3,
        term_yrs .- 0.1,
        wp_re_pct,
        label = "Whole Pop (ρ=" * rho_w_str * ")",
        markersize = 7.0,
        marker = :circle,
        color = :royalblue
    )
    plot!(
        p3,
        term_yrs .- 0.1,
        wp_re_pct,
        label = "",
        linewidth = 1.5,
        color = :royalblue
    )
    scatter!(
        p3,
        term_yrs .+ 0.1,
        fs_re_pct,
        label = "Fishable (ρ=" * rho_f_str * ")",
        markersize = 7.0,
        marker = :diamond,
        color = :seagreen
    )
    plot!(
        p3,
        term_yrs .+ 0.1,
        fs_re_pct,
        label = "",
        linewidth = 1.5,
        color = :seagreen
    )

    # Panel 4: Parameter Stability Across Data Peels
    k_whole = [wp.params[i].K for i in length(peels):-1:1]
    push!(
        k_whole,
        (retro_result.model_type === :delay_difference) ?
        wp.base_fit.s : wp.base_fit.K
    )
    data_yrs = [retro_result.terminal_years[i] for i in length(peels):-1:1]
    push!(data_yrs, years[end])

    param_label = (retro_result.model_type === :delay_difference) ?
                  "Survival s" : "Carrying Capacity K"
    p4 = plot(
        data_yrs,
        k_whole,
        label = "Whole Pop " * param_label,
        xlabel = "Available Data Horizon (Terminal Year)",
        ylabel = param_label,
        title = "D: Parameter Stability Across Sequential Reanalyses",
        linewidth = 2.5,
        marker = :circle,
        markersize = 6.0,
        color = :royalblue
    )

    k_fish = [fs.params[i].K for i in length(peels):-1:1]
    push!(
        k_fish,
        (retro_result.model_type === :delay_difference) ?
        fs.base_fit.s : fs.base_fit.K
    )
    plot!(
        p4,
        data_yrs,
        k_fish,
        label = "Fishable " * param_label,
        linewidth = 2.0,
        linestyle = :dash,
        marker = :diamond,
        markersize = 5.0,
        color = :seagreen
    )

    p_combined = plot(
        p1, p2, p3, p4,
        layout = (2, 2),
        size = (1400, 950),
        plot_title = title,
        margin = 6Plots.mm
    )

    return p_combined
end

"""
    plot_stage_specific_dynamics(fit::LeslieMatrixFit; title="Snow Crab Stage-Specific Dynamics & Transition Analysis")

Render a 4-panel diagnostic figure visualizing generalized Leslie stage-transition dynamics:
- **Panel A (Stage Trajectories)**: Historical stage abundance timeseries.
- **Panel B (Stable Stage Distribution)**: Observed average stage structure vs asymptotic right eigenvector (SSD w_1).
- **Panel C (Reproductive Value Profile)**: Left eigenvector (v_1) showing relative reproductive value across stages.
- **Panel D (Transition Elasticity Heatmap)**: Proportional elasticity matrix E indicating demographic sensitivity.

# Arguments
- `fit::LeslieMatrixFit`: Fitted stage-transition model.
- `title::String`: Main plot super-title.

# Returns
`Plots.Plot`: 4-panel diagnostic figure.
"""
function plot_stage_specific_dynamics(
    fit::LeslieMatrixFit;
    title::String = "Snow Crab Stage-Specific Dynamics & Transition Analysis"
)
    stages = fit.stages
    S = length(stages)
    years = fit.years
    T = length(years)

    # Panel 1: Stage Abundance Trajectories (Key Instars)
    p1 = plot(
        xlabel = "Year",
        ylabel = "Abundance (Numbers)",
        title = "A: Stage Abundance Dynamics Across Instars",
        legend = :outertopright,
        linewidth = 2.0
    )

    plot!(p1, years, fit.observed_stages[1, :], label = "Instar 4 (Recruit)",
          linewidth = 2.5, color = :seagreen, marker = :circle, markersize = 4.0)
    if S >= 2
        plot!(p1, years, fit.observed_stages[2, :], label = "Instar 5 (Juv)",
              linewidth = 2.0, color = :teal, marker = :square, markersize = 3.5)
    end
    for st in stages
        if st.is_legal_male
            plot!(p1, years, fit.observed_stages[st.index, :], label = st.name * " [Legal]",
                  linewidth = 2.2, linestyle = :solid, marker = :diamond, markersize = 4.0)
        end
    end
    for st in stages
        if st.is_reproductive && (st.instar == 9 || st.instar == 11)
            plot!(p1, years, fit.observed_stages[st.index, :], label = st.name * " [Repro]",
                  linewidth = 2.0, linestyle = :dash, marker = :circle, markersize = 3.5)
        end
    end

    # Panel 2: Observed Stage Distribution vs Stable Stage Distribution (SSD w_1)
    obs_mean = [mean(fit.observed_stages[i, :]) for i in 1:S]
    sum_obs = sum(obs_mean)
    obs_prop = (sum_obs > 0.0) ? (obs_mean ./ sum_obs) .* 100.0 : zeros(Float64, S)
    ssd_pct = fit.stable_stage_distribution .* 100.0

    stage_indices = collect(1:S)
    p2 = bar(
        stage_indices .- 0.18,
        obs_prop,
        bar_width = 0.35,
        label = "Observed Mean (%)",
        color = :royalblue,
        alpha = 0.85,
        xlabel = "Stage Index (1: Instar 4, 8-9: Legal Males, 13-15: Repro Females)",
        ylabel = "Proportion of Population (%)",
        title = "B: Observed vs Asymptotic Stable Stage Distribution (SSD)"
    )
    bar!(
        p2,
        stage_indices .+ 0.18,
        ssd_pct,
        bar_width = 0.35,
        label = "Theoretical SSD (w₁)",
        color = :darkorange,
        alpha = 0.85
    )

    # Panel 3: Stage Reproductive Value (RV v_1)
    rv_vals = fit.reproductive_values
    p3 = plot(
        stage_indices,
        rv_vals,
        xlabel = "Stage Index",
        ylabel = "Relative Reproductive Value (v₁ / v₁[1])",
        title = "C: Stage Reproductive Value Profile (Fisher's RV)",
        linewidth = 2.5,
        marker = :circle,
        markersize = 6.0,
        color = :crimson,
        label = "Reproductive Value"
    )
    repro_idx = [st.index for st in stages if st.is_reproductive]
    if !isempty(repro_idx)
        scatter!(
            p3,
            repro_idx,
            rv_vals[repro_idx],
            label = "Reproductive Females",
            markersize = 8.0,
            marker = :star5,
            color = :gold
        )
    end

    # Panel 4: Stage Elasticity Matrix Heatmap (E)
    E_mat = fit.elasticity_matrix
    p4 = heatmap(
        1:S,
        1:S,
        E_mat,
        xlabel = "From Stage j",
        ylabel = "To Stage i",
        title = "D: Stage Transition Elasticity Matrix (∑ E_ij = 1.0)",
        color = :viridis,
        aspect_ratio = :equal
    )

    p_combined = plot(
        p1, p2, p3, p4,
        layout = (2, 2),
        size = (1400, 950),
        plot_title = title,
        margin = 6Plots.mm
    )

    return p_combined
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
`Plots.Plot`: 4-panel comparative figure.
"""
function plot_macro_model_comparison(
    comp::MacroModelComparison;
    title::String = "Three-Way Macro-Ecological Model Comparison: Fishable Stock"
)
    yrs = comp.years

    # Panel 1: Fishable Biomass Trajectories
    p1 = plot(
        yrs,
        comp.biomass_obs_kg,
        label = "Observed Fishable Biomass",
        xlabel = "Year",
        ylabel = "Biomass (kg)",
        title = "A: Fishable Biomass Trajectories",
        linewidth = 2.5,
        marker = :circle,
        markersize = 6.0,
        color = :black
    )
    plot!(
        p1,
        yrs,
        comp.biomass_schaefer_kg,
        label = "Schaefer Logistic (R²=" * string(round(comp.r2_schaefer, digits = 2)) * ")",
        linewidth = 2.2,
        linestyle = :dash,
        color = :royalblue
    )
    plot!(
        p1,
        yrs,
        comp.biomass_delay_diff_kg,
        label = "Delay-Difference (R²=" * string(round(comp.r2_delay_diff, digits = 2)) * ")",
        linewidth = 2.2,
        linestyle = :dashdot,
        color = :seagreen
    )
    plot!(
        p1,
        yrs,
        comp.biomass_leslie_kg,
        label = "Leslie Matrix (R²=" * string(round(comp.r2_leslie, digits = 2)) * ")",
        linewidth = 2.2,
        linestyle = :solid,
        color = :darkorange
    )

    # Panel 2: Model Residuals Across Years
    res_sch = comp.biomass_obs_kg .- comp.biomass_schaefer_kg
    res_dd = comp.biomass_obs_kg .- comp.biomass_delay_diff_kg
    res_les = comp.biomass_obs_kg .- comp.biomass_leslie_kg

    p2 = hline(
        [0.0],
        linestyle = :dot,
        color = :gray,
        linewidth = 1.5,
        label = "",
        xlabel = "Year",
        ylabel = "Residual (Obs - Pred, kg)",
        title = "B: Prediction Residuals Across Years"
    )
    plot!(p2, yrs, res_sch, label = "Schaefer", linewidth = 2.0, marker = :square, color = :royalblue)
    plot!(p2, yrs, res_dd, label = "Delay-Diff", linewidth = 2.0, marker = :diamond, color = :seagreen)
    plot!(p2, yrs, res_les, label = "Leslie Matrix", linewidth = 2.0, marker = :circle, color = :darkorange)

    # Panel 3: Biomass Variations & Coefficient of Variation
    cat_names = ["Observed", "Schaefer", "Delay-Diff", "Leslie Matrix"]
    cvs = [comp.cv_obs, comp.cv_schaefer, comp.cv_delay_diff, comp.cv_leslie]
    sds = [comp.sd_obs, comp.sd_schaefer, comp.sd_delay_diff, comp.sd_leslie]

    x_cats = 1:4
    p3 = bar(
        x_cats .- 0.18,
        cvs,
        bar_width = 0.35,
        label = "Coeff of Variation (CV = σ / μ)",
        color = :royalblue,
        alpha = 0.85,
        xticks = (x_cats, cat_names),
        ylabel = "Variation Metric",
        title = "C: Fishable Biomass Variability & Dispersion"
    )
    bar!(
        p3,
        x_cats .+ 0.18,
        sds,
        bar_width = 0.35,
        label = "Standard Deviation σ (kg)",
        color = :mediumpurple,
        alpha = 0.85
    )

    # Panel 4: Model Performance & AIC Scores
    model_labels = ["Schaefer", "Delay-Diff", "Leslie Matrix"]
    r2_vals = [comp.r2_schaefer, comp.r2_delay_diff, comp.r2_leslie]
    rmse_vals = [comp.rmse_schaefer, comp.rmse_delay_diff, comp.rmse_leslie]
    aic_vals = [comp.aic_schaefer, comp.aic_delay_diff, comp.aic_leslie]

    x_models = 1:3
    p4 = bar(
        x_models .- 0.20,
        r2_vals,
        bar_width = 0.25,
        label = "Goodness of Fit R²",
        color = :seagreen,
        alpha = 0.85,
        xticks = (x_models, model_labels),
        ylabel = "Metric Score",
        title = "D: Model Performance & AIC Ranking (Pref: " * uppercase(string(comp.preferred_model)) * ")"
    )
    bar!(
        p4,
        x_models,
        rmse_vals,
        bar_width = 0.25,
        label = "RMSE (kg)",
        color = :royalblue,
        alpha = 0.85
    )
    min_aic = minimum(aic_vals)
    delta_aics = aic_vals .- min_aic
    bar!(
        p4,
        x_models .+ 0.20,
        delta_aics,
        bar_width = 0.25,
        label = "ΔAIC (0 = Preferred)",
        color = :crimson,
        alpha = 0.85
    )

    p_combined = plot(
        p1, p2, p3, p4,
        layout = (2, 2),
        size = (1400, 950),
        plot_title = title,
        margin = 6Plots.mm
    )

    return p_combined
end

# ==============================================================================
# Agents.jl Native Visualizations and Animation Tools
# ==============================================================================

"""
    crab_color(crab::SnowCrab)

Determine the discrete visualization color for an individual snow crab based
on its life history stage, sex, maturity, and reproductive status:
- Larval instars (`zoea`, `megalopa`): Amber (`#E69F00`)
- Benthic immature females: Pink (`#CC79A7`)
- Benthic immature males: Sky blue (`#56B4E9`)
- Adult ovigerous females carrying egg clutches: Crimson (`#D55E00`)
- Adult non-ovigerous females: Purple (`#785EF0`)
- Commercial legal-sized adult males (CW >= 95 mm): Dark blue (`#003B73`)
- Sublegal adult males: Royal blue (`#0072B2`)

# Arguments
- `crab::SnowCrab`: Individual crab agent.

# Returns
`String`: Hex color code for graphical representation.
"""
function organism_color(org::Organism)
    if org.stage === zoea || org.stage === megalopa || org.stage === egg
        return "#E69F00"
    elseif org.stage === juvenile
        return (org.sex === female) ? "#CC79A7" : "#56B4E9"
    elseif org.stage === adult
        if is_ovigerous(org)
            return "#D55E00"
        elseif org.sex === female
            return "#785EF0"
        else
            return is_legal_male(org) ? "#003B73" : "#0072B2"
        end
    end
    return "#888888"
end

"""
    crab_size(crab::SnowCrab; min_size=4.0, max_size=24.0)

Compute marker display size in points scaled from individual carapace width
(CW in mm):
``\\text{size} = \\text{clamp}(CW / 4.0, min\\_size, max\\_size)``

# Arguments
- `crab::SnowCrab`: Individual crab agent.
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
    return clamp(org.size / 4.0, min_size, max_size)
end

"""
    crab_marker(crab::SnowCrab)

Select scatter marker shape based on sex classification:
- Female (`female`): Circle (`:circle`)
- Male (`male`): Square (`:rect`)
- Undifferentiated larvae (`undifferentiated`): Diamond (`:diamond`)

# Arguments
- `crab::SnowCrab`: Individual crab agent.

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
    crab_heatarray(model, field::Symbol=:temperature_field)

Extract a 2D matrix from the ABM spatial environment to serve as a background
heatarray for `Agents.abmplot` or `Agents.abmvideo`.

# Arguments
- `model`: Snow crab simulation model.
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
        return model.temperature_field
    end
end

"""
    plot_abm_summary(model; heat_field=:temperature, output_path=nothing,
                     figure_size=(800, 800), title="Snow Crab Spatial Ecosystem")

Render a publication-ready spatial ABM summary snapshot of the snow crab population
using `Agents.abmplot` over the specified environmental landscape (temperature or food).

# Arguments
- `model`: Configured snow crab simulation model.
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
    title::String = "Snow Crab Spatial Ecosystem"
)
    field_sym = Symbol(heat_field)
    cbar_label = if field_sym in (:food, :food_availability, :food_availability_field)
        "Benthic Food Density"
    else
        "Bottom Temperature (°C)"
    end

    heat_fn = m -> crab_heatarray(m, field_sym)
    n_crabs = nagents(model)
    sim_time = hasproperty(model, :current_time) ? model.current_time : 0.0

    full_title = "$(title) (Day $(round(sim_time, digits=1)), Living Crabs: $(n_crabs))"

    fig, ax, abmobs = Agents.abmplot(
        model;
        agent_color = crab_color,
        agent_size = crab_size,
        agent_marker = crab_marker,
        heatarray = heat_fn,
        add_colorbar = true,
        colorbar_label = cbar_label,
        figure = (size = figure_size,),
        axis = (title = full_title, titlealign = :left)
    )

    if output_path !== nothing
        mkpath(dirname(abspath(output_path)))
        CairoMakie.save(output_path, fig)
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
function animate_organism_simulation(model; filepath="organism_animation.mp4",
                                    frames=120, dt=2, framerate=15,
                                    heat_field=:temperature, figure_size=(700, 700),
                                    title="Organism Ecosystem Dynamics")
    Agents.abmvideo(model;
        filepath = filepath,
        frames = frames,
        dt = dt,
        framerate = framerate,
        heatarray = heat_field,
        figure_size = figure_size,
        title = title,
        agent_color = organism_color,
        agent_size = organism_size,
        agent_marker = organism_marker
    )
end
function plot_functional_relationships(
    params_list::Vector{IBMParameters};
    type::Symbol = :maturity,
    label_prefix::String = "Set",
    title::Union{String, Nothing} = nothing,
    xlabel::Union{String, Nothing} = nothing,
    ylabel::Union{String, Nothing} = nothing
)
    p = plot(legend = :outertopright, size = (800, 600))

    if type === :maturity
        # CW range for plotting: from 10mm to 250mm
        cw_vals = exp.(range(log(10.0), log(250.0), length=100))
        for (i, params) in enumerate(params_list)
            # We use deterministic mode for clean functional curves
            probs = [maturity(cw; params=params, deterministic=true) for cw in cw_vals]
            plot!(p, cw_vals, probs, label="$(label_prefix)$i", linewidth=2.5)
        end
        title === nothing ? title = "Maturity Ogives (Functional Space)" : title nothing
        xlabel === nothing ? xlabel = "Carapace Width (mm)" : xlabel nothing
        ylabel === nothing ? ylabel = "P(Mature)" : ylabel nothing

    elseif type === :fecundity
        cw_vals = collect(10.0:5.0:250.0)
        for (i, params) in enumerate(params_list)
            # Using primiparous method for standard comparison
            eggs = [fecundity(cw; fecundity_parameterization="moncton", is_primiparous=true, params=params, deterministic=true) for cw in cw_vals]
            plot!(p, cw_vals, eggs, label="$(label_prefix)$i", linewidth=2.5)
        end
        title === nothing ? title = "Fecundity Curves (Functional Space)" : title nothing
        xlabel === nothing ? xlabel = "Carapace Width (mm)" : xlabel nothing
        ylabel === nothing ? ylabel = "Total Eggs" : ylabel nothing

    elseif type === :selectivity
        size_vals = collect(20.0:2.0:180.0)
        for (i, params) in enumerate(params_list)
            probs = selectivity.(size_vals; params=params)
            plot!(p, size_vals, probs, label="$(label_prefix)$i", linewidth=2.5)
        end
        title === nothing ? title = "Fishery Selectivity (Functional Space)" : title nothing
        xlabel === nothing ? xlabel = "Carapace Width (mm)" : xlabel nothing
        ylabel === nothing ? ylabel = "Retention Probability" : ylabel nothing

    else
        error("Unsupported functional type: $type. Use :maturity, :fecundity, or :selectivity.")
    end

    plot!(p, title=title, xlabel=xlabel, ylabel=ylabel)
    return p
end
