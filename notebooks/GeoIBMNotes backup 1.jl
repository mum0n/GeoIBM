### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ a0000001-0001-0001-0001-000000000002
begin
    # Ensure package environment is active and load core libraries
    using Pkg
    Pkg.activate(normpath(joinpath(@__DIR__, "..")))
    
    using GeoIBM
    using DataFrames
    import CairoMakie: Figure, Axis, heatmap!, lines!, scatter!, barplot!, hist!, vlines!, hlines!, axislegend, Colorbar, Theme, set_theme!, AxisAspect, Label, Point2f, linesegments!
    using PlutoUI
    using Random
    using Statistics
    using Printf
    using TOML
    using LinearAlgebra
    
    # Configure Makie plot aesthetics for high-resolution notebook display
    set_theme!(
        Theme(
            fontsize = 12,
            font = "sans-serif",
            Axis = (
                xgridvisible = true,
                ygridvisible = true,
                xgridcolor = (:gray, 0.18),
                ygridcolor = (:gray, 0.18),
                titlesize = 13,
                titlealign = :left
            )
        )
    )
    "Environment active & GeoIBM loaded successfully."
end

# ╔═╡ a0000001-0001-0001-0001-000000000001
md"""
# GeoIBM: Spatial Individual-Based Life-History Modeling (*Chionoecetes opilio*)
### Interactive Scientific Simulation, Macro-Ecological Dynamics, and Calibration Workflow

This **Pluto.jl** reactive notebook provides an end-to-end research environment for simulating,
visualizing, and calibrating benthic-demersal snow crab (*Chionoecetes opilio*) populations
using `configs/snowcrab.toml`.

#### Scope and Architectural Highlights:
1. **Biological Realism**: 20 discrete life-history stages (0–19: brooded eggs, zoea I–II,
   megalopa, immature instars IV–IX, through terminal molt stages up to instar XVI).
2. **Empirical Process Realism**:
   - Bioenergetic allometry: Standard metabolic rate $M_0 \propto W^{0.75}$ with Arrhenius/Q10
     thermal modulation.
   - Stage-dependent growth: Hiatt linear increment matrices validated against Sainte-Marie
     and Alunno-Bruscia empirical growth schedules.
   - Phenological gating: Degree-day accumulation and photoperiod-cued spring molting windows.
   - Dimorphic maturity ogives: Sex-specific log-logistic transition probabilities ($L_{50} \approx 91.8$ mm
     males, $51.6$ mm females).
3. **Reactive Multi-Panel Dashboards**:
   - Interactive parameter steering via reactive controls.
   - Dynamic 2D spatial habitat and agent distribution map.
   - Real-time demographic trajectories (abundance, biomass, cumulative fishery yield).
   - Population size-frequency spectra (dimorphism & legal size selectivity).
   - Empirical vs theoretical cumulative maturity ogives.
4. **Macro-Ecological Synthesis & Benchmarks**:
   - Schaefer logistic surplus production fitting.
   - Deriso-Schnute delay-difference dynamic model estimation.
   - Generalized Leslie / Lefkovitch stage-structured matrix modeling.
   - Retrospective peeling with Mohn's $\rho$ calculation for stock assessment bias.
5. **Systematic Multi-Attribute Calibration**:
   - Parameter exploration and multi-criteria evaluation (`evaluate_population_criteria`)
     for persistence, bounded carrying capacity, size stability, and growth fidelity.
"""

# ╔═╡ a0000001-0001-0001-0001-000000000003
md"""
## 1. Snow Crab Configuration & Interactive Steering

The simulation parameters are ingested directly from [`configs/snowcrab.toml`](file:///c:/home/jae/projects/GeoIBM/configs/snowcrab.toml).
Use the reactive widgets below to adjust external environmental drivers and fishery pressure.
Every modification automatically ripples through all downstream spatial, demographic,
and macro-ecological analyses.
"""

# ╔═╡ a0000001-0001-0001-0001-000000000004
begin
    # Ingest baseline configuration file
    snowcrab_toml_path = normpath(joinpath(@__DIR__, "..", "configs", "snowcrab.toml"))
    raw_config = TOML.parsefile(snowcrab_toml_path)
    
    # Extract baseline settings
    base_env = raw_config["environment"]
    base_sim = raw_config["simulation"]
    base_fish = raw_config["fishery"]
    
    md"""
    Loaded baseline config from: `$(basename(snowcrab_toml_path))`
    - Default Grid: $(base_sim["grid_nx"]) × $(base_sim["grid_ny"]) cells
    - Default Carrying Capacity $K$: $(base_env["benthic_carrying_capacity"]) organisms
    - Baseline Bottom Temp: $(base_env["temp_base"]) °C (Amplitude: $(base_env["temp_amplitude"]) °C)
    """
end

# ╔═╡ a0000001-0001-0001-0001-000000000005
md"""
### Interactive Controls & Area of Interest (AOI) Geographic Map

Select the geographic center $(\lambda, \phi)$ on the Scotian Shelf / Laurentian Channel domain.
The chosen coordinates inform the regional bathymetric and thermal climatology lookup,
updating the simulated environmental drivers and Habitat Suitability Index (HSI) in real time:
"""

# ╔═╡ a0000001-0001-0001-0001-000000000030
@bind aoi_coords HTML(aoi_coords_widget(default_lon = -63.5, default_lat = 44.5))

# ╔═╡ a0000001-0001-0001-0001-000000000033
begin
    # Environmental Hydrographic Lookup & Real-time Habitat Suitability Index (HSI)
    # Handle initial reactive binding where aoi_coords may be nothing or missing
    aoi_lon = if aoi_coords !== nothing && !ismissing(aoi_coords) && hasproperty(aoi_coords, :lon)
        Float64(aoi_coords.lon)
    else
        -63.5
    end
    
    aoi_lat = if aoi_coords !== nothing && !ismissing(aoi_coords) && hasproperty(aoi_coords, :lat)
        Float64(aoi_coords.lat)
    else
        44.5
    end
    
    # Regional shelf bathymetric and thermal climatology formulation
    aoi_depth = max(25.0, 180.0 + 80.0 * sin(0.8 * aoi_lon) + 60.0 * cos(1.1 * aoi_lat))
    aoi_t_bottom = 2.0 + 2.5 * exp(-((aoi_depth - 150.0)/60.0)^2) + 0.3 * (45.0 - aoi_lat)
    
    # Compute HSI using GeoIBM empirical suitability formulation
    computed_hsi = calculate_habitat_suitability(aoi_t_bottom, aoi_depth)
end

# ╔═╡ a0000001-0001-0001-0001-000000000006
@bind steering_controls HTML(steering_controls_widget(default_interval = 5))

# ╔═╡ a0000001-0001-0001-0001-000000000015
md"""
## 2. Model Initialization & Simulation Execution

We construct the model parameters using `load_ibm_configuration` layered with our
interactive overrides. The simulation advances in discrete daily steps ($\Delta t = 1.0$ day),
recording individual agent transitions, energetic budgets, molting events, and annual fishery catch.
"""

# ╔═╡ a0000001-0001-0001-0001-000000000016
sim_results = begin
    # 1. Ingest baseline dictionary from snowcrab.toml
    cfg = load_ibm_configuration(snowcrab_toml_path)
    
    # 2. Extract settings from unified interactive steering control with initial defaults
    valid_ctrl = steering_controls !== nothing && !ismissing(steering_controls)
    
    sim_status = (valid_ctrl && hasproperty(steering_controls, :sim_status)) ?
        String(steering_controls.sim_status) : "stopped"
    current_day = (valid_ctrl && hasproperty(steering_controls, :current_day)) ?
        Int(steering_controls.current_day) : 0
    sim_years = (valid_ctrl && hasproperty(steering_controls, :sim_years)) ?
        Int(steering_controls.sim_years) : 2
    grid_res = (valid_ctrl && hasproperty(steering_controls, :grid_res)) ?
        Int(steering_controls.grid_res) : 50
    
    # Cap simulated horizon to current interactive day when stopped/paused/running
    total_horizon_days = round(Int, sim_years * 365)
    effective_days = clamp(current_day, 0, total_horizon_days)
    
    cfg[:days] = max(1, total_horizon_days)
    cfg[:grid] = (grid_res, grid_res)
    cfg[:temp_base] = (valid_ctrl && hasproperty(steering_controls, :temp_base)) ?
        Float64(steering_controls.temp_base) : 5.0
    cfg[:temp_amplitude] = (valid_ctrl && hasproperty(steering_controls, :temp_amplitude)) ?
        Float64(steering_controls.temp_amplitude) : 4.0
    cfg[:food_max] = (valid_ctrl && hasproperty(steering_controls, :food_max)) ?
        Float64(steering_controls.food_max) : 10.0
    cfg[:benthic_carrying_capacity] = (valid_ctrl && hasproperty(steering_controls, :benthic_carrying_capacity)) ?
        Float64(steering_controls.benthic_carrying_capacity) : 6000.0
    cfg[:fishing_probability_daily] = (valid_ctrl && hasproperty(steering_controls, :fishing_rate)) ?
        Float64(steering_controls.fishing_rate) : 0.002
    legal_size_val = (valid_ctrl && hasproperty(steering_controls, :legal_size)) ?
        Float64(steering_controls.legal_size) : 95.0
    cfg[:legal_size_male] = legal_size_val
    
    # Apply modular biological process toggles
    cfg[:use_energetics] = (valid_ctrl && hasproperty(steering_controls, :use_energetics)) ?
        Bool(steering_controls.use_energetics) : true
    cfg[:use_thermal_stress] = (valid_ctrl && hasproperty(steering_controls, :use_thermal_stress)) ?
        Bool(steering_controls.use_thermal_stress) : true
    cfg[:use_cannibalism] = (valid_ctrl && hasproperty(steering_controls, :use_cannibalism)) ?
        Bool(steering_controls.use_cannibalism) : false
    cfg[:use_senescence] = (valid_ctrl && hasproperty(steering_controls, :use_senescence)) ?
        Bool(steering_controls.use_senescence) : true
    cfg[:use_movement] = (valid_ctrl && hasproperty(steering_controls, :use_movement)) ?
        Bool(steering_controls.use_movement) : true
    cfg[:use_directed_movement] = cfg[:use_movement]
    cfg[:use_reproduction] = (valid_ctrl && hasproperty(steering_controls, :use_reproduction)) ?
        Bool(steering_controls.use_reproduction) : true
    
    # Preserve scale consistency for the smaller interactive grid
    scale_factor = (grid_res * grid_res) / (1000.0 * 1000.0)
    adjusted_seed_agents = max(60, round(Int, 600 * sqrt(scale_factor)))
    cfg[:n_agents] = adjusted_seed_agents
    
    # Explicit override tracking to ensure grid synchrony
    explicit_keys = Set([
        :days, :grid, :temp_base, :temp_amplitude,
        :food_max, :benthic_carrying_capacity,
        :fishing_probability_daily, :legal_size_male, :n_agents,
        :use_energetics, :use_thermal_stress, :use_cannibalism,
        :use_senescence, :use_movement, :use_directed_movement,
        :use_reproduction
    ])
    sync_grid_dimensions!(cfg; explicit = explicit_keys)
    
    # 3. Construct structured IBMParameters
    params = IBMParameters(cfg)
    
    # 4. Initialize agent-based model
    model = initialize_model(
        n_agents = adjusted_seed_agents,
        grid = (grid_res, grid_res),
        dt = 1.0,
        params = params,
        seed = 1234
    )
    
    # 5. Define agent and model observation collectors
    adata = default_agent_observations(params)
    mdata = [
        :total_eggs_spawned,
        :total_recruits_settled,
        :total_catch_numbers,
        :total_catch_biomass_kg
    ]
    
    # 6. Execute simulation up to current interactive day
    # When notebook initializes, sim_status == "stopped" & effective_days == 0
    agent_df, model_df = run_simulation(
        model, effective_days;
        adata = adata,
        mdata = mdata
    )
    
    # 7. Extract annual demographic and fishery production table
    annual_df = extract_annual_production(agent_df, model_df)
    
    (
        model = model,
        params = params,
        agent_df = agent_df,
        model_df = model_df,
        annual_df = annual_df,
        sim_days = effective_days,
        target_horizon_days = total_horizon_days,
        sim_years = sim_years,
        sim_status = sim_status,
        legal_size = legal_size_val
    )
end

# ╔═╡ a0000001-0001-0001-0001-000000000018
md"""
## 3. Interactive Makie Dashboard Components

The panels below recreate the exact layouts, analytical formulations, and styling of the
native `GLMakie` dashboard (`src/gui.jl`) in **Pluto.jl**:
- **Spatial Habitat & Organism Distribution**: Ambient thermal field ($\Omega$) with agent particles sized by carapace width ($L_i$) and colored by demographic stage / sex via `organism_color`.
- **Abundance Trajectories**: Dynamic population trajectory on a logarithmic scale ($\log_{10}(\text{count})$) for Total, Mature Females, Mature Males, and Legal Fishable Males.
- **Stage Abundance Structure**: Instar stage demographics visualized as paired lollipop stems (`stem!`), split 50/50 for unsexed egg/aggregate stages and by realized counts for individuated stages.
- **Size Frequency by Sex and Maturity**: Normalized Gaussian kernel densities (`kde_density`, $h = 3.0\text{ mm}$, `scale = :probability`) comparing immature vs mature components of both sexes alongside the legal fishery cut ($L_{\text{legal}}$).
- **Maturity Ogives & Fishery Selectivity**: Empirical maturity fractions and legal harvest retention contrasted against theoretical single-ecdysis, cumulative, and knife-edge selectivity curves.
- **Biomass, Catch & Macro-Models**: Fishable stock biomass ($B_{\text{legal}}$) and annual catch alongside fitted surplus production (Schaefer), delay-difference (Deriso-Schnute), and Leslie matrix trajectories.
- **Stock-Recruitment Relationship**: Realized spawner-recruit pairs $(\log_{10}(S), \log_{10}(R))$ overlaid with parametric Ricker and Beverton-Holt density-dependent recruitment curves.
"""

# ╔═╡ a0000001-0001-0001-0001-000000000019
plot_dashboard_spatial(
    sim_results.model, sim_results.params;
    sim_days = sim_results.sim_days,
    horizon_days = sim_results.target_horizon_days,
    sim_status = sim_results.sim_status
)

# ╔═╡ a0000001-0001-0001-0001-000000000020
plot_dashboard_abundance(
    sim_results.agent_df;
    sim_days = sim_results.sim_days
)

# ╔═╡ a0000001-0001-0001-000000000034
plot_dashboard_stages(
    sim_results.model,
    sim_results.params
)

# ╔═╡ a0000001-0001-0001-0001-000000000021
plot_dashboard_size_density(
    sim_results.model,
    sim_results.params
)

# ╔═╡ a0000001-0001-0001-0001-000000000022
plot_dashboard_ogive(
    sim_results.model,
    sim_results.params
)

# ╔═╡ a0000001-0001-0001-0001-000000000035
plot_dashboard_macro(
    sim_results.agent_df,
    sim_results.model_df,
    sim_results.annual_df
)

# ╔═╡ a0000001-0001-0001-0001-000000000036
plot_dashboard_stock_recruitment(
    sim_results.annual_df;
    sim_days = sim_results.sim_days
)

# ╔═╡ a0000001-0001-0001-0001-000000000023
md"""
## 4. Macro-Ecological Population Dynamic Visuals

We fit discrete-time aggregate assessment models to the annual demographic output:
- **Schaefer Logistic Surplus Production Model**:
  ```math
  N_{t+1} = N_t + r N_t \left(1 - \frac{N_t}{K}\right) - C_t
  ```
- **Deriso-Schnute Delay-Difference Dynamic Model**:
  ```math
  N_{t+1} = (1 + \rho) s N_t - \rho s^2 N_{t-1} + \alpha S_{t-k} - C_t
  ```
- **Generalized Leslie (Lefkovitch) Stage Transition Matrix**:
  ```math
  \mathbf{n}_{t+1} = \mathbf{A} \mathbf{n}_t - \mathbf{C}_t
  ```
"""

# ╔═╡ a0000001-0001-0001-0001-000000000024
macro_results = begin
    ann_df = sim_results.annual_df
    if nrow(ann_df) >= 3
        # Fit Schaefer and Delay-Difference dynamics
        macro_fits = fit_population_dynamics(ann_df; lag = 1)
        
        # Fit Generalized Leslie matrix
        stage_mat, catch_mat, yrs = extract_annual_stage_matrix(ann_df)
        leslie_fit = fit_generalized_leslie_model(stage_mat, catch_mat; years = yrs)
        
        # Compare models on legal fishable stock
        comparison = compare_macro_models(ann_df, leslie_fit; lag = 1)
        
        (
            macro_fits = macro_fits,
            leslie_fit = leslie_fit,
            comparison = comparison,
            has_fits = true
        )
    else
        (
            macro_fits = nothing,
            leslie_fit = nothing,
            comparison = nothing,
            has_fits = false
        )
    end
end

# ╔═╡ a0000001-0001-0001-0001-000000000025
begin
    if macro_results.has_fits
        # Render 4-panel macro dynamic diagnostic figure
        p_macro = plot_population_dynamics_fits(macro_results.macro_fits)
        p_macro
    else
        fig_placeholder = Figure(size = (600, 150))
        Label(fig_placeholder[1, 1],
              "Minimum 3 annual time steps required for macro-model fitting (current: $(nrow(sim_results.annual_df))).\n" *
              "Advance the simulation duration in the transport panel to 3 or more years.",
              fontsize = 13, color = :darkred)
        fig_placeholder
    end
end

# ╔═╡ a0000001-0001-0001-0001-000000000026
begin
    if macro_results.has_fits
        # Three-way comparative evaluation (Schaefer vs Delay-Diff vs Leslie)
        p_comp = plot_macro_model_comparison(macro_results.comparison)
        p_comp
    else
        fig_comp_empty = Figure(size = (600, 120))
        Label(fig_comp_empty[1, 1], "Three-way comparison awaiting >= 3 annual periods.",
              color = :gray)
        fig_comp_empty
    end
end

# ╔═╡ a0000001-0001-0001-0001-000000000027
md"""
## 5. Tabular Summaries

This section presents the tabular outputs of the simulation and calibration evaluations
following all graphical panels and model comparisons.

### 5.1 Annual Production & Assessment Summary Table
"""

# ╔═╡ a0000001-0001-0001-0001-000000000017
sim_results.annual_df

# ╔═╡ a0000001-0001-0001-0001-000000000029
md"""
### 5.2 Multi-Attribute Calibration & Stopping Criteria Table

The orchestrator (`scripts/organism_orchestrator.jl`) evaluates candidate parameterizations
against five simultaneous biological criteria:
- **Persistence**: Population remains viable ($N_{\text{living}} \ge N_{\text{min}}$).
- **Bounds**: Population does not exceed carrying capacity ceilings ($N_{\text{living}} \le N_{\text{max}}$).
- **Size Structure Stability**: Drift in mean carapace width does not exceed tolerance.
- **Maturity Ogive Fidelity**: Empirical maturity schedule matches theoretical logistic curves ($\text{RMSE} \le 0.35$).
- **Growth Schedule Fidelity**: Stage sizes match empirical reference medians ($\text{RMSE} \le 15.0$ mm).
"""

# ╔═╡ a0000001-0001-0001-0001-000000000028
criteria_eval = begin
    crit = evaluate_population_criteria(sim_results.model, sim_results.agent_df)
    
    crit_table = DataFrame(
        Criterion = [
            "1. Population Persistence",
            "2. Population Bounds",
            "3. Steady Size Structure",
            "4. Maturity Ogive Fidelity",
            "5. Growth Schedule Fidelity",
            "Overall Composite Status"
        ],
        Metric_Value = [
            "$(crit.n_living) organisms (floor: $(crit.n_min))",
            "$(crit.n_living) organisms (ceiling: $(crit.n_max))",
            "$(round(crit.size_drift * 100.0, digits = 1))% drift",
            "RMSE = $(round(crit.maturity_rmse, digits = 3))",
            "RMSE = $(round(crit.growth_rmse, digits = 1)) mm",
            "Loss Score = $(round(crit.composite_score, digits = 2))"
        ],
        Status = [
            crit.is_persistent ? "PASS" : "FAIL",
            crit.is_bounded ? "PASS" : "FAIL",
            crit.is_size_steady ? "PASS" : "FAIL",
            crit.is_maturity_realistic ? "PASS" : "FAIL",
            crit.is_growth_realistic ? "PASS" : "FAIL",
            crit.passed_all ? "ALL CRITERIA PASSED" : "CALIBRATION INCOMPLETE"
        ]
    )
    crit_table
end

# ╔═╡ Cell order:
# ╟─a0000001-0001-0001-0001-000000000001
# ╟─a0000001-0001-0001-0001-000000000002
# ╟─a0000001-0001-0001-0001-000000000003
# ╟─a0000001-0001-0001-0001-000000000004
# ╟─a0000001-0001-0001-0001-000000000005
# ╟─a0000001-0001-0001-0001-000000000030
# ╟─a0000001-0001-0001-0001-000000000033
# ╟─a0000001-0001-0001-0001-000000000006
# ╟─a0000001-0001-0001-0001-000000000015
# ╟─a0000001-0001-0001-0001-000000000016
# ╟─a0000001-0001-0001-0001-000000000018
# ╟─a0000001-0001-0001-0001-000000000019
# ╟─a0000001-0001-0001-0001-000000000020
# ╟─a0000001-0001-0001-0001-000000000034
# ╟─a0000001-0001-0001-0001-000000000021
# ╟─a0000001-0001-0001-0001-000000000022
# ╟─a0000001-0001-0001-0001-000000000035
# ╟─a0000001-0001-0001-0001-000000000036
# ╟─a0000001-0001-0001-0001-000000000023
# ╟─a0000001-0001-0001-0001-000000000024
# ╟─a0000001-0001-0001-0001-000000000025
# ╟─a0000001-0001-0001-0001-000000000026
# ╟─a0000001-0001-0001-0001-000000000027
# ╟─a0000001-0001-0001-0001-000000000017
# ╟─a0000001-0001-0001-0001-000000000029
# ╟─a0000001-0001-0001-0001-000000000028
