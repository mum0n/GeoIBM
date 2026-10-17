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
    using JLD2
    
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

*Environmental data lakehouse:* Bathymetry and ocean hydrography are resolved via
`GeoData.lakehouse_dir("GeoIBM")`, honoring `ENV["GEOIBM_STORE_ROOT"]`,
`ENV["GEODATA_LAKEHOUSE"]`, `LocalPreferences.toml`, and the workspace fallback.
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
    t_offset_val = get(base_env, "temp_offset", get(base_env, "temp_base", 0.0))
    t_mult_val = get(base_env, "temp_multiplier", get(base_env, "temp_amplitude", 1.0))
    
    md"""
    Loaded baseline config from: `$(basename(snowcrab_toml_path))`
    - Bathymetry Product: $(get(base_env, "bathymetry_source", "nwa_bathymetry")) (loaded via `GeoData.geoload`)
    - Default Grid: $(base_sim["grid_nx"]) × $(base_sim["grid_ny"]) cells
    - Default Carrying Capacity $K$: $(base_env["benthic_carrying_capacity"]) organisms
    - Temperature Offset: $(t_offset_val) °C (Seasonal Multiplier: $(t_mult_val)×)
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

# ╔═╡ a0000001-0001-0001-0001-000000000006
@bind steering_sliders HTML(parameter_steering_sliders_widget(default_interval = 10, default_years = 30))

# ╔═╡ a0000001-0001-0001-0001-000000000033
begin
    # Environmental Hydrographic Lookup & Real-time Habitat Suitability Index (HSI)
    # Handle initial reactive binding where aoi_coords may be a Dict, NamedTuple, or missing
    aoi_lon = Float64(extract_ctrl_val(aoi_coords, :lon, -63.5))
    aoi_lat = Float64(extract_ctrl_val(aoi_coords, :lat, 44.5))
    
    # Estimate seabed depths over surrounding ±500 m area (1000 m × 1000 m domain)
    # using bathymetry source (GeoData store with linear interpolation between points)
    aoi_bathy = estimate_aoi_bathymetry(aoi_lon, aoi_lat; span_meters = 1000.0, n_points = 25)
    aoi_depth = aoi_bathy.mean_depth
    aoi_depth_grid = aoi_bathy.depths
    
    # Regional shelf bathymetric and thermal climatology formulation
    aoi_t_bottom = 2.0 + 2.5 * exp(-((aoi_depth - 150.0)/60.0)^2) + 0.3 * (45.0 - aoi_lat)
    
    # Compute HSI across surrounding area depths with interactive modifiers
    raw_hsi_grid = calculate_habitat_suitability.(aoi_t_bottom, aoi_depth_grid)
    raw_hsi = mean(raw_hsi_grid)
    hab_off = Float64(extract_ctrl_val(steering_sliders, :habitat_offset, 0.0))
    hab_mult = Float64(extract_ctrl_val(steering_sliders, :habitat_multiplier, 1.0))
    computed_hsi_grid = clamp.(raw_hsi_grid .* hab_mult .+ hab_off, 0.0, 1.0)
    computed_hsi = mean(computed_hsi_grid)

    md"""
    **Local AOI Environmental Summary:**
    - Mean Seabed Depth: **$(round(aoi_depth, digits = 1)) m**
    - Bottom Temperature: **$(round(aoi_t_bottom, digits = 2)) °C**
    - Realized Habitat Suitability Index (HSI): **$(round(computed_hsi, digits = 4))** (Baseline: $(round(raw_hsi, digits = 4)))
    """
end

# ╔═╡ a0000001-0001-0001-0001-000000000037
plot_bottom_temperature_cycle(
    aoi_lon,
    aoi_lat;
    temp_offset = Float64(extract_ctrl_val(steering_sliders, :temp_offset, 0.0)),
    temp_multiplier = Float64(extract_ctrl_val(steering_sliders, :temp_multiplier, 1.0))
)

# ╔═╡ a0000001-0001-0001-0001-000000000015
md"""
## 2. Model Initialization & Interactive Dashboard Execution

We construct the model parameters using `load_ibm_configuration` layered with our
interactive overrides. The simulation advances in discrete daily steps ($\Delta t = 1.0$ day),
recording individual agent transitions, energetic budgets, molting events, and annual fishery catch.

Toggle `reuse_saved_simulation` to reload cached simulation outputs when available, skipping full multi-year integration on initial page load.

Use the transport controls below to execute, step, pause, or rewind the simulation across multi-year horizons:
"""

# ╔═╡ a0000001-0001-0001-0001-000000000050
@bind reuse_saved_simulation PlutoUI.CheckBox(default = true)

# ╔═╡ a0000001-0001-0001-0001-000000000031
@bind transport_controls HTML(simulation_transport_widget(default_interval = 10, default_years = 30))

# ╔═╡ a0000001-0001-0001-0001-000000000016
sim_results = begin
    # 1. Ingest baseline dictionary from snowcrab.toml
    cfg = load_ibm_configuration(snowcrab_toml_path)
    
    # 2. Extract settings from transport widget and parameter steering sliders
    sim_status = String(extract_ctrl_val(transport_controls, :sim_status, "stopped"))
    current_day = Int(extract_ctrl_val(transport_controls, :current_day, 0))
    sim_years = Int(extract_ctrl_val(transport_controls, :sim_years, 30))
    grid_res = Int(extract_ctrl_val(steering_sliders, :grid_res, 50))
    
    # Cap simulated horizon to current interactive day when stopped/paused/running
    total_horizon_days = round(Int, sim_years * 365)
    effective_days = clamp(current_day, 0, total_horizon_days)
    
    cfg[:days] = max(1, total_horizon_days)
    cfg[:grid] = (grid_res, grid_res)
    cfg[:temp_offset] = Float64(extract_ctrl_val(
        steering_sliders, :temp_offset,
        extract_ctrl_val(steering_sliders, :temp_base, 0.0)
    ))
    cfg[:temp_multiplier] = Float64(extract_ctrl_val(
        steering_sliders, :temp_multiplier,
        extract_ctrl_val(steering_sliders, :temp_amplitude, 1.0)
    ))
    cfg[:temp_base] = cfg[:temp_offset]
    cfg[:temp_amplitude] = cfg[:temp_multiplier]
    cfg[:temp_noise_sd] = Float64(extract_ctrl_val(steering_sliders, :temp_noise_sd, 0.0))
    cfg[:habitat_offset] = Float64(extract_ctrl_val(steering_sliders, :habitat_offset, 0.0))
    cfg[:habitat_multiplier] = Float64(extract_ctrl_val(steering_sliders, :habitat_multiplier, 1.0))
    cfg[:food_max] = Float64(extract_ctrl_val(steering_sliders, :food_max, 10.0))
    cfg[:benthic_carrying_capacity] = Float64(extract_ctrl_val(steering_sliders, :benthic_carrying_capacity, 6000.0))
    cfg[:fishing_season_start] = Float64(extract_ctrl_val(steering_sliders, :fishing_season_start, 90.0))
    cfg[:fishing_season_end] = Float64(extract_ctrl_val(steering_sliders, :fishing_season_end, 150.0))
    cfg[:fishing_probability_daily] = Float64(extract_ctrl_val(steering_sliders, :fishing_rate, 0.002))
    legal_size_val = Float64(extract_ctrl_val(steering_sliders, :legal_size, 95.0))
    cfg[:legal_size_male] = legal_size_val
    
    # Apply modular biological process toggles
    cfg[:use_energetics] = Bool(extract_ctrl_val(steering_sliders, :use_energetics, true))
    cfg[:use_thermal_stress] = Bool(extract_ctrl_val(steering_sliders, :use_thermal_stress, true))
    cfg[:use_cannibalism] = Bool(extract_ctrl_val(steering_sliders, :use_cannibalism, false))
    cfg[:use_senescence] = Bool(extract_ctrl_val(steering_sliders, :use_senescence, true))
    cfg[:use_movement] = Bool(extract_ctrl_val(steering_sliders, :use_movement, true))
    cfg[:use_directed_movement] = cfg[:use_movement]
    cfg[:use_reproduction] = Bool(extract_ctrl_val(steering_sliders, :use_reproduction, true))
    
    # Preserve scale consistency for the smaller interactive grid
    scale_factor = (grid_res * grid_res) / (1000.0 * 1000.0)
    adjusted_seed_agents = max(60, round(Int, 600 * sqrt(scale_factor)))
    cfg[:n_agents] = adjusted_seed_agents
    
    # Dynamically bind spatial domain bounds around selected AOI coordinates
    sel_lon = Float64(extract_ctrl_val(aoi_coords, :lon, -63.5))
    sel_lat = Float64(extract_ctrl_val(aoi_coords, :lat, 44.5))
    span_m = 1000.0
    half_m = span_m / 2.0
    lat_deg_len = 111139.0
    lon_deg_len = max(1000.0, 111139.0 * cos(deg2rad(sel_lat)))
    dlat = half_m / lat_deg_len
    dlon = half_m / lon_deg_len

    cfg[:domain_lon] = [sel_lon - dlon, sel_lon + dlon]
    cfg[:domain_lat] = [sel_lat - dlat, sel_lat + dlat]

    # Explicit override tracking to ensure grid synchrony
    explicit_keys = Set([
        :days, :grid, :domain_lon, :domain_lat, :temp_offset, :temp_multiplier,
        :temp_base, :temp_amplitude, :temp_noise_sd, :habitat_offset, :habitat_multiplier,
        :food_max, :benthic_carrying_capacity,
        :fishing_season_start, :fishing_season_end,
        :fishing_probability_daily, :legal_size_male, :n_agents,
        :use_energetics, :use_thermal_stress, :use_cannibalism,
        :use_senescence, :use_movement, :use_directed_movement,
        :use_reproduction
    ])
    sync_grid_dimensions!(cfg; explicit = explicit_keys)
    
    # 3. Construct structured IBMParameters
    params = IBMParameters(cfg)
    
    # 4. Initialize agent-based model with exact AOI bathymetric depth grid
    aoi_model_bathy = (grid_res == size(aoi_depth_grid, 1) && grid_res == size(aoi_depth_grid, 2)) ?
        aoi_depth_grid :
        estimate_aoi_bathymetry(sel_lon, sel_lat; span_meters = span_m, n_points = grid_res).depths

    model = initialize_model(
        n_agents = adjusted_seed_agents,
        grid = (grid_res, grid_res),
        dt = 1.0,
        params = params,
        depth_map = aoi_model_bathy,
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
    
    # 6. Execute simulation up to current interactive day or load cached instance
    cache_file = normpath(joinpath(@__DIR__, "..", "scratch", "saved_ibm_run.jld2"))
    can_load_cache = reuse_saved_simulation && isfile(cache_file) && (sim_status == "stopped" || current_day == 0)
    
    agent_df, model_df = if can_load_cache
        try
            cached = JLD2.load(cache_file)
            (cached["agent_df"], cached["model_df"])
        catch
            run_simulation(model, effective_days; adata = adata, mdata = mdata)
        end
    else
        res_a, res_m = run_simulation(model, effective_days; adata = adata, mdata = mdata)
        if effective_days > 0
            mkpath(dirname(cache_file))
            try
                JLD2.save(cache_file, "agent_df", res_a, "model_df", res_m)
            catch
            end
        end
        (res_a, res_m)
    end
    
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

# ╔═╡ a0000001-0001-0001-0001-000000000019
plot_dashboard_spatial(
    sim_results.model, sim_results.params;
    sim_days = sim_results.sim_days,
    horizon_days = sim_results.target_horizon_days,
    sim_status = sim_results.sim_status,
    background_layer = Symbol(extract_ctrl_val(steering_sliders, :background_layer, "habitat_quality"))
)

# ╔═╡ a0000001-0001-0001-0001-000000000041
md"""
*Figure 1: Spatial Habitat & Organism Distribution.* Realized Habitat Suitability Index (HSI) field with agent particles sized by carapace width ($L_i$) and colored by demographic stage and sex via `organism_color`.
"""

# ╔═╡ a0000001-0001-0001-0001-000000000048
plot_dashboard_environmental_distributions(
    sim_results.model;
    sim_days = sim_results.sim_days
)

# ╔═╡ a0000001-0001-0001-0001-000000000049
md"""
*Figure 1b: Environmental Driver Distributions at Day $(round(Int, sim_results.sim_days)).* Empirical cell distributions across the spatial grid for seabed bathymetric depth, ambient bottom temperature, and realized Habitat Suitability Index (HSI).
"""

# ╔═╡ a0000001-0001-0001-0001-000000000034
plot_dashboard_stages(
    sim_results.model,
    sim_results.params;
    sim_days = sim_results.sim_days
)

# ╔═╡ a0000001-0001-0001-0001-000000000042
md"""
*Figure 2: Instar Stage Abundance Structure.* Instar stage demographics visualized as paired lollipop stems (`stem!`), split 50/50 for unsexed egg/aggregate stages and by realized counts for individuated stages.
"""

# ╔═╡ a0000001-0001-0001-0001-000000000021
plot_dashboard_size_density(
    sim_results.model,
    sim_results.params;
    sim_days = sim_results.sim_days
)

# ╔═╡ a0000001-0001-0001-0001-000000000043
md"""
*Figure 3: Size Frequency by Sex and Maturity.* Normalized Gaussian kernel densities (`kde_density`, $h = 3.0\text{ mm}$, `scale = :probability`) comparing immature vs mature components of both sexes alongside the legal fishery cut ($L_{\text{legal}}$).
"""

# ╔═╡ a0000001-0001-0001-0001-000000000022
plot_dashboard_ogive(
    sim_results.model,
    sim_results.params;
    sim_days = sim_results.sim_days
)

# ╔═╡ a0000001-0001-0001-0001-000000000044
md"""
*Figure 4: Maturity Ogives & Fishery Selectivity.* Empirical maturity fractions and legal harvest retention contrasted against theoretical single-ecdysis, cumulative, and knife-edge selectivity curves.
"""

# ╔═╡ a0000001-0001-0001-0001-000000000020
plot_dashboard_abundance(
    sim_results.agent_df;
    sim_days = sim_results.sim_days
)

# ╔═╡ a0000001-0001-0001-0001-000000000045
md"""
*Figure 5: Abundance Trajectories.* Dynamic population trajectory on a logarithmic scale ($\log_{10}(\text{count})$) for Total, Mature Females, Mature Males, and Legal Fishable Males.
"""

# ╔═╡ a0000001-0001-0001-0001-000000000035
plot_dashboard_macro(
    sim_results.agent_df,
    sim_results.model_df,
    sim_results.annual_df;
    sim_days = sim_results.sim_days
)

# ╔═╡ a0000001-0001-0001-0001-000000000046
md"""
*Figure 6: Biomass, Catch & Macro-Models.* Fishable stock biomass ($B_{\text{legal}}$) and annual catch alongside fitted surplus production (Schaefer), delay-difference (Deriso-Schnute), and Leslie matrix trajectories.
"""

# ╔═╡ a0000001-0001-0001-0001-000000000036
plot_dashboard_stock_recruitment(
    sim_results.annual_df;
    sim_days = sim_results.sim_days
)

# ╔═╡ a0000001-0001-0001-0001-000000000047
md"""
*Figure 7: Stock-Recruitment Relationship.* Realized spawner-recruit pairs $(\log_{10}(S), \log_{10}(R))$ overlaid with parametric Ricker and Beverton-Holt density-dependent recruitment curves.
"""

# ╔═╡ a0000001-0001-0001-0001-000000000023
md"""
## 4. Macro-Ecological Population Dynamic Visuals & Theory

This section links the bottom-up individual-based simulation to classical top-down
macro-ecological and fisheries population dynamics models. Discrete-time aggregate models
are fitted directly to the annual demographic output series generated by the IBM.

### Biomass Dynamics (Schaefer Surplus Production Model)

Surplus production models treat the aggregate population as a single homogeneous biomass unit,
abstracting age and size structures. The discrete-time Schaefer formulation (Schaefer, 1954)
with catch removed at time $t$ is expressed as:

```math
B_{t+1} = B_t + r B_t \left(1 - \frac{B_t}{K}\right) - C_t
```

Where:
- $B_t$: Aggregate fishable biomass at time period $t$ (kg or metric tons).
- $r$: Intrinsic rate of population increase ($\text{year}^{-1}$), capturing the maximum per capita reproductive and somatic surplus capacity.
- $K$: Environmental carrying capacity (kg), representing the unexploited asymptotic equilibrium biomass supported by habitat resources.
- $r B_t \left(1 - \frac{B_t}{K}\right)$: Surplus production, representing net somatic growth and recruitment balance above natural mortality; maximum surplus production (Maximum Sustainable Yield, $\text{MSY} = \frac{r K}{4}$) occurs at $B_{\text{MSY}} = \frac{K}{2}$.
- $C_t$: Aggregate fishery catch removals during interval $t$ (kg).

### Deriso-Schnute Delay-Difference Dynamic Model

The delay-difference equation (Deriso, 1980; Schnute, 1985) bridges surplus production and age-structured
dynamics by incorporating somatic growth (Ford-Walford or Brody-Bertalanffy growth parameters) and age/stage-at-recruitment delays without tracking individual cohorts explicitly:

```math
B_{t+1} = (1 + \rho) s B_t - \rho s^2 B_{t-1} + R(S_{t-k}) - C_t
```

Where:
- $B_t, B_{t-1}$: Exploitable fishable stock biomass in years $t$ and $t-1$.
- $s$: Annual fraction of natural survival ($s = e^{-M}$, where $M$ is the instantaneous natural mortality rate).
- $\rho$: Brody growth coefficient from the von Bertalanffy weight allometry:
  ```math
  w_{a} - w_{\infty} = \rho (w_{a-1} - w_{\infty}) \quad \Longrightarrow \quad w_{a} = (1+\rho) w_{a-1} - \rho w_{a-2}
  ```
  accounting for the persistence of somatic growth momentum from the preceding annual stanza.
- $(1 + \rho) s B_t$: Projected biomass surviving from $t$ to $t+1$ augmented by somatic weight increment.
- $-\rho s^2 B_{t-1}$: Correction factor discounting the compound survival and growth of stock from two periods prior ($t-1$), preventing double counting of accumulated tissue.
- $R(S_{t-k})$: Density-dependent recruitment biomass entering the fishable stock at year $t+1$, lagged by $k$ years after spawning by parental stock $S_{t-k}$. For snow crab, $k$ corresponds to the multi-year developmental delay from larval extrusion to legal recruitment ($L \ge 95$ mm, typically 7–9 years).
- $C_t$: Harvested biomass removals in year $t$.

### Generalized Leslie (Lefkovitch) Stage-Structured Matrix Model

For stage-structured benthic decapods where molt frequency decreases with size and terminal molt fixes adult morphology, populations are represented by a stage-vector $\mathbf{n}_t \in \mathbb{R}^p$ across $p$ developmental stages:

```math
\mathbf{n}_{t+1} = \mathbf{A} \mathbf{n}_t - \mathbf{C}_t
```

Where:
- $\mathbf{n}_t = \begin{bmatrix} n_{1,t} & n_{2,t} & \cdots & n_{p,t} \end{bmatrix}^T$: Vector of organism abundances across the 20 discrete life-history stages (instars 0–19).
- $\mathbf{A}$: Transition projection matrix (Lefkovitch, 1965) partitioned into reproductive fecundity, survival with stage retention, and growth transitions:
  ```math
  \mathbf{A} = \begin{bmatrix}
  P_1 & F_2 & F_3 & \cdots & F_p \\
  G_1 & P_2 & 0   & \cdots & 0 \\
  0   & G_2 & P_3 & \cdots & 0 \\
  \vdots & \ddots & \ddots & \ddots & \vdots \\
  0   & \cdots & 0 & G_{p-1} & P_p
  \end{bmatrix}
  ```
- $F_j$: Effective stage-specific fecundity contribution from stage $j$ to stage 1 ($F_j = \sigma_j m_j$, with $\sigma_j$ maternal survival and $m_j$ egg production).
- $P_j$: Probability that an individual in stage $j$ survives and remains in stage $j$ during the time interval ($P_j = \sigma_j (1 - \gamma_j)$, where $\gamma_j$ is the ecdysis/molting probability).
- $G_j$: Probability that an individual in stage $j$ survives and transitions (molts) into stage $j+1$ ($G_j = \sigma_j \gamma_j$).
- $\mathbf{C}_t$: Stage-specific harvest removal vector ($\mathbf{C}_t = \begin{bmatrix} 0 & \cdots & 0 & C_{\text{legal}, t} \end{bmatrix}^T$).
- Asymptotic population growth rate is determined by the dominant eigenvalue $\lambda_1 = e^r$ of $\mathbf{A}$, while the right eigenvector $\mathbf{w}_1$ defines the stable stage distribution, and the left eigenvector $\mathbf{v}_1$ defines the stage-specific reproductive values.
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
# ╟─a0000001-0001-0001-0001-000000000037
# ╟─a0000001-0001-0001-0001-000000000006
# ╟─a0000001-0001-0001-0001-000000000015
# ╟─a0000001-0001-0001-0001-000000000050
# ╟─a0000001-0001-0001-0001-000000000031
# ╟─a0000001-0001-0001-0001-000000000016
# ╟─a0000001-0001-0001-0001-000000000019
# ╟─a0000001-0001-0001-0001-000000000041
# ╟─a0000001-0001-0001-0001-000000000048
# ╟─a0000001-0001-0001-0001-000000000049
# ╟─a0000001-0001-0001-0001-000000000034
# ╟─a0000001-0001-0001-0001-000000000042
# ╟─a0000001-0001-0001-0001-000000000021
# ╟─a0000001-0001-0001-0001-000000000043
# ╟─a0000001-0001-0001-0001-000000000022
# ╟─a0000001-0001-0001-0001-000000000044
# ╟─a0000001-0001-0001-0001-000000000020
# ╟─a0000001-0001-0001-0001-000000000045
# ╟─a0000001-0001-0001-0001-000000000035
# ╟─a0000001-0001-0001-0001-000000000046
# ╟─a0000001-0001-0001-0001-000000000036
# ╟─a0000001-0001-0001-0001-000000000047
# ╟─a0000001-0001-0001-0001-000000000023
# ╟─a0000001-0001-0001-0001-000000000024
# ╟─a0000001-0001-0001-0001-000000000025
# ╟─a0000001-0001-0001-0001-000000000026
# ╟─a0000001-0001-0001-0001-000000000027
# ╟─a0000001-0001-0001-0001-000000000017
# ╟─a0000001-0001-0001-0001-000000000029
# ╟─a0000001-0001-0001-0001-000000000028
