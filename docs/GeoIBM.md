# Individual and Agent Based Models (GeoIBM) Documentation

This document describes the architecture, mathematical formulations, command-line interface, and public API of the biological Individual and Agent Based Model (`GeoIBM`) engine. The engine implements a high-performance, modular individual- and cohort-based simulation system for organismal population dynamics, bioenergetics, and harvest management.

Species-specific biological processes (allometry, developmental thresholds, ecdysis, and reproduction) are fully decoupled from generic individual-based model (IBM) simulation logic via an extensible behavioral dispatch interface (`SpeciesBehavior`). A complete concrete implementation is provided for the snow crab (*Chionoecetes*).

---

## 1. Overview & Modular Architecture

The simulation engine tracks population cohorts and spatial individuals across their full life-history cycle, integrating environmental forcing fields (temperature, food availability) on discrete 2D spatial grids.

### Architectural Pillars

1. **Generic Simulation Engine**:
   - Discrete 2D spatial representation (`GridSpace`).
   - Agent scheduling and lifecycle integration using `Agents.jl`.
   - Hybrid spatial architecture combining pre-recruit aggregate cohort groups (`AggregateGroup`) with individuated spatial agents (`Organism`).
   - Environmental field dynamics with local resource replenishment and advection-diffusion-taxis dispersal.

2. **Species Behavior Dispatch Layer**:
   - Abstract types: `SpeciesBehavior` and `SpeciesParams`.
   - Generic interface functions defined with rigorous mathematical assumptions and fallback handling.
   - Dynamic dispatch routes species-specific allometry, phenology, development, and reproduction to concrete subtypes (e.g. `SpeciesBehaviorImpl`).
   - Default behavior resolution via `default_species_behavior()`.

```
                      [ SpeciesBehavior ]  <── abstract interface
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
      [ SpeciesBehaviorImpl ]          [ OtherSpeciesBehavior ]
      (Chionoecetes)              (Extensible)
               │
   ┌───────────┼────────────────────────────────────────┐
   ▼           ▼                    ▼                   ▼
Allometry   Development         Phenology         Reproduction
(Hiatt)   (Degree-Days)   (Intermolt Thresholds)  (Spermathecal Store)
```

### High-Level Life-History Cycle (Snow Crab Example)

`
       [ Adult Females ] ──(Mating Encounter)──> [ Spermathecae Storage ]
              │                                           │
              ▼                                           ▼
       [ Brood Incubation ] <──────────────────── [ Egg Extrusion ]
       (Stage 0 on Pleopods, Thermal Rate)       (Allometric Fecundity)
              │
              ▼ (Hatching: Stage 1 Inflow)
       [ Pelagic Larvae (Stages 1–3) ]
       (Zoea I, Zoea II, Megalopa; ~800 DD, T_opt = 11°C)
              │
              ▼ (Settlement & Benthic Metamorphosis at Stage 4)
       [ Aggregate Cohorts (Stages 4–11) ] ──> [ Early Benthic Instars ]
                                                         │
                                                         ▼ (Individuation at Stage 12)
                                              [ Spatial Agents (Stages 12–19) ]
                                                         │
                                                         ▼ (Continuous Maturation Hazard)
                                               [ Morphometric Adults ]
                                              ┌──────────┴──────────┐
                                              ▼                     ▼
                                       [ Adult Females ]     [ Adult Males ]
                                       (Breeding Stock)    (Commercial Fishery)
                                       (Terminal Stage 16) (Terminal Stage 19)
`

---


> **Species dispatch is not defaulted.** `SpeciesBehavior` is the extension
> point for a second species, and three methods in the generic layer used to defeat
> it: `stage_size_lower_bound`, `growth_increment` and `build_stages` each built a
> `SpeciesBehaviorImpl()` and called *that*, discarding the behaviour they were
> handed. With one species the mistake is invisible, because every behaviour *is* a
> `SpeciesBehaviorImpl` and dispatch already selects the right method, and because
> an altered `SpeciesParams` does change the answer — which is what makes it easy to
> miss. The cost appears only when a second species is written: it would be measured
> against snow crab growth coefficients and the snow crab stage-size curve, and
> produce plausible, wrong animals. All three now fail at the call site with the
> offending type named.
>
> **Two reporting invariants are not visible in the data.** `catch_total` and
> `catch_legal` are the same series, because the fishery gate removes an animal only
> when `is_legal_male` passes; a test asserts the identity so that admitting sublegal
> harvest forces the two columns apart. And the fishery has exactly one switch:
> `fishery.fishing_probability_daily` is `0.0` by default, and that alone means no
> harvest. There is no `toggles.use_fishery` beside it — the two used to express one
> thing, so the model could report the fishery as enabled while nothing was
> catchable. `fishery_active(params)` is the predicate.

## 2. Package Organization

The source repository is organized into focused, modular components:

```
GeoIBM/
├── Project.toml              # Dependencies and compatibility bounds
├── docs/
│   ├── GeoIBM.md             # Comprehensive architecture and API reference (this file)
│   └── deep_check_2026-10.md # Methodological audit of documentation claims (2026-10)
├── src/
│   ├── GeoIBM.jl             # Module definition, includes, and exported public APIs
│   ├── types.jl              # Core type system: Organism, StageDefinition, IBMParameters
│   ├── organism_dynamics.jl  # Generic growth, allometry fallbacks, bioenergetics, molt
│   ├── species.jl            # Species behaviour: allometry, phenology, reproduction
│   ├── simulation.jl         # Grid initialization, agent/model steppers, simulation runner
│   ├── cli.jl                # CLI entry points: parsing, configuration layers, reporting
│   ├── assimilation.jl       # Environmental store: GeoData fetch, manifest, read-back
│   ├── notebook_support.jl   # Pluto widget builders and notebook-facing dashboards
│   ├── analysis.jl           # KDE, OLS/IRLS estimation, Leslie matrices, macro models
│   ├── visualization.jl      # Diagnostic figures and video animations (CairoMakie)
│   └── gui.jl                # Real-time interactive 3-dashboard GUI (GLMakie)
├── configs/
│   ├── default.toml          # Every declared parameter, at its struct default
│   └── snowcrab.toml         # Snow crab overlay: biology plus operating scale
├── scripts/
│   ├── organism_ibm.jl       # CLI simulation application
│   ├── organism_orchestrator.jl # Parameter variation and calibration orchestrator
│   └── fit_macro_models.jl   # Post-hoc macro-model fitting CLI for saved results
├── notebooks/
│   ├── README.md             # Pluto launch and headless-run instructions
│   └── GeoIBMNotes.jl        # Pluto research notebook
└── test/
    ├── runtests.jl           # Suite entry point: biology, macro, config, simulation, vis, habitat
    ├── test_fixtures.jl      # Shared fixtures and synthetic datasets
    ├── test_allometry_biology.jl
    ├── test_habitat_environment.jl
    ├── test_macro_dynamics.jl
    ├── test_simulation_agents.jl
    ├── test_config_parameters.jl
    └── test_visualization_gui.jl
```

### Component Roles

| Source File | Primary Responsibility |
|---|---|
| [`src/types.jl`](src/types.jl) | Declares abstract interface (`SpeciesBehavior`, `SpeciesParams`), agent structs (`Organism`), cohort containers (`AggregateGroup`), configuration (`IBMParameters`), and matrix types (`StageDefinition`, `LeslieMatrixFit`, `MacroModelComparison`). |
| [`src/organism_dynamics.jl`](src/organism_dynamics.jl) | Implements generic biological interfaces and fallbacks, Kleiber metabolic bioenergetics ($W^{0.75}$), Holling Type II consumption, ecdysis mechanics, mortality hazards, and spatial dispersal. |
| [`src/species.jl`](src/species.jl) | Implements `SpeciesParams`, `SpeciesBehaviorImpl`, empirical Hiatt increments, log-logistic maturity schedules, allometric fecundity, thermal ecdysial thresholds, and primiparous/multiparous reproductive cycles. |
| [`src/simulation.jl`](src/simulation.jl) | Manages model instantiation (`initialize_model`), environmental field generation, scheduled time stepping (`agent_step!`, `model_step!`), and batch execution (`run_simulation`). |
| [`src/cli.jl`](src/cli.jl) | Resolves the two-layer configuration (baseline `default.toml` plus a run overlay), parses CLI switches into typed parameters, and provides terminal reporting, snapshot telemetry, batch figure export and `julia_main` binary entry points. |
| [`src/assimilation.jl`](src/assimilation.jl) | Assimilates the published environmental inputs (ETOPO 2022 bathymetry, WOA23/GLORYS12V1 hydrography) into an idempotent GeoData store with a manifest, and reads the store back for a run. |
| [`src/notebook_support.jl`](src/notebook_support.jl) | Standardized HTML/JS widget builders (domain picker, steering sliders, transport bar) and standalone CairoMakie dashboard panels for the Pluto notebook, mirroring the `gui.jl` dashboards. |
| [`src/analysis.jl`](src/analysis.jl) | Computes kernel density estimations (KDE), iterative logistic fits (IRLS), demographic summaries, Schaefer and Delay-Difference biomass models, generalized Leslie matrices, and retrospective analyses. |
| [`src/visualization.jl`](src/visualization.jl) | Generates executive diagnostic figures, maturity ogives, stock-recruitment plots, and video animations via CairoMakie (`Figure` output). |
| [`src/gui.jl`](src/gui.jl) | Provides interactive real-time multi-dashboard steering GUI with reactive particle streaming, parameter steering, and macro-ecological model fitting via GLMakie (`launch_organism_gui`). |
| [`scripts/fit_macro_models.jl`](../scripts/fit_macro_models.jl) | Post-hoc CLI fitting Schaefer and delay-difference macro models to saved simulation results — whole population and fishable component — with optional retrospective analysis and Mohn's ρ. |
| [`notebooks/GeoIBMNotes.jl`](../notebooks/GeoIBMNotes.jl) | Pluto research notebook walking the model's life history, calibration, macro fitting and outputs through the `notebook_support.jl` widgets; `notebooks/README.md` covers launching it and running it headless. |

---

## 3. Mathematical Formulations & References

### 3.1 Allometric Growth and Post-Molt Prediction

In crustaceans, growth occurs discontinuously through ecdysis. Post-molt carapace width is modeled via linear or bivariate allometric relations on the logarithmic scale (Hiatt model):

$$\ln(\text{size}_{t+1}) = \beta_0 + \beta_1 \ln(\text{size}_t) + \delta_{\text{mat}} \cdot \mathbb{I}_{\text{mat}} + \Delta_{\text{temp}}(T) + \epsilon$$

where $\beta_0$ and $\beta_1$ are sex-specific empirical coefficients, $\delta_{\text{mat}}$ is the change in growth increment associated with adolescent/terminal molt, and $\epsilon \sim \mathcal{N}(0, \sigma^2)$ represents individual growth variation (Sainte-Marie et al. 1995).

### 3.2 Logistic Morphometric Maturity Schedules

The probability that an individual transitions to morphometric maturity during ecdysis is governed by a logistic ogive parameterized by $L_{50}$ (carapace size at 50% maturity) and logistic slope $\beta$:

$$P(\text{mature} \mid \text{size}, T) = \frac{1}{1 + \exp\left(-\beta \left(\ln(\text{size}) - \ln(L_{50}(T))\right) - \kappa_T (T - T_{\text{ref}})\right)}$$

where $\kappa_T$ accounts for the Temperature-Size Rule (TSR), causing individuals in colder regimes to mature at larger sizes or different rates (Sainte-Marie 1993).

### 3.3 Brood Fecundity Allometry

Brood fecundity (extruded egg count) is modeled as a power-law function of female carapace width, differentiated by maternal reproductive history:

$$\log_{10}(\text{Fecundity}) = a + b \log_{10}(\text{size}) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

- **Primiparous females** (first breeding clutch): $a = -0.7229, b = 2.94, \sigma = 0.090$
- **Multiparous females** (second or subsequent clutches): $a = 0.0158, b = 2.5811, \sigma = 0.145$
- Senescence attenuation: For multiparous females with shell age $\tau_{\text{shell}} > 3.0$ years, fecundity is discounted by $1 - 0.08(\tau_{\text{shell}} - 3.0)$, bounded below by 0.60.

### 3.4 Bioenergetics and Kleiber Consumption

Daily energy balance is integrated through standard allometric metabolic consumption and Holling Type II intake:

$$\frac{dE}{dt} = \left( I_{\text{actual}} \cdot \alpha_{\text{assim}} \cdot (1 - \text{SDA}) \right) - C_{\text{maint}}$$

$$C_{\text{maint}} = R_{\text{meta}} \cdot W^{0.75} \cdot \exp(0.08 T) \cdot \exp\left(-\frac{(T - T_{\text{opt}})^2}{2 \sigma_T^2}\right)$$

$$I_{\text{actual}} = \min\left( R_{\text{intake}} \cdot \frac{F}{F + K_F} \cdot W^{0.75}, F \right)$$

where $W$ is wet body mass in grams, $F$ is local forage concentration, $K_F$ is half-saturation food density, and $\text{SDA} = 0.15$ accounts for specific dynamic action.

### 3.5 Natural Mortality Hazard Integration

Total daily natural mortality probability integrates baseline, predatory, physiological, and environmental risks:

$$Z_{\text{daily}} = \left( Z_{\text{base}} \cdot \mu_{\text{stage}} \right) + Z_{\text{pred}}(\text{size}) + Z_{\text{soft}}(\Delta t_{\text{molt}}) + Z_{\text{starve}}(E) + Z_{\text{thermal}}(T) + Z_{\text{senescence}}(\tau) + Z_{\text{cannibal}}$$

1. **Size-Dependent Predation**: $Z_{\text{pred}} = \frac{0.15}{365} \exp(-0.03 \max(0, \text{size} - 5.0))$.
2. **Soft-Shell Ecdysial Vulnerability**: $Z_{\text{soft}} = \frac{0.12}{365} \exp(-\Delta t_{\text{molt}} / 15.0)$.
3. **Starvation Risk**: For energetic reserve $E < 0$, $Z_{\text{starve}} = \frac{0.20}{365} (1 - \exp(E / (0.05 W)))$.
4. **Thermal Stress**: For $T > T_{\text{crit}}$, $Z_{\text{thermal}} = \frac{0.05}{365} (T - T_{\text{crit}})^{1.5}$.
5. **Adult Senescence**: For shell age $\tau > \tau_{\text{thresh}}$, $Z_{\text{senescence}} = \frac{0.08}{365} \exp(0.80 (\tau - \tau_{\text{thresh}}))$.

### 3.6 Pelagic Larval Ontogeny, Thermal Viability, and Developmental Gates

Pre-recruit cohorts develop through discrete pelagic instars (egg, Zoea I, Zoea II, and Megalopa) before benthic metamorphosis and settlement (Instar 1 juvenile). In accordance with empirical rearing and southern Gulf of St. Lawrence field observations (Ouellet & Sainte-Marie 2018):

1. **Thermal Degree-Day Accumulation**:
   $$\Delta \text{DD}_t = \max(0.0, T_t - T_0) \Delta t, \quad T_0 = -1.5^\circ\text{C}$$
   Zoea I, Zoea II, and Megalopa each require $\approx 270\text{ DD}$ (empirical range $225\text{--}350\text{ DD}$, typically $260\text{--}280\text{ DD}$ at $2.8\text{--}7.8^\circ\text{C}$), with minimum intermolt durations $\Delta t_{\min} = [0, 36, 36, 42]\text{ days}$, bringing cumulative pelagic development to $\sim 800\text{ DD}$.

2. **Parabolic Thermal Survival**:
   Pelagic larval survival is maximal at $10\text{--}12^\circ\text{C}$ with symmetric concave-down thermal viability away from optimum $T_{\text{opt}} = 11.0^\circ\text{C}$ ($\sigma_T = 3.5^\circ\text{C}$):
   $$h_{\text{temp}}(T) = \exp\left( \frac{1}{2} \left( \frac{T - T_{\text{opt}}}{\sigma_T} \right)^2 \right)$$

3. **Over-Age Developmental Hazard Gate**:
   Larvae that exceed maximum viable stage duration without molting experience an escalating attrition hazard:
   $$Z_{\text{overage}} = \begin{cases} 0.05 \cdot \frac{t_{\text{stage}} - 84.0}{10.0}, & \text{if } t_{\text{stage}} > 84.0\text{ days} \\ 0.0, & \text{otherwise} \end{cases}$$

---


## 4. Operating Scale and Performance

`configs/snowcrab.toml` configures a **1000 × 1000** grid, which at one cell per
metre is a 1 km × 1 km footprint — the scale of a single snow crab survey
stratum — with `benthic_carrying_capacity = 10000`, roughly 10000 organisms per
square kilometre. The shipped seeding holds one 2000-member aggregate clutch at
stage 1 and an empty individuated stock, so day one runs exactly 2000 benthic
organisms and the density multiplier is $1 + (2000/10000)^2 = 1.04$.

### Measured run cost (benchmarked 2026-10-23)

Median milliseconds per simulated day from a three-replicate benchmark of 1500
organisms at the seeded distribution, warm-up excluded (`scratch/bench_perf.jl`,
replicate log in `scratch/bench_perf.log`):

| Grid (cells) | Dynamics only (ms/day) | With the CLI's default observation set (ms/day) |
|---|---|---|
| 1000 × 1000 | 40.3 | 49.9 |
| 200 × 200 | 30.7 | 29.3 |
| 100 × 100 | 36.5 | 28.5 |
| 30 × 30 | 22.6 | 20.9 |

Replicate spread is about ±25%, so read the table in halves, not decimals. A
separate end-to-end timing of the shipped snowcrab configuration measured
32.6 ms of bare dynamics per simulated day: a two-year, 730-day run is then
roughly **24 s** of dynamics, or roughly **36 s** with the CLI's default
per-step observation columns collected.

### Where the time goes, and what has been done

| Cost | Detail |
|---|---|
| Aggregate cohorts | The largest win, and a real defect. A cohort that both survives and transitions kept its survivor group and emitted a *new* group for the transitions, at the same cell and the stage it would occupy next day. Those children fragmented in turn, so the group count grew geometrically: a 500-day run reached **1135 groups for six stages**, and the per-step cost is linear in the group count. Cohorts at the same stage and cell are now merged, with abundances summed and size, mass, age, degree-days and last-growth time abundance-weighted. A run now holds **12 groups**, and this step went from **143 ms to 0.3 ms**. |
| Temperature field | Drawn in place with `randn!` and scaled by broadcast. It used to be `rand(rng, Normal(0, sd), dims)`, which allocates a *second* full-size matrix every step: 7.6 MB per day and 5.4 GB over two years. Measured 9.0 ms/day → 2.3 ms/day, with no allocation. |
| Food field | Rewritten as the affine map it is, `F = (1 - r·dt)·F + r·f_max·dt`: one multiply-add per cell instead of a subtract, two multiplies and an add. |
| Density multiplier | `1 + (N/K)^exponent` was evaluated once per organism per day, walking the cohorts each time; profiling put a quarter of the run there. It is now computed once per step, after the cohorts have advanced, and cached on the model. |
| `model.params` | Hoisted out of `agent_step!`. Every access is a symbol lookup into the properties dictionary, and the function read it about a dozen times per organism per day. |
| `IBMParameters` property reads | Its `getproperty` resolved a nested access with `fieldnames(IBMParameters)` — which returns a fresh `Vector{Symbol}` on every call — plus a chain of `hasfield` checks. It is now one dictionary lookup. |
| Cell temperature | One organism read its own cell's temperature four times per day, for energetics, degree-day accrual, the molt decision and mortality. It is read once and passed down. |
| Day of year | Recomputed per organism; `model_step!` already computes it for the seasonal cycle and now publishes it. |
| Seasonal flags and thresholds | The fishery season test and the legal-size thresholds are properties of the day, not of the organism, and are resolved once per step. Legality is also checked before the harvest draw rather than after, which stops a random number being spent on every sublegal animal. |
| Cannibalism density query | Now gated on `use_cannibalism`; it ran for every organism every day regardless. |
| Observation columns | `run_simulation` collects `adata` and `mdata` on every step unless `when` is supplied. Each column is a pass over the population, so pass an interval for long runs. |

### Three counter-intuitive properties

**Neither grid extreme dominates.** At 1500 organisms the sweep moved both ways:
the cheapest grid measured was 30 × 30 (22.6 ms/day bare) and the most expensive
was 1000 × 1000 (40.3 ms/day). Two cost drivers pull against each other along
the sweep — per-cell environmental field work scales with grid area, while
per-organism bookkeeping and neighbourhood queries grow with local occupancy,
which is highest on coarse grids — and the ±25% replicate noise brackets the
ordering between them.

**Observation width adds measurable cost only on the heaviest grid.** The
CLI's default per-step observation columns are extra population passes
beside the dynamics. At 1000 × 1000 the with-observation runs measured about
10 ms/day more than the bare runs; on the smaller grids the difference sat
inside the replicate noise, so the honest bound is "no more than ~10 ms/day,
and often less".

**Pin the activation order for stored runs.** `initialize_model` defaults to
`Agents.Schedulers.Randomly()`, which draws the activation order from the model's
RNG, so a fixed seed fixes the order by construction while still using every
thread. The Agents.jl default, `Schedulers.fastest`, is `allids(model)` — the
agent container's own order, which for a `Dict` container is whatever the hash
table happens to hold and is not derived from the RNG at all. That matters here
because organisms interact: mortality, mate search and cannibalism all read
neighbours, so the activation order is part of the state transition rather than an
implementation detail. Two same-seed `Randomly` runs agree on every observation
column.

### Calibration constraints worth knowing

* **`K` must exceed the seeded stock.** The density term `1 + (N/K)^exponent`
  multiplies every hazard directly, so a stock above K is not a slow calibration
  problem: with the shipped 2000-member clutch at K = 300 the day-one multiplier
  is already about 45, which presents as an extinction the model never modelled.
  `initialize_model` warns when the stock already exceeds K and names the fields
  to change.
* **Asymptotic size bounds everything.** `species.max_size_male` and
  `max_size_female` are the ceiling on any modelled or drawn size, and the
  saturating limit of the stage-size curve. The growth rule has a log-size slope
  slightly above 1, so without the ceiling the size increment compounds with each
  ecdysis and a 20 mm individual reaches 1.4 m in twelve molts.
* **The stage-size curve saturates.** It is
  `ln S(stage) = ln L + (intercept - ln L)·exp(-decay·(stage - anchor))`, fitted
  to the lower edge of each tabulated stage. The previous form,
  `exp(intercept + slope·(stage - anchor))`, is unbounded for every positive
  slope, and put a stage-12 female at 112 mm against a 95 mm ceiling.
* **The fishery is off by default, and the probability is the switch.**
  `fishery.fishing_probability_daily` is `0.0`, so no animal is ever harvested and
  the stock grows until the density term regulates it. `fishery_active(params)` is
  the predicate; there is no separate toggle to also set.

---

## 5. Command-Line Interface (CLI)

The CLI tool is located at [`scripts/organism_ibm.jl`](scripts/organism_ibm.jl).

### Syntax

```bash
julia --project=. scripts/organism_ibm.jl [OPTIONS]
```

### Options Reference

| Option | Type | Default | Description |
|---|---|---|---|
| `--config=<path>` | String | — | Overlay TOML. `default.toml` is always loaded first; if this names the baseline, its `snowcrab.toml` sibling is overlaid on it. |
| `--days=<int>` | Integer | `730` | Simulation duration in days. |
| `--n_agents=<int>` | Integer | sex-vector cohort sum | Initial individuated population; an explicit value overrides the sum of the sex-structured cohort vectors, rescaling their shape. Shipped vectors are all-zero, so unwritten the default is `0` individuated agents and the aggregate clutch seeds alone. |
| `--grid=<nx,ny>` | Int,Int | `100,100` | 2D spatial grid dimensions. |
| `--grid_nx=<int>`, `--grid_ny=<int>` | Integer | `100` | Grid dimensions, individually. |
| `--seed=<int>` | Integer | `1234` | Random seed. |
| `--temp_offset=<float>` | Float | `0.0` | Baseline bottom-temperature offset (°C); anomaly on top of the GeoData hydrographic mean. |
| `--temp_multiplier=<float>` | Float | `1.0` | Seasonal temperature multiplier on the synthetic oscillation; scales GeoData anomalies. |
| `--temp_base=<float>`, `--temp_amplitude=<float>` | Float | — | Legacy aliases for `--temp_offset` and `--temp_multiplier`. |
| `--benthic_carrying_capacity=<float>` | Float | `12000.0` | Total benthic capacity for the whole model, not per cell. |
| `--frequency_days=<float>` | Float | `365.0` | Output sampling interval. |
| `--snapshot_day=<float>` | Float | `274.0` | Annual survey snapshot day of year. |
| `--output_dir=<path>` | String | `outputs` | Destination directory for exported outputs. |
| `--save=<bool>` | Bool | `true` | Write annual indicators to CSV. |
| `--plot=<bool>` | Bool | `false` | Export diagnostic figures. |
| `--animate=<bool>` | Bool | `false` | Record an animation. |
| `--gui=<bool>` | Bool | `false` | Launch the interactive dashboard. |

Flag names are the TOML key names verbatim. There are no abbreviations or
alternative spellings; an unrecognized name is reported as a warning and ignored
rather than being guessed at. Any declared parameter is settable, for example
`--maturity_l50_male=88.0`, `--rate_predation=0.1`, `--use_energetics=false`, or
a whole schedule as `--intermolt_days_schedule=60,75,90`.

The fishery ships switched off: `fishery.fishing_probability_daily` is `0.0`, and
that is the only switch -- there is no `use_fishery` toggle -- so no animal is
harvested until you set that probability.

### Parameter Variation & Calibration

The orchestrator script at [`scripts/organism_orchestrator.jl`](scripts/organism_orchestrator.jl) executes systematic grid and sensitivity searches across multi-criteria objective targets:

```bash
julia --project=. scripts/organism_orchestrator.jl --vary="temp_offset=2.0:1.0:5.0;fishing_prob=0.0,0.02,0.05" --days=1095 --runs=3
```

---

## 6. Complete API Reference

### 6.1 Types & Structures (`src/types.jl`)

#### `SpeciesBehavior` and `SpeciesParams`
Abstract base types governing modular species dispatch.

```julia
abstract type SpeciesBehavior end
abstract type SpeciesParams end
```

#### `Organism <: AbstractAgent`
Spatial agent structure tracked by `Agents.jl`.

```julia
@agent struct Organism(GridAgent{2})
    stage::Int                             # Developmental stage index (individuated agents: 12 through 19 male, 12 through 16 female)
    sex::Sex                               # male, female, or undifferentiated
    behavior::SpeciesBehavior              # Behavioral dispatch instance
    age_days::Float64                      # Total age in days
    size::Float64                          # Carapace size in millimeters
    mass::Float64                          # Wet body mass in grams
    energy_reserve::Float64                # Stored energy pool (J or normalized)
    last_growth_event_time::Float64        # Simulation time of last ecdysis (days)
    degree_days::Float64                   # Cumulative physiological thermal units
    is_mating_ready::Bool                  # Reproductive receptivity flag
    egg_count::Int                         # Extruded brood count
    hatch_day::Float64                     # Scheduled incubation completion day
    sperm_clutches::Int                    # Number of stored spermathecal clutches
    is_mature::Bool                        # Terminal morphometric maturity flag
    trait_growth_quantile::Float64         # Phenotypic growth rate quantile
    trait_maturity_quantile::Float64       # Phenotypic maturity quantile
    trait_fecundity_quantile::Float64      # Phenotypic fecundity quantile
    trait_mortality_quantile::Float64      # Phenotypic risk tolerance quantile
end
```

#### `StageDefinition`
Container for discrete demographic life-history stage matrices.

```julia
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
```
*Note: Includes property forwarding for `cw_mean`, `cw_range`, `mean_weight`, and `instar`.*

#### `IBMParameters`
Top-level configuration structure with eleven nested parameter sections:
- `toggles::ProcessToggles`: Granular biological and operational feature switches.
- `simulation::SimulationParams`: Time step $\Delta t$, random seed, initial abundance distributions, calendar duration.
- `environment::EnvironmentParams`: Baseline temperature, seasonal amplitude, food replenishment, spatial habitat weights, carrying capacity.
- `organism::OrganismParams`: Metabolic budgets, Kleiber scaling, ecdysis energy thresholds, asymptotic size bounds, stage labels.
- `phenology::PhenologyParams`: Molting and mating seasonal calendar windows, thermal brooding duration, spermathecal storage.
- `mortality::MortalityParams`: Stage-dependent baseline natural mortality, predation, soft-shell vulnerability, starvation, thermal stress, terminal adult senescence, juvenile cannibalism.
- `fishery::FisheryParams`: Minimum legal size threshold ($L_{50}$), fishing season calendar window, daily harvest probability switch.
- `output::OutputParams`: Periodic snapshot intervals, CSV results destination, plotting and animation controls.
- `species::SpeciesParams`: Species-specific allometry (Hiatt increment, weight-length powers), log-logistic maturity ogives, Sainte-Marie fecundity.
- `leslie::LeslieParams`: Generalized Lefkovitch stage-structured projection-matrix parameters and multi-year fitting controls.
- `criteria::CriteriaParams`: Multi-criteria population viability and fidelity thresholds (persistence, size drift, maturity RMSE, growth RMSE).

There is no binning section. Demographic histogramming was configured but never
implemented, so the section only ever advertised settings that had no effect; it has
been removed rather than left as an inert control.

---

### 6.2 Snow Crab Species Implementation (`src/species.jl`)

All biological processes, allometric constants, and parameterizations specific to the snow crab (*Chionoecetes*) are centralized within `src/species.jl`.

#### Parameters: `SpeciesParams`
- **Maturity Ogives**:
  - `maturity_l50_male::Float64 = 91.8` (mm CW), `maturity_l50_male_sd::Float64 = 1.2`
  - `maturity_slope_male::Float64 = 6.9957`, `maturity_slope_male_sd::Float64 = 0.25`
  - `maturity_l50_female::Float64 = 51.6` (mm CW), `maturity_l50_female_sd::Float64 = 0.8`
  - `maturity_slope_female::Float64 = 12.9812`, `maturity_slope_female_sd::Float64 = 0.40`
- **Fecundity Allometry** (Sainte-Marie 1993 log-linear egg production):
  - `fecundity_primiparous_intercept::Float64 = -0.7229`, `slope = 2.94`, `err = 0.090`
  - `fecundity_multiparous_intercept::Float64 = 0.0158`, `slope = 2.5811`, `err = 0.145`
- **Hiatt Growth Increments**:
  - Male: Intercept $\beta_0 = 0.30153$, slope $\beta_1 = 1.00027$, covariance terms $\beta_2 = -0.00643655$, $\beta_3 = 0.0017492$.
  - Female: Intercept $\beta_0 = 0.1931$, slope $\beta_1 = 1.0372$, covariance terms $\beta_2 = -0.017376$, $\beta_3 = 0.005068$.
- **Fishery Selectivity**:
  - $S_{50} = 95.0$ mm, logistic slope bandwidth $\sigma = 5.0$ mm.

#### Behavioral Dispatch: `SpeciesBehaviorImpl <: SpeciesBehavior`
```julia
struct SpeciesBehaviorImpl <: SpeciesBehavior
    params::SpeciesParams
end
SpeciesBehaviorImpl() = SpeciesBehaviorImpl(SpeciesParams())
default_species_behavior() = SpeciesBehaviorImpl()
```

#### Concrete Methods:
- `stage_size_lower_bound(behavior, stage, sex)`: Lower edge of the carapace-width band (mm) for a stage, saturating at the asymptotic size.
- `llbm(stage)` / `llbf(stage)`: Male / female lower bound evaluated against `SpeciesBehaviorImpl()`.
- `get_stage_size(behavior, stage, sex; type=:mid, rng=Random.default_rng())`: Lower (`:lb`), upper (`:ub`), midpoint (`:mid`) or beta-distributed (`:random`) carapace width within the stage band. `stage_size` is the exported alias, and convenience methods on the default implementation exist as `get_stage_size(stage, sex; ...)`.
- `growth_increment(behavior, log_size; sex=0, mat=0, temp=nothing, condition=1.0, rng=Random.default_rng(), growth_quantile=nothing)`: Post-molt natural logarithm $\ln(\text{size}_{t+1})$; `growth_quantile` pins the phenotypic quantile deterministically.
- `predict_postgrowth_size(...)`: Linear (unlogged) post-molt size in millimeters.
- `growth_schedule(behavior)`: Reference table of stage sizes and increments.
- `maturity(behavior, sex, log_size; temp=nothing, maturity_quantile=nothing, rng=Random.default_rng(), deterministic=false)`: Log-logistic morphometric maturity probability, guarded by the absolute size gate (`species.maturity_size_gate`); convenience `maturity(size; sex=0, params=nothing, ...)` works in raw size.
- `fecundity(behavior, size; is_primiparous=true, method=nothing, shell_age=1.0, rng=Random.default_rng(), deterministic=false, fecundity_quantile=nothing)`: Brood egg count under the Sainte-Marie parameterization, with convenience wrappers on raw size.
- `selectivity(behavior, size; s50=nothing, s50_sigma=nothing)`: Logistic gear-retention probability, vectorized; convenience `selectivity(size; params=nothing, ...)` accepts a `SpeciesParams`.
- `calculate_mass(behavior, size, sex; mat=0, stochastic=false, rng=Random.default_rng())`: Wet body mass (g) from carapace width (mm) under the sex- and maturity-stratified allometries; convenience `calculate_mass(size, sex; ...)`.
- `build_stages(behavior; max_male_stage=nothing, max_female_stage=nothing)`: Stage table for the generalized Leslie/Lefkovitch model — pooled-sex recruit/juvenile bins first, then sex-specific rows up to each sex's configured terminal stage (`leslie_max_male_stage` / `leslie_max_female_stage`, overridable). The four stage-table vectors are read in parallel, so a length mismatch is an error.
- `min_intermolt_days(behavior, stage)` / `required_degree_days(behavior, stage)`: Chronological and thermal ecdysis gates.
- `perform_development!(behavior, org, model)` / `perform_reproduction!(behavior, org, model)`: Physiological development and molting; seasonal copulation, spermathecal storage, brood extrusion and hatching.

---

### 6.3 Individual State Queries & Predicates (`src/types.jl`)

Inline predicates for demographic filtering:
- `is_adult(org::Organism)`: True once morphometrically mature (`org.is_mature`).
- `is_mature_male(org)` / `is_mature_female(org)`: True if mature and of that sex; `is_adult_male` / `is_adult_female` are the same checks under the maturity-first naming.
- `is_ovigerous(org::Organism)`: True if carrying an extruded clutch (`org.egg_count > 0`).
- `is_legal_male(org::Organism, legal_size::Real, adult_size::Real)`: True if male and `size >= legal_size` and `size >= adult_size`. Call sites pass the thresholds from `FisheryParams.legal_size_male` and `OrganismParams.adult_size_threshold` rather than hardcoding them.
- `is_legal(...)`: Alias for `is_legal_male`.
- `is_stage(org::Organism, stage::Integer)`: True at that developmental stage.
- `mass_if_mature_female(org)` / `mass_if_mature_male(org)`: `org.mass` if mature of that sex, else `0.0`.
- `mass_if_legal_male(org, legal_size, adult_size)`: `org.mass` for a legal male, else `0.0`.

---

### 6.4 Organism Dynamics & Bioenergetics (`src/organism_dynamics.jl`)

- `calculate_energy_change(temp, food, size; params=IBMParameters(), mass=nothing)`: Net energy change $\frac{dE}{dt}$ and actual forage intake via Kleiber metabolic scaling ($W^{0.75}$) and a Holling Type II functional response. When `mass=nothing` the mass used is the male immature allometry at the given size; pass `mass` explicitly for sex-correct scaling.
- `calculate_energy_change(org::Organism, temp, food; params=IBMParameters())`: Agent-level method using the organism's stored mass.
- `molt_probability(org, current_temp, current_time=0.0; params=IBMParameters())`: Daily ecdysis probability from intermolt days, degree-days, energy reserves and temperature; zero for mature animals or when `toggles.use_molting` is false.
- `check_molt_conditions(org, current_temp, current_time=0.0; params=IBMParameters(), rng=nothing)`: Bernoulli draw on the daily molt probability.
- `perform_molt!(org, current_time, temp=nothing; params=IBMParameters(), rng=Random.default_rng())`: Executes ecdysis — advances the stage, evaluates the maturity transition, resamples size and mass from the stage band, and resets degree-days and shell age.
- `calculate_mortality_probability(org, temp, local_density, current_time; params=IBMParameters(), density_feedback=1.0, on_clamp=nothing)`: Aggregate daily hazard combining stage-dependent baseline natural mortality, senescence (restricted to terminally molted adults, evaluated on unfloored post-terminal shell age), thermal stress, starvation, and the density term passed in `density_feedback`. `on_clamp` observes each hazard-cap clamp event.
- `shell_condition(org, current_time; params=IBMParameters())`: Days since the last ecdysis translated into shell-condition categories 1-5 over `organism.shell_condition_days`.
- `step_aggregate_groups!(model; behavior=get(abmproperties(model), :species_behavior, default_species_behavior()))`: Advances the pre-individuation aggregate cohorts (stages `FIRST_COHORT_STAGE` through `juvenile_max_stage - 1`) through survival, growth and transition hazards, then individuates the terminal cohort into `Organism` agents at `juvenile_max_stage` with a 1:1 sex split.
- `organism_disperse!(org, model; rng=abmrng(model))`: Spatial movement gated on `use_movement` / `use_directed_movement`, toward thermal and forage habitat gradients under the softmax temperature.

---

### 6.5 Simulation Engine (`src/simulation.jl`)

- `default_ibm_configuration()`: Complete baseline configuration `Dict{Symbol, Any}` — grid geometry, runtime duration, environmental forcings, bioenergetics, life-history phenology, fishery dynamics, process toggles and output handling.
- `load_ibm_configuration(filepath)`: Reads a configuration from a structured TOML file.
- `save_ibm_configuration(filepath::String, config::AbstractDict)`: Serializes the configuration dictionary to TOML.
- `initialize_model(; n_agents=nothing, grid=(50, 50), temp_map, food_map, habitat_map, dt=1.0, params, behavior, seed, scheduler=Agents.Schedulers.Randomly(), kwargs...)`: Constructs and seeds the model on a closed-domain `GridSpace(grid; periodic = false)`. Individuals are seeded from the two sex-structured cohort vectors across their benthic stage range; `n_agents` overrides their sum, rescaling the vector shape per benthic cell (rounding is applied per cell on purpose — the stage shape is preserved, not the exact total). The aggregate cohort is seeded in either case from `initial_aggregate_cohort`. Any value left at `nothing` resolves to its built-in default (synthetic environment maps, struct `IBMParameters`, random seed, `SpeciesBehaviorImpl`), an explicit `dt` and `seed` are recorded into the returned simulation parameters, and extra `kwargs` (e.g. `depth_map`) are attached as model properties.
- `initialize_model(config_filepath::String; kwargs...)` / `initialize_model(config::AbstractDict; kwargs...)`: Build the model from a configuration file or dictionary with the same keyword overrides.
- `agent_step!(org::Organism, model)`: One individual time step: energetics, degree-days, molting, mortality, harvest, dispersal, reproduction.
- `model_step!(model)`: Steps the global environment fields (temperature seasonality, food replenishment), advances the aggregate cohorts, and refreshes the demographic caches.
- `run_simulation(model, n_steps; adata=nothing, mdata=nothing, when=nothing, step_callback=nothing)`: Runs `n_steps` through `Agents.run!` and returns `(agent_df, model_df)`. With `adata` omitted the default observation set from `default_agent_observations` is collected; with `mdata` omitted the model's cumulative counters (eggs, recruits, catches, molts, mortality causes) are collected. `step_callback` is stored on the model and invoked after each step.
- `default_agent_observations(params)`: The default per-step agent observation tuples, including per-stage counts (`stage_1`, `stage_2`, ...).
- ABM utility accessors — `nagents(model)`, `allagents(model)`, `abmrng(model)` (Agents.jl), plus `safe_mean` / `safe_sum` empty-safe reductions.

---

### 6.6 Empirical Biology, Surveys, & Demographics (`src/analysis.jl`)

- `gaussian_kde(data; eval_points=nothing, bandwidth=nothing)`: One-dimensional Gaussian kernel density estimate, returning `(grid, density)`.
- `derive_stage_proportions_kde(sizes, sexes=nothing; stages=build_stages())`: Discretizes a size distribution into per-stage proportions and counts.
- `generate_mock_survey_data(; n_samples=500, rng=Random.default_rng(), behavior=default_species_behavior())`: Synthesizes survey catch records across size, sex, maturity, shell condition and fecundity.
- `fit_empirical_biology_model(data::DataFrame; params=SpeciesParams())`: Calibrates allometric mass regressions, maturity ogives and phenotypic covariances from survey data (required columns `:size`, `:sex`, `:maturity`, `:mass`, `:temp_c`, `:fecundity`).
- `estimate_demographics(model; size, sex, temp=4.0, shell_age=1.0, params=SpeciesParams(), rng=...)`: Predicts mass, maturity probability and fecundity for an individual crab.
- `sample_organism_traits(model; size, sex, temp=4.0, shell_age=1.0, params=SpeciesParams(), rng=...)`: Samples correlated individual phenotypic trait draws.
- `summarize_empirical_model(model; io=stdout)`: Formatted ASCII parameter summary.
- `compute_empirical_ogive(sizes, mature_flags; n_bins=15)`: Binned proportions mature for empirical ogive fitting.
- `summarize_population(model)`: Demographic summary from a live model — counts by sex, maturity and stage with biomass totals.

---

### 6.7 Macro-Ecological Models & Stock Assessment (`src/analysis.jl`)

- `extract_stock_recruitment(model)`: Stock-recruitment pairs from the model's clutch recruitment records (spawners in thousands of eggs, settlement-time recruits, and the density feedback at settlement).
- `extract_stock_recruitment(agent_df, model_df; lag_days=nothing, window_days=nothing, days_per_year=365.0)`: Annual spawner and lagged-recruit pairs from run outputs; `nothing` lags and windows fall back to the configuration.
- `fit_stock_recruitment(spawners, recruits; model_type=:both)`: Fits Beverton-Holt ($R = \frac{\alpha S}{1 + \beta S}$) and/or Ricker ($R = \alpha S e^{-\beta S}$); with `:both`, model selection is by AIC.
- `summarize_stock_recruitment(sr_df::DataFrame; fits=nothing)`: Prints the stock-recruitment table and, when fits are supplied, the parameter estimates ($R^2$, $\alpha$, $\beta$, $S_{\max}$).
- `extract_annual_production(agent_df, model_df; dt_year=365.0, frequency_days=dt_year, snapshot_day=274.0)`: Annual time series of totals and survey-snapshot biomass, catches and egg production, plus per-stage columns (`stage_1`, ...) from `ANNUAL_PRODUCTION_SCHEMA` — an empty short run returns the same columns as a populated long one.
- `save_simulation_results(filepath, agent_df, model_df; dt_year=365.0, frequency_days=dt_year, snapshot_day=274.0, meta=Dict{Symbol, Any}())`: Writes both data frames and metadata to one JLD2 file; returns the path.
- `load_simulation_results(filepath::String)`: Loads a results file written this way.
- `fit_schaefer(time, numbers, catches; label="Population")`: Surplus-production fit $\frac{dB}{dt} = rB\left(1 - \frac{B}{K}\right) - C$ in the units of `numbers`; returns `r`, `K`, `MSY`, `N_msy`, `U_msy` with trajectory, $R^2$, RMSE and AIC.
- `fit_delay_difference(time, numbers, catches; lag=1, label="Population")`: Deriso-Schnute delay-difference fit returning survival `s`, natural mortality `M` and the recruitment parameters.
- `fit_population_dynamics(source; lag=1)`: Unified whole-population dispatch — `source` may be a saved-results path, an `(agent_df, model_df)` tuple, or an annual-production table; returns a `NamedTuple` with the Schaefer, delay-difference and Leslie fits plus the annual table.
- `summarize_population_dynamics(macro_fits)`: Prints parameter estimates and diagnostics ($r$, $K$, MSY, $N_{\text{MSY}}$, $R^2$, RMSE) for each macro fit.
- `mohn_rho(terminal_peeled, terminal_base)`: Mohn's $\rho$ retrospective bias; the two peel vectors must have equal length.
- `retrospective_analysis(source; n_peels=5, model_type=:schaefer, lag=1)`: Sequentially peels terminal years, refits, and quantifies retrospective stability; `model_type = :both` is not supported here and warns before falling back.
- `summarize_retrospective(retro_result)`: Prints the peel table and Mohn's $\rho$.
- `extract_annual_stage_matrix(source; stages=build_stages())`: Returns `(stage_matrix, catch_matrix, years)` from a live model, an `(agent_df, model_df)` tuple or a saved-results path.
- `fit_generalized_leslie_model(stage_matrix, catch_matrix=zeros(size(stage_matrix)); stages=build_stages(), years=1:size(stage_matrix, 2))`: Calibrates the stage-transition projection matrix $\mathbf{A}$, dominant eigenvalue $\lambda$, damping ratio, reproductive values and elasticities into a `LeslieMatrixFit`.
- `summarize_leslie_model(fit::LeslieMatrixFit)`: Formatted Leslie diagnostics ($\lambda$, $r$, damping ratio).
- `compare_macro_models(annual_data, leslie_fit; lag=1)`: Three-way comparison (`MacroModelComparison`) of Schaefer, delay-difference and Leslie formulations on the same series.
- `summarize_macro_comparison(comp::MacroModelComparison)`: Comparative table of $R^2$, RMSE and AIC across the macro models.
- `evaluate_population_criteria(model, agent_df; min_population=nothing, max_population=nothing, max_size_drift=nothing, max_maturity_rmse=nothing, max_growth_rmse=nothing, window_steps=nothing, maturity_ogive_min_count=nothing)`: Population-fidelity scoring against the five biological criteria; `nothing` thresholds fall back to `params.criteria`, and the configured `benthic_carrying_capacity` is read from the model properties.
- `write_dataframe_csv(df, filepath)` / `read_dataframe_csv(filepath)`: CSV I/O helpers.

---

### 6.8 Visualization & Animation (`src/visualization.jl`)

CairoMakie throughout:
- `plot_population_timeseries(agent_df; model_df=nothing)`: Abundance and recruit trajectories; the recruits panel appears when `model_df` carries `:total_recruits_settled`.
- `plot_stage_distribution(model)`: Stage histogram partitioned by sex.
- `plot_size_by_sex(model)`: Kernel density and histogram of carapace widths across sexes.
- `plot_stage_allometry(model)`: Carapace size vs. stage against the theoretical lower and upper bounds.
- `plot_maturity_ogives(model; size_bins=20:10:140, ecdyses=nothing, min_count=1)`: Empirical maturity proportions vs. size with theoretical logistic ogives.
- `plot_reproduction_dynamics(agent_df, model_df)`: Egg counts, ovigerous female abundance and larval production over time.
- `plot_spatial_habitat(model)`: Bottom-temperature heatmap with agent positions.
- `plot_simulation_dashboard(model, agent_df; model_df=nothing)`: Multi-panel executive dashboard — timeseries, stage composition, spatial positions, maturity.
- `plot_stock_recruitment(sr_data; fits=nothing, title=...)`: Fitted stock-recruitment curves over the empirical annual points, with as many panels as fits provide. Convenience method `(model, agent_df; model_df=nothing, title=...)` extracts the pairs from a run first.
- `plot_population_dynamics_fits(macro_fits; title=...)`: Observed trajectory against the Schaefer, delay-difference and Leslie predictions from `fit_population_dynamics`.
- `plot_macro_model_comparison(comp::MacroModelComparison; title=...)`: Three-way comparison figure for the result of `compare_macro_models` — a separate function, not an alias.
- `plot_retrospective_analysis(retro_res)`: Retrospective peel trajectories with Mohn's $\rho$ annotation.
- `plot_stage_specific_dynamics(fit::LeslieMatrixFit; title=...)`: Stable stage distribution, reproductive values and elasticity matrix.
- `plot_functional_relationships(params_list; type=:maturity, ...)`: Species-function diagnostic panels (maturity, growth, mass, fecundity) for one or more parameter sets — `params_list` is a vector of `IBMParameters`.
- `export_all_plots(model, agent_df, output_dir; model_df=nothing, ext="png")`: Generates and exports the diagnostic figure set.
- `organism_color(org)` / `organism_size(org)` / `organism_marker(org)` / `organism_heatarray(model[, field])`: Plot-attribute mappers by sex, size, maturity and field.
- `plot_abm_summary(model; heat_field=:temperature, output_path=nothing)`: High-resolution spatial summary; written to `output_path` when given, otherwise returned as a figure.
- `animate_organism_simulation(model; filepath="organism_animation.mp4", frames=120, dt=2, framerate=15, heat_field=:temperature)`: `.mp4` animation of spatial movement and environmental dynamics.

---

### 6.9 Interactive GUI Dashboard (`src/gui.jl`)

- `launch_organism_gui(; display_window=true, config_path="")`: Launches the interactive GLMakie dashboards (control & dynamics, demographics & macro models, process functions). A `config_path` naming a readable TOML seeds the initial state from that configuration instead of struct defaults.
- `launch_organism_gui(config::Dict{Symbol, Any}; display_window=true)`: Variant taking an already-loaded configuration dictionary.
- `IBMGuiState`: Reactive container managing simulation state, model stepping and `Observables` signals.
- `step_gui_simulation!(state::IBMGuiState, days::Float64)`: Advances the model and notifies subscribers.

---

### 6.10 Backward Compatibility & Domain Aliases

Canonical naming, plus the alias surface that remains:

| Exported Symbol | Canonical Target | Category |
|---|---|---|
| `stage_size` | `get_stage_size` | Function Alias |
| `is_legal` | `is_legal_male` | Demographic Query Alias |
| `DEFAULT_FECUNDITY_PRIMIPAROUS_*` | `species.jl` constants | Allometric Invariant |
| `DEFAULT_FECUNDITY_MULTIPAROUS_*` | `species.jl` constants | Allometric Invariant |

The older alias names are gone: the `crab_*` visualization and dispersal aliases, the `weight_if_*` / `mass_if_adult_*` demographic aliases, the `plot_instar_*` plot names, `animate_snowcrab_simulation!` and `sample_crab_traits` no longer exist. The canonical `organism_*`, `mass_if_*`, `plot_stage_*` and `animate_organism_simulation` forms above are the API.

---

## 7. Biological Soundness, Parameter Audit & Verification Report

### 7.1 Life History Sequence & Developmental Indexing
The simulation architecture enforces strict structural consistency across developmental phases:
- **Egg / Brood Phase (Stage 0):** Brooded on the pleopods of mature females (`Organism.egg_count`). Thermal incubation follows:
  $$\tau_{\text{brood}} = \tau_{\text{base}} \cdot \left(\frac{T_{\text{ref}}}{\max(T_{\text{floor}}, T)}\right)^{\theta}$$
  Eggs are explicitly prohibited from existing as independent unattached cohort objects (`AggregateGroup`), ensuring that unhatched eggs do not mix with free-swimming larvae.
- **Pelagic Larval Phase (Stages 1–3):**
  - Stage 1: Zoea I
  - Stage 2: Zoea II
  - Stage 3: Megalopa
  - Governed by `intermolt_days_larval` ($[0, 36, 36, 42]$ days) and `degree_days_larval` ($[0, 270, 270, 270]$ degree-days above $T_0 = -1.5^\circ\text{C}$), reflecting empirical snow crab larval development (*Ouellet & Sainte-Marie 2018*, *Incze et al. 1987*).
  - Parabolic larval survival centered at $T_{\text{opt}} = 11.0^\circ\text{C}$ ($\sigma = 3.5^\circ\text{C}$).
  - `larval_overage_hazard`: Larvae remaining in pelagic stages beyond 84 days face escalating developmental mortality.
- **Benthic Juvenile Cohorts (Stages 4–11 / Instars 1–8):**
  - Instars 1–4 require $182.5$ days (semi-annual molts); instars 5+ require $365.0$ days (annual molts).
  - Represented as `AggregateGroup` cohorts through stage 11 (`juvenile_max_stage - 1`). Transitions follow a Poisson cumulative distribution over accumulated degree-days:
    $$h(\text{DD}) = h_0 \cdot \text{CDF}(\text{Poisson}(\text{DD}_{\text{req}}), \text{DD}_{\text{acc}})$$
    scaled by `aggregate_transition_hazard` ($h_0 = 0.35$), producing natural cohort desynchronization.
- **Individuated Agents (Stages 12–19 for males, 12–16 for females):**
  - Recruits individuate cleanly into `Organism` agents at `juvenile_max_stage = 12` with a 1:1 sex ratio (`male_split_fraction = 0.5`).
  - Spatial settlement occurs within `settlement_dispersal_radius = 2` cells with log-normal size variability ($\sigma = 0.04$).
  - Asymptotic size bounds ($L_{\infty,\text{male}} = 160.0$ mm, $L_{\infty,\text{female}} = 95.0$ mm) constrain carapace expansion via saturating log-decay curves.

### 7.2 Bioenergetic Dynamics & Respiration Budgets
- **Allometric Budget:** Kleiber-type scaling ($M^{0.75}$) models standard metabolic rate.
- **Temperature Response:** Modulated by $e^{0.08 T} \times \exp(-(T - T_{\text{opt}})^2 / (2\sigma^2))$ with $T_{\text{opt}} = 3.0^\circ\text{C}$, $\sigma = 2.5^\circ\text{C}$, capturing high metabolic performance in typical cold intermediate layer (CIL) waters.
- **Feeding & Dynamic Action:** Holling Type II functional feeding response ($F / (F + K_h)$), where ingested energy is scaled by assimilation efficiency $\eta = 0.75$, with specific dynamic action ($\text{SDA} = 0.15$) deducted before contributing to reserves.
- **Molt Energetics:** Ecdysis is permitted only when reserve exceeds $\max(E_{\text{floor}}, E_{\text{slope}} \cdot \text{size})$. Surplus energy post-molt is retained at `molt_energy_retention = 0.10`, preventing runaway accumulation.

### 7.3 Maturity, Senescence & Terminal Molt
- **Maturity Ogives:** Evaluated as log-logistic curves in size, with sexual dimorphism ($L_{50,\text{male}} = 91.8$ mm, $L_{50,\text{female}} = 51.6$ mm) and temperature sensitivity ($dL_{50}/dT = -2.0$ mm/$^\circ$C).
- **Daily Hazard Evaluation:** Evaluated continuously across daily steps as a hazard $h = -\ln(1 - P_{\text{mat}}) / \text{days\_per\_year}$, eliminating artificial seasonal pulse artifacts.
- **Terminal Molt & Senescence:** Mature females cease growth upon reaching maturity or terminal stage 16; males cease growth at maturity or stage 19. Senescence is restricted to terminally molted adults and is governed by unfloored post-terminal shell age ($t - t_{\text{terminal}}$) rather than total chronological age, preventing immature animals from suffering senescence penalties.

### 7.4 Comprehensive Parameter Inventory & Utilization Audit

| Section | Parameter | Default | Biological Role & Operational Mechanism | Used In Code |
| :--- | :--- | :--- | :--- | :--- |
| **`[toggles]`** | `use_energetics` | `true` | Enables Holling Type II forage ingestion and Kleiber metabolic expenditure. | `organism_dynamics.jl`, `simulation.jl` |
| | `use_thermal_stress` | `true` | Activates excess mortality above 8.0°C and ecdysis thermal inhibition. | `organism_dynamics.jl` |
| | `use_cannibalism` | `false` | Enables density-dependent predation on small juveniles (CW < 35 mm). | `organism_dynamics.jl`, `simulation.jl` |
| | `use_senescence` | `true` | Increases mortality on terminally molted adults past shell age threshold. | `organism_dynamics.jl` |
| | `use_spermatheca` | `true` | Enables multi-annual sperm storage across consecutive extruded clutches. | `species.jl` |
| | `use_movement` | `true` | Enables spatial locomotion across the 2D grid. | `simulation.jl` |
| | `use_directed_movement`| `true` | Biases dispersal toward cold, food-rich benthic cells via softmax taxis. | `simulation.jl` |
| | `use_degree_days` | `true` | Molting and cohort transitions require physiological thermal degree-days. | `organism_dynamics.jl`, `species.jl`, `simulation.jl` |
| | `use_molting` | `true` | Enables ecdysis and post-growth morphometric expansion. | `organism_dynamics.jl`, `species.jl`, `simulation.jl` |
| | `use_reproduction` | `true` | Governs copulation, clutch extrusion, and hatching release. | `species.jl` |
| **`[simulation]`** | `dt` | `1.0` | Discrete simulation time step in days. | `simulation.jl` |
| | `seed` | `1234` | Master pseudo-random number generator seed. | `simulation.jl` |
| | `days` | `730` | Total duration of simulation run (days). | `simulation.jl`, `cli.jl` |
| | `grid_nx`, `grid_ny` | `100`, `100` | Grid spatial resolution in cells. Synchronized via `sync_grid_dimensions!`. | `simulation.jl`, `cli.jl` |
| | `days_per_year` | `365.0` | Calendar normalization constant for rates and annual modulos. | `simulation.jl`, `organism_dynamics.jl`, `analysis.jl` |
| | `temp_seasonal_phase_day`| `80.0` | Day-of-year offset for annual temperature sinusoidal peak. | `simulation.jl` |
| | `initial_aggregate_cohort` | `[0, 2000, 0, ...]`| Initial cohort abundance distribution. Element 2 (stage 1) holds 2000 larvae. | `simulation.jl` |
| | `initial_female/male_cohort`| `fill(0.0, 20)` | Initial abundance of post-individuation cohorts by sex (element `i` is stage `i-1`); both shipped empty, so day-one individuals come solely from the aggregate cohort. | `simulation.jl` |
| | `initial_cohort_days_per_stage`| `90.0` | Base age scaling (days) for seeded aggregate cohorts. | `simulation.jl` |
| | `initial_age_days_per_stage`| `180.0` | Base age scaling (days) for seeded post-individuation individuals. | `simulation.jl` |
| | `initial_age_first_stage`| `3.0` | Stage offset for initial age assignment. | `simulation.jl` |
| | `initial_age_jitter_days`| `90.0` | Uniform random jitter applied to seeded individual age. | `simulation.jl` |
| | `initial_degree_days_min/max`| `50.0`, `300.0`| Range of pre-ecdysial degree-days assigned to seeded agents. | `simulation.jl` |
| **`[environment]`** | `temp_offset` | `0.0` | Additive temperature control: bottom-water mean in synthetic mode, anomaly on top of the hydrographic mean in GeoData mode (°C). Legacy `temp_base` remains an accepted alias. | `simulation.jl`, `cli.jl`, `gui.jl` |
| | `temp_multiplier` | `1.0` | Multiplicative temperature control: seasonal oscillation amplitude in synthetic mode; scales GeoData anomalies. Legacy `temp_amplitude` remains an accepted alias. | `simulation.jl`, `cli.jl`, `gui.jl` |
| | `temp_noise_sd` | `0.5` | Spatial standard deviation of environmental temperature noise. | `simulation.jl` |
| | `temp_base_degreedays` | `-1.5` | Threshold temperature $T_0$ (°C) below which degree-days do not accrue. | `organism_dynamics.jl`, `simulation.jl` |
| | `food_max` | `10.0` | Carrying capacity ceiling on local benthic food density. | `simulation.jl` |
| | `food_replenishment_rate`| `0.1` | Daily linear food renewal rate ($d^{-1}$). | `simulation.jl` |
| | `benthic_carrying_capacity`| `12000.0` | Total benthic capacity ($K$) across the entire grid for crowding feedback. | `organism_dynamics.jl`, `simulation.jl` |
| | `benthic_carrying_capacity_exponent`| `2.0` | Nonlinearity exponent in $(N/K)^p$ mortality feedback. | `organism_dynamics.jl`, `simulation.jl` |
| | `half_saturation_food` | `2.0` | Forage concentration where intake is at 50% saturation. | `organism_dynamics.jl`, `simulation.jl` |
| | `assimilation_efficiency`| `0.75` | Gross energetic assimilation fraction from ingested food. | `organism_dynamics.jl` |
| | `habitat_quality_mean/sd`| `0.5`, `0.1` | Gaussian background parameters for habitat suitability index (HSI). | `simulation.jl` |
| | `habitat_offset` | `0.0` | Additive habitat control: spatial-noise mean in synthetic mode (falling back to `habitat_quality_mean` when 0.0), added after scaling in GeoData mode. | `simulation.jl`, `gui.jl` |
| | `habitat_multiplier` | `1.0` | Multiplicative habitat control: scales the synthetic noise SD or the GeoData HSI field before the offset is applied. | `simulation.jl`, `gui.jl` |
| | `habitat_temperature_optimum`| `2.5` | Preferred temperature for spatial dispersal taxis (°C). | `simulation.jl` |
| | `habitat_temperature_sigma`| `1.5` | Width of thermal preference curve for dispersal. | `simulation.jl` |
| | `habitat_food_weight` | `0.5` | Relative weighting of local food availability vs temperature in taxis. | `simulation.jl` |
| | `movement_softmax_temperature`| `0.5` | Softmax temperature regulating heading stochasticity. | `simulation.jl` |
| | `settlement_dispersal_radius`| `2` | Cell radius for recruit settlement around parental/cohort position. | `organism_dynamics.jl` |
| | `settlement_size_log_sd` | `0.04` | Log-normal standard deviation in carapace size at recruitment. | `organism_dynamics.jl` |
| | `use_geodata` | `false` | Switches the run from the synthetic climatology to assimilated GeoData environmental fields. | `simulation.jl` |
| | `domain_lon`, `domain_lat` | `[-65.0, -63.0]`, `[43.0, 45.0]` | Longitude and latitude extents (°) of the GeoData extraction domain. | `simulation.jl` |
| | `time_range` | `["1993-01-01", "1993-12-31"]` | Calendar range of the GeoData extraction. | `simulation.jl` |
| | `pelagic_depth` | `15.0` | Depth (m) defining the pelagic (larval) environmental layer. | `simulation.jl` |
| | `temperature_source` | `"woa23"` | GeoData hydrography product for the bottom and surface temperature fields. | `simulation.jl` |
| | `temperature_month` | `0` | Month selector for the hydrography extract. | `simulation.jl` |
| | `bathymetry_source` | `"etopo2022"` | GeoData bathymetry product. | `simulation.jl` |
| | `geodata_input_dir` | `"inputs"` | Run-local cache directory for downloaded GeoData inputs. | `simulation.jl` |
| | `geodata_store_dir` | `"../lakehouse/GeoIBM"` | Shared environmental store written by `assimilate_environmental_inputs`; an empty string falls back to `geodata_input_dir`. | `simulation.jl`, `assimilation.jl` |
| **`[organism]`** | `max_stage_male/female` | `19`, `16` | Terminal developmental stage ceilings (instar 16 and 13 + 3 pelagic). | `types.jl`, `organism_dynamics.jl`, `simulation.jl` |
| | `stage_labels` | Vector | Consistent plot and axis labels across the full life cycle. | `types.jl`, `gui.jl`, `visualization.jl` |
| | `metabolic_rate_base` | `0.01` | Basal metabolic scaling coefficient. | `organism_dynamics.jl` |
| | `intake_rate_base` | `0.05` | Maximum daily ingestion rate coefficient. | `organism_dynamics.jl` |
| | `metabolic_mass_exponent`| `0.75` | Kleiber allometric scaling exponent on body mass. | `organism_dynamics.jl` |
| | `metabolic_temperature_coefficient`| `0.08` | Exponential thermal sensitivity factor for metabolism. | `organism_dynamics.jl` |
| | `metabolic_optimal_temperature`| `3.0` | Thermal optimum for energetic scope (°C). | `organism_dynamics.jl` |
| | `metabolic_optimal_sigma`| `2.5` | Thermal breadth for metabolic performance curve. | `organism_dynamics.jl` |
| | `specific_dynamic_action`| `0.15` | Respiratory loss fraction due to ingestion and digestion. | `organism_dynamics.jl` |
| | `energy_molt_slope` | `0.03` | Slope of energetic requirement per mm carapace width. | `organism_dynamics.jl` |
| | `energy_molt_floor` | `2.0` | Minimum baseline energy reserve required to initiate ecdysis. | `organism_dynamics.jl` |
| | `molt_energy_retention` | `0.10` | Residual reserve fraction carried forward post-ecdysis. | `organism_dynamics.jl` |
| | `molt_energy_base_fraction`| `0.85` | Invariant fraction of molt energetic requirement pre-committed. | `organism_dynamics.jl` |
| | `molt_hazard_coefficient`| `0.15` | Base rate coefficient for daily molt probability. | `organism_dynamics.jl` |
| | `molt_progress_saturation`| `3.0` | Saturation cap on molt progress ratio ($\text{DD} / \text{DD}_{\text{req}}$). | `organism_dynamics.jl` |
| | `initial_energy` | `10.0` | Initial energy reserve assigned to newly individuated recruits. | `organism_dynamics.jl`, `simulation.jl` |
| | `stage_size_sample_a/b` | `2.5`, `2.5` | Shape parameters of symmetric Beta distribution for size within stage bounds. | `organism_dynamics.jl` |
| | `shell_age_stage_offset` | `180.0` | Days accrued per stage used to infer time since last molt. | `organism_dynamics.jl` |
| | `shell_age_first_stage` | `6` | Anchor stage for shell age calculation. | `organism_dynamics.jl` |
| | `shell_condition_days` | `[30, 365, 730, 1277.5]` | Breakpoints in days for shell condition categories (1 to 5). | `organism_dynamics.jl` |
| | `egg_mass_g`, `egg_size_mm` | `7e-5`, `0.60` | Mass (g) and diameter (mm) of an individual ovum. | `types.jl`, `species.jl`, `simulation.jl` |
| | `energy_deficit_removal_probability`| `0.75` | Probability of staying in place (not moving) when energy reserves drop into deficit. | `simulation.jl` |
| | `juvenile_max_stage` | `12` | Stage boundary where cohorts transition to individual agents. | `organism_dynamics.jl`, `simulation.jl`, `types.jl` |
| | `adult_size_threshold` | `68.0` | Size threshold (mm CW) for morphometric adult categorization. | `types.jl`, `simulation.jl` |
| | `trait_quantile_min/max`| `0.01`, `0.99`| Quantile bounds for drawing individual biological traits. | `types.jl`, `simulation.jl` |
| **`[phenology]`** | `molt_season_start/end` | `15.0`, `140.0`| Day-of-year window for spring molt season. | `organism_dynamics.jl` |
| | `mating_season_start/end`| `30.0`, `120.0`| Day-of-year window for copulation and clutch extrusion. | `species.jl` |
| | `mating_search_radius` | `5` | Grid search radius in cells for males to locate receptive females. | `species.jl` |
| | `mating_energy_cost` | `8.0` | Energetic expenditure incurred by a male upon mating. | `species.jl` |
| | `mating_size_weight_exponent`| `2.0` | Exponent favoring larger males during female partner selection. | `species.jl` |
| | `sperm_stores_legal/sublegal`| `3`, `2` | Clutches stored in spermatheca based on sire size legality. | `species.jl` |
| | `minimum_clutch_size` | `1000` | Lower bound on extruded clutch size. | `species.jl` |
| | `brood_temperature_floor`| `1.0` | Minimum temperature floor preventing divergence in incubation scaling. | `species.jl` |
| | `brood_duration_base` | `365.0` | Nominal egg incubation duration (days) at $T_{\text{ref}}$. | `species.jl` |
| | `brood_temperature_reference`| `5.0` | Reference temperature for egg incubation scaling (°C). | `species.jl` |
| | `brood_temperature_exponent`| `0.30` | Power-law exponent for thermal acceleration of embryo development. | `species.jl` |
| | `brood_duration_sd` | `5.0` | Gaussian noise SD (days) added to hatch timing. | `species.jl` |
| | `molt_season_min_stage` | `13` | Lowest stage constrained by the seasonal molt gate (instars 10+). | `organism_dynamics.jl` |
| | `molt_thermal_onset/bandwidth`| `10.0`, `3.0` | Parameters for thermal inhibition of ecdysis above 10.0°C. | `organism_dynamics.jl` |
| **`[mortality]`** | `hazard_cap` | `1.0` | Maximum cap on daily individual mortality hazard. | `organism_dynamics.jl` |
| | `baseline_rate_by_stage`| Vector (0.35 to 0.06) | Stage-specific natural background mortality rates ($a^{-1}$). | `organism_dynamics.jl` |
| | `rate_predation` | `0.05` | Base predation mortality rate ($a^{-1}$). | `organism_dynamics.jl` |
| | `rate_soft_shell` | `0.12` | Post-molt soft-shell vulnerability rate ($a^{-1}$). | `organism_dynamics.jl` |
| | `rate_starvation` | `0.20` | Starvation mortality rate under energetic deficit ($a^{-1}$). | `organism_dynamics.jl` |
| | `rate_thermal` | `0.05` | Base thermal stress mortality rate ($a^{-1}$) above threshold. | `organism_dynamics.jl` |
| | `rate_senescence` | `0.05` | Senescence acceleration rate for terminally molted adults ($a^{-1}$). | `organism_dynamics.jl` |
| | `rate_cannibalism` | `0.05` | Juvenile cannibalism encounter rate ($a^{-1}$). | `organism_dynamics.jl` |
| | `rate_handling` | `0.05` | Discard handling mortality rate ($a^{-1}$) applied upon legal capture. | `simulation.jl` |
| | `handling_mortality_days`| `365.0` | Duration window where handling mortality applies. | `simulation.jl` |
| | `baseline_trait_sigma` | `0.25` | Inter-individual variability scale for baseline mortality trait quantile. | `organism_dynamics.jl` |
| | `predation_size_reference/decay`| `5.0`, `0.03`| Size-refuge parameters attenuating predation: $e^{-0.03 (\text{CW} - 5)}$. | `organism_dynamics.jl` |
| | `soft_shell_duration/decay_days`| `30.0`, `15.0`| Vulnerability duration (days) and exponential decay time post-molt. | `organism_dynamics.jl` |
| | `starvation_critical_fraction`| `0.05` | Critical biomass fraction determining starvation hazard scaling. | `organism_dynamics.jl` |
| | `temp_thermal_stress` | `8.0` | Temperature threshold (°C) above which thermal mortality begins. | `organism_dynamics.jl` |
| | `thermal_exponent` | `1.5` | Power exponent $(T - 8.0)^{1.5}$ for thermal stress scaling. | `organism_dynamics.jl` |
| | `senescence_age_threshold`| `3.0` | Shell age threshold (years post-terminal molt) where senescence begins. | `organism_dynamics.jl` |
| | `senescence_exp_rate` | `0.80` | Exponential acceleration rate of senescence hazard with shell age. | `organism_dynamics.jl` |
| | `cannibalism_size_threshold`| `35.0` | Carapace size ceiling (mm) below which individuals are vulnerable. | `organism_dynamics.jl` |
| | `cannibalism_density_ref/cap`| `10.0`, `2.0` | Density normalization and maximum multiplier for cannibalism. | `organism_dynamics.jl` |
| | `aggregate_transition_hazard`| `0.35` | Daily rate coefficient for aggregate cohort stage molting. | `organism_dynamics.jl` |
| | `aggregate_baseline_hazards`| Vector | Stage-specific baseline daily mortality for aggregate cohorts. | `organism_dynamics.jl` |
| | `larval_overage_hazard` | `0.05` | Excess mortality rate for larvae exceeding 84 days in pelagic stages. | `organism_dynamics.jl` |
| | `larval_temp_opt/sigma` | `11.0`, `3.5` | Optimal temperature (11°C) and tolerance curve for larval survival. | `organism_dynamics.jl` |
| **`[species]`** | Maturity parameters | $L_{50,\text{m}}=91.8, L_{50,\text{f}}=51.6$ | Logistic ogives with SDs and slopes calibrated to DFO surveys. | `species.jl` |
| | `maturity_size_gate` | `35.0` | Size gate (mm) below which maturity is impossible. | `species.jl` |
| | `maturity_l50_temp_slope`| `-2.0` | Shift in $L_{50}$ per °C deviation from reference. | `species.jl` |
| | `maturity_temp_reference`| `4.0` | Reference ambient temperature (°C). | `species.jl` |
| | Fecundity parameters | Intercepts & slopes | Moncton and Sainte-Marie allometric models for primiparous and multiparous clutches. | `species.jl` |
| | `primiparous_shell_age_max`| `2.0` | Shell age cutoff (years) classifying a female as primiparous. | `species.jl` |
| | Growth increment parameters| Intercepts, slopes, betas | Hiatt growth increment model with condition and temperature modifiers. | `species.jl` |
| | `growth_maturation_penalty_*`| Mean: $-0.135$, SD: $0.009$ | Somatic growth reduction during the terminal molt into maturity. | `species.jl` |
| | `growth_condition_coefficient`| `0.05` | Positive modifier on growth increment from nutritional condition. | `species.jl` |
| | `growth_temperature_coefficient`| `-0.004` | Growth increment sensitivity to ambient temperature. | `species.jl` |
| | `growth_min_expansion_*`| Immature: 0.12, Mature: 0.06 | Minimum fractional size expansion per molt. | `species.jl` |
| | `max_size_male/female` | `160.0`, `95.0` | Asymptotic physiological size ceilings. | `species.jl`, `organism_dynamics.jl` |
| | `size_lb_*` | Anchor, intercept, decay | Empirical lower bounds per stage saturating toward `max_size_*`. | `species.jl` |
| | `benthic_anchor_stage` | `5.0` | First benthic stage where sex-specific size curves take effect. | `species.jl`, `simulation.jl` |
| | `selectivity_s50/sigma` | `95.0`, `5.0` | Commercial fishery gear selectivity ogive parameters. | `species.jl` |
| | `mass_alpha/beta_*` | Immature & mature | Length-weight power relationship coefficients ($W = \alpha L^\beta$). | `species.jl` |
| | `intermolt_days_schedule`| Vector | Minimum intermolt days for benthic stages (182.5d early, 365d late). | `species.jl` |
| | `degree_days_schedule` | Vector | Pre-ecdysial physiological thermal degree-day requirements. | `species.jl` |
| | `intermolt_days_larval` | `[0, 36, 36, 42]` | Pelagic intermolt durations for egg, zoea 1, zoea 2, megalopa. | `species.jl` |
| | `degree_days_larval` | `[0, 270, 270, 270]`| Pelagic thermal degree-day requirements above -1.5°C. | `species.jl` |
| | `leslie_max_male/female_stage`| `15`, `14` | Stage ceiling for Leslie stage definitions. | `species.jl` |
| | `leslie_legal_size_male` | `95.0` | Minimum legal size threshold in Leslie projection. | `species.jl` |
| | `recruitment_stage` | `13` | Biological recruitment stage entering the delay-difference model. | `species.jl`, `analysis.jl` |
| **`[fishery]`** | `legal_size_male` | `95.0` | Commercial legal carapace width threshold (mm). | `types.jl`, `simulation.jl` |
| | `no_take_proportion` | `0.0` | Fraction of the grid's cells partitioned into a no-take zone that the fishery does not harvest. | `simulation.jl` |
| | `fishing_probability_daily`| `0.0` | Daily capture probability per legal male; master fishery switch. | `types.jl`, `simulation.jl` |
| | `fishing_season_start/end`| `90.0`, `150.0` | Day-of-year window when the fishery is actively harvesting. | `types.jl`, `simulation.jl` |
| **`[output]`** | `save`, `save_file` | `true`, `"simulation_results.csv"` | Annual CSV indicator persistence controls. | `simulation.jl`, `cli.jl` |
| | `snapshot_day` | `274.0` | Day-of-year for annual demographic snapshot (~1 September). | `simulation.jl`, `cli.jl` |
| | `frequency_days` | `365.0` | Sampling interval (days) between periodic observations. | `simulation.jl`, `cli.jl` |
| | `plot`, `output_dir` | `false`, `"outputs"` | Diagnostic plotting controls and destination path. | `cli.jl`, `visualization.jl` |
| | `animate`, `animation_file`| `false`, `"organism_animation.mp4"` | Video rendering switch and target file. | `cli.jl`, `visualization.jl` |
| | `anim_frames/dt/fps/field`| `120`, `2`, `15`, `"temperature"` | Animation length, step jump, frame rate, and field mapped. | `cli.jl`, `visualization.jl` |
| **`[leslie]`** | `survival_stage1/2`, `gamma_stage1/2`| 0.74–0.78 | Early recruit and juvenile survival and transition rates. | `types.jl`, `analysis.jl` |
| | `male_split_fraction` | `0.50` | Recruits male fraction upon individuation. | `types.jl`, `organism_dynamics.jl`, `simulation.jl` |
| | Male/female survival & transitions | Various | Sex-specific survival, transition decays, and terminal maturation rates. | `types.jl`, `analysis.jl` |
| | Fecundity parameters | Base: 0.85, Exp: 2.5 | Size-dependent per-capita egg contribution into stage 1. | `types.jl`, `analysis.jl` |
| | `trajectory_error_weight`| `0.5` | Weight on recursive trajectory relative to one-step process error. | `analysis.jl` |
| | `first_sex_specific_stage`| `9` | Developmental stage where sex-specific decay anchors. | `analysis.jl` |
| **`[criteria]`** | `min_population` | `1` | Persistence floor (not-extinct threshold). | `types.jl`, `analysis.jl` |
| | `max_population_multiple`| `3.0` | Explosion ceiling as multiple of carrying capacity $K$. | `types.jl`, `analysis.jl` |
| | `max_size_drift` | `0.20` | Maximum allowable fractional shift in mean size across window. | `types.jl`, `analysis.jl` |
| | `max_maturity_rmse` | `0.35` | Maximum allowable RMSE vs cumulative maturity curve. | `types.jl`, `analysis.jl` |
| | `max_growth_rmse` | `15.0` | Maximum allowable RMSE vs reference growth schedule (mm). | `types.jl`, `analysis.jl` |
| | `window_steps` | `10` | Recent steps used for evaluating size stability window. | `types.jl`, `analysis.jl` |
| | `maturity_ogive_min_count`| `10` | Minimum sample count required per size bin to evaluate maturity ogive. | `types.jl`, `analysis.jl` |
| | Penalty & weight terms | Various | Composite optimization objective loss weights. | `analysis.jl` |
| | `ogive_bin_start/step/stop`| `20.0`, `10.0`, `140.0` | Carapace size binning range (mm) for empirical ogives. | `analysis.jl` |

---

## 8. Deep Logic, Consistency and Correctness Check (2026-10)

> **Status (2026-10-23):** every finding below has been addressed in `src/`, `configs/`
> and the documentation, and the fixes are covered by the test suite's current
> baseline. This section is kept as the dated audit record. The resolution of each
> finding, the re-audit results, and corrections to three of the audit's own claims —
> M8 (the damping-ratio row had the standard convention backwards: the code's
> ρ = λ₁/|λ₂| ≥ 1 is the textbook Caswell 2001 §9.1 definition, "larger is faster"),
> M2 (the denominator is benthic-only and the per-cell rounding is intentional) and
> M12 (stage-column parity holds on the empty-results path) — are recorded in
> **[`docs/deep_check_2026-10.md`](deep_check_2026-10.md)** §10 and the 2026-10-23
> addendum in §11.

A full audit of `src/`, `configs/` and `test/` was carried out on 2026-10-22:
every module was read in full, cross-module call graphs were traced, the
third-party contracts the model depends on were verified against the installed
package sources, and the critical findings were reproduced by execution.

The complete report — with file-and-line references, reproductions, and the
justification for each judgement — is in
**[`docs/deep_check_2026-10.md`](deep_check_2026-10.md)**.

**Summary of the audit:** 3 critical, 22 major and 31 minor findings, plus 3
places where a docstring and the code disagree. The package's biology and
parameterisation are internally consistent; the defects are concentrated in the
Agents.jl integration boundary, the configuration plumbing, and the statistics
layer.

### 8.1 Critical findings

| ID | Location | Finding | Evidence |
|---|---|---|---|
| **C1** | `src/simulation.jl:1597` | The fishery harvest path reads `model.properties`, which is not a model field. Agents.jl 7.1.0 resolves model properties out of the properties container (`core/model_abstract.jl:120`), so this is a lookup of the key `:properties` and raises `KeyError: key :properties not found`. Any run with a fishery enabled aborts on the first in-season day. | Reproduced by execution; also fails in the project's own suite at `test/test_config_parameters.jl:364`. The correct accessor `abmproperties(model)` is used at `src/simulation.jl:1626` and `:1715`. |
| **C2** | `src/assimilation.jl:216`, `:188`, `:110-123` | `assimilate_environmental_inputs` never writes the hydrography artefact it reports as written: the `:woa23` provider returns interpolators and paths, not a `GeoDataset`, so `ds_hydro isa GeoDataset` is false. The writer, the presence check and the reader use three different names, so `read_assimilated_environment` raises "no [:hydrography]" immediately after a successful assimilation. | `GeoDataSources/src/woa23.jl:313-325` (return contract), `:137-139` (artefact naming) vs `src/assimilation.jl:188` (writer), `:110-123` (presence check), `src/simulation.jl:770` (reader). |
| **C3** | `src/simulation.jl:960` | `GridSpace(grid)` defaults to `periodic = true` in Agents.jl 7.1.0, so `nearby_positions` returns *wrapped* neighbours and dispersal can move an organism from one side of the domain to the other in one step — including across the no-take / fished partition, which is defined on the x index. | The model's own type parameter records it: `StandardABM{GridSpace{2, true}, ...}`. Default at `~/.julia/packages/Agents/lKaNl/src/spaces/grid_single.jl:22,35`; consumer at `src/simulation.jl:1415`. |

### 8.2 Major findings

| ID | Location | Finding |
|---|---|---|
| M1 | `src/types.jl:514-515`, `src/simulation.jl:1215,1289-1297` | The per-sex cohort vectors are length 12 (stages 0-11) but individual seeding enumerates stages 12+, so **no individual organism is ever seeded from them**. Reproduced: 0 agents seeded. |
| M2 | `src/simulation.jl:1231,1297` | `seed_sum` counts stages that seed nothing, and the per-cell rounding means the seeded population need not equal `n_agents`. |
| M3 | `src/simulation.jl:1100-1108`, `:1793` | Agents.jl defaults `agents_first = true`, so `model_step!` runs **after** the agent phase: day 1 is simulated against the initial field, and the calendar the agents read lags by one step. |
| M4 | `src/simulation.jl:962-966` vs `src/organism_dynamics.jl:919` | `model.dt` and `params.simulation.dt` come from different places; on the keyword path they differ (2.0 vs 1.0), so ageing and hazards disagree. |
| M5 | `src/organism_dynamics.jl:674-690`, `src/simulation.jl:1096` | `current_day_of_year`'s documented fallback is unreachable — `:day_of_year` is always present — so the GUI-tick path reads a stale calendar. |
| M6 | `src/simulation.jl:962-966` | The seed actually used is never written to `params.simulation.seed`. |
| M7 | `src/analysis.jl:1509,1563` | `optimize_blocks` lacks the "every trial was penalised" guard that `optimize_model` has, so a wholly diverging Leslie fit is reported as a successful one. |
| M8 | `src/analysis.jl:2700-2703` | The Leslie damping ratio is inverted: λ₁/\|λ₂\| instead of \|λ₂\|/λ₁ (Caswell 2001, §9.1). *(Row corrected 2026-10-23: the code's λ₁/\|λ₂\| ≥ 1 **is** the standard §9.1 damping ratio — larger means faster convergence — and this finding was a mis-statement of the convention; see [`deep_check_2026-10.md`](deep_check_2026-10.md) §11.)* |
| M9 | `src/analysis.jl:3574-3575`, `:3933-3937` | The calibrated maturity curve applies the temperature response twice (fitted logit term *and* an L50 shift), and the reported (L50, slope) is not the fitted curve. |
| M10 | `src/analysis.jl:3759-3761`, `:3929` | The maturity fallback stores a mm/°C quantity in a dimensionless logit-coefficient slot. |
| M11 | `src/analysis.jl:687-712` | The Ricker and Beverton-Holt predictors stop reproducing the fitted line when the slope is non-positive, so R²/AIC/`best_model` are computed on a curve that was not fitted. |
| M12 | `src/analysis.jl:2404-2413` | The Leslie fit is estimated on a fabricated stage decomposition built from an undocumented constant (`exp(-0.28·(stage-4))`). |
| M13 | `src/analysis.jl:470-489`, `src/simulation.jl:1084-1091` | The per-clutch stock-recruitment extraction is always empty: nothing ever populates `:clutch_recruitment_records`. |
| M14 | `src/cli.jl:309-310`, `:241-260` | `--temp_base` / `--temp_amplitude` are advertised as legacy aliases but are silently ignored (the alias is handled everywhere *except* the CLI parser). Reproduced, and fails at `test/test_config_parameters.jl:175` and `:1263`. |
| M15 | `src/types.jl:913`, `src/simulation.jl:1127` | `no_take_proportion` is declared and read but absent from both TOML files, so the no-take line is not configurable. Fails at `test/test_config_parameters.jl:244`. |
| M16 | `src/visualization.jl:54-58`, `:614-619` | The mature-adult series is never plotted: the guarded column `count_is_adult` does not exist in the observation schema. |
| M17 | `src/visualization.jl:860-861` | `round(Int, bh.R_max)` throws `InexactError` when `R_max = Inf` (a non-positive fitted slope), aborting the whole plot export. |
| M18 | `src/gui.jl:2179-2212`, `:1791-1813` | GUI "Reset" does not clear the snapshot cache, so "Restore Year" can reinstate the previous run's model. |
| M19 | `src/assimilation.jl:190-203`, `:229-243` | The manifest records a product and a domain that the copied hydrography artefact does not have. |
| M20 | `src/simulation.jl:1799-1802` | The food field is not clamped, so it goes negative for `dt > 1/replenishment_rate`. |
| M21 | `src/simulation.jl:1611-1637` | A caught-and-released organism skips dispersal and reproduction for the rest of the step. |
| M22 | `configs/default.toml:94,101` vs `configs/snowcrab.toml:109,115` | Two undocumented differences between the shipped configurations (`use_geodata`, `bathymetry_source`), breaking the project's own configuration-diff invariant. |

### 8.3 Test-suite baseline at the time of the audit

`julia --project=. test/runtests.jl config` reports
`1456 passed, 4 failed, 3 errored`. The failures correspond to C1, M14, M15
and M22; the full mapping is tabulated in
[`docs/deep_check_2026-10.md` §7](deep_check_2026-10.md#7-current-test-suite-failures-baseline).

### 8.4 Reproducing the findings

The verification scripts are kept in `scratch/`:

```
scratch/verify_agents.jl    # Agents.jl API: model.properties, GridSpace periodicity
scratch/verify_claims.jl    # seeding, dt divergence, calendar staleness, the fishery KeyError
scratch/verify_order.jl     # the agents_first step ordering
scratch/verify_cli.jl       # the --temp_base / --temp_amplitude alias behaviour
```

### 8.5 Recommended order of work

1. **C1** — one-line fix; unblocks the fishery, which is currently unusable.
2. **C3** — one-line fix, but it changes every dispersal result.
3. **M3** — decide the phase order explicitly and align the comments.
4. **M1 + M2** — extend the cohort vectors, add a length validation, correct `seed_sum`.
5. **M14 + M15 + M22** — restore the CLI alias, add the two missing TOML keys, annotate the undocumented differences.
6. **M7-M12** — the statistics layer; reported numbers change, so affected figures need regenerating.
7. **M16, M17, M21** — plotting and export paths.
8. **C2 + M19** — the assimilation path; the store contract needs designing rather than patching.
