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
│   ├── GeoIBM.md        # Comprehensive architecture and API reference (this file)
│   └── notes.md              # Research notes and design specifications
├── src/
│   ├── GeoIBM.jl        # Module definition, includes, and exported public APIs
│   ├── types.jl              # Core type system: Organism, StageDefinition, IBMParameters
│   ├── organism_dynamics.jl  # Generic growth, allometry fallbacks, bioenergetics, molt
│   ├── species.jl            # Species behaviour: allometry, phenology, reproduction
│   ├── simulation.jl         # Grid initialization, agent/model steppers, simulation runner
│   ├── analysis.jl           # KDE, OLS/IRLS estimation, Leslie matrices, macro models
│   ├── visualization.jl      # Diagnostic figures and video animations (CairoMakie)
│   └── gui.jl                # Real-time interactive 3-dashboard GUI (GLMakie)
├── configs/
│   ├── default.toml          # Every declared parameter, at its struct default
│   └── snowcrab.toml         # Snow crab overlay: biology plus operating scale
├── scripts/
│   ├── organism_ibm.jl       # CLI simulation application
│   └── organism_orchestrator.jl # Parameter variation and calibration orchestrator
└── test/
    └── runtests.jl           # Complete automated test suite
```

### Component Roles

| Source File | Primary Responsibility |
|---|---|
| [`src/types.jl`](src/types.jl) | Declares abstract interface (`SpeciesBehavior`, `SpeciesParams`), agent structs (`Organism`), cohort containers (`AggregateGroup`), configuration (`IBMParameters`), and matrix types (`StageDefinition`, `LeslieMatrixFit`, `MacroModelComparison`). |
| [`src/organism_dynamics.jl`](src/organism_dynamics.jl) | Implements generic biological interfaces and fallbacks, Kleiber metabolic bioenergetics ($W^{0.75}$), Holling Type II consumption, ecdysis mechanics, mortality hazards, and spatial dispersal. |
| [`src/species.jl`](src/species.jl) | Implements `SpeciesParams`, `SpeciesBehaviorImpl`, empirical Hiatt increments, log-logistic maturity schedules, allometric fecundity, thermal ecdysial thresholds, and primiparous/multiparous reproductive cycles. |
| [`src/simulation.jl`](src/simulation.jl) | Manages model instantiation (`initialize_model`), environmental field generation, scheduled time stepping (`agent_step!`, `model_step!`), and batch execution (`run_simulation`). |
| [`src/analysis.jl`](src/analysis.jl) | Computes kernel density estimations (KDE), iterative logistic fits (IRLS), demographic summaries, Schaefer and Delay-Difference biomass models, generalized Leslie matrices, and retrospective analyses. |
| [`src/visualization.jl`](src/visualization.jl) | Generates executive diagnostic figures, maturity ogives, stock-recruitment plots, and video animations via CairoMakie (`Figure` output). |
| [`src/gui.jl`](src/gui.jl) | Provides interactive real-time multi-dashboard steering GUI with reactive particle streaming, parameter steering, and macro-ecological model fitting via GLMakie (`launch_organism_gui`). |

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
square kilometre. The seeded stock is 2000 individuals plus 4500 in aggregate
cohorts, so the density multiplier starts at 1.42 and rises as the population
grows toward K.

Three `Float64` fields of that size cost 23 MB and are the only per-cell state.
The measured cost is about **3 ms per simulated day**, so a two-year run is
roughly 2 s of dynamics plus whatever the observation columns add.

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

**A sparse grid is cheap per organism.** At 1000 × 1000 with a few thousand
organisms the cells are almost all empty, so `nearby_agents` returns nothing and
the per-organism cost is low. The same organisms on a 30 × 30 grid were roughly
three times more expensive *per organism*, because each neighbourhood query
returned hundreds of occupants instead of none. Grid size does not track
population cost the way you would expect: the per-cell field updates scale with
the grid, and the per-organism neighbour queries scale with local occupancy.

**Cell count is now nearly irrelevant.** After the cohort merge the cost is 3.2
ms/day at 1000 × 1000 and 1.0 ms/day at 200 × 200 for the same population — the
two environmental fields are 85% of what is left.

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
  problem: at K = 300 against the same seed the multiplier is 2050 and every
  hazard is a certainty on day one, which presents as an extinction the model
  never modelled. `initialize_model` warns when the stock already exceeds K and
  names the fields to change.
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
| `--n_agents=<int>` | Integer | `300` | Initial population size. |
| `--grid=<nx,ny>` | Int,Int | `100,100` | 2D spatial grid dimensions. |
| `--grid_nx=<int>`, `--grid_ny=<int>` | Integer | `100` | Grid dimensions, individually. |
| `--seed=<int>` | Integer | `1234` | Random seed. |
| `--temp_base=<float>` | Float | `5.0` | Baseline bottom water temperature (°C). |
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
julia --project=. scripts/organism_orchestrator.jl --vary="temp_base=2.0:1.0:5.0;fishing_prob=0.0,0.02,0.05" --days=1095 --runs=3
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
    stage::Int                             # Developmental stage index (0 to 16)
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
Top-level configuration structure with modular nested settings:
- `toggles::ProcessToggles`: Granular biological and operational feature switches.
- `simulation::SimulationParams`: Time step $\Delta t$, random seed, agent count, duration.
- `environment::EnvironmentParams`: Baseline temperature, seasonal amplitude, food dynamics.
- `organism::OrganismParams`: Metabolic and consumption rates, molt energy thresholds.
- `phenology::PhenologyParams`: Molting and mating seasonal calendar bounds, brooding period.
- `mortality::MortalityParams`: Senescence thresholds, post-molt vulnerability duration.
- `fishery::FisheryParams`: Legal size thresholds ($L_{50}$), daily harvest probability.
- `output::OutputParams`: Reporting frequencies, output directory paths, visualization flags.

There is no binning section. Demographic histogramming was configured but never implemented,
so the section only ever advertised settings that had no effect; it has been removed rather
than left as an inert control.
- `binning::BinningParams`: Demographic histogram binning configurations.

---

### 6.2 Snow Crab Species Implementation (`src/species.jl`)

All biological processes, allometric constants, and parameterizations specific to the snow crab (*Chionoecetes*) are centralized within `src/species.jl`.

#### Parameters: `SpeciesParams <: SpeciesParams`
- **Maturity Ogives**:
  - `maturity_l50_male::Float64 = 91.8` (mm CW), `maturity_l50_male_sd::Float64 = 1.2`
  - `maturity_slope_male::Float64 = 6.9957`, `maturity_slope_male_sd::Float64 = 0.25`
  - `maturity_l50_female::Float64 = 51.6` (mm CW), `maturity_l50_female_sd::Float64 = 0.8`
  - `maturity_slope_female::Float64 = 12.9812`, `maturity_slope_female_sd::Float64 = 0.40`
- **Fecundity Allometry**:
  - `fecundity_parameterization::String = "moncton"` (Sainte-Marie 1993)
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
const Organism = Organism
```

#### Concrete Methods:
- `stage_size_lower_bound(behavior::SpeciesBehaviorImpl, stage::Real, sex)`: Lower bound of carapace width (mm) across stages 1 to 16.
- `llbm(stage)`: Convenience shortcut for male lower bound: `stage_size_lower_bound(SpeciesBehaviorImpl(), stage, :male)`.
- `llbf(stage)`: Convenience shortcut for female lower bound: `stage_size_lower_bound(SpeciesBehaviorImpl(), stage, :female)`.
- `get_stage_size(behavior::SpeciesBehaviorImpl, stage::Real, sex; type=:mid, rng=...)`: Bounds (`:lb`, `:ub`), midpoint (`:mid`), or stochastic beta-distributed (`:random`) carapace width.
- `growth_increment(behavior::SpeciesBehaviorImpl, log_size::Real; sex, mat, temp, condition, rng, growth_quantile)`: Evaluates post-molt natural logarithm $\ln(\text{size}_{t+1})$.
- `predict_postgrowth_size(behavior::SpeciesBehaviorImpl, size::Real; ...)`: Linear post-molt size in millimeters.
- `growth_schedule(behavior::SpeciesBehaviorImpl)`: Reference `DataFrame` of stage sizes and increments.
- `maturity(behavior::SpeciesBehaviorImpl, sex, log_size::Real; temp, maturity_quantile, rng, deterministic)`: Log-logistic morphometric maturity probability.
- `fecundity(behavior::SpeciesBehaviorImpl, size::Real; is_primiparous, method, shell_age, rng, deterministic, fecundity_quantile)`: Brood egg count estimation.
- `selectivity(behavior::SpeciesBehaviorImpl, size::Real; s50, s50_sigma)`: Logistic gear retention probability.
- `calculate_mass(behavior::SpeciesBehaviorImpl, size::Real, sex; mat, stochastic, rng)`: Wet body mass (grams) from carapace width (mm).
- `build_stages(behavior::SpeciesBehaviorImpl; max_male_stage=12, max_female_stage=11)`: Constructs the demographic stage hierarchy.
- `min_intermolt_days(behavior::SpeciesBehaviorImpl, stage::Int)`: Minimum chronological days required between molts.
- `required_degree_days(behavior::SpeciesBehaviorImpl, stage::Int)`: Thermal degree-days required before ecdysis.
- `perform_development!(behavior::SpeciesBehaviorImpl, org::Organism, model)`: Advances physiological development and molting.
- `perform_reproduction!(behavior::SpeciesBehaviorImpl, org::Organism, model)`: Executes seasonal copulation, spermathecal storage, brood extrusion, and hatching.

---

### 6.3 Individual State Queries & Predicates (`src/types.jl`)

High-performance inline predicates for demographic filtering:
- `is_adult(org::Organism)`: True if individual has undergone terminal morphometric molt (`org.is_mature`).
- `is_adult_male(org::Organism)`: True if male and mature.
- `is_adult_female(org::Organism)`: True if female and mature.
- `is_ovigerous(org::Organism)`: True if carrying an extruded clutch (`org.egg_count > 0`).
- `is_legal_male(org::Organism; legal_size=95.0)`: True if male and `size >= legal_size`.
- `is_legal(org::Organism; legal_size=95.0)`: Alias for `is_legal_male`.
- `mass_if_adult_female(org::Organism)`: Returns `org.mass` if adult female, otherwise `0.0`.
- `mass_if_adult_male(org::Organism)`: Returns `org.mass` if adult male, otherwise `0.0`.
- `mass_if_legal_male(org::Organism; legal_size=95.0)`: Returns `org.mass` if legal male, otherwise `0.0`.
- `weight_if_*`: Backwards-compatible aliases for `mass_if_*`.

---

### 6.4 Organism Dynamics & Bioenergetics (`src/organism_dynamics.jl`)

- `calculate_energy_change(temp::Real, food::Real, size::Real; params=IBMParameters(), mass=nothing)`:
  Evaluates net energy assimilation $\frac{dE}{dt}$ and actual forage intake $I_{\text{actual}}$ via Kleiber metabolic scaling ($W^{0.75}$) and Holling Type II functional response.
- `calculate_energy_change(org::Organism, temp::Real, food::Real; params=IBMParameters())`:
  Agent-level method wrapping the above.
- `molt_probability(org::Organism, current_temp::Real, current_time::Real=0.0; params=IBMParameters())`:
  Computes daily probability of undergoing ecdysis based on intermolt days, degree-days, energy reserves, and temperature.
- `check_molt_conditions(org::Organism, current_temp::Real, current_time::Real=0.0; params=IBMParameters(), rng=Random.default_rng())`:
  Evaluates boolean criteria for ecdysis.
- `perform_molt!(org::Organism, current_time::Real, current_temp::Real; params=IBMParameters(), rng=Random.default_rng())`:
  Executes ecdysis, advances stage, evaluates maturity transition, updates size and mass via Hiatt models, and resets degree-days.
- `calculate_mortality_probability(org::Organism, current_temp::Real, local_density::Real, current_time::Real=0.0; params=IBMParameters())`:
  Computes aggregate daily mortality hazard combining baseline natural mortality, senescence, thermal stress, and density-dependent competition.
- `shell_condition(days_since_molt::Real)`:
  Translates days post-ecdysis into standard fishery shell condition categories (1: soft, 2: new hard, 3: intermediate, 4: old, 5: very old).
- `step_aggregate_groups!(model; behavior=model.species_behavior)`:
  Advances pre-individuation aggregate cohorts (stages 0 to 5) through survival, growth, and stage graduation into spatial agents at stage 6.
- `organism_disperse!(org::Organism, model; rng=abmrng(model))`:
  Executes spatial movement governed by thermal and forage habitat taxis.
- `crab_disperse!(args...; kwargs...)`: Alias for `organism_disperse!`.

---

### 6.5 Simulation Engine (`src/simulation.jl`)

- `default_ibm_configuration()`:
  Returns complete baseline configuration `Dict{Symbol, Any}` encompassing spatial grid geometry, runtime duration, environmental forcings, bioenergetics, life-history phenology, fishery dynamics, process toggles, and output handling.
- `load_ibm_configuration(filepath::String)`:
  Reads simulation configuration from a structured TOML file.
- `save_ibm_configuration(config::AbstractDict, filepath::String)`:
  Serializes configuration dictionary to TOML format.
- `initialize_model(; N_agents=100, dims=(50, 50), temp_map=nothing, food_map=nothing, dt_days=1.0, params=IBMParameters(), behavior=default_species_behavior(), rng_seed=nothing)`:
  Constructs and seeds an `Agents.StandardABM` spatial simulation model.
- `initialize_model(config_filepath::String; kwargs...)`:
  Initializes model from a TOML configuration file with keyword overrides.
- `initialize_model(config::AbstractDict; kwargs...)`:
  Initializes model from a configuration dictionary with keyword overrides.
- `agent_step!(org::Organism, model)`:
  Advances an individual agent through one time step (energetics, degree-days, molting, mortality, harvest, dispersal, reproduction).
- `model_step!(model)`:
  Advances global environment fields (temperature seasonality, food replenishment), steps aggregate cohorts, and updates demographic caches.
- `run_simulation(model::StandardABM, total_days::Real; agent_collect=[], model_collect=[])`:
  Executes the simulation across `total_days` and collects agent- and model-level time-series `DataFrames`.
- ABM Utility Accessors:
  - `nagents(model)`: Total active spatial agents.
  - `allagents(model)`: Iterator over all agents.
  - `abmrng(model)`: Model random number generator.
  - `safe_mean(x)`: Mean calculation returning `0.0` for empty collections.

---

### 6.6 Empirical Biology, Surveys, & Demographics (`src/analysis.jl`)

- `gaussian_kde(data::AbstractVector{<:Real}; eval_points=nothing, bandwidth=nothing)`:
  1D Gaussian kernel density estimation.
- `derive_stage_proportions_kde(sizes::AbstractVector{<:Real}, sexes::AbstractVector; stages=build_stages())`:
  Discretizes empirical size distributions into stage proportions.
- `generate_mock_survey_data(; n_samples=500, rng=Random.default_rng(), behavior=default_species_behavior())`:
  Synthesizes research survey catch records across size, sex, maturity, shell condition, and fecundity.
- `fit_empirical_biology_model(data::DataFrame)`:
  Calibrates allometric exponents, maturity ogives, and phenotypic covariance matrices from survey data.
- `estimate_demographics(model::EmpiricalBiologicalModel; size::Real, sex, temp=nothing, shell_age=1.0, rng=Random.default_rng())`:
  Predicts mass, maturity probability, and fecundity for an individual crab.
- `sample_organism_traits(model::EmpiricalBiologicalModel; rng=Random.default_rng())`:
  Samples correlated individual phenotypic trait quantiles.
- `sample_crab_traits(args...; kwargs...)`: Alias for `sample_organism_traits`.
- `summarize_empirical_model(model::EmpiricalBiologicalModel; io=stdout)`:
  Prints formatted ASCII parameter summary.
- `compute_empirical_ogive(sizes::AbstractVector{<:Real}, mature_flags::AbstractVector{<:Integer}; n_bins::Int=15)`:
  Computes binned proportions mature for empirical ogive fitting.
- `summarize_population(agent_df::DataFrame; frequency_days=365.0)`:
  Aggregates simulation results into annual demographic summaries.

---

### 6.7 Macro-Ecological Models & Stock Assessment (`src/analysis.jl`)

- `extract_stock_recruitment(agent_df::DataFrame, model_df::DataFrame; lag_years::Int=4, frequency_days=365.0)`:
  Computes annual spawner biomass and lagged recruitment pairs.
- `fit_stock_recruitment(spawners::AbstractVector{<:Real}, recruits::AbstractVector{<:Real}; model_type::Symbol=:beverton_holt)`:
  Fits Beverton-Holt ($R = \frac{\alpha S}{1 + \beta S}$) or Ricker ($R = \alpha S e^{-\beta S}$) models.
- `summarize_stock_recruitment(fit::NamedTuple; io=stdout)`:
  Prints formatted stock-recruitment parameters ($R^2$, $\alpha$, $\beta$, $S_{\text{max}}$).
- `extract_annual_production(agent_df::DataFrame, model_df::DataFrame; frequency_days=365.0)`:
  Extracts annual timeseries of legal biomass, survey biomass, catches, and egg production.
- `fit_schaefer(biomass_obs::AbstractVector{<:Real}, catches::AbstractVector{<:Real}; dt=1.0)`:
  Fits Schaefer continuous surplus production model: $\frac{dB}{dt} = r B (1 - B/K) - C$.
- `fit_delay_difference(biomass_obs::AbstractVector{<:Real}, catches::AbstractVector{<:Real}; w_recruit=0.25, alpha=0.30, rho=0.85)`:
  Fits Deriso-Schnute delay-difference model.
- `fit_population_dynamics(biomass_obs::AbstractVector{<:Real}, catches::AbstractVector{<:Real}; model_type::Symbol=:schaefer, kwargs...)`:
  Unified dispatch wrapper for biomass dynamic models.
- `summarize_population_dynamics(fit::NamedTuple; io=stdout)`:
  Prints parameter estimates and diagnostics ($r$, $K$, MSY, $B_{\text{MSY}}$, RMSE, $R^2$).
- `mohn_rho(retrospective_trajectories::Vector{<:AbstractVector{<:Real}})`:
  Calculates Mohn's $\rho$ metric quantifying retrospective assessment bias:
  $$\rho = \frac{1}{P} \sum_{p=1}^P \frac{B_{T-p}^{(-p)} - B_{T-p}^{(0)}}{B_{T-p}^{(0)}}$$
- `retrospective_analysis(biomass_obs::AbstractVector{<:Real}, catches::AbstractVector{<:Real}; n_peels::Int=5, model_type::Symbol=:schaefer, kwargs...)`:
  Executes retrospective peels and evaluates assessment stability.
- `summarize_retrospective(retro_res::NamedTuple; io=stdout)`:
  Prints retrospective peeling table and Mohn's $\rho$.
- `extract_annual_stage_matrix(agent_df::DataFrame; stages=build_stages(), frequency_days=365.0)`:
  Constructs annual abundance matrix $\mathbf{N}_{t,s}$ across discrete life-history stages.
- `fit_generalized_leslie_model(stage_matrix::AbstractMatrix{<:Real}, catch_matrix::AbstractMatrix{<:Real}; stages=build_stages(), years=Int[])`:
  Calibrates stage-transition projection matrix $\mathbf{A}$, dominant eigenvalue $\lambda$, damping ratio, reproductive values, and elasticity matrix.
- `summarize_leslie_model(fit::LeslieMatrixFit; io=stdout)`:
  Prints formatted Leslie demographic diagnostics ($\lambda$, $r$, damping ratio).
- `compare_macro_models(biomass_obs::AbstractVector{<:Real}, catches::AbstractVector{<:Real}, leslie_fit::LeslieMatrixFit; stages=build_stages(), years=Int[])`:
  Constructs comparative evaluation (`MacroModelComparison`) across Schaefer, Delay-Difference, and Leslie formulations.
- `summarize_macro_comparison(comp::MacroModelComparison; io=stdout)`:
  Prints comparative table of $R^2$, RMSE, and AIC across macro models.
- `evaluate_population_criteria(model::StandardABM, agent_df::DataFrame; min_biomass_threshold=100.0, target_sex_ratio=0.50)`:
  Evaluates population viability and harvest sustainability criteria.
- `save_simulation_results(agent_df::DataFrame, model_df::DataFrame, output_dir::String="outputs"; prefix="simulation")`:
  Saves simulation outputs to CSV format.
- `load_simulation_results(output_dir::String="outputs"; prefix="simulation")`:
  Loads simulation outputs from CSV files.
- `write_dataframe_csv(df::DataFrame, filepath::String)` / `read_dataframe_csv(filepath::String)`:
  CSV I/O helpers.

---

### 6.8 Visualization & Animation (`src/visualization.jl`)

- `plot_population_timeseries(model_df::DataFrame)`:
  Multi-panel plot of total abundance, mature biomass, recruit counts, and catches over time.
- `plot_stage_distribution(model::StandardABM)`:
  Histogram of organism stages partitioned by sex.
- `plot_instar_distribution`: Backwards-compatible alias for `plot_stage_distribution`.
- `plot_size_by_sex(model::StandardABM)`:
  Kernel density and histogram of carapace sizes across sexes.
- `plot_stage_allometry(model::StandardABM)`:
  Carapace size vs. stage compared against theoretical empirical lower and upper bounds.
- `plot_instar_allometry`: Backwards-compatible alias for `plot_stage_allometry`.
- `plot_maturity_ogives(model::StandardABM)`:
  Empirical maturity proportions vs. size compared with theoretical logistic ogives.
- `plot_reproduction_dynamics(agent_df::DataFrame, model_df::DataFrame)`:
  Egg counts, ovigerous female abundance, and larval production over time.
- `plot_spatial_habitat(model::StandardABM)`:
  2D heatmap of bottom temperature overlaid with agent spatial positions.
- `plot_simulation_dashboard(model::StandardABM, agent_df::DataFrame; model_df=DataFrame())`:
  Executive 6-panel summary dashboard visualizing timeseries, stage composition, spatial positions, and maturity.
- `plot_stock_recruitment(sr_df::DataFrame, fit::NamedTuple)`:
  Fitted stock-recruitment curve overlaid on empirical annual points.
- `plot_population_dynamics_fits(comp::MacroModelComparison)`:
  Observed survey biomass trajectories compared against Schaefer, Delay-Difference, and Leslie predictions.
- `plot_macro_model_comparison`: Alias for `plot_population_dynamics_fits`.
- `plot_retrospective_analysis(retro_res::NamedTuple)`:
  Retrospective biomass peel trajectories with Mohn's $\rho$ annotation.
- `plot_stage_specific_dynamics(leslie_fit::LeslieMatrixFit)`:
  Stable stage distribution, reproductive values, and elasticity matrix heatmap.
- `plot_functional_relationships(behavior::SpeciesBehavior=default_species_behavior())`:
  Diagnostic panels showing growth increments, maturity ogives, mass allometry, and fecundity relationships.
- `export_all_plots(model::StandardABM, agent_df::DataFrame, model_df::DataFrame, output_dir::String="outputs")`:
  Generates and exports all diagnostic figures to disk as PNG files.
- `organism_color(org::Organism)`: Color mapper based on sex and maturity status.
- `organism_size(org::Organism)`: Marker size scaling proportional to carapace width.
- `organism_marker(org::Organism)`: Shape mapper (`:circle` for female, `:square` for male).
- `organism_heatarray(model::StandardABM)`: Spatial temperature or food grid matrix extractor.
- `crab_color`, `crab_size`, `crab_marker`, `crab_heatarray`: Domain aliases.
- `plot_abm_summary(model::StandardABM; heat_field="temperature", output_path="outputs/abm_spatial_summary.png")`:
  High-resolution spatial summary rendering using CairoMakie.
- `animate_organism_simulation(model::StandardABM; filepath="organism_animation.mp4", frames=120, dt=2, framerate=15, field_name="temperature")`:
  Renders `.mp4` video recording spatial agent movement and environmental dynamics.
- `animate_snowcrab_simulation`: Backwards-compatible alias for `animate_organism_simulation`.

---

### 6.9 Interactive GUI Dashboard (`src/gui.jl`)

- `launch_organism_gui(model=nothing; figure_size=(1600, 1000), target_fps=30)`:
  Launches interactive multi-panel GUI window (`GLMakie.Screen`) featuring three interlinked dashboards:
  1. **Control & Dynamics View**: Parameter steering sliders, temperature amplitude, fishing mortality, benthic carrying capacity, real-time spatial heatmaps with animated particle scatter overlays, and live population timeseries.
  2. **Demographics & Macro Models View**: Real-time stage distribution histograms, size distributions by sex, and dynamic Schaefer logistic vs. Deriso-Schnute surplus production fits.
  3. **Process Functions View**: Empirical maturity ogives, fishery retention selectivity, and allometric scaling curves plotted against theoretical baselines.
- `IBMGuiState`: Reactive container managing simulation state, model stepping, and `GLMakie.Observable` signals.
- `step_gui_simulation!(state::IBMGuiState, n_steps::Int)`: Advances model state and notifies UI subscribers.

---

### 6.10 Backward Compatibility & Domain Aliases

To maintain compatibility with existing analytical scripts while establishing canonical naming:

| Exported Symbol | Canonical Target | Category |
|---|---|---|
| `Organism` | `Organism` | Type Alias |
| `stage_size` | `get_stage_size` | Function Alias |
| `crab_color` | `organism_color` | Visualization Alias |
| `crab_size` | `organism_size` | Visualization Alias |
| `crab_marker` | `organism_marker` | Visualization Alias |
| `crab_heatarray` | `organism_heatarray` | Visualization Alias |
| `crab_disperse!` | `organism_disperse!` | Dynamic Dispersal Alias |
| `sample_crab_traits` | `sample_organism_traits` | Phenotypic Trait Alias |
| `weight_if_adult_female` | `mass_if_adult_female` | Demographic Query Alias |
| `weight_if_adult_male` | `mass_if_adult_male` | Demographic Query Alias |
| `weight_if_legal_male` | `mass_if_legal_male` | Demographic Query Alias |
| `animate_snowcrab_simulation` | `animate_organism_simulation` | Animation Alias |
| `plot_instar_allometry` | `plot_stage_allometry` | Visualization Alias |
| `plot_instar_distribution` | `plot_stage_distribution` | Visualization Alias |
| `FECUNDITY_PRIMIPAROUS_*` | `SpeciesParams` empirical constants | Allometric Invariant |
| `FECUNDITY_MULTIPAROUS_*` | `SpeciesParams` empirical constants | Allometric Invariant |
#### IBMParameters
Top-level configuration structure with 11 modular nested settings:
- 	oggles::ProcessToggles: Granular biological and operational feature switches.
- simulation::SimulationParams: Time step $\Delta t$, random seed, initial abundance distributions, calendar duration.
- environment::EnvironmentParams: Baseline temperature, seasonal amplitude, food replenishment, spatial habitat peaks, carrying capacity.
- organism::OrganismParams: Metabolic budgets, Kleiber scaling, ecdysis energy thresholds, asymptotic size bounds, stage labels.
- phenology::PhenologyParams: Molting and mating seasonal calendar windows, thermal brooding duration, spermathecal storage.
- mortality::MortalityParams: Stage-dependent baseline natural mortality, predation, soft-shell vulnerability, starvation, thermal stress, terminal adult senescence, juvenile cannibalism.
- ishery::FisheryParams: Minimum legal size threshold ({50}$), fishing season calendar window, daily harvest probability switch.
- output::OutputParams: Periodic snapshot intervals, CSV results destination, plotting and animation controls.
- species::SpeciesParams: Species-specific allometry (Hiatt increment, weight-length powers), log-logistic maturity ogives, Sainte-Marie fecundity.
- leslie::LeslieParams: Generalized Lefkovitch stage-structured projection matrix parameters and multi-year fitting controls.
- criteria::CriteriaParams: Multi-criteria population viability and fidelity thresholds (persistence, size drift, maturity RMSE, growth RMSE).

---

### 6.2 Snow Crab Species Implementation (`src/species.jl`)

All biological processes, allometric constants, and parameterizations specific to the snow crab (*Chionoecetes*) are centralized within `src/species.jl`.

#### Parameters: `SpeciesParams <: SpeciesParams`
- **Maturity Ogives**:
  - `maturity_l50_male::Float64 = 91.8` (mm CW), `maturity_l50_male_sd::Float64 = 1.2`
  - `maturity_slope_male::Float64 = 6.9957`, `maturity_slope_male_sd::Float64 = 0.25`
  - `maturity_l50_female::Float64 = 51.6` (mm CW), `maturity_l50_female_sd::Float64 = 0.8`
  - `maturity_slope_female::Float64 = 12.9812`, `maturity_slope_female_sd::Float64 = 0.40`
- **Fecundity Allometry**:
  - `fecundity_parameterization::String = "moncton"` (Sainte-Marie 1993)
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
const Organism = Organism
```

#### Concrete Methods:
- `stage_size_lower_bound(behavior::SpeciesBehaviorImpl, stage::Real, sex)`: Lower bound of carapace width (mm) across stages 1 to 16.
- `llbm(stage)`: Convenience shortcut for male lower bound: `stage_size_lower_bound(SpeciesBehaviorImpl(), stage, :male)`.
- `llbf(stage)`: Convenience shortcut for female lower bound: `stage_size_lower_bound(SpeciesBehaviorImpl(), stage, :female)`.
- `get_stage_size(behavior::SpeciesBehaviorImpl, stage::Real, sex; type=:mid, rng=...)`: Bounds (`:lb`, `:ub`), midpoint (`:mid`), or stochastic beta-distributed (`:random`) carapace width.
- `growth_increment(behavior::SpeciesBehaviorImpl, log_size::Real; sex, mat, temp, condition, rng, growth_quantile)`: Evaluates post-molt natural logarithm $\ln(\text{size}_{t+1})$.
- `predict_postgrowth_size(behavior::SpeciesBehaviorImpl, size::Real; ...)`: Linear post-molt size in millimeters.
- `growth_schedule(behavior::SpeciesBehaviorImpl)`: Reference `DataFrame` of stage sizes and increments.
- `maturity(behavior::SpeciesBehaviorImpl, sex, log_size::Real; temp, maturity_quantile, rng, deterministic)`: Log-logistic morphometric maturity probability.
- `fecundity(behavior::SpeciesBehaviorImpl, size::Real; is_primiparous, method, shell_age, rng, deterministic, fecundity_quantile)`: Brood egg count estimation.
- `selectivity(behavior::SpeciesBehaviorImpl, size::Real; s50, s50_sigma)`: Logistic gear retention probability.
- `calculate_mass(behavior::SpeciesBehaviorImpl, size::Real, sex; mat, stochastic, rng)`: Wet body mass (grams) from carapace width (mm).
- `build_stages(behavior::SpeciesBehaviorImpl; max_male_stage=12, max_female_stage=11)`: Constructs the demographic stage hierarchy.
- `min_intermolt_days(behavior::SpeciesBehaviorImpl, stage::Int)`: Minimum chronological days required between molts.
- `required_degree_days(behavior::SpeciesBehaviorImpl, stage::Int)`: Thermal degree-days required before ecdysis.
- `perform_development!(behavior::SpeciesBehaviorImpl, org::Organism, model)`: Advances physiological development and molting.
- `perform_reproduction!(behavior::SpeciesBehaviorImpl, org::Organism, model)`: Executes seasonal copulation, spermathecal storage, brood extrusion, and hatching.

---

### 6.3 Individual State Queries & Predicates (`src/types.jl`)

High-performance inline predicates for demographic filtering:
- `is_adult(org::Organism)`: True if individual has undergone terminal morphometric molt (`org.is_mature`).
- `is_adult_male(org::Organism)`: True if male and mature.
- `is_adult_female(org::Organism)`: True if female and mature.
- `is_ovigerous(org::Organism)`: True if carrying an extruded clutch (`org.egg_count > 0`).
- `is_legal_male(org::Organism; legal_size=95.0)`: True if male and `size >= legal_size`.
- `is_legal(org::Organism; legal_size=95.0)`: Alias for `is_legal_male`.
- `mass_if_adult_female(org::Organism)`: Returns `org.mass` if adult female, otherwise `0.0`.
- `mass_if_adult_male(org::Organism)`: Returns `org.mass` if adult male, otherwise `0.0`.
- `mass_if_legal_male(org::Organism; legal_size=95.0)`: Returns `org.mass` if legal male, otherwise `0.0`.
- `weight_if_*`: Backwards-compatible aliases for `mass_if_*`.

---

### 6.4 Organism Dynamics & Bioenergetics (`src/organism_dynamics.jl`)

- `calculate_energy_change(temp::Real, food::Real, size::Real; params=IBMParameters(), mass=nothing)`:
  Evaluates net energy assimilation $\frac{dE}{dt}$ and actual forage intake $I_{\text{actual}}$ via Kleiber metabolic scaling ($W^{0.75}$) and Holling Type II functional response.
- `calculate_energy_change(org::Organism, temp::Real, food::Real; params=IBMParameters())`:
  Agent-level method wrapping the above.
- `molt_probability(org::Organism, current_temp::Real, current_time::Real=0.0; params=IBMParameters())`:
  Computes daily probability of undergoing ecdysis based on intermolt days, degree-days, energy reserves, and temperature.
- `check_molt_conditions(org::Organism, current_temp::Real, current_time::Real=0.0; params=IBMParameters(), rng=Random.default_rng())`:
  Evaluates boolean criteria for ecdysis.
- `perform_molt!(org::Organism, current_time::Real, current_temp::Real; params=IBMParameters(), rng=Random.default_rng())`:
  Executes ecdysis, advances stage, evaluates maturity transition, updates size and mass via Hiatt models, and resets degree-days.
- `calculate_mortality_probability(org::Organism, current_temp::Real, local_density::Real, current_time::Real=0.0; params=IBMParameters())`:
  Computes aggregate daily mortality hazard combining baseline natural mortality, senescence, thermal stress, and density-dependent competition.
- `shell_condition(days_since_molt::Real)`:
  Translates days post-ecdysis into standard fishery shell condition categories (1: soft, 2: new hard, 3: intermediate, 4: old, 5: very old).
- `step_aggregate_groups!(model; behavior=model.species_behavior)`:
  Advances pre-individuation aggregate cohorts (stages 0 to 5) through survival, growth, and stage graduation into spatial agents at stage 6.
- `organism_disperse!(org::Organism, model; rng=abmrng(model))`:
  Executes spatial movement governed by thermal and forage habitat taxis.
- `crab_disperse!(args...; kwargs...)`: Alias for `organism_disperse!`.

---

### 6.5 Simulation Engine (`src/simulation.jl`)

- `default_ibm_configuration()`:
  Returns complete baseline configuration `Dict{Symbol, Any}` encompassing spatial grid geometry, runtime duration, environmental forcings, bioenergetics, life-history phenology, fishery dynamics, process toggles, and output handling.
- `load_ibm_configuration(filepath::String)`:
  Reads simulation configuration from a structured TOML file.
- `save_ibm_configuration(config::AbstractDict, filepath::String)`:
  Serializes configuration dictionary to TOML format.
- `initialize_model(; N_agents=100, dims=(50, 50), temp_map=nothing, food_map=nothing, dt_days=1.0, params=IBMParameters(), behavior=default_species_behavior(), rng_seed=nothing)`:
  Constructs and seeds an `Agents.StandardABM` spatial simulation model.
- `initialize_model(config_filepath::String; kwargs...)`:
  Initializes model from a TOML configuration file with keyword overrides.
- `initialize_model(config::AbstractDict; kwargs...)`:
  Initializes model from a configuration dictionary with keyword overrides.
- `agent_step!(org::Organism, model)`:
  Advances an individual agent through one time step (energetics, degree-days, molting, mortality, harvest, dispersal, reproduction).
- `model_step!(model)`:
  Advances global environment fields (temperature seasonality, food replenishment), steps aggregate cohorts, and updates demographic caches.
- `run_simulation(model::StandardABM, total_days::Real; agent_collect=[], model_collect=[])`:
  Executes the simulation across `total_days` and collects agent- and model-level time-series `DataFrames`.
- ABM Utility Accessors:
  - `nagents(model)`: Total active spatial agents.
  - `allagents(model)`: Iterator over all agents.
  - `abmrng(model)`: Model random number generator.
  - `safe_mean(x)`: Mean calculation returning `0.0` for empty collections.

---

### 6.6 Empirical Biology, Surveys, & Demographics (`src/analysis.jl`)

- `gaussian_kde(data::AbstractVector{<:Real}; eval_points=nothing, bandwidth=nothing)`:
  1D Gaussian kernel density estimation.
- `derive_stage_proportions_kde(sizes::AbstractVector{<:Real}, sexes::AbstractVector; stages=build_stages())`:
  Discretizes empirical size distributions into stage proportions.
- `generate_mock_survey_data(; n_samples=500, rng=Random.default_rng(), behavior=default_species_behavior())`:
  Synthesizes research survey catch records across size, sex, maturity, shell condition, and fecundity.
- `fit_empirical_biology_model(data::DataFrame)`:
  Calibrates allometric exponents, maturity ogives, and phenotypic covariance matrices from survey data.
- `estimate_demographics(model::EmpiricalBiologicalModel; size::Real, sex, temp=nothing, shell_age=1.0, rng=Random.default_rng())`:
  Predicts mass, maturity probability, and fecundity for an individual crab.
- `sample_organism_traits(model::EmpiricalBiologicalModel; rng=Random.default_rng())`:
  Samples correlated individual phenotypic trait quantiles.
- `sample_crab_traits(args...; kwargs...)`: Alias for `sample_organism_traits`.
- `summarize_empirical_model(model::EmpiricalBiologicalModel; io=stdout)`:
  Prints formatted ASCII parameter summary.
- `compute_empirical_ogive(sizes::AbstractVector{<:Real}, mature_flags::AbstractVector{<:Integer}; n_bins::Int=15)`:
  Computes binned proportions mature for empirical ogive fitting.
- `summarize_population(agent_df::DataFrame; frequency_days=365.0)`:
  Aggregates simulation results into annual demographic summaries.

---

### 6.7 Macro-Ecological Models & Stock Assessment (`src/analysis.jl`)

- `extract_stock_recruitment(agent_df::DataFrame, model_df::DataFrame; lag_years::Int=4, frequency_days=365.0)`:
  Computes annual spawner biomass and lagged recruitment pairs.
- `fit_stock_recruitment(spawners::AbstractVector{<:Real}, recruits::AbstractVector{<:Real}; model_type::Symbol=:beverton_holt)`:
  Fits Beverton-Holt ($R = \frac{\alpha S}{1 + \beta S}$) or Ricker ($R = \alpha S e^{-\beta S}$) models.
- `summarize_stock_recruitment(fit::NamedTuple; io=stdout)`:
  Prints formatted stock-recruitment parameters ($R^2$, $\alpha$, $\beta$, $S_{\text{max}}$).
- `extract_annual_production(agent_df::DataFrame, model_df::DataFrame; frequency_days=365.0)`:
  Extracts annual timeseries of legal biomass, survey biomass, catches, and egg production.
- `fit_schaefer(biomass_obs::AbstractVector{<:Real}, catches::AbstractVector{<:Real}; dt=1.0)`:
  Fits Schaefer continuous surplus production model: $\frac{dB}{dt} = r B (1 - B/K) - C$.
- `fit_delay_difference(biomass_obs::AbstractVector{<:Real}, catches::AbstractVector{<:Real}; w_recruit=0.25, alpha=0.30, rho=0.85)`:
  Fits Deriso-Schnute delay-difference model.
- `fit_population_dynamics(biomass_obs::AbstractVector{<:Real}, catches::AbstractVector{<:Real}; model_type::Symbol=:schaefer, kwargs...)`:
  Unified dispatch wrapper for biomass dynamic models.
- `summarize_population_dynamics(fit::NamedTuple; io=stdout)`:
  Prints parameter estimates and diagnostics ($r$, $K$, MSY, $B_{\text{MSY}}$, RMSE, $R^2$).
- `mohn_rho(retrospective_trajectories::Vector{<:AbstractVector{<:Real}})`:
  Calculates Mohn's $\rho$ metric quantifying retrospective assessment bias:
  $$\rho = \frac{1}{P} \sum_{p=1}^P \frac{B_{T-p}^{(-p)} - B_{T-p}^{(0)}}{B_{T-p}^{(0)}}$$
- `retrospective_analysis(biomass_obs::AbstractVector{<:Real}, catches::AbstractVector{<:Real}; n_peels::Int=5, model_type::Symbol=:schaefer, kwargs...)`:
  Executes retrospective peels and evaluates assessment stability.
- `summarize_retrospective(retro_res::NamedTuple; io=stdout)`:
  Prints retrospective peeling table and Mohn's $\rho$.
- `extract_annual_stage_matrix(agent_df::DataFrame; stages=build_stages(), frequency_days=365.0)`:
  Constructs annual abundance matrix $\mathbf{N}_{t,s}$ across discrete life-history stages.
- `fit_generalized_leslie_model(stage_matrix::AbstractMatrix{<:Real}, catch_matrix::AbstractMatrix{<:Real}; stages=build_stages(), years=Int[])`:
  Calibrates stage-transition projection matrix $\mathbf{A}$, dominant eigenvalue $\lambda$, damping ratio, reproductive values, and elasticity matrix.
- `summarize_leslie_model(fit::LeslieMatrixFit; io=stdout)`:
  Prints formatted Leslie demographic diagnostics ($\lambda$, $r$, damping ratio).
- `compare_macro_models(biomass_obs::AbstractVector{<:Real}, catches::AbstractVector{<:Real}, leslie_fit::LeslieMatrixFit; stages=build_stages(), years=Int[])`:
  Constructs comparative evaluation (`MacroModelComparison`) across Schaefer, Delay-Difference, and Leslie formulations.
- `summarize_macro_comparison(comp::MacroModelComparison; io=stdout)`:
  Prints comparative table of $R^2$, RMSE, and AIC across macro models.
- `evaluate_population_criteria(model::StandardABM, agent_df::DataFrame; min_biomass_threshold=100.0, target_sex_ratio=0.50)`:
  Evaluates population viability and harvest sustainability criteria.
- `save_simulation_results(agent_df::DataFrame, model_df::DataFrame, output_dir::String="outputs"; prefix="simulation")`:
  Saves simulation outputs to CSV format.
- `load_simulation_results(output_dir::String="outputs"; prefix="simulation")`:
  Loads simulation outputs from CSV files.
- `write_dataframe_csv(df::DataFrame, filepath::String)` / `read_dataframe_csv(filepath::String)`:
  CSV I/O helpers.

---

### 6.8 Visualization & Animation (`src/visualization.jl`)

- `plot_population_timeseries(model_df::DataFrame)`:
  Multi-panel plot of total abundance, mature biomass, recruit counts, and catches over time.
- `plot_stage_distribution(model::StandardABM)`:
  Histogram of organism stages partitioned by sex.
- `plot_instar_distribution`: Backwards-compatible alias for `plot_stage_distribution`.
- `plot_size_by_sex(model::StandardABM)`:
  Kernel density and histogram of carapace sizes across sexes.
- `plot_stage_allometry(model::StandardABM)`:
  Carapace size vs. stage compared against theoretical empirical lower and upper bounds.
- `plot_instar_allometry`: Backwards-compatible alias for `plot_stage_allometry`.
- `plot_maturity_ogives(model::StandardABM)`:
  Empirical maturity proportions vs. size compared with theoretical logistic ogives.
- `plot_reproduction_dynamics(agent_df::DataFrame, model_df::DataFrame)`:
  Egg counts, ovigerous female abundance, and larval production over time.
- `plot_spatial_habitat(model::StandardABM)`:
  2D heatmap of bottom temperature overlaid with agent spatial positions.
- `plot_simulation_dashboard(model::StandardABM, agent_df::DataFrame; model_df=DataFrame())`:
  Executive 6-panel summary dashboard visualizing timeseries, stage composition, spatial positions, and maturity.
- `plot_stock_recruitment(sr_df::DataFrame, fit::NamedTuple)`:
  Fitted stock-recruitment curve overlaid on empirical annual points.
- `plot_population_dynamics_fits(comp::MacroModelComparison)`:
  Observed survey biomass trajectories compared against Schaefer, Delay-Difference, and Leslie predictions.
- `plot_macro_model_comparison`: Alias for `plot_population_dynamics_fits`.
- `plot_retrospective_analysis(retro_res::NamedTuple)`:
  Retrospective biomass peel trajectories with Mohn's $\rho$ annotation.
- `plot_stage_specific_dynamics(leslie_fit::LeslieMatrixFit)`:
  Stable stage distribution, reproductive values, and elasticity matrix heatmap.
- `plot_functional_relationships(behavior::SpeciesBehavior=default_species_behavior())`:
  Diagnostic panels showing growth increments, maturity ogives, mass allometry, and fecundity relationships.
- `export_all_plots(model::StandardABM, agent_df::DataFrame, model_df::DataFrame, output_dir::String="outputs")`:
  Generates and exports all diagnostic figures to disk as PNG files.
- `organism_color(org::Organism)`: Color mapper based on sex and maturity status.
- `organism_size(org::Organism)`: Marker size scaling proportional to carapace width.
- `organism_marker(org::Organism)`: Shape mapper (`:circle` for female, `:square` for male).
- `organism_heatarray(model::StandardABM)`: Spatial temperature or food grid matrix extractor.
- `crab_color`, `crab_size`, `crab_marker`, `crab_heatarray`: Domain aliases.
- `plot_abm_summary(model::StandardABM; heat_field="temperature", output_path="outputs/abm_spatial_summary.png")`:
  High-resolution spatial summary rendering using CairoMakie.
- `animate_organism_simulation(model::StandardABM; filepath="organism_animation.mp4", frames=120, dt=2, framerate=15, field_name="temperature")`:
  Renders `.mp4` video recording spatial agent movement and environmental dynamics.
- `animate_snowcrab_simulation`: Backwards-compatible alias for `animate_organism_simulation`.

---

### 6.9 Interactive GUI Dashboard (`src/gui.jl`)

- `launch_organism_gui(model=nothing; figure_size=(1600, 1000), target_fps=30)`:
  Launches interactive multi-panel GUI window (`GLMakie.Screen`) featuring three interlinked dashboards:
  1. **Control & Dynamics View**: Parameter steering sliders, temperature amplitude, fishing mortality, benthic carrying capacity, real-time spatial heatmaps with animated particle scatter overlays, and live population timeseries.
  2. **Demographics & Macro Models View**: Real-time stage distribution histograms, size distributions by sex, and dynamic Schaefer logistic vs. Deriso-Schnute surplus production fits.
  3. **Process Functions View**: Empirical maturity ogives, fishery retention selectivity, and allometric scaling curves plotted against theoretical baselines.
- `IBMGuiState`: Reactive container managing simulation state, model stepping, and `GLMakie.Observable` signals.
- `step_gui_simulation!(state::IBMGuiState, n_steps::Int)`: Advances model state and notifies UI subscribers.

---

### 6.10 Backward Compatibility & Domain Aliases

To maintain compatibility with existing analytical scripts while establishing canonical naming:

| Exported Symbol | Canonical Target | Category |
|---|---|---|
| `Organism` | `Organism` | Type Alias |
| `stage_size` | `get_stage_size` | Function Alias |
| `crab_color` | `organism_color` | Visualization Alias |
| `crab_size` | `organism_size` | Visualization Alias |
| `crab_marker` | `organism_marker` | Visualization Alias |
| `crab_heatarray` | `organism_heatarray` | Visualization Alias |
| `crab_disperse!` | `organism_disperse!` | Dynamic Dispersal Alias |
| `sample_crab_traits` | `sample_organism_traits` | Phenotypic Trait Alias |
| `weight_if_adult_female` | `mass_if_adult_female` | Demographic Query Alias |
| `weight_if_adult_male` | `mass_if_adult_male` | Demographic Query Alias |
| `weight_if_legal_male` | `mass_if_legal_male` | Demographic Query Alias |
| `animate_snowcrab_simulation` | `animate_organism_simulation` | Animation Alias |
| `plot_instar_allometry` | `plot_stage_allometry` | Visualization Alias |
| `plot_instar_distribution` | `plot_stage_distribution` | Visualization Alias |
| `FECUNDITY_PRIMIPAROUS_*` | `SpeciesParams` empirical constants | Allometric Invariant |
| `FECUNDITY_MULTIPAROUS_*` | `SpeciesParams` empirical constants | Allometric Invariant |


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
| | `initial_female/male_cohort`| `fill(0.0, 12)` | Initial abundance of post-individuation adults by stage. | `simulation.jl` |
| | `initial_cohort_days_per_stage`| `90.0` | Base age scaling (days) for seeded aggregate cohorts. | `simulation.jl` |
| | `initial_age_days_per_stage`| `180.0` | Base age scaling (days) for seeded post-individuation individuals. | `simulation.jl` |
| | `initial_age_first_stage`| `3.0` | Stage offset for initial age assignment. | `simulation.jl` |
| | `initial_age_jitter_days`| `90.0` | Uniform random jitter applied to seeded individual age. | `simulation.jl` |
| | `initial_degree_days_min/max`| `50.0`, `300.0`| Range of pre-ecdysial degree-days assigned to seeded agents. | `simulation.jl` |
| **`[environment]`** | `temp_base` | `5.0` | Baseline bottom temperature (°C). | `simulation.jl` |
| | `temp_amplitude` | `4.0` | Amplitude of seasonal thermal oscillations (°C). | `simulation.jl` |
| | `temp_noise_sd` | `0.5` | Spatial standard deviation of environmental temperature noise. | `simulation.jl` |
| | `temp_base_degreedays` | `-1.5` | Threshold temperature $T_0$ (°C) below which degree-days do not accrue. | `organism_dynamics.jl`, `simulation.jl` |
| | `food_max` | `10.0` | Carrying capacity ceiling on local benthic food density. | `simulation.jl` |
| | `food_replenishment_rate`| `0.1` | Daily linear food renewal rate ($d^{-1}$). | `simulation.jl` |
| | `benthic_carrying_capacity`| `12000.0` | Total benthic capacity ($K$) across the entire grid for crowding feedback. | `organism_dynamics.jl`, `simulation.jl` |
| | `benthic_carrying_capacity_exponent`| `2.0` | Nonlinearity exponent in $(N/K)^p$ mortality feedback. | `organism_dynamics.jl`, `simulation.jl` |
| | `half_saturation_food` | `2.0` | Forage concentration where intake is at 50% saturation. | `organism_dynamics.jl`, `simulation.jl` |
| | `assimilation_efficiency`| `0.75` | Gross energetic assimilation fraction from ingested food. | `organism_dynamics.jl` |
| | `habitat_quality_mean/sd`| `0.5`, `0.1` | Gaussian background parameters for habitat suitability index (HSI). | `simulation.jl` |
| | `habitat_peaks` | `3` | Number of coherent spatial Gaussian habitat hotspots. | `simulation.jl` |
| | `habitat_peak_height` | `0.3` | HSI elevation added at the center of each Gaussian hotspot. | `simulation.jl` |
| | `habitat_peak_width_fraction`| `0.15` | Spatial width ($\sigma$) of hotspots relative to grid dimension. | `simulation.jl` |
| | `habitat_temperature_optimum`| `2.5` | Preferred temperature for spatial dispersal taxis (°C). | `simulation.jl` |
| | `habitat_temperature_sigma`| `1.5` | Width of thermal preference curve for dispersal. | `simulation.jl` |
| | `habitat_food_weight` | `0.5` | Relative weighting of local food availability vs temperature in taxis. | `simulation.jl` |
| | `movement_softmax_temperature`| `0.5` | Softmax temperature regulating heading stochasticity. | `simulation.jl` |
| | `settlement_dispersal_radius`| `2` | Cell radius for recruit settlement around parental/cohort position. | `organism_dynamics.jl` |
| | `settlement_size_log_sd` | `0.04` | Log-normal standard deviation in carapace size at recruitment. | `organism_dynamics.jl` |
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
| | `energy_deficit_removal_probability`| `0.75` | Daily mortality probability when energy reserves drop into deficit. | `simulation.jl` |
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
