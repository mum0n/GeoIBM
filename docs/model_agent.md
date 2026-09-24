# Biological Individual-Based Model (BIBM) Engine Documentation

This document describes the architecture, mathematical formulations, command-line interface, and public API of the Biological Individual-Based Model (`model_agent`) engine. The engine implements a high-performance, modular individual- and cohort-based simulation system for organismal population dynamics, bioenergetics, and harvest management.

Species-specific biological processes (allometry, developmental thresholds, ecdysis, and reproduction) are fully decoupled from generic individual-based model (IBM) simulation logic via an extensible behavioral dispatch interface (`SpeciesBehavior`). A complete concrete implementation is provided for the snow crab (*Chionoecetes opilio*).

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
   - Dynamic dispatch routes species-specific allometry, phenology, development, and reproduction to concrete subtypes (e.g. `SnowCrabBehavior`).
   - Default behavior resolution via `default_species_behavior()`.

```
                      [ SpeciesBehavior ]  <── abstract interface
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
      [ SnowCrabBehavior ]          [ OtherSpeciesBehavior ]
      (Chionoecetes opilio)              (Extensible)
               │
   ┌───────────┼────────────────────────────────────────┐
   ▼           ▼                    ▼                   ▼
Allometry   Development         Phenology         Reproduction
(Hiatt)   (Degree-Days)   (Intermolt Thresholds)  (Spermathecal Store)
```

### High-Level Life-History Cycle (Snow Crab Example)

```
       [ Adult Females ] ──(Mating Encounter)──> [ Spermathecae Storage ]
              │                                           │
              ▼                                           ▼
       [ Brood Incubation ] <──────────────────── [ Egg Extrusion ]
       (Thermal Incubation Rate)                  (Allometric Fecundity)
              │
              ▼
       [ Larval Hatching & Pelagic Drift ]
              │
              ▼ (Settlement & Benthic Metamorphosis)
       [ Aggregate Groups (Stages 0–5) ] ──> [ Early Benthic Cohorts ]
                                                         │
                                                         ▼ (Individual Emergence at Stage 6)
                                              [ Spatial Agents (Stages 6–16) ]
                                                         │
                                                         ▼ (Terminal Ecdysis)
                                               [ Morphometric Adults ]
                                              ┌──────────┴──────────┐
                                              ▼                     ▼
                                       [ Adult Females ]     [ Adult Males ]
                                       (Breeding Stock)    (Commercial Fishery)
```

---

## 2. Package Organization

The source repository is organized into focused, modular components:

```
model_agent/
├── Project.toml              # Dependencies and compatibility bounds
├── docs/
│   ├── model_agent.md        # Comprehensive architecture and API reference (this file)
│   └── notes.md              # Research notes and design specifications
├── src/
│   ├── model_agent.jl        # Module definition, includes, and exported public APIs
│   ├── types.jl              # Core type system: Organism, StageDefinition, IBMParameters
│   ├── organism_dynamics.jl  # Generic growth, allometry fallbacks, bioenergetics, molt
│   ├── snowcrab.jl           # Snow crab parameters, allometry, phenology, reproduction
│   ├── simulation.jl         # Grid initialization, agent/model steppers, simulation runner
│   ├── analysis.jl           # KDE, OLS/IRLS estimation, Leslie matrices, macro models
│   ├── visualization.jl      # Diagnostic figures and video animations (CairoMakie)
│   └── gui.jl                # Real-time interactive 3-dashboard GUI (GLMakie)
├── scripts/
│   ├── organism_ibm.config   # Default TOML simulation configuration
│   ├── organism_ibm.jl       # CLI simulation application
│   └── organism_orchestrator.jl # Parameter variation and calibration orchestrator
└── test/
    └── runtests.jl           # Complete automated test suite
```

### Component Roles

| Source File | Primary Responsibility |
|---|---|
| [`src/types.jl`](file:///c:/home/jae/projects/model_agent/src/types.jl) | Declares abstract interface (`SpeciesBehavior`, `SpeciesParams`), agent structs (`Organism`), cohort containers (`AggregateGroup`), configuration (`IBMParameters`), and matrix types (`StageDefinition`, `LeslieMatrixFit`, `MacroModelComparison`). |
| [`src/organism_dynamics.jl`](file:///c:/home/jae/projects/model_agent/src/organism_dynamics.jl) | Implements generic biological interfaces and fallbacks, Kleiber metabolic bioenergetics ($W^{0.75}$), Holling Type II consumption, ecdysis mechanics, mortality hazards, and spatial dispersal. |
| [`src/snowcrab.jl`](file:///c:/home/jae/projects/model_agent/src/snowcrab.jl) | Implements `SnowCrabParams`, `SnowCrabBehavior`, empirical Hiatt increments, log-logistic maturity schedules, allometric fecundity, thermal ecdysial thresholds, and primiparous/multiparous reproductive cycles. |
| [`src/simulation.jl`](file:///c:/home/jae/projects/model_agent/src/simulation.jl) | Manages model instantiation (`initialize_model`), environmental field generation, scheduled time stepping (`agent_step!`, `model_step!`), and batch execution (`run_simulation`). |
| [`src/analysis.jl`](file:///c:/home/jae/projects/model_agent/src/analysis.jl) | Computes kernel density estimations (KDE), iterative logistic fits (IRLS), demographic summaries, Schaefer and Delay-Difference biomass models, generalized Leslie matrices, and retrospective analyses. |
| [`src/visualization.jl`](file:///c:/home/jae/projects/model_agent/src/visualization.jl) | Generates executive diagnostic figures, maturity ogives, stock-recruitment plots, and video animations via CairoMakie (`Figure` output). |
| [`src/gui.jl`](file:///c:/home/jae/projects/model_agent/src/gui.jl) | Provides interactive real-time multi-dashboard steering GUI with reactive particle streaming, parameter steering, and macro-ecological model fitting via GLMakie (`launch_organism_gui`). |

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

---

## 4. Command-Line Interface (CLI)

The CLI tool is located at [`scripts/organism_ibm.jl`](file:///c:/home/jae/projects/model_agent/scripts/organism_ibm.jl).

### Syntax

```bash
julia --project=. scripts/organism_ibm.jl [OPTIONS]
```

### Options Reference

| Option | Type | Default | Description |
|---|---|---|---|
| `--config=<path>`, `-c=<path>` | String | `""` | Path to TOML configuration file. |
| `--days=<int>`, `-d=<int>` | Integer | `730` | Simulation duration in days. |
| `--n_agents=<int>`, `-n=<int>` | Integer | `300` | Initial population size. |
| `--grid=<nx,ny>`, `-g=<nx,ny>` | Int,Int | `100,100` | 2D spatial grid dimensions. |
| `--temp_base=<float>` | Float | `3.0` | Baseline bottom temperature (°C). |
| `--fishing_prob=<float>` | Float | `0.02` | Daily harvest probability on legal males. |
| `--legal_size_male=<float>` | Float | `95.0` | Minimum legal harvest carapace width (mm). |
| `--plot`, `--no-plot` | Flag | `false` | Enable or disable diagnostic figure generation. |
| `--save`, `--no-save` | Flag | `true` | Export tabular simulation results (CSV). |
| `--output_dir=<path>` | String | `outputs/` | Destination directory for exported outputs. |
| `--seed=<int>` | Integer | `42` | Random seed for stochastic reproducibility. |

### Parameter Variation & Calibration

The orchestrator script at [`scripts/organism_orchestrator.jl`](file:///c:/home/jae/projects/model_agent/scripts/organism_orchestrator.jl) executes systematic grid and sensitivity searches across multi-criteria objective targets:

```bash
julia --project=. scripts/organism_orchestrator.jl --vary="temp_base=2.0:1.0:5.0;fishing_prob=0.0,0.02,0.05" --days=1095 --runs=3
```

---

## 5. Complete API Reference

### 5.1 Types & Structures (`src/types.jl`)

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
- `binning::BinningParams`: Demographic histogram binning configurations.

---

### 5.2 Snow Crab Species Implementation (`src/snowcrab.jl`)

All biological processes, allometric constants, and parameterizations specific to the snow crab (*Chionoecetes opilio*) are centralized within `src/snowcrab.jl`.

#### Parameters: `SnowCrabParams <: SpeciesParams`
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

#### Behavioral Dispatch: `SnowCrabBehavior <: SpeciesBehavior`
```julia
struct SnowCrabBehavior <: SpeciesBehavior
    params::SnowCrabParams
end
SnowCrabBehavior() = SnowCrabBehavior(SnowCrabParams())
default_species_behavior() = SnowCrabBehavior()
const SnowCrab = Organism
```

#### Concrete Methods:
- `stage_size_lower_bound(behavior::SnowCrabBehavior, stage::Real, sex)`: Lower bound of carapace width (mm) across stages 1 to 16.
- `llbm(stage)`: Convenience shortcut for male lower bound: `stage_size_lower_bound(SnowCrabBehavior(), stage, :male)`.
- `llbf(stage)`: Convenience shortcut for female lower bound: `stage_size_lower_bound(SnowCrabBehavior(), stage, :female)`.
- `get_stage_size(behavior::SnowCrabBehavior, stage::Real, sex; type=:mid, rng=...)`: Bounds (`:lb`, `:ub`), midpoint (`:mid`), or stochastic beta-distributed (`:random`) carapace width.
- `growth_increment(behavior::SnowCrabBehavior, log_size::Real; sex, mat, temp, condition, rng, growth_quantile)`: Evaluates post-molt natural logarithm $\ln(\text{size}_{t+1})$.
- `predict_postgrowth_size(behavior::SnowCrabBehavior, size::Real; ...)`: Linear post-molt size in millimeters.
- `growth_schedule(behavior::SnowCrabBehavior)`: Reference `DataFrame` of stage sizes and increments.
- `maturity(behavior::SnowCrabBehavior, sex, log_size::Real; temp, maturity_quantile, rng, deterministic)`: Log-logistic morphometric maturity probability.
- `fecundity(behavior::SnowCrabBehavior, size::Real; is_primiparous, method, shell_age, rng, deterministic, fecundity_quantile)`: Brood egg count estimation.
- `selectivity(behavior::SnowCrabBehavior, size::Real; s50, s50_sigma)`: Logistic gear retention probability.
- `calculate_mass(behavior::SnowCrabBehavior, size::Real, sex; mat, stochastic, rng)`: Wet body mass (grams) from carapace width (mm).
- `build_stages(behavior::SnowCrabBehavior; max_male_stage=12, max_female_stage=11)`: Constructs the demographic stage hierarchy.
- `min_intermolt_days(behavior::SnowCrabBehavior, stage::Int)`: Minimum chronological days required between molts.
- `required_degree_days(behavior::SnowCrabBehavior, stage::Int)`: Thermal degree-days required before ecdysis.
- `perform_development!(behavior::SnowCrabBehavior, org::Organism, model)`: Advances physiological development and molting.
- `perform_reproduction!(behavior::SnowCrabBehavior, org::Organism, model)`: Executes seasonal copulation, spermathecal storage, brood extrusion, and hatching.

---

### 5.3 Individual State Queries & Predicates (`src/types.jl`)

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

### 5.4 Organism Dynamics & Bioenergetics (`src/organism_dynamics.jl`)

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

### 5.5 Simulation Engine (`src/simulation.jl`)

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

### 5.6 Empirical Biology, Surveys, & Demographics (`src/analysis.jl`)

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

### 5.7 Macro-Ecological Models & Stock Assessment (`src/analysis.jl`)

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

### 5.8 Visualization & Animation (`src/visualization.jl`)

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

### 5.9 Interactive GUI Dashboard (`src/gui.jl`)

- `launch_organism_gui(model=nothing; figure_size=(1600, 1000), target_fps=30)`:
  Launches interactive multi-panel GUI window (`GLMakie.Screen`) featuring three interlinked dashboards:
  1. **Control & Dynamics View**: Parameter steering sliders, temperature amplitude, fishing mortality, benthic carrying capacity, real-time spatial heatmaps with animated particle scatter overlays, and live population timeseries.
  2. **Demographics & Macro Models View**: Real-time stage distribution histograms, size distributions by sex, and dynamic Schaefer logistic vs. Deriso-Schnute surplus production fits.
  3. **Process Functions View**: Empirical maturity ogives, fishery retention selectivity, and allometric scaling curves plotted against theoretical baselines.
- `IBMGuiState`: Reactive container managing simulation state, model stepping, and `GLMakie.Observable` signals.
- `step_gui_simulation!(state::IBMGuiState, n_steps::Int)`: Advances model state and notifies UI subscribers.

---

### 5.10 Backward Compatibility & Domain Aliases

To maintain compatibility with existing analytical scripts while establishing canonical naming:

| Exported Symbol | Canonical Target | Category |
|---|---|---|
| `SnowCrab` | `Organism` | Type Alias |
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
| `FECUNDITY_PRIMIPAROUS_*` | `SnowCrabParams` empirical constants | Allometric Invariant |
| `FECUNDITY_MULTIPAROUS_*` | `SnowCrabParams` empirical constants | Allometric Invariant |

