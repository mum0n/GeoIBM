# Biological Individual-Based Model (BIBM) Engine Documentation

This documentation describes the general architecture of the Biological Individual-Based Model (BIBM) engine, which provides a high-performance framework for simulating organismal life history. The engine is designed to be species-agnostic through an abstraction layer that decouples simulation logic from specific biological processes. Detailed implementation examples are provided using Snow crab (*Chionoecetes opilio*) dynamics.

---

## 1. Overview & Modular Biological Architecture

The BIBM engine simulates the complete life history, spatial ecology, bioenergetics, developmental schedules, closed-loop reproduction, and demographic stock dynamics of a population. The architecture is built upon two primary pillars:

1.  **Generic Simulation Core**: Handles spatial grids (`GridSpace`), time integration, agent management via `Agents.jl`, and demographic aggregate caching.
2.  **Species Behavior Abstraction Layer**: Uses the `SpeciesBehavior` abstract type to dispatch species-specific logic (e.g., molting in crustaceans vs. metabolic changes in mammals) without modifying the core execution loop.

### High-Level Life-History Flowchart (Crustacean Example)

```
       [ Adult Females ] ──(Mating Cycle)──> [ Gamete/Sperm Storage ]
              │                                      │
              ▼                                      ▼
       [ Brood Incubation ] <────────────── [ Egg Production ]
       (~365 days external)                  (Empirical Fecundity)
              │
              ▼
       [ Larval Hatching & Pelagic Drift ]
              │
              ▼ (Density-Dependent Recruitment/Settlement)
       [ Benthic Recruits ] ──> [ Developmental Stages (Stages/Growth Steps) ]
                                                │
                                                ▼ (Maturity Transition)
                                         [ Terminal Maturity ]
                                                │
                                     ┌──────────────┴──────────────┐
                                     ▼                             ▼
                              [ Adult Females ]              [ Adult Males ]
                            (Reproductive Stock)           (Commercial/Predator Interaction)
```

### Modular Biological Processes (Feature Toggles)
Rather than choosing between "simple" or "complex" modes, the engine uses a feature-toggle system within `IBMParameters`. This allows users to construct highly customized simulation environments by enabling or disabling specific biological drivers:

*   **`use_energetics`**: Enables bioenergetic modeling (metabolic scaling, food intake, energy reserves).
*   **`use_thermal_stress`**: Integrates thermal degree-day accumulation and stenothermic mortality.
*   **`use_food_dynamics`**: Simulates local resource replenishment and carrying capacity.
*   **`use_habitat_taxis`**: Enables directed movement toward environmental optima (e.g., temperature or forage).
*   **`use_senescence`**: Adds exponentially increasing mortality as organisms age beyond a threshold.
*   **`use_cannibalism`**: Simulates density-dependent conspecific predation.

---

## 2. Package & Codebase Structure

The simulation is organized into modular files within `src/`:

```
model_agent/
├── Project.toml              # Dependencies: Agents, DataFrames, Distributions, Plots, etc.
├── docs/
│   ├── model_agent_new.md    # Primary documentation and user guide (this file)
│   └── organism_ibm_research_paper.md 
├── src/
│   ├── model_agent.jl        # Root package module & public API exports
│   ├── types.jl              # Structs: Organism, IBMParameters, SpeciesBehavior; Enums
│   ├── empirical_biology.jl  # Empirical rules (e.g., growth curves, maturity ogives)
   │   ├── organism_dynamics.jl # Bioenergetics, molting/growth thresholds, mortality
│   ├── simulation.jl         # Spatial GridSpace, core steppers, config loaders
│   ├── analysis.jl           # Summary statistics, stock-recruitment, and macro models
│   └── visualization.jl      # Diagnostic plots, interactive dashboards, and animations
├── scripts/
│   ├── organism_ibm.config   # Default TOML configuration file
│   ├── organism_ibm.jl       # Standalone simulation CLI tool
│   ├── organism_orchestrator.jl # Automated parameter calibration & exploration
│   └── fit_macro_models.jl   # Post-hoc macro model fitting CLI tool
└── test/
    └── runtests.jl           # Comprehensive unit and integration test suite
```

### Module Responsibilities

| Module File | Purpose & Key Exports |
|---|---|
| [`src/types.jl`](file:///c:/home/jae/projects/model_agent/src/types.jl) | Defines `@agent struct Organism`, `LifeStage`, `Sex`, `IBMParameters`, and the `SpeciesBehavior` interface. |
| [`src/empirical_biology.jl`](file:///c:/home/jae/projects/model_agent/src/empirical_biology.jl) | Implements empirical equations for growth, maturity (ogives), fecundity, and multivariate covariance modeling. |
| [`src/organism_dynamics.jl`](file:///c:/home/jae/projects/model_agent/src/organism_dynamics.jl) | Evaluates metabolic scaling ($W^{0.75}$), energy-dependent growth/molting thresholds, and multi-driver mortality rates. |
| [`src/simulation.jl`](file:///c:/home/jae/projects/model_agent/src/simulation.jl) | Implements spatial GridSpace, core `agent_step!` loop (dispatching to behavior), and configuration management. |
| [`src/analysis.jl`](file:///c:/home/jae/projects/model_agent/src/analysis.jl) | Computes demographic snapshots, stock-recruitment curves, and macro-ecological models (Schaefer, Delay-Difference). |
| [`src/visualization.jl`](file:///c:/home/jae/projects/model_agent/src/visualization.jl) | Produces diagnostic figures, interactive dashboards, and video animations of the simulation. |

---

## 3. Empirical Formulations & Mathematical Reference (Example: Crustaceans)

The following mathematical models are implemented within the `CrustaceanBehavior` module as a primary example.

### 3.1 Growth and Maturity
**Hiatt Growth Increment Model**:
Post-molt size is predicted from pre-molt size on a logarithmic scale, modulated by temperature:
$$\ln(\text{size}_{t+1}) = \beta_0 + \beta_1 \ln(\text{size}_t) + \delta_{\text{mat}} \cdot \mathbb{I}_{\text{mat}} + \Delta_{\text{temp}}(T) + \epsilon$$

**Maturity Ogives**:
The probability of attaining morphometric maturity is modeled as a logistic function:
$$P(\text{mature} \mid \text{size}, \text{sex}) = \frac{1}{1 + \exp(-(\alpha + \beta \ln(\text{size})))}$$

### 3.2 Mortality Drivers
The total mortality rate $M_{\text{total}}$ integrates multiple modular drivers:
$$M_{\text{total}} = M_{\text{base}}(\text{stage}) + M_{\text{predation}}(\text{size}) + M_{\text{soft\_shell}}(\Delta t_{molt}) + M_{\text{starvation}}(E) + M_{\text{thermal}}(T) + M_{\text{senescence}}(\text{age})$$

*   **Bioenergetic Starvation**: Triggered if metabolic reserves $E < 0$.
*   **Stenothermic Thermal Stress**: Applied when ambient temperature exceeds critical thresholds.
*   **Terminal Senescence**: Exponentially increasing hazard for adults once age exceeds a threshold $\tau_{\text{sen}}$.

---

## 4. Command-Line Interface (CLI) User Guide

The primary entry point is [`scripts/organism_ibm.jl`](file:///c:/home/jae/projects/model_agent/scripts/organism_ibm.jl).

### 4.1 Parameter Specification & Configuration
Parameters are managed via a three-tier precedence hierarchy:
1.  **Internal Defaults**: Hardcoded baseline values within `IBMParameters`.
2.  **Configuration File**: Loaded from `scripts/organism_ibm.config` or specified via `--config=<path>`.
3.  **Command-Line Arguments**: Explicit flags (e.g., `--temp_base=4.5`) override all previous settings.

### 4.2 CLI Options

| Option | Type | Default | Description |
|---|---|---|---|
| `--config=<path>` | Path | `organism_ibm.config` | Custom TOML configuration file |
| `--n_agents=<Int>` | Integer | `300` | Initial population size |
| `--days=<Int>` | Integer | `730` | Total simulation duration (days) |
| `--grid=<nx,ny>` | `nx,ny` | `100,100` | 2D spatial grid dimensions |
| `--param=<k>=<v>` | String | — | Dynamically override any specific parameter |
| `--plot` / `--no-plot` | Flag | `false` | Generate and export diagnostic plots |
| `--save` / `--no-save` | Flag | `true` | Save annual production results to CSV |

---

## 5. Generated Visualizations & Interactive Tools

When executed with the `--plot` flag, the engine generates a suite of diagnostics including:
*   **Executive Dashboard**: A comprehensive multi-panel overview of abundance, demographics, and spatial distribution.
*   **Demographic Profiles**: Stage distributions, size dimorphism (male vs female), and maturity ogives.
*   **Spatial Maps**: 2D heatmaps showing organism density or environmental variables (e.g., temperature).
*   **Stock-Recruitment Plots**: Empirical fits of population dynamics to macro models.

### Interactive Tools
*   **Interactive Dashboard**: `interactive_abm_dashboard(model)` provides a real-time GUI for exploring spatial and demographic state.
*   **Animations**: `animate_organism_ibm(model, ...)` exports high-frame-rate video of the simulation processes.

---

## 6. Interactive Julia API Usage

```julia
using model_agent

# 1. Initialize custom parameters using feature toggles
params = IBMParameters(
    use_energetics = true,
    use_thermal_stress = true,
    temp_base = 4.5,
    fishing_probability_daily = 0.0
)

# 2. Create the model with a specific species behavior (e.g., Crustacean)
model = initialize_model(
    N_agents = 500,
    dims = (50, 50),
    params = params,
    behavior = CrustaceanBehavior(), # Modular dispatch
    rng_seed = 42
)

# 3. Run the simulation for a specified duration
agent_df, model_df = run_simulation(model, 1095) # 3 years

# 4. Export and Analyze results
export_all_plots(model, agent_df, "outputs")
```

---

## 7. Verification & Test Suite

The package includes an extensive automated test suite:
```powershell
julia --project=. test/runtests.jl
```
Tests cover mathematical correctness of empirical rules, the integrity of the modular dispatch mechanism, parameter sensitivity, and demographic conservation laws.
