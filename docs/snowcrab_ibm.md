# Snow Crab Individual-Based Model (SnowCrab IBM)

A spatially explicit, end-to-end individual-based simulation model (IBM) for the snow crab (*Chionoecetes opilio*), implemented in Julia using the [Agents.jl](https://juliadynamics.github.io/Agents.jl/) ecosystem and [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/).

Authored by **Jae S. Choi**.

---

## 1. Overview & Modular Biological Architecture [L9-88]

The Snow Crab IBM simulates the complete life history, spatial ecology, bioenergetics, developmental molting schedule, closed-loop reproduction, and demographic stock dynamics of snow crab populations.

To support both high-throughput demographic assessment and detailed eco-physiological investigation, the simulation provides **two parallel model architectures** selectable via configuration (`model = "simple"` or `"complex"`) or CLI flag (`--model simple` or `--model complex`):

```
                      ┌──────────────────────────────────────────────┐
                      │   Dual Model Architecture Selection          │
                      │   (Config: model = "simple" | "complex")     │
                      └──────────────────────┬───────────────────────┘
                                             │
               ┌─────────────────────────────┴─────────────────────────────┐
               ▼                                                           ▼
    ┌─────────────────────────────┐                             ┌─────────────────────────────┐
    │       model = "simple"      │                             │      model = "complex"      │
    │     (Default Baseline)      │                             │   (Eco-Physiological Real)  │
    ├─────────────────────────────┤                             ├─────────────────────────────┤
    │ • Time/age-based molting    │                             │ • Thermal degree-days (DD)  │
    │ • No temperature interaxns  │                             │ • Holling Type II foraging  │
    │ • No food limitation        │                             │ • W^0.75 metabolic scaling  │
    │ • Direct annual mating      │                             │ • Carapace condition (1–5)  │
    │ • No spermathecal buffer    │                             │ • Multi-driver mortality    │
    │ • Random walk dispersal     │                             │ • Habitat taxis movement    │
    │ • Baseline M + harvest      │                             │ • Size-assortative mating   │
    │ • High-throughput execution │                             │ • 3-clutch sperm storage    │
    └─────────────────────────────┘                             └─────────────────────────────┘
```

### Complete Life-History Flowchart

```
       [ Adult Females ] ──(Spring Mating)──> [ Spermathecal Storage (Complex) ]
              │                                      │
              ▼                                      ▼
       [ Brood Incubation ] <────────────── [ Egg Clutch Extrusion ]
       (~365 days external)                  (Empirical Fecundity)
              │
              ▼
       [ Larval Hatching & Regional Pelagic Drift ]
              │
              ▼ (Density-Dependent Compensatory Settlement)
       [ Benthic Recruits (Instar 4) ] ──> [ Juvenile Molting (Instars 5–10+) ]
                                                    │
                                                    ▼ (Logistic Maturity Transition)
                                             [ Terminal Molt ]
                                                    │
                                     ┌──────────────┴──────────────┐
                                     ▼                             ▼
                              [ Adult Females ]              [ Adult Males ]
                            (Reproductive Stock)           (Commercial Fishery)
```

### Key Biological Invariants & Empirical Rules

1. **Dual Parallel Model Paradigms**:
   - **`model = "simple"` (Default)**: A parsimonious distillation of the complex model. Operates without temperature interactions, thermal degree-day accumulation, food limitations, shell degradation, or multi-year spermathecal storage. Molting uses minimum intermolt durations with fixed daily hazard probabilities; maturation follows empirical logistic ogives; mating is direct with immediate 365-day brood incubation; movement follows 2D random walk dispersal; and mortality is constant baseline natural mortality plus size-selective legal male harvest.
   - **`model = "complex"`**: Fully featured eco-physiological model incorporating 2D thermal degree-day accumulation ($\text{DD}_{\text{req}}$), allometric bioenergetics ($W^{0.75}$ metabolic scaling, Holling Type II foraging), dynamic carapace condition wear (Categories 1 to 5), multi-driver mortality accounting (senescence, predation, starvation, thermal stress, shell fouling), habitat taxis (gradient movement toward thermal optimum and forage), and size-assortative mating competition with 3-clutch spermathecal buffering.

2. **Discrete Developmental Instars**:
   - Instars 1–3 represent pelagic larval stages (Zoea I, Zoea II, Megalopa).
   - Instar 4 is the primary benthic post-larval recruit ($\approx 10.7$ mm carapace width).
   - Instars 4 through 14+ represent benthic juveniles and adults.

3. **Terminal Molt Invariance**:
   - Snow crabs undergo a terminal molt (ecdysis ceases permanently) upon attaining morphometric maturity. Adult crabs allocate energy exclusively to maintenance, reproduction, and locomotion.

4. **Empirical Hiatt Growth Increments**:
   - Carapace width (CW) progression upon molting follows empirical linear-logarithmic Hiatt models with sex-specific slopes, intercepts, and maturity decrements ($\delta_{\text{mat}}$). In complex mode, increments are modulated by ambient temperature and condition.

5. **Closed-Loop Reproduction & Brood Incubation**:
   - Receptive adult females encounter mature adult males during the spring mating window (days 30–120).
   - In complex mode, females store sperm in spermathecae for up to 3 consecutive clutches, enabling autonomous multiparous oviposition even under male depletion.
   - In simple mode, mating directly initiates clutch extrusion with fixed 365-day incubation.
   - Allometric fecundity models estimate clutch size (40,000–150,000 eggs) with lognormal environmental stochasticity.
   - Hatching larvae undergo density-dependent compensatory survival to settle into Instar 4 benthic recruits.

---

## 2. Package & Codebase Structure

The simulation is organized into modular files within `src/`:

```
model_agent/
├── Project.toml              # Dependencies: Agents, DataFrames, Distributions, Plots, etc.
├── Manifest.toml             # Pinned dependency environment
├── README.md                 # Project overview and quickstart guide
├── docs/
│   ├── snowcrab_ibm.md       # Primary documentation and user guide (this file)
│   └── snowcrab_ibm_research_paper.md # Scientific research paper specification
├── src/
│   ├── model_agent.jl        # Root package module & public API exports
│   ├── types.jl              # Structs: SnowCrab, IBMParameters, EmpiricalCovModel; Enums
│   ├── empirical_biology.jl  # Empirical rules, bounds, growth, covariance modeling
│   ├── crab_dynamics.jl      # Bioenergetics, molt thresholds, ecdysis, and mortality
│   ├── simulation.jl         # Spatial GridSpace, simple/complex steppers, config loaders
│   ├── analysis.jl           # Summary statistics, stock-recruitment, and macro models
│   └── visualization.jl      # Diagnostic plots, interactive dashboards, and animations
├── scripts/
│   ├── snowcrab_ibm.config   # Default TOML configuration file (model = "simple")
│   ├── snowcrab_ibm.jl       # Standalone simulation CLI tool (--model simple|complex)
│   ├── snowcrab_orchestrator.jl # Automated parameter calibration & multi-attribute exploration
│   └── fit_macro_models.jl   # Standalone post-hoc macro model fitting CLI tool
└── test/
    └── runtests.jl           # Test suite (553 unit and integration tests)
```

### Module Responsibilities

| Module File | Purpose & Key Exports |
|---|---|
| [`src/types.jl`](file:///c:/home/jae/projects/model_agent/src/types.jl) | Defines `@agent struct SnowCrab`, `CrabStage`, `CrabSex`, `IBMParameters`, `EmpiricalCovModel`, and stage predicates (`is_adult`, `is_adult_male`, `is_adult_female`, `is_ovigerous`, `is_legal_male`). |
| [`src/empirical_biology.jl`](file:///c:/home/jae/projects/model_agent/src/empirical_biology.jl) | Implements empirical equations: `llbm`, `llbf`, `carapacewidth`, `growth_increment`, `predict_postmolt_cw`, `maturity`, `fecundity`, `selectivity`, `growth_schedule`, `fit_empirical_cov_model`, `predict_from_cov_model`, `generate_mock_crab_survey`. |
| [`src/crab_dynamics.jl`](file:///c:/home/jae/projects/model_agent/src/crab_dynamics.jl) | Evaluates feeding intake, Kleiber metabolic scaling ($W^{0.75}$), size-dependent molt energetic thresholds, ecdysis execution, carapace condition wear, and multi-driver mortality (`natural_mortality_rate`, `fishing_mortality_rate`). |
| [`src/simulation.jl`](file:///c:/home/jae/projects/model_agent/src/simulation.jl) | Implements spatial GridSpace, cached demographics, habitat taxis movement (`crab_disperse!`), simple and complex steppers (`agent_step_simple!`, `agent_step_complex!`, `model_step_simple!`, `model_step_complex!`), configuration persistence (`load_ibm_configuration`, `save_ibm_configuration`), and model initialization (`initialize_model`). |
| [`src/analysis.jl`](file:///c:/home/jae/projects/model_agent/src/analysis.jl) | Computes population demographic snapshots (`summarize_population`), empirical maturity ogives, stock-recruitment fitting, macro models (Schaefer & Delay-Difference), retrospective peels, and Generalized Leslie stage models. |
| [`src/visualization.jl`](file:///c:/home/jae/projects/model_agent/src/visualization.jl) | Produces 13 diagnostic figures, executive dashboard, interactive GUI exploration (`interactive_abm_dashboard`), and video animations (`animate_snowcrab_ibm`). |

---

## 3. Empirical Formulations & Mathematical Reference

### 3.1 Instar Carapace Width Bounds
Lower and upper carapace width boundaries for developmental instars are derived from empirical modal analysis:

$$\text{Males (llbm)}: \quad CW_{\min}(i) = \exp(1.917564 + 0.298914 \cdot (i - 3))$$
$$\text{Females (llbf)}: \quad CW_{\min}(i) = \exp(2.198848 + 0.315026 \cdot (i - 4))$$

### 3.2 Hiatt Growth Increment Model & Temperature Modulation
Post-molt carapace width is predicted from pre-molt size on the logarithmic scale:

$$\ln(CW_{t+1}) = \beta_0 + \beta_1 \ln(CW_t) + \delta_{\text{mat}} \cdot \mathbb{I}_{\text{mat}} + \Delta_{\text{temp}}(T) + \epsilon$$

Empirical coefficients:
- **Males**: $\beta_0 \sim \mathcal{N}(0.30153, 0.02198)$, $\beta_1 \sim \mathcal{N}(1.00027, 0.00667)$, $\delta_{\text{mat}} \sim \mathcal{N}(-0.13564, 0.00922)$
- **Females**: $\beta_0 \sim \mathcal{N}(0.1931, 0.0545)$, $\beta_1 \sim \mathcal{N}(1.0372, 0.0181)$, $\delta_{\text{mat}} \sim \mathcal{N}(-0.1310, 0.0252)$
- **Phenotypic Residual Variation**: $\epsilon \sim \mathcal{N}(0, \sigma^2)$ with $\sigma = 0.045\text{ ln(mm)}$.
- **Temperature Modulation ($\Delta_{\text{temp}}$)**: In complex mode, ambient thermal anomalies adjust the increment:
  $$\Delta_{\text{temp}}(T) = \gamma_T \cdot (T - T_{\text{ref}})$$
  capturing thermal growth acceleration within optimal temperature envelopes ($1.0^\circ\text{C} \le T \le 4.5^\circ\text{C}$).

### 3.3 Maturity Ogives
The probability of attaining morphometric maturity at ecdysis is modeled as a logistic function of carapace width:

$$P(\text{mature} \mid CW, \text{sex}) = \frac{1}{1 + \exp(-(\alpha + \beta \ln(CW)))}$$

- **Males**: $\alpha \sim \mathcal{N}(-31.620, 0.121)$, $\beta \sim \mathcal{N}(6.9957, 0.0270)$ ($L_{50} \approx 91.8$ mm)
- **Females**: $\alpha \sim \mathcal{N}(-51.1947, 0.2551)$, $\beta \sim \mathcal{N}(12.9812, 0.0641)$ ($L_{50} \approx 51.6$ mm)

Upon molting, a stochastic Bernoulli trial is evaluated: $u \sim \mathcal{U}(0, 1) < P(\text{mature} \mid CW, \text{sex})$. If true, the individual transitions to adult terminal maturity and ecdysis ceases permanently.

### 3.4 Allometric Fecundity
Clutch size is estimated based on female carapace width with lognormal variability:

- **Primiparous Females**:
  $$\log_{10}(F) = 2.94 \cdot \log_{10}(CW) - 0.7229 + \epsilon, \quad R^2 = 0.88$$
- **Multiparous Females**:
  $$\log_{10}(F) = 2.5811 \cdot \log_{10}(CW) + 0.0158 + \epsilon, \quad R^2 = 0.607$$

### 3.5 Multivariate Empirical Covariance Matrix Modeling
To infer joint biological distributions from empirical field survey data without relying solely on isolated univariate equations, the package provides a unified multivariate covariance estimation module:

$$\mathbf{y} = \left[ \ln(CW), \, \ln(W), \, \text{logit}(P_{\text{mat}}), \, \iota, \, \ln(F) \right]^T \sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{\Sigma})$$

Functions:
- `fit_empirical_cov_model(data_df)`: Fits mean vector $\boldsymbol{\mu}$ and positive-definite covariance matrix $\mathbf{\Sigma}$ from biological samples.
- `predict_from_cov_model(model, N)`: Generates $N$ synthetic individuals preserving empirical correlation structures across carapace width, body mass, maturity, developmental stage, and reproductive potential.
- `generate_mock_crab_survey(N)`: Creates standardized synthetic survey datasets for testing and model calibration.

### 3.6 Dynamic Carapace Condition (Shell Wear) Scoring
In complex mode, individual shell condition is tracked across five standard fisheries classification categories:
1. **Category 1 (New Soft)**: Freshly molted cuticle, vulnerable to cannibalism and physical injury, high hydration, low meat yield.
2. **Category 2 (Clean Hard)**: Hard carapace, light coloration, fully hardened chelae, prime commercial condition.
3. **Category 3 (Intermediate Hard)**: Carapace with initial epiphytic colonization, minimal shell wear.
4. **Category 4 (Old Hard)**: Darkened, scratched carapace with moderate epibiont encrustation (barnacles, bryozoans).
5. **Category 5 (Very Old / Decaying)**: Heavily fouled, brittle shell with bacterial pitting and loss of limb mobility; high natural mortality risk.

#### Shell Progression Hazard:
Shell wear progresses as a function of time post-ecdysis:
$$P(\text{advance } CC) = 1 - \exp(-\lambda_{cc} \cdot \Delta t), \quad \lambda_{cc} \approx \frac{0.35}{365.0} \text{ day}^{-1}$$

Individuals in Category 5 experience elevated senescence mortality due to cuticular erosion.

### 3.7 Natural, Shell Wear, and Multi-Driver Mortality
Natural mortality integrates multiple biological and environmental drivers:

$$M_{\text{total}} = M_{\text{base}}(stage) + M_{\text{predation}}(CW) + M_{\text{soft\_shell}}(\Delta t_{molt}) + M_{\text{starvation}}(E) + M_{\text{thermal}}(T) + M_{\text{senescence}}(age) + M_{\text{cannibalism}}(\rho)$$

1. **Base Physiological Mortality**: $M_0 / 365$ ($M_0 \approx 0.20$ yr$^{-1}$), scaled by life-history stage (e.g., higher for zoea/megalopa).
2. **Size-Attenuated Predation Risk**: Exponential attenuation with increasing carapace width:
   $$M_{\text{predation}} = \frac{0.15}{365} \exp(-0.03 \cdot (CW - 5.0))$$
3. **Post-molt Soft-shell Vulnerability**: Increased hazard immediately following ecdysis, decaying exponentially as the new shell hardens: $M_{\text{soft\_shell}} = \frac{0.12}{365} \exp(-\Delta t_{molt}/15)$.
4. **Bioenergetic Starvation Mortality**: Applied when metabolic reserves $E < 0$ (complex mode).
5. **Stenothermic Thermal Stress**: Applied when bottom temperature exceeds critical threshold ($T > T_{\text{crit}}$) in complex mode.
6. **Terminal Adult Senescence**: Exponentially increasing mortality for adult crabs once carapace age $\ge \tau_{sen}$ years.
7. **Conspecific Cannibalism**: Density-dependent risk affecting small juveniles (CW < 35 mm).

All mortality events are logged to dedicated model counters (`deaths_predation`, `deaths_starvation`, etc.) for auditing. In simple mode, mortality is simplified to a parameterized senescence risk and size-attenuated juvenile predation.
In simple mode, mortality includes a parameterized senescence risk for adults and size-attenuated juvenile predation to ensure demographic stability.

### 3.8 Modular Biological Processes (Feature Toggles)
The simulation uses a unified logic engine where specific biological processes are toggled via configuration parameters (`use_energetics`, `use_thermal_stress`, etc.):
- **Thermal Stress & Environmental Forcing**: If enabled, the model simulates sinusoidal seasonal temperature oscillations and localized thermal noise.
- **Food Dynamics**: If enabled, local food resources replenish toward carrying capacity up to a maximum level.
- **Senescence**: If enabled, adult mortality includes an exponential increase in hazard as shell age exceeds `senescence_age_threshold`.
- **Cannibalism & Energetics**: (Currently being integrated).

### 3.9 Spatial Movement: Habitat Taxis
Movement is executed by `crab_disperse!(crab, model)`. Depending on whether habitat taxis are enabled in the configuration, crabs will either perform directed movement towards optimal habitats or simple random walks.
- **Simple Mode**: Crabs perform unbiased 2D random walks to adjacent Moore neighborhood cells ($dx, dy \in \{-1, 0, 1\}$).
- **Complex Mode**: Crabs perform directed habitat taxis toward cells that minimize thermal stress ($|T - T_{\text{opt}}|$ where $T_{\text{opt}} \approx 2.5^\circ\text{C}$) and maximize available benthic food resources ($F(x, y)$).

### 3.10 Mating Dynamics & Size-Assortative Competition
- **Simple Mode**: Receptive females search within neighborhood radius $r = 2$. Encounter with any mature adult male triggers copulation; the female extrudes eggs immediately without sperm storage, initiating a 365-day brood incubation.
- **Complex Mode**: Receptive females evaluate all mature males in the local neighborhood. Mating features size-assortative competition: larger males have competitive priority in mate-guarding and pairing. Copulation fully charges the female spermatheca ($S \leftarrow 3$ clutches). Females autonomously extrude clutches using stored sperm even if males become scarce.

### 3.11 Recruitment and Egg Production Metrics
To facilitate both population forecasting and Stock-Recruitment (SR) modeling, the simulation tracks two distinct egg production metrics:
1. **Cumulative Total Eggs Spawned**: The total number of eggs produced by all females since $t=0$. Useful for long-term biomass audits.
2. **Annual Egg Production**: Calculated as the difference in cumulative counts between annual sampling intervals ($\Delta \text{Eggs}_{\text{annual}} = E_{\text{cum}}(t) - E_{\text{cum}}(t-\Delta t)$). This metric is critical for fitting classical SR models like Schaefer or Beverton-Holt.

---

## 4. Command-Line Interface (CLI) User Guide

The primary simulation entry point is [`scripts/snowcrab_ibm.jl`](file:///c:/home/jae/projects/model_agent/scripts/snowcrab_ibm.jl).

### 4.1 Parameter Specification & Configuration Precedence

Parameters are specified using `=` (e.g. `--key=value`) or space separation (`--key value`). The CLI operates under a strict three-tier precedence hierarchy:
1. **Internal Defaults**: Hardcoded baseline values built into the package.
2. **Configuration File**: Automatically detects `scripts/snowcrab_ibm.config` in the project root, or a custom file supplied via `--config=<path>` (or `-c=<path>`). Any parameter in the configuration file overrides internal defaults.
3. **Command-Line Arguments**: Explicit flags provided at execution time (e.g. `--model=complex --temp_base=3.5`) override both configuration file settings and internal defaults.

```powershell
julia --project=. scripts/snowcrab_ibm.jl [OPTIONS]
```

### 4.2 CLI Options

| Option | Type | Default | Description |
|---|---|---|---|
| `--model=<mode>` | String | `simple` | Simulation model architecture: `simple` (default distillation) or `complex` (full eco-physiology) |
| `--config=<path>`, `-c=<path>` | Path | `scripts/snowcrab_ibm.config` | Path to TOML configuration file |
| `--n_agents=<Int>` | Integer | `300` | Initial population size seeded at startup |
| `--days=<Int>` | Integer | `730` | Total duration of the simulation in days |
| `--grid=<nx,ny>` | `nx,ny` | `100,100` | 2D spatial grid dimensions (cells) |
| `--temp_base=<Float>` | Float | `5.0` | Annual mean bottom water temperature (°C) |
| `--temp_amp=<Float>` | Float | `4.0` | Amplitude of seasonal temperature oscillation (°C) |
| `--dt=<Float>` | Float | `1.0` | Time step in days |
| `--fishing_prob=<Float>` | Float | `0.0` | Daily commercial harvest probability on legal males |
| `--carrying_capacity=<Flt>`| Float | `300.0` | Benthic carrying capacity parameter ($K$) |
| `--molt_season_start=<Flt>`| Float | `15.0` | Day-of-year when spring molt window opens |
| `--molt_season_end=<Flt>`  | Float | `140.0` | Day-of-year when spring molt window closes |
| `--seed=<Int>` | Integer | `1234` | RNG seed for reproducible simulations |
| `--param=<key>=<value>` | String | — | Dynamically override any specific parameter |
| `--save` / `--no-save` | Flag | `true` | Save annual production results to CSV |
| `--save_file=<Filename>` | Path | `simulation_results.csv` | Output filename for CSV results |
| `--snapshot_day=<Float>` | Float | `274.0` | Day of calendar year for survey snapshot (~1 September) |
| `--frequency=<Float>` | Float | `365.0` | Output sampling frequency in days (default: 365.0 = 1 year) |
| `--plot` / `--no-plot` | Flag | `false` | Export all diagnostic PNG plots and dashboard |
| `--output_dir=<Path>` | Path | `outputs` | Target directory for generated figures |
| `--help`, `-h` | Flag | `false` | Display command-line help text |

### 4.3 Example Runs
  
#### Standard Run with Default Configuration
```powershell
julia --project=. scripts/snowcrab_ibm.jl
```
  
#### Custom Scenario via Command-Line Arguments (e.g., high fishing mortality)
```powershell
julia --project=. scripts/snowcrab_ibm.jl --fishing_prob=0.1 --days=1095 --plot
```
  
#### Multi-Year Commercial Fishery Scenario with Plotting and Custom Config File
```powershell
julia --project=. scripts/snowcrab_ibm.jl --config=my_custom_setup.config --n_agents=300 --days=1825 --plot --output_dir=outputs
```

---

## 5. Generated Visualizations & Interactive Tools

When executed with `--plot` or via `export_all_plots`, up to 13 diagnostic figures are generated in the output directory:

1. **`simulation_dashboard.png`**: High-impact 6-panel executive summary (abundance, instar breakdown, size dimorphism, spatial habitat map, allometry, recruitment).
2. **`instar_distribution.png`**: Grouped bar chart showing male and female crab counts across Instars 4–14+.
3. **`size_distribution_by_sex.png`**: Overlaid carapace width distributions with commercial legal size line ($CW = 95$ mm).
4. **`instar_allometry.png`**: Simulated individual carapace widths plotted against empirical `llbm` and `llbf` curves.
5. **`spatial_habitat.png`**: 2D bottom temperature heatmap with individual crabs color- and shape-coded by life stage.
6. **`reproduction_dynamics.png`**: 3-panel timeseries of ovigerous female prevalence, cumulative eggs spawned, and benthic recruitment pulses.
7. **`maturity_ogive.png`**: Simulated empirical proportion mature by size bin contrasted with theoretical logistic curves.
8. **`population_dynamics.png`**: Multi-panel timeseries of total abundance, mature adults, ovigerous females, mean carapace width, and recruits.
9. **`stock_recruitment.png`**: 3-panel empirical stock-recruitment diagnostic figure showing fitted Beverton-Holt and Ricker curves.
10. **`macro_population_dynamics.png`**: 4-panel diagnostic figure visualizing post-hoc Schaefer and Delay-Difference fits.
11. **`macro_retrospective.png`**: 4-panel diagnostic figure visualizing retrospective patterns and Mohn's $\rho$.
12. **`stage_specific_dynamics.png`**: 4-panel diagnostic figure of stage abundance dynamics, SSD, reproductive values, and elasticity matrix.
13. **`fishable_macro_comparison.png`**: 4-panel diagnostic figure comparing observed fishable biomass against Schaefer, Delay-Difference, and Leslie models.

### Interactive Exploration & Video Animations

- **Interactive ABM Dashboard**: `interactive_abm_dashboard(model)` launches an interactive Makie/Agents.jl GUI with live sliders for spatial navigation and property tracking.
- **Simulation Animation**: `animate_snowcrab_ibm(model, 180; filename = "outputs/snowcrab_simulation.mp4")` exports a high-frame-rate video animation of spatial crab movement, molting events, and demographic metrics over time.

---

## 6. Interactive Julia API Usage

```julia
using DrWatson
@quickactivate "model_agent"
using model_agent

# 1. Configure simulation parameters
params = IBMParameters(
    temp_base = 4.5,
    temp_amplitude = 3.5,
    fishing_probability_daily = 0.0,
    benthic_carrying_capacity = 300.0,
    dt = 1.0
)

# 2. Initialize spatial model in simple (default) or complex mode
model_simple = initialize_model(
    N_agents = 200,
    dims = (50, 50),
    params = params,
    model_mode = :simple,
    rng_seed = 42
)

model_complex = initialize_model(
    N_agents = 200,
    dims = (50, 50),
    params = params,
    model_mode = :complex,
    rng_seed = 42
)

# 3. Specify data collectors
adata = [
    (:stage, length),
    (:size_mm, safe_mean),
    (:weight_g, safe_mean),
    (is_adult, count),
    (is_adult_male, count),
    (is_adult_female, count),
    (is_ovigerous, count),
    (is_legal_male, count),
    (:instar, safe_mean)
]
mdata = [:total_eggs_spawned, :total_recruits_settled, :total_catch_numbers]

# 4. Execute simulation (e.g. 3 years = 1095 days)
agent_df, model_df = run_simulation(model_simple, 1095; adata = adata, mdata = mdata)

# 5. Summarize population demographics and biomass
summary = summarize_population(model_simple)
println("Final living crabs:    ", summary.total_crabs)
println("Total Biomass (kg):    ", round(summary.total_biomass_kg, digits = 2))
println("Female SSB (kg):       ", round(summary.female_ssb_kg, digits = 2))
println("Mature Male Biomass:   ", round(summary.male_mature_biomass_kg, digits = 2))
println("Legal Biomass (>=95mm):", round(summary.legal_biomass_kg, digits = 2))

# 6. Fit empirical multivariate covariance models
mock_survey = generate_mock_crab_survey(500; rng_seed = 42)
cov_fit = fit_empirical_cov_model(mock_survey)
pred_crabs = predict_from_cov_model(cov_fit, 100)

# 7. Fit macro-ecological dynamic models
save_simulation_results("outputs/simulation_results.csv", agent_df, model_df)
macro_fits = fit_population_dynamics("outputs/simulation_results.csv"; lag = 1)
println(summarize_population_dynamics(macro_fits))

# 8. Fit Generalized Leslie stage-transition matrix model
stages = build_instar_stages()
stage_mat, catch_mat, yrs = extract_annual_stage_matrix("outputs/simulation_results.csv"; stages = stages)
leslie_fit = fit_generalized_leslie_model(stage_mat, catch_mat; stages = stages, years = yrs)
println(summarize_leslie_model(leslie_fit))

# 9. Batch export all diagnostic plots to disk (13 figures)
export_all_plots(model_simple, agent_df, "outputs"; model_df = model_df)
```

---

## 7. Verification & Test Suite

Run the full automated test suite:
```powershell
julia --project=. test/runtests.jl
```
All **553 automated unit and integration tests** verify:
- Monotonicity and bounds ordering of empirical instar functions (`llbm`, `llbf`).
- Hiatt growth increments with phenotypic residual variance and temperature modulation.
- Logistic maturity schedules and allometric fecundity methods.
- Commercial gear selectivity response.
- Instar incrementation on ecdysis and terminal molt invariance.
- Simple model stepping: time/age-based molting, direct annual mating without multi-year spermathecal buffers, random walk dispersal, and no food limitation.
- Complex model stepping: thermal degree-days ($\text{DD}_{\text{req}}$), allometric bioenergetics ($W^{0.75}$), Holling Type II foraging, shell condition wear (Categories 1–5), multi-driver mortality accounting, habitat taxis, and size-assortative mating competition.
- Empirical covariance matrix estimation (`fit_empirical_cov_model`) and multi-trait predictive sampling (`predict_from_cov_model`).
- Stock biomass aggregations: Total Biomass, Female SSB, Mature Male Biomass, and Legal Biomass.
- Empirical stock-recruitment extraction, Beverton-Holt and Ricker non-linear model fitting, and AIC selection.
- Schaefer logistic and Delay-Difference parameter estimation for Whole Population and Fishable Stock.
- Retrospective pattern reanalysis, terminal relative bias calculation, and Mohn's $\rho$ quantification against $|\rho| \le 0.20$.
- CSV simulation results persistence, loading, and post-hoc analysis.
- Instar stage definitions (15 stages, pooled instars 4–5, sex-differentiated instars 6+).
- Gaussian Kernel Density Estimation (KDE) and stage derivation.
- Generalized Leslie / Lefkovitch stage-transition matrix estimation ($\mathbf{n}_{t+1} = \mathbf{A}\mathbf{n}_t - \mathbf{C}_t$).
- Demographic spectral decomposition: dominant eigenvalue $\lambda_1$, growth rate $r$, stable stage distribution (SSD), reproductive values (RV), damping ratio $\rho$, and elasticity matrix $\mathbf{E}$.
- Three-way macro model comparison for fishable biomass (Schaefer vs Delay-Difference vs Leslie matrix model) evaluating variability ($CV$, $SD$, $\text{Var}(\Delta B)$), $R^2$, RMSE, and AIC ranking.
- All 13 biological visualization tools, interactive dashboards, animations, and batch file exports.
