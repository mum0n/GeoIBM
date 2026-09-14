# Snow Crab Individual-Based Model (SnowCrab IBM)

A spatially explicit, multi-agent eco-physiological simulation model for the snow crab (*Chionoecetes opilio*), implemented in Julia using the [Agents.jl](https://juliadynamics.github.io/Agents.jl/) ecosystem and [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/).

Authored by **Jae S. Choi**.

---

## 1. Overview & Dual Biological Architectures

The Snow Crab IBM simulates the complete life history, spatial ecology, bioenergetics, molting phenology, closed-loop reproduction, and stock dynamics of snow crab populations.

To support both high-throughput demographic assessment and detailed eco-physiological investigation, the simulation provides **two parallel model architectures** selectable via `model = "simple"` (default) or `model = "complex"`:

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

### Biological Life-Cycle Diagram

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

### Key Biological Features & Empirical Rules

1. **Dual Parallel Model Paradigms**:
   - **`model = "simple"` (Default)**: A parsimonious distillation omitting thermal dependencies, degree-day accumulation, food constraints, shell degradation, and multi-year sperm storage. Uses minimum intermolt durations with fixed daily hazard probabilities, baseline morphometric maturation, direct annual mating with immediate 365-day brood incubation, random walk dispersal, and constant baseline natural mortality + legal male harvest. Ideal for rapid demographic exploration and macro-model calibration.
   - **`model = "complex"`**: Fully featured eco-physiological model incorporating 2D thermal degree-day accumulation ($\text{DD}_{\text{req}}$), allometric bioenergetics ($W^{0.75}$ metabolic scaling, Holling Type II foraging), dynamic carapace condition wear (Categories 1 to 5), multi-driver mortality accounting (senescence, predation, starvation, thermal stress, shell fouling), habitat taxis (gradient movement toward thermal optimum and forage), and size-assortative mating competition with 3-clutch spermathecal buffering.

2. **Discrete Developmental Instars & Modal Bounds**:
   - Pelagic larval stages (Instars 1–3: Zoea I, Zoea II, Megalopa).
   - Primary benthic post-larval recruit (Instar 4, $\approx 10.7$ mm carapace width).
   - Benthic juveniles and adults (Instars 4 through 14+) with empirical modal bounds (`llbm`, `llbf`).

3. **Probabilistic Hiatt Growth Increments & Temperature Modulation**:
   - Post-molt carapace width follows empirical linear-logarithmic Hiatt models with sex-specific slopes, intercepts, maturity decrements ($\delta_{\text{mat}}$), and phenotypic residual error ($\epsilon \sim \mathcal{N}(0, 0.045^2)$).
   - In complex mode, growth increments are modulated by ambient thermal conditions and individual body condition.

4. **Terminal Molt Invariance & Logistic Maturity**:
   - Morphometric maturation probability is evaluated via empirical logistic ogives ($CW_{50} \approx 91.8$ mm for males, $51.6$ mm for females) with stochastic Bernoulli trials. Post-maturation ecdysis ceases permanently.

5. **Empirical Covariance Matrix Predictive Modeling**:
   - Integrated multivariate log-scale covariance model (`fit_empirical_cov_model`, `predict_from_cov_model`) estimating joint distributions of carapace width, wet mass, maturity probability, instar, and fecundity directly from biological survey data.

6. **Dynamic Carapace Condition Scoring (Complex Mode)**:
   - Explicit tracking of shell condition across 5 standard fisheries categories (1: new soft, 2: clean hard, 3: intermediate, 4: old hard, 5: very old/decaying), governing shell-wear mortality, fouling, and molting eligibility.

7. **Stock Biomass Accounting & Allometric Scaling**:
   - Power scaling $W = \alpha \cdot CW^\beta$ differentiating immature/mature males and females.
   - Dynamic tracking of Total Biomass ($B_{\text{total}}$), Female Spawning Stock Biomass ($\text{SSB}_F$), Mature Male Biomass ($\text{MMB}$), and Legal Commercial Male Biomass ($B_{\text{legal}}$, males $\ge 95$ mm).

8. **Macro-Ecological Dynamic Model Fitting & Retrospective Diagnostics**:
   - Integrated estimation of discrete-time Schaefer logistic surplus production models and Deriso-Schnute delay-difference models for both the Whole Population and the Commercial Fishable Stock.
   - Automated sequential retrospective peeling and Mohn's $\rho$ diagnostic calculation against the $|\rho| \le 0.20$ threshold.

9. **Generalized Leslie Stage-Transition Matrix & 3-Way Model Comparison**:
   - Continuous size distributions mapped to 15 instar stages via Gaussian Kernel Density Estimation (KDE).
   - Generalized transition matrix $\mathbf{n}_{t+1} = \mathbf{A}\mathbf{n}_t - \mathbf{C}_t$, dominant eigenvalue $\lambda_1$, growth rate $r = \ln(\lambda_1)$, stable stage distribution (SSD), reproductive values (RV), damping ratio $\rho$, and elasticity matrix $\mathbf{E}$.
   - Rigorous three-way comparison of fishable biomass variations across Schaefer, Delay-Difference, and Leslie matrix models (Mean, SD, CV, $\text{Var}(\Delta B)$, $R^2$, RMSE, AIC).

---

## 2. Directory Structure

```
model_agent/
├── README.md                          # Project overview, quickstart, and CLI guide (this file)
├── Project.toml                       # Dependencies: Agents, DataFrames, Distributions, Plots, LinearAlgebra
├── Manifest.toml                      # Pinned reproducible dependency environment
├── docs/
│   ├── snowcrab_ibm.md                # Comprehensive documentation and technical user manual
│   └── snowcrab_ibm_research_paper.md # Scientific research paper specification and mathematical methods
├── scripts/
│   ├── snowcrab_ibm.config            # Default TOML configuration file (model = "simple")
│   ├── snowcrab_ibm.jl                # Standalone simulation CLI tool (--model simple|complex)
│   ├── snowcrab_orchestrator.jl       # Automated parameter calibration & multi-attribute exploration
│   └── fit_macro_models.jl            # Post-hoc macro model fitting CLI tool (--retro, --leslie)
├── src/
│   ├── model_agent.jl                 # Root module and public API exports
│   ├── types.jl                       # Structs (SnowCrab, IBMParameters, LeslieMatrixFit) and Enums
│   ├── empirical_biology.jl           # Empirical equations, KDE derivation, and covariance models
│   ├── crab_dynamics.jl               # Bioenergetics, molt hazard, ecdysis, and multi-driver mortality
│   ├── simulation.jl                  # Spatial GridSpace, simple/complex steppers, config loaders
│   ├── analysis.jl                    # Macro models, retrospective peeling, and Leslie matrix estimation
│   └── visualization.jl               # Diagnostic figures, interactive dashboards, and animations
├── test/
│   └── runtests.jl                    # Automated test suite (553 unit and integration tests)
└── outputs/                           # Directory for simulation results, CSV tables, and exported figures
```

---

## 3. Quickstart & Installation

This project utilizes [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/) to ensure 100% scientific reproducibility.

### Step 1: Clone & Instantiate Dependencies
Open a terminal in the project directory:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Step 2: Run the Automated Test Suite
Verify that all 553 unit and integration tests pass:

```bash
julia --project=. test/runtests.jl
```

---

## 4. Command-Line Interface (CLI) Usage

### 4.1 Running the IBM Simulation (`scripts/snowcrab_ibm.jl`)

The primary simulation runner supports execution parameterized via configuration files (`snowcrab_ibm.config`) and command-line arguments:

```powershell
julia --project=. scripts/snowcrab_ibm.jl [OPTIONS]
```

#### Configuration File Precedence:
The CLI uses a strict three-tier precedence hierarchy:
1. **Internal Defaults**: Hardcoded baseline parameters in the package.
2. **Configuration File**: Automatically detects `scripts/snowcrab_ibm.config` in the working directory/project root, or a custom file supplied via `--config <path>`. All parameters in the config file override internal defaults.
3. **Command-Line Arguments**: Explicit CLI flags (e.g. `--model complex --days 1095`) override values from both the config file and internal defaults.

#### Common Examples:

- **Run with Default Configuration (`model = "simple"` auto-detected)**:
  ```powershell
  julia --project=. scripts/snowcrab_ibm.jl
  ```

- **Run with the Detailed Eco-Physiological Model (`model = "complex"`)**:
  ```powershell
  julia --project=. scripts/snowcrab_ibm.jl --model complex --plot
  ```

- **Specify a Custom Configuration File**:
  ```powershell
  julia --project=. scripts/snowcrab_ibm.jl --config custom_run.config
  ```

- **Run with Custom Config and Specific Parameter Overrides**:
  ```powershell
  julia --project=. scripts/snowcrab_ibm.jl --config scripts/snowcrab_ibm.config --days 1095 --plot --output_dir outputs
  ```

- **Commercial Harvest Scenario (2% daily fishing probability on legal males)**:
  ```powershell
  julia --project=. scripts/snowcrab_ibm.jl --n_agents 300 --days 1825 --fishing_prob 0.02 --plot
  ```

- **Multi-Decadal Long Horizon Run (10 Years)**:
  ```powershell
  julia --project=. scripts/snowcrab_ibm.jl --n_agents 250 --days 3650 --carrying_capacity 350
  ```

#### Supported CLI Options:

| Option | Type | Default | Description |
|---|---|---|---|
| `--model` | String | `simple` | Model architecture mode: `simple` (default distillation) or `complex` (full eco-physiology) |
| `--config`, `-c` | Path | `scripts/snowcrab_ibm.config` | Path to TOML configuration file (overrides internal defaults) |
| `--n_agents` | Integer | `300` | Initial population size seeded at startup |
| `--days` | Integer | `730` | Duration of the simulation in days |
| `--grid` | `nx,ny` | `100,100` | 2D spatial grid dimensions (cells) |
| `--temp_base` | Float | `5.0` | Annual mean bottom water temperature (°C) |
| `--temp_amp` | Float | `4.0` | Seasonal temperature oscillation amplitude (°C) |
| `--dt` | Float | `1.0` | Simulation time step in days |
| `--fishing_prob` | Float | `0.0` | Daily harvest probability on legal males ($\ge 95$ mm) |
| `--carrying_capacity`| Float | `300.0` | Benthic carrying capacity parameter ($K$) |
| `--molt_season_start`| Float | `15.0` | Day-of-year when spring molt window opens |
| `--molt_season_end`  | Float | `140.0` | Day-of-year when spring molt window closes |
| `--seed` | Integer | `1234` | RNG seed for reproducible execution |
| `--param <k>=<v>` | String | — | Override any individual parameter dynamically |
| `--save` / `--no-save` | Flag | `true` | Enable/disable saving annual simulation results to CSV |
| `--save_file` | Path | `simulation_results.csv` | Output filename for saved results |
| `--snapshot_day` | Float | `274.0` | Day of calendar year for survey snapshot (~1 September) |
| `--frequency` | Float | `365.0` | Output sampling frequency in days (default: 365.0 = 1 year) |
| `--plot` / `--no-plot` | Flag | `false` | Generate and export all diagnostic figures |
| `--output_dir` | Path | `outputs` | Target directory for generated files |
| `--help`, `-h` | Flag | `false` | Display command-line help text |

#### Saved Simulation Output Metrics (`simulation_results.csv`):
The simulation records time-series indicators at each sampling interval (`--frequency`, default 365 days; day-of-year `--snapshot_day`, default 274.0):
- `total_numbers`: Total living crab abundance ($N_t$)
- `legal_numbers`: Commercial legal-sized adult males ($CW \ge 95$ mm)
- `mature_females`: Total abundance of morphometrically mature females
- `mature_males`: Total abundance of morphometrically mature males
- `fishable_male_biomass_kg`: Total biomass of legal fishable males in kg
- `mature_male_biomass_kg`: Total biomass of mature males in kg
- `mature_female_biomass_kg`: Total biomass of mature females (spawning stock biomass) in kg
- `total_egg_production`: Cumulative egg production across all spawning events
- `catch_total` / `catch_legal`: Fishery removals during the interval
- `surplus_production_total` / `surplus_production_legal`: Realized surplus production ($N_{t+1} - N_t + C_t$)

---

### 4.2 Post-Hoc Macro-Ecological Model Fitting, Retrospective & Leslie Analysis (`scripts/fit_macro_models.jl`)

Fit discrete-time Schaefer logistic surplus production models, Deriso-Schnute delay-difference models, and Generalized Leslie stage-transition matrix models to saved simulation results, with optional retrospective peel analysis and 3-way biomass comparison:

```powershell
# Standard fitting
julia --project=. scripts/fit_macro_models.jl --file outputs/simulation_results.csv --lag 1 --output_dir outputs

# Complete macro suite: Schaefer, Delay-Diff, Retrospective (5 peels), Leslie Matrix & 3-Way Comparison
julia --project=. scripts/fit_macro_models.jl --file outputs/simulation_results.csv --retro --peels 5 --leslie --output_dir outputs
```

The script estimates:
- **Schaefer Surplus Production**: Intrinsic growth rate $r$, carrying capacity $K$, maximum sustainable yield $MSY = rK/4$, biomass at MSY $N_{MSY} = K/2$, harvest rate $U_{MSY} = r/2$, $R^2$, and AIC.
- **Deriso-Schnute Delay-Difference**: Annual survival $s$, natural mortality $M = -\ln(s)$, recruitment parameters $(\alpha, \beta)$, $R^2$, and AIC.
- **Retrospective Peels & Mohn's $\rho$**: Computes sequential terminal peels ($p=1,\dots,P$), calculates Mohn's $\rho$ relative to full baseline, and evaluates stability against the Hurtado-Ferro et al. (2015) $|\rho| \le 0.20$ threshold.
- **Generalized Leslie Matrix Model**: Fits $15 \times 15$ transition matrix $\mathbf{A}$, extracts dominant eigenvalue $\lambda_1$, growth rate $r = \ln(\lambda_1)$, stable stage distribution (SSD), reproductive values, damping ratio $\rho$, and elasticity matrix.
- **Three-Way Macro Model Comparison**: Directly compares observed fishable stock biomass against Schaefer, Delay-Difference, and Leslie projections, reporting variability ($CV$, $SD$, $\text{Var}(\Delta B)$), $R^2$, RMSE, and AIC rankings.
- Generates 4-panel diagnostic figures: `macro_population_dynamics.png`, `macro_retrospective.png`, `stage_specific_dynamics.png`, and `fishable_macro_comparison.png`.

---

### 4.3 Simulation Orchestrator & Automated Parameter Calibration (`scripts/snowcrab_orchestrator.jl`)

The orchestrator script systematically varies user-selected model parameters across ranges or discrete candidate grids while fixing all other parameters to baseline values (from `snowcrab_ibm.config`). Simulations are executed until multi-attribute biological and ecological criteria are achieved:

```powershell
# 1. Sensitivity grid over temperature and food abundance:
julia --project=. scripts/snowcrab_orchestrator.jl --vary="temp_base=3.5,5.0;food_max=8.0,12.0"

# 2. Linear parameter sweep across a range:
julia --project=. scripts/snowcrab_orchestrator.jl --range="temp_base=3.0:1.0:6.0" --days=1095

# 3. Exhaustive search (no early stop) saving best calibrated config and figures:
julia --project=. scripts/snowcrab_orchestrator.jl \
    --vary="temp_base=4.0,5.0;food_max=8.0,12.0;carrying_capacity=250.0,350.0" \
    --max_trials=10 --no-stop_on_first_pass --save_best=calibrated.config --plot_best
```

---

## 5. Interactive Julia API Usage

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

# 3. Define data collectors
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

# 4. Execute simulation (e.g. 5 years = 1825 days)
agent_df, model_df = run_simulation(model_simple, 1825; adata = adata, mdata = mdata)

# 5. Extract demographic summary & stock biomass
summary = summarize_population(model_simple)
println("Final living crabs:       ", summary.total_crabs)
println("Total Biomass (kg):       ", round(summary.total_biomass_kg, digits = 2))
println("Female SSB (kg):          ", round(summary.female_ssb_kg, digits = 2))
println("Mature Male Biomass (kg): ", round(summary.male_mature_biomass_kg, digits = 2))
println("Legal Biomass (kg):       ", round(summary.legal_biomass_kg, digits = 2))

# 6. Multivariate Covariance Modeling from Biological Survey Data
mock_data = generate_mock_crab_survey(500; rng_seed = 42)
cov_model = fit_empirical_cov_model(mock_data)
pred_crabs = predict_from_cov_model(cov_model, 100)
println("Fitted Covariance Model Variables: ", cov_model.variable_names)

# 7. Fit empirical stock-recruitment relationships
sr_df = extract_stock_recruitment(model_simple)
sr_fits = fit_stock_recruitment(sr_df.spawners, sr_df.recruits)
println(summarize_stock_recruitment(sr_df; fits = sr_fits))

# 8. Fit macro-ecological models & export persistence file
save_simulation_results("outputs/simulation_results.csv", agent_df, model_df)
macro_fits = fit_population_dynamics("outputs/simulation_results.csv"; lag = 1)
println(summarize_population_dynamics(macro_fits))

# 9. Export all diagnostic figures to disk
export_all_plots(model_simple, agent_df, "outputs"; model_df = model_df)
```

---

## 6. Generated Visualizations & Interactive Tools

When executed with `--plot` or via `export_all_plots`, up to 13 diagnostic figures are generated:

1. **`simulation_dashboard.png`**: 6-panel executive summary (abundance timeseries, instar breakdown, size dimorphism, spatial habitat map, allometric scaling, cumulative recruitment).
2. **`instar_distribution.png`**: Grouped bar chart of male and female counts across Instars 4–14+.
3. **`size_distribution_by_sex.png`**: Overlaid carapace width distributions with legal size threshold line ($CW = 95$ mm).
4. **`instar_allometry.png`**: Simulated individual sizes plotted against empirical `llbm` and `llbf` boundaries.
5. **`spatial_habitat.png`**: 2D bottom water temperature heatmap with individual crabs color- and shape-coded by life stage.
6. **`reproduction_dynamics.png`**: 3-panel timeseries of ovigerous females, eggs spawned, and benthic recruitment pulses.
7. **`maturity_ogive.png`**: Simulated empirical proportion mature by size bin vs. theoretical logistic curves.
8. **`population_dynamics.png`**: Demographic timeseries of total abundance, adults, ovigerous females, mean CW, and recruits.
9. **`stock_recruitment.png`**: 3-panel diagnostic plot of fitted Beverton-Holt and Ricker curves, compensatory survival ratios, and cohort timeseries.
10. **`macro_population_dynamics.png`**: 4-panel diagnostic plot of observed abundance vs. fitted Schaefer logistic and Deriso-Schnute delay-difference trajectories, plus surplus production parabolas.
11. **`macro_retrospective.png`**: 4-panel diagnostic plot of retrospective peel spaghetti trajectories, relative percentage error trends vs. $|\rho| \le 0.20$ bounds, and parameter stability ($r, K$).
12. **`stage_specific_dynamics.png`**: 4-panel diagnostic plot of stage abundance dynamics, observed vs. asymptotic stable stage distribution (SSD), stage reproductive values (RV), and elasticity matrix heatmap.
13. **`fishable_macro_comparison.png`**: 4-panel diagnostic plot comparing observed fishable biomass against Schaefer, Delay-Difference, and Generalized Leslie models, residuals, variability metrics ($CV, \sigma$), and AIC model selection ranking.

### Interactive Exploration & Video Animations

- **Interactive ABM Dashboard**: `interactive_abm_dashboard(model)` launches an interactive GUI with live sliders for spatial exploration, agent filtering, and stepping.
- **Simulation Animation**: `animate_snowcrab_ibm(model, 180; filename = "outputs/snowcrab_simulation.mp4")` exports a high-frame-rate video animation of spatial crab movement, molting events, and demographic metrics over time.

---

## 7. Documentation & References

- **Comprehensive User Guide**: [docs/snowcrab_ibm.md](file:///c:/home/jae/projects/model_agent/docs/snowcrab_ibm.md)
- **Scientific Research Paper Specification**: [docs/snowcrab_ibm_research_paper.md](file:///c:/home/jae/projects/model_agent/docs/snowcrab_ibm_research_paper.md)
- **Primary Methodological Literature**:
  - Alunno-Bruscia, M., & Sainte-Marie, B. (1998). *CJFAS*, 55(2), 459–477.
  - Beverton, R. J. H., & Holt, S. J. (1957). *On the Dynamics of Exploited Fish Populations*.
  - Caswell, H. (2001). *Matrix Population Models: Construction, Analysis, and Interpretation*. 2nd Ed.
  - Conan, G. Y., & Comeau, M. (1986). *CJFAS*, 43(9), 1710–1719.
  - Datseris, G., et al. (2022). *Simulation*, 98(9), 803–813 (Agents.jl).
  - Dawe, E. G., et al. (2012). *Fisheries Research*, 121–122, 61–70.
  - Deriso, R. B. (1980). *CJFAS*, 37(2), 268–282.
  - Grimm, V., et al. (2020). *JASSS*, 23(2), 7 (ODD Protocol).
  - Hurtado-Ferro, F., et al. (2015). *ICES JMS*, 72(1), 99–110 (Retrospective patterns).
  - Lefkovitch, L. P. (1965). *Biometrics*, 21(1), 1–18.
  - Leslie, P. H. (1945). *Biometrika*, 33(3), 183–212.
  - Mohn, R. (1999). *ICES JMS*, 56(4), 473–488 (Mohn's rho).
  - Sainte-Marie, B., et al. (1995). *CJFAS*, 52(5), 903–924.
  - Schaefer, M. B. (1954). *IATTC Bulletin*, 1(2), 27–56.
  - Schnute, J. (1985). *CJFAS*, 42(3), 414–429.
