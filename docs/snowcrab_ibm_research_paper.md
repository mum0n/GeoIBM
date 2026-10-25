# An End-to-End Individual-Based Model for Snow Crab (*Chionoecetes opilio*): Integrating Bioenergetics, Thermal Molt Phenology, Allometric Growth, Closed-Loop Reproduction, and Configurable Process Toggles

Jae S. Choi
Bedford Institute of Oceanography, Fisheries and Oceans Canada
Dartmouth, NS, Canada


## Document Metadata
- **Working Title**: An End-to-End Individual-Based Model for Snow Crab (*Chionoecetes opilio*): Integrating Bioenergetics, Thermal Molt Phenology, Allometric Growth, Closed-Loop Reproduction, and Configurable Process Toggles
- **Target Venues**: *Canadian Journal of Fisheries and Aquatic Sciences*, *Ecological Modelling*, or *ICES Journal of Marine Science*
- **Document Status**: Comprehensive Research Paper Specification & Mathematical Methods
- **Software Implementation**: `model_agent.jl` (Julia / Agents.jl ecosystem)

---

## Abstract

1. **Background**: The snow crab (*Chionoecetes opilio*) supports premier commercial cold-water fisheries across the Northwest Atlantic (Scotian Shelf, Gulf of St. Lawrence, Newfoundland and Labrador) and subarctic Pacific (Bering Sea). Managing this resource amidst rapid ocean warming, changing cold intermediate layer (CIL) dynamics, and shifting predation landscapes requires mechanistic tools capable of resolving complex decapod life-history features: determinate growth via terminal molt, extreme sexual size dimorphism, long-term female spermathecal sperm storage, temperature-dependent egg incubation, size-dependent predation, shell wear dynamics, and strict thermal degree-day controls on ecdysis.
2. **Objectives**: We develop, mathematically specify, and validate an open-source, spatially explicit individual-based model (IBM) for snow crab implemented in Julia using `Agents.jl`. To bridge theoretical ecological exploration, management testing, and high-throughput assessment, the framework introduces **a configurable toggle-based architecture**:
   - a parsimonious distilled reference model (`baseline configuration`, default) driven by time/age-based molting and maturation probabilities, direct annual reproduction, and baseline mortality without thermal or food dependencies; and
   - a comprehensive eco-physiological model (`fully-featured configuration`) coupling allometric bioenergetics ($W^{0.75}$), 2D thermal degree-day integration, dynamic carapace condition wear (Categories 1–5), multi-driver natural mortality, habitat taxis, size-assortative mate competition, and 3-clutch spermathecal buffering.
3. **Methods**: The unified modeling suite integrates:
   - Aggregate dynamics for pre-benthic stages (Egg to pre-stage 1) and discrete developmental stages for benthic individuals (Stages 4–14+) with empirical modal size boundaries;
   - Linear-logarithmic Hiatt growth increments with maturity decrements and environmental temperature adjustments;
   - Multivariate empirical covariance matrix modeling (`fit_empirical_cov_model`) parameterizing joint distributions of size, wet mass, maturity, stage, and fecundity directly from biological surveys;
   - Dynamic carapace condition scoring (Categories 1 to 5) capturing shell fouling, cuticular degradation, and shell-wear mortality;
   - Physiological thermal degree-day accumulation ($\text{DD}_{\text{req}}$) and late-winter/spring adolescent molting windows (days 15–140) generating annual skip-molt phenology;
   - Closed-loop reproduction with temperature-mediated incubation ($\approx 365$–$730$ d) and Beverton-Holt compensatory benthic settlement;
   - Multi-driver natural mortality accounting (senescence, predation, starvation, thermal stress, and shell degradation);
   - Complete stock biomass accounting ($B_{\text{total}}$, $\text{SSB}_F$, $\text{MMB}$, $B_{\text{legal}}$), macro-ecological dynamic model fitting (Schaefer, Delay-Difference, Generalized Leslie matrix models), and retrospective peeling diagnostics (Mohn's $\rho$).
4. **Key Findings / Capabilities**: The IBM reproduces wild multi-year developmental chronology (6–8 years to maturity), generates realistic bimodal sexual size dimorphism, and captures emergent stock-recruitment density dependence. Comparative simulation demonstrates that while the simple model provides an agile, robust baseline for macro-model calibration, the complex model uncovers critical vulnerabilities to marine heatwaves, localized male depletion, and skip-molt accumulation under ocean warming.
5. **Significance & Prospects**: This open-source framework provides a versatile computational foundation for evaluating climate change impacts on benthic decapods, testing alternative harvest control rules, and serving as an individual-based operating model for management strategy evaluation (MSE).

**Keywords**: *Chionoecetes opilio*, individual-based model, Agents.jl, configurable process toggles, terminal molt, thermal degree-days, carapace condition, skip-molting, spermathecal storage, fisheries assessment.

---

## 1. Introduction

### 1.1 Ecological & Economic Context
The snow crab (*Chionoecetes opilio*, Brachyura: Oregoniidae) is an abundant, stenothermic cold-water decapod distributed throughout the Northwest Atlantic, the Arctic Ocean, the northern Bering Sea, and the Sea of Japan (Alunno-Bruscia & Sainte-Marie 1998; Tremblay 1997; Dawe et al. 2012). It occupies a pivotal ecological role as a dominant benthic omnivore and macro-invertebrate predator feeding on polychaetes, bivalves, crustaceans, and ophiuroids, while serving as critical forage for commercial groundfish, particularly Atlantic cod (*Gadus morhua*) and Atlantic halibut (*Hippoglossus hippoglossus*) (Boudreau et al. 2011; Dionne et al. 2003).

Economically, *C. opilio* sustains multi-hundred-million-dollar trap fisheries across Eastern Canada and Alaska. Management relies primarily on a "3-S" framework: **Size** (minimum size $size \ge 95$ mm), **Sex** (harvest exclusively targeting males, protecting maternal spawning biomass), and **Season** (spring/summer fisheries avoiding soft-shell post-molt periods) (Choi & Zisserson 2012; DFO 2022). However, recent dramatic fluctuations—such as the unprecedented 2018–2022 collapse of the Eastern Bering Sea stock linked to marine heatwaves and metabolic starvation (Szuwalski et al. 2021, 2023; Fedewa et al. 2020)—highlight acute management vulnerabilities to climate-driven environmental changes.

### 1.2 Life-History Complexities of Snow Crab
Accurate simulation of snow crab population dynamics requires resolving multiple unusual biological characteristics:
1. **Determinate Growth via Terminal Molt**: In contrast to lobsters (*Homarus americanus*) which molt throughout their lifespan, *C. opilio* undergoes a terminal molt to morphometric maturity, after which ecdysis ceases permanently (Conan & Comeau 1986; Sainte-Marie et al. 1995; Sainte-Marie & Sainte-Marie 1999).
2. **Extreme Sexual Size Dimorphism**: Mature males continue molting through stages 11–14, reaching 95–150 mm size with disproportionately enlarged chelae. Mature females undergo terminal molt earlier at stages 9–11 (50–85 mm size), developing an expanded circular abdomen for egg brooding (Alunno-Bruscia & Sainte-Marie 1998; Comeau et al. 1998).
3. **Protracted Juvenile Chronology & Skip-Molting**: Early benthic stages molt multiple times per year, but adolescent stages ($\ge 7$) transition to an annual molting cycle strictly synchronized with late-winter/spring. When thermal degree-days or energetic reserves are insufficient, adolescents skip-molt, remaining in their current stage for an additional year (Sainte-Marie et al. 1995; Dawe et al. 2012). Total development from benthic settlement to terminal maturity requires 6–8 years.
4. **Carapace Condition & Shell Aging**: Adult crabs that have completed their terminal molt remain in the population for 4–7 years. Over time, their cuticles advance through discrete shell condition categories (Categories 1 to 5), exhibiting progressive wear, epibiont fouling, cuticular breakdown, and escalating natural mortality.
5. **Complex Mating System & Spermathecal Storage**: Receptive females mate in spring (February–May). Females store viable sperm in paired ectodermal spermathecae, enabling autonomous multiparous egg extrusion for up to 3 consecutive annual clutches even in the absence of mature males (Sainte-Marie et al. 1997, 2000; Rondeau & Sainte-Marie 2001).
6. **Temperature-Mediated Embryogenesis**: Females brood clutches of 20,000–150,000 eggs beneath their abdomen. Incubation duration is tightly governed by ambient bottom water temperatures, requiring $\approx 12$ months at 3–5°C, but expanding to 24 months (biennial reproduction) in sub-zero waters ($< 1^\circ\text{C}$) (Kuhn & Choi 2011; Webb et al. 2007).

### 1.3 Limitations of Existing Population Models & The Parsimony-Realism Dilemma
Traditional fisheries assessment models—such as lumped-biomass surplus production or age-structured delay-difference models—assume continuous growth and homogeneous mixing, omitting discontinuous ecdysis, terminal molt invariance, and spatial thermal constraints. Conversely, highly complex individual-based models often face severe parameter inflation, high computational cost, and sensitivity to unobserved physiological constants.

To address this challenge systematically, ecological modelers require a **configurable framework**: a parsimonious distilled reference model that strips away environmental dependencies to establish clean baseline demographics, contrasted directly with a fully articulated eco-physiological model that introduces thermal gating, bioenergetics, and behavioral competition.

### 1.4 Objectives of the Present Study
Our specific objectives are to:
1. Formulate and implement a modular, high-performance IBM for *C. opilio* in Julia using `Agents.jl`;
2. Establish **a configurable toggle-based architecture** (`baseline configuration` vs. `fully-featured configuration`) within a single unified simulation engine;
3. Incorporate empirical modal stage boundaries, Hiatt growth increments, and logistic maturity ogives;
4. Formulate multivariate empirical covariance matrix modeling (`fit_empirical_cov_model`) to parameterize joint morphometric and reproductive distributions directly from survey data;
5. Integrate dynamic carapace condition scoring (Categories 1–5), allometric bioenergetics ($W^{0.75}$), habitat taxis, and multi-driver natural mortality accounting;
6. Implement macro-ecological dynamic model fitting (Schaefer, Delay-Difference, Generalized Leslie stage-transition matrix) and retrospective peeling diagnostics (Mohn's $\rho$);
7. Provide comprehensive model verification across 550+ automated unit and integration tests.

---

## 2. Model Description (ODD Protocol Formulation)

The model is formulated in accordance with the **ODD (Overview, Design Concepts, Details)** protocol for individual-based models (Grimm et al. 2006, 2020).

### 2.1 Overview

#### 2.1.1 Purpose
The purpose of the Snow Crab IBM is to simulate individual bioenergetics, seasonal thermal molting schedules, spatial movement, reproductive success, natural/fishing mortality, and emergent stock biomass dynamics of *C. opilio* under variable ocean temperatures and fishing exploitation rates. A key methodological feature is the **configurable architecture**:
- **`baseline configuration` (Default Baseline)**: A distilled, high-throughput model stripping away temperature dependencies, degree-day accumulation, food limitations, shell wear, and spermathecal storage. Provides an analytical reference baseline for stock assessment and macro-model calibration.
- **`fully-featured configuration` (Eco-Physiological Realism)**: Fully resolves 2D thermal degree-days ($\text{DD}_{\text{req}}$), allometric bioenergetics ($W^{0.75}$), Holling Type II feeding, carapace condition wear (Categories 1–5), multi-driver mortality, habitat taxis, size-assortative mate competition, and 3-clutch spermathecal buffering.

#### 2.1.2 Entities, State Variables, and Scales
- **Individual Agents (`SnowCrab`)**:
  Modeled as mobile agents in a 2D discrete grid space $\Omega = [1, N_x] \times [1, N_y]$. Each crab $i$ is characterized by 16 state variables:
  $$\mathbf{x}_i = \left( \text{id}, \mathbf{p}_i, \text{stage}_i, \text{sex}_i, \iota_i, a_i, CW_i, W_i, E_i, t_{\text{molt}, i}, \text{DD}_i, CC_i, \mu_{\text{ready}, i}, N_{\text{eggs}, i}, t_{\text{hatch}, i}, S_i \right)$$
  - $\mathbf{p}_i = (x_i, y_i) \in \Omega$: Spatial grid coordinate.
  - $\text{stage}_i \in \{\text{zoea}, \text{megalopa}, \text{juvenile}, \text{adult}, \text{egg}\}$: Lifecycle stage.
  - $\text{sex}_i \in \{\text{male}, \text{female}, \text{undifferentiated}\}$: Morphological sex.
  - $\iota_i \in \{1, 2, \dots, 14+\}$: Discrete developmental stage index.
  - $a_i \in \mathbb{R}^+$: Chronological age in days post-hatch.
  - $CW_i \in \mathbb{R}^+$: Size in millimeters.
  - $W_i \in \mathbb{R}^+$: Individual mass in grams.
  - $E_i \in \mathbb{R}$: Internal metabolic energy reserve budget.
  - $t_{\text{molt}, i} \in \mathbb{R}$: Simulation timestamp of most recent ecdysis (days).
  - $\text{DD}_i \in \mathbb{R}^+$: Cumulative thermal degree-days since last ecdysis (°C·days).
  - $CC_i \in \{1, 2, 3, 4, 5\}$: Carapace condition category (1=soft, 2=clean hard, 3=intermediate, 4=old, 5=very old/decaying).
  - $\mu_{\text{ready}, i} \in \{0, 1\}$: Physiological mating receptivity flag.
  - $N_{\text{eggs}, i} \in \mathbb{N}_0$: Brooded external egg count.
  - $t_{\text{hatch}, i} \in \mathbb{R}^+$: Scheduled simulation day for brood release.
  - $S_i \in \{0, 1, 2, 3\}$: Number of viable clutches remaining in spermathecal storage.

- **Environment (`GridSpace{2}`)**:
  The benthic environment consists of an $N_x \times N_y$ discrete lattice. Each spatial cell $(x, y)$ tracks:
  1. Bottom water temperature $T(x, y, t)$ (°C):
     $$T(x, y, t) = \bar{T}(x, y) + A_T \sin\left(\frac{2\pi (t - t_0)}{365}\right)$$
  2. Benthic food density $F(x, y, t)$ (arbitrary biomass units), replenished daily via logistic renewal:
     $$\Delta F(x, y, t) = r_F \cdot F(x, y, t) \cdot \left(1 - \frac{F(x, y, t)}{K_F}\right) \Delta t - \sum_{i \in \text{cell}} I_i$$

- **Model Properties & Counters**:
  The simulation container maintains cached demographic lookups (`cached_adult_males`, `cached_receptive_females`, `cached_benthic_count`) and multi-driver mortality counters:
  `natural_deaths`, `predation_deaths`, `starvation_deaths`, `thermal_deaths`, `shell_deaths`, `harvest_deaths`.

- **Scales**:
  - Temporal resolution: $\Delta t = 1.0$ day.
  - Spatial resolution: 1 grid cell $\equiv 2\text{ km} \times 2\text{ km}$ (scalable).
  - Extent: Regional shelf management areas (e.g., $50 \times 50$ to $100 \times 100$ cells $\equiv 10,000$–$40,000\text{ km}^2$).
  - Simulation horizon: Multi-year to multi-decadal (e.g., 2–15 years; 730–5,475 days).

#### 2.1.3 Process Scheduling (Dual Stepping Pipelines)

```
========================================================================================
SIMULATION PIPELINE: DUAL EXECUTION FLOW
========================================================================================

           [ Model Step Initialization: Refresh Environment & Clear Caches ]
                                          │
                                          ▼
                         [ Agent Stepping: Loop over Crabs ]
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
      ┌─────────────────────────┐                     ┌─────────────────────────┐
      │   agent_step_simple!    │                     │   agent_step_complex!   │
      ├─────────────────────────┤                     ├─────────────────────────┤
      │ 1. Random Walk Dispersal│                     │ 1. Habitat Taxis Move   │
      │ 2. Simple Aging (dt)    │                     │ 2. Degree-Day Accum     │
      │ 3. Time-Based Molting   │                     │ 3. Holling II Forage    │
      │ 4. Baseline Maturation  │                     │ 4. W^0.75 Basal Metab   │
      │ 5. Baseline Mortality   │                     │ 5. Degree-Day Ecdysis   │
      │ 6. Legal Male Harvest   │                     │ 6. Shell Wear Progress  │
      └─────────────────────────┘                     │ 7. Multi-Driver Mort    │
                                                      │ 8. Legal Male Harvest   │
                                                      └─────────────────────────┘
                                          │
                                          ▼
                         [ Model Step: Closed-Loop Processes ]
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
      ┌─────────────────────────┐                     ┌─────────────────────────┐
      │   model_step_simple!    │                     │   model_step_complex!   │
      ├─────────────────────────┤                     ├─────────────────────────┤
      │ 1. Direct Pairing       │                     │ 1. Size-Assortative     │
      │ 2. Immed 365d Extrusion │                     │    Mate Competition     │
      │ 3. Density-Dep Recruits │                     │ 2. Spermatheca Storage  │
      │ 4. Cache Demographics   │                     │ 3. Autonomous Extrusion │
      └─────────────────────────┘                     │ 4. Temp-Dep Incubation  │
                                                      │ 5. Regional Larval Drift│
                                                      │ 6. Density-Dep Recruits │
                                                      └─────────────────────────┘
```

---

## 3. Detailed Mathematical Formulations

### 3.1 Discrete Developmental Stages & Modal Bounds
Snow crabs grow through discrete stages. Stages 1 and 2 represent pelagic Zoea I and Zoea II stages ($size \approx 1.8 \pm 0.1$ mm and $2.8 \pm 0.15$ mm, respectively). Stage 3 is the pelagic Megalopa ($size \approx 3.5 \pm 0.2$ mm). Stage 4 is the first benthic post-larval recruit ($size \approx 10.7$ mm).

#### Pelagic Larval Development & Thermal Viability
In accordance with laboratory rearing and field observations in the southern Gulf of St. Lawrence (Ouellet & Sainte-Marie 2018):
- **Developmental Degree-Days**: Stage progression requires thermal degree-day accumulation $\Delta \text{DD} = \max(0, T - T_0) \Delta t$ ($T_0 = -1.5^\circ\text{C}$). Zoea I $\to$ Zoea II, Zoea II $\to$ Megalopa, and Megalopa $\to$ benthic recruit each require $\approx 270\text{ DD}$ (range $225\text{--}350\text{ DD}$, typically $260\text{--}280\text{ DD}$ at $2.8\text{--}7.8^\circ\text{C}$), with minimum intermolt durations of 36, 36, and 42 days respectively (cumulative pelagic development $\sim 800\text{ DD}$).
- **Over-Age Developmental Gate**: An excess hazard of $+0.05 \cdot (\text{days} - 84)/10$ is applied if an instar exceeds 84 days without molting, penalizing prolonged developmental delay.
- **Concave-Down Thermal Survival**: Larval survival peaks at $10\text{--}12^\circ\text{C}$ (modeled at $T_{\text{opt}} = 11.0^\circ\text{C}$, $\sigma_T = 3.5^\circ\text{C}$):
  $$h_{\text{temp}}(T) = \exp\left( 0.5 \left( \frac{T - T_{\text{opt}}}{\sigma_T} \right)^2 \right)$$

For benthic stages $i \ge 4$, empirical lower modal bounds are defined by linear-logarithmic models (Sainte-Marie et al. 1995; DFO 2020):


$$\text{Males (llbm)}: \quad \ln(CW_{\min}(i)) = 1.917564 + 0.298914 \cdot (i - 3) \implies CW_{\min}(i) = \exp(1.917564 + 0.298914 \cdot (i - 3))$$

$$\text{Females (llbf)}: \quad \ln(CW_{\min}(i)) = 2.198848 + 0.315026 \cdot (i - 4) \implies CW_{\min}(i) = \exp(2.198848 + 0.315026 \cdot (i - 4))$$

Inverted continuous stage classification:
$$\iota(size, \text{male}) = 3 + \frac{\ln(size) - 1.917564}{0.298914}, \quad \iota(size, \text{female}) = 4 + \frac{\ln(size) - 2.198848}{0.315026}$$

---

### 3.2 Hiatt Growth Increments, Bias Corrections & Temperature Modulation
Post-ecdysial size ($CW_{t+1}$) is related to pre-molt width ($CW_t$) via the empirical Hiatt growth model on the logarithmic scale (Hiatt 1948; Sainte-Marie et al. 1995):

$$\ln(CW_{t+1}) = \beta_0 + \beta_1 \ln(CW_t) + \delta_{\text{mat}} \cdot \mathbb{I}_{\text{mat}} + \Delta_{\text{temp}}(T) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

where $\mathbb{I}_{\text{mat}} \in \{0, 1\}$ indicates the terminal molt to maturity.

#### Empirical Parameters:
- **Males**: $\beta_0 = 0.30153 \pm 0.02198$, $\beta_1 = 1.00027 \pm 0.00667$, $\delta_{\text{mat}} = -0.13564 \pm 0.00922$, $\sigma = 0.040$.
- **Females**: $\beta_0 = 0.19310 \pm 0.05450$, $\beta_1 = 1.03720 \pm 0.01810$, $\delta_{\text{mat}} = -0.13100 \pm 0.02520$, $\sigma = 0.045$.

#### Unbiased Conditional Expectation in Natural Units:
$$\mathbb{E}[CW_{t+1} \mid CW_t, \mathbb{I}_{\text{mat}}] = \exp\left(\beta_0 + \beta_1 \ln(CW_t) + \delta_{\text{mat}} \mathbb{I}_{\text{mat}} + \Delta_{\text{temp}}(T) + \frac{\sigma^2}{2}\right)$$

In complex mode, thermal anomalies modulate growth increments:
$$\Delta_{\text{temp}}(T) = \gamma_T \cdot (T - T_{\text{ref}}), \quad \gamma_T = 0.015\text{ °C}^{-1}, \quad T_{\text{ref}} = 3.0^\circ\text{C}$$

---

### 3.3 Morphometric Maturation Ogives & Terminal Molt Invariance
Morphometric maturity probability at ecdysis is modeled as an empirical logistic function of size:

$$P(\text{mature} \mid size, \text{sex}) = \frac{1}{1 + \exp(-(\alpha_{\text{mat}} + \beta_{\text{mat}} \ln(size)))}$$

- **Males**: $\alpha_{\text{mat}} = -31.620 \pm 0.121$, $\beta_{\text{mat}} = 6.9957 \pm 0.0270 \implies CW_{50} = \exp(31.620 / 6.9957) = 91.82$ mm.
- **Females**: $\alpha_{\text{mat}} = -51.1947 \pm 0.2551$, $\beta_{\text{mat}} = 12.9812 \pm 0.0641 \implies CW_{50} = \exp(51.1947 / 12.9812) = 51.61$ mm.

**Terminal Molt Invariant**: Upon attaining maturity ($u \sim \mathcal{U}(0, 1) < P_{\text{mature}}$), the crab transitions permanently to `stage = adult` and ecdysis ceases permanently ($\text{can\_molt} = \text{false}$).

---

### 3.4 Allometric Length-Mass Scaling
Individual mass $W$ (g) scales allometrically with size $size$ (mm):

$$W = \alpha_{\text{wt}} \cdot size^{\beta_{\text{wt}}}$$

- **Immature Males**: $\alpha = 0.0002665$, $\beta = 3.098$
- **Mature Males**: $\alpha = 0.0001830$, $\beta = 3.160$ (hyper-allometric chela enlargement)
- **Immature Females**: $\alpha = 0.0003440$, $\beta = 2.980$
- **Mature Females**: $\alpha = 0.0004500$, $\beta = 2.920$ (broadened abdomen)

---

### 3.5 Multivariate Empirical Covariance Matrix Modeling
To derive parameter estimates and initial population states directly from empirical survey datasets, the framework provides a multivariate normal covariance module:

$$\mathbf{y} = \left[ \ln(size), \, \ln(W), \, \text{logit}(P_{\text{mat}}), \, \iota, \, \ln(F) \right]^T \sim \mathcal{N}_k(\boldsymbol{\mu}, \mathbf{\Sigma})$$

where $\boldsymbol{\mu} \in \mathbb{R}^k$ is the empirical sample mean vector and $\mathbf{\Sigma} \in \mathbb{R}^{k \times k}$ is the positive-definite empirical covariance matrix:

$$\hat{\boldsymbol{\mu}} = \frac{1}{N} \sum_{i=1}^N \mathbf{y}_i, \quad \hat{\mathbf{\Sigma}} = \frac{1}{N-1} \sum_{i=1}^N (\mathbf{y}_i - \hat{\boldsymbol{\mu}})(\mathbf{y}_i - \hat{\boldsymbol{\mu}})^T$$

Synthetic crabs sampled via `predict_from_cov_model(model, N)` preserve all pairwise biological covariances (e.g., between size, mass, maturity state, and reproductive potential), providing self-consistent initializations for simulation runs.

---

### 3.6 Molting Phenology & Stepping Mechanics: Baseline vs. Fully-Featured

#### 3.6.1 Baseline Configurationl Molt Mechanics
The simple model abstracts away thermal tracking and environmental dependencies:
- **Eligibility**: $t - t_{\text{molt}} \ge \Delta t_{\min}(\iota)$.
- **Transition Hazard**: Evaluated as a constant daily hazard:
  $$p_{\text{molt}} = 0.05 \cdot \Delta t$$
- **Ecdysis**: If $u \sim \mathcal{U}(0, 1) < p_{\text{molt}}$, the crab executes ecdysis, advances to post-molt size via `predict_postmolt_cw`, updates stage index, tests for logistic maturation, and resets $t_{\text{molt}} \leftarrow t$.

#### 3.6.2 Fully-Featured Configurationl Molt Mechanics
The complex model integrates physiological degree-days, bioenergetics, and seasonal gates:
- **Thermal Degree-Days**: $\text{DD}(t) = \sum_{\tau} \max(0, T(\tau) - T_0) \Delta t$, with $T_0 = 0.0^\circ\text{C}$.
- **Multi-Tier Gate**:
  $$\Phi_{\text{molt}} = (\text{stage} \neq \text{adult}) \land (t - t_{\text{molt}} \ge \Delta t_{\min}) \land (\text{DD} \ge \text{DD}_{\text{req}}) \land \Psi_{\text{season}}(\iota, t) \land (E \ge E_{\text{req}})$$
- **Seasonal Window ($\Psi_{\text{season}}$)**: For subadults ($\iota \ge 7$), ecdysis is restricted to days 15–140. Failure to meet criteria induces an annual skip-molt.
- **Degree-Day Hazard**: $p_{\text{molt}} = 1 - \exp(-\lambda_{\text{molt}} \Delta t)$, $\lambda_{\text{molt}} = 0.15 \times \min(\text{DD} / \text{DD}_{\text{req}}, 3.0)$.
- **Energy Cost**: Apolysis and ecdysis retain only $\kappa_{\text{retain}} = 0.20$ of accumulated reserves ($E \leftarrow 0.20 E$).

---

### 3.7 Individual Bioenergetics & Metabolic Scaling
In complex mode, individual energy dynamics follow:

$$\frac{dE_i}{dt} = \eta_{\text{assim}} \cdot I_i(CW_i, F, T) - M_{\text{basal}}(W_i, T) - C_{\text{locomotion}}$$

#### Ingestion Functional Response ($I_i$):
Holling Type II response allometrically scaled:
$$I_i(CW_i, F, T) = I_{\max} \cdot \left(\frac{F(x, y)}{F(x, y) + K_F}\right) \cdot CW_i^{\theta_I} \cdot Q_{10, I}^{(T - T_{\text{ref}})/10}$$
where $I_{\max} = 0.10$, $K_F = 100.0$, $\theta_I = 0.80$, $Q_{10, I} = 2.0$.

#### Basal Metabolism with Kleiber 3/4-Power Mass Scaling ($M_{\text{basal}}$):
$$M_{\text{basal}}(W_i, T) = m_0 \cdot W_i^{0.75} \cdot Q_{10, M}^{(T - T_{\text{ref}})/10}$$
where $m_0 = 0.008$ and $Q_{10, M} = 2.1$.

In simple mode, food limitation is omitted and crabs do not incur energetic deficits.

---

### 3.8 Reproduction, Spermathecal Storage & Incubation

#### Baseline Configurationl Reproduction:
Receptive mature females search within Moore radius $r = 2$. Encountering any mature male triggers copulation; the female extrudes an egg clutch immediately without long-term sperm storage ($S = 0$). Eggs incubate for a fixed duration $\tau = 365$ days.

#### Fully-Featured Configurationl Reproduction:
1. **Size-Assortative Mating Competition**: Mature adult males within the local neighborhood compete for receptive females. Males with larger size have competitive priority in mate-guarding and pairing.
2. **Spermathecal Buffer**: Copulation fully charges the female spermatheca ($S_i \leftarrow 3$ clutches).
3. **Autonomous Multiparous Oviposition**: In subsequent years, females with $S_i \ge 1$ fertilize and extrude clutches autonomously even in the absence of mature males ($S_{i, t+1} = S_{i, t} - 1$).
4. **Allometric Fecundity**:
   $$F_{\text{primi}}(size) \approx 0.1893 \cdot size^{2.9400}, \quad F_{\text{multi}}(size) \approx 1.0371 \cdot size^{2.5811}$$
5. **Temperature-Dependent Incubation**:
   $$\tau_{\text{incubation}} = 365.0 \cdot \left(\frac{5.0}{\max(1.0, \bar{T}_{\text{bottom}})}\right)^{0.30}$$
   extending from 365 days at 5°C to 730 days (biennial reproduction) at $< 0.5^\circ\text{C}$.

---

### 3.9 Dynamic Carapace Condition Scoring & Multi-Driver Natural Mortality

#### 3.9.1 Carapace Condition (Shell Wear) Progression
In complex mode, post-molt cuticular aging progresses through five fisheries categories ($CC_1 \to CC_5$):
- $CC_1$: New soft shell (post-ecdysis, high water content).
- $CC_2$: Clean hard shell (prime commercial quality).
- $CC_3$: Intermediate hard shell (light epibionts).
- $CC_4$: Old hard shell (darkened, scratched, encrusted).
- $CC_5$: Very old decaying shell (pitted, brittle, limb loss).

Shell transition hazard:
$$P(\text{advance } CC) = 1 - \exp(-\lambda_{cc} \cdot \Delta t), \quad \lambda_{cc} \approx \frac{0.35}{365.0}\text{ day}^{-1}$$

#### 3.9.2 Multi-Driver Mortality Formulation
Natural mortality integrates senescence, size-dependent predation, starvation, thermal stress, and shell degradation:

$$M_{\text{total}} = M_{\text{senescence}} + M_{\text{predation}}(size) + M_{\text{starvation}}(E) + M_{\text{thermal}}(T) + M_{\text{shell}}(CC)$$

1. **Adult Baseline Senescence**: $M_0 / 365$ ($M_0 = 0.20$ yr$^{-1}$).
2. **Size-Attenuated Predation**: $M_{\text{predation}} = \frac{0.15}{365} \exp(-0.03 \cdot (size - 5.0))$.
3. **Starvation Mortality**: $\mu_{\text{starv}} \cdot |E_i|$ when $E_i < 0$.
4. **Thermal Stress**: $\mu_T \cdot \max(0, T - 7.0^\circ\text{C})^2$.
5. **Shell Degradation**: Additional mortality of $\Delta M_{CC5} \approx 0.25$ yr$^{-1}$ for Category 5 carapaces.

In simple mode, natural mortality reduces strictly to baseline adult senescence plus juvenile predation attenuation.

---

### 3.10 Spatial Movement: Random Walk vs. Habitat Taxis
Movement is executed by `crab_disperse!(crab, model)`:
- **Baseline Configuration**: Crabs perform unbiased 2D random walks to adjacent Moore neighborhood cells ($dx, dy \in \{-1, 0, 1\}$).
- **Fully-Featured Configuration**: Crabs execute directed habitat taxis. For each candidate neighboring cell $(x', y')$, a habitat utility score is calculated:
  $$U(x', y') = -\omega_T |T(x', y') - T_{\text{opt}}| + \omega_F \frac{F(x', y')}{K_F}$$
  where $T_{\text{opt}} \approx 2.5^\circ\text{C}$, $\omega_T = 1.0$, and $\omega_F = 0.5$. Movement steps are selected via multinomial logit probabilities over candidate cells.

---

### 3.11 Commercial Exploitation, Stock Biomass & Macro Models
1. **Commercial Selectivity & Yield**:
   $$S_{\text{trap}}(size) = \frac{1}{1 + \exp(-(size - 95.0) / 5.0)}, \quad Y(t) = \frac{1}{1000} \sum_{i \in \text{harvested}} W_i \text{ (kg)}$$
2. **Biomass Accounting**:
   - $B_{\text{total}} = \sum_{\text{all}} W_i / 1000$ kg
   - $\text{SSB}_F = \sum_{\text{adult females}} W_i / 1000$ kg
   - $\text{MMB} = \sum_{\text{adult males}} W_i / 1000$ kg
   - $B_{\text{legal}} = \sum_{\text{males, } size \ge 95} W_i / 1000$ kg
3. **Macro-Ecological Model Fitting**:
   - Discrete-time Schaefer surplus production: $N_{t+1} = N_t + r N_t (1 - N_t/K) - C_t$
   - Deriso-Schnute delay-difference: $N_{t+1} = s (N_t - C_t) + R_{t+1}$
   - Generalized Leslie matrix model: $\mathbf{n}_{t+1} = \mathbf{A}\mathbf{n}_t - \mathbf{C}_t$ ($S = 15$ stages)
   - Retrospective peeling diagnostics & Mohn's $\rho$: evaluates terminal assessment stability against $|\rho| \le 0.20$.
   - Three-way comparison of fishable biomass variations ($CV$, $SD$, $\text{Var}(\Delta B)$, $R^2$, RMSE, AIC).

---

## 4. Software Architecture & Implementation

### 4.1 Julia & Agents.jl Ecosystem
The IBM is implemented in Julia (v1.10+ / v1.13+) utilizing the [`Agents.jl`](https://juliadynamics.github.io/Agents.jl/) framework (Datseris et al. 2022). Julia was selected for its high computational performance, just-in-time (JIT) LLVM compilation, zero-cost type abstractions, native multi-threading, and package integration with `DataFrames.jl`, `Distributions.jl`, and `Plots.jl`.

### 4.2 Computational Benchmarks & Memory Efficiency
- **Memory Footprint**: The `SnowCrab` struct occupies 104 bytes in memory. A population of $100,000$ active individual agents consumes only $\approx 10.4$ MB of RAM.
- **Execution Speed**:
  - **Baseline Configuration (`baseline configuration`)**: A 3-year simulation (1,095 time steps, 300 initial agents on a $100 \times 100$ grid) executes in **~2.9 seconds**.
  - **Fully-Featured Configuration (`fully-featured configuration`)**: The full eco-physiological model executes the same 3-year scenario in **~3.8 seconds**.

---

## 5. Model Assumptions: Baseline vs. Fully-Featured Paradigms

| Feature / Process | Baseline Configurationl (`baseline configuration`, Default) | Fully-Featured Configurationl (`fully-featured configuration`) |
|---|---|---|
| **Ecdysis Eligibility** | Minimum intermolt days ($\Delta t \ge \Delta t_{\min}$) | Degree-days ($\text{DD} \ge \text{DD}_{\text{req}}$) + intermolt + energy + spring window |
| **Growth Increments** | Hiatt linear-log model with phenotypic error | Hiatt model + thermal modulation $\Delta_{\text{temp}}(T)$ + condition |
| **Foraging & Energy** | Unlimited food; bioenergetics bypassed | Holling Type II foraging, $W^{0.75}$ metabolic scaling, starvation |
| **Temperature Effects** | Omitted | Modulates degree-days, ingestion, metabolism, incubation, stress |
| **Carapace Condition** | Omitted (shell wear invariant) | 5 fisheries categories ($CC_1 \dots CC_5$) with wear hazard & mortality |
| **Mating Mechanics** | Direct local pairing; immediate clutch extrusion | Size-assortative competition; 3-clutch spermathecal storage |
| **Brood Incubation** | Fixed $\tau = 365$ days | Temperature-dependent $\tau = 365 (5.0 / T)^{0.30}$ (365–730 days) |
| **Spatial Movement** | Unbiased 2D random walk | Habitat taxis (thermal optimum and forage gradient pursuit) |
| **Natural Mortality** | Adult senescence ($M_0 = 0.20$) + juvenile predation | Multi-driver (senescence, predation, starvation, thermal, shell wear) |

---

## 6. Model Verification & Benchmark Validation

### 6.1 Automated Verification Suite
The simulation codebase is supported by **553 automated unit and integration tests** in [`test/runtests.jl`](file:///c:/home/jae/projects/model_agent/test/runtests.jl):
1. **Empirical Invariants**: Modal bounds ordering ($CW_{\min}(i) < CW_{\max}(i)$) and monotonic progression.
2. **Growth Verification**: Hiatt predicted sizes against empirical survey regression intervals and phenotypic residual variation ($\sigma = 0.045$).
3. **Maturity Ogives**: Verifying that $P(\text{mature} \mid CW_{50}) \approx 0.50$ for males and females, and stochastic Bernoulli terminal molt execution.
4. **Allometric Mass Equations**: Testing $W = \alpha size^\beta$ for all 4 demographic groups against published tables.
5. **Baseline Configurationl Stepping Dynamics**: Verifying pure time/age-based molting, direct annual mating without multi-year spermathecal buffers, random walk dispersal, and no food limitation.
6. **Fully-Featured Configurationl Stepping Dynamics**: Degree-days ($\text{DD}_{\text{req}}$), allometric bioenergetics ($W^{0.75}$), Holling Type II foraging, shell condition wear (Categories 1–5), multi-driver mortality accounting, habitat taxis, and size-assortative mating competition.
7. **Empirical Covariance Matrix Modeling**: Verifying `fit_empirical_cov_model` positive-definiteness, joint sampling fidelity in `predict_from_cov_model`, and mock survey dataset generation.
8. **Stock Biomass Accounting**: Mathematical consistency ($B_{\text{total}} \ge \text{SSB}_{\text{total}} = \text{SSB}_F + \text{MMB}$) and legal commercial male biomass.
9. **Stock-Recruitment Parameter Fitting**: Non-linear estimation of Beverton-Holt and Ricker curves, $R^2$, and AIC calculation.
10. **Macro-Ecological Dynamic Model Estimation**: Process-error regression and trajectory grid search for Schaefer and Delay-Difference models across both whole population and fishable components.
11. **Retrospective Peeling & Mohn's $\rho$ Diagnostic**: Sequential peel execution, degrees of freedom bounds ($P \le T - 3$), and diagnostic threshold classification ($|\rho| \le 0.20$).
12. **Generalized Leslie Stage-Transition Matrix & Spectral Demography**: Non-negative transition matrix estimation $\mathbf{n}_{t+1} = \mathbf{A}\mathbf{n}_t - \mathbf{C}_t$, dominant eigenvalue $\lambda_1 > 0$, growth rate $r = \ln(\lambda_1)$, damping ratio $\rho \ge 1.0$, stable stage distribution ($\sum w_{1,i} = 1.0$), and elasticity matrix ($\sum E_{ij} = 1.0$).
13. **Three-Way Macro Model Comparison**: Evaluating fishable stock biomass trajectories and variability metrics ($CV$, $SD$, $\text{Var}(\Delta B)$), $R^2$, RMSE, and AIC rankings across Schaefer, Delay-Difference, and Leslie matrix models.
14. **Visualizations & Interactive Tools**: All 13 biological diagnostic figures, interactive ABM dashboards, and video animation generation.

---

## 7. Future Prospects & Research Horizons

1. **Coupling with 3D Hydrodynamic Circulation**: Integrating daily 3D velocity and bottom temperature fields from ocean reanalysis models (e.g., NEMO, FVCOM) to simulate advective larval drift and spatio-temporal juvenile recruitment corridors.
2. **Climate Change & Marine Heatwave Scenarios**: Evaluating physiological impacts of Cold Intermediate Layer (CIL) warming ($> 3^\circ\text{C}$), elevated metabolic maintenance, and starvation collapse risks.
3. **Multi-Species Trophic Modeling**: Explicit coupling with groundfish predator fields (Atlantic cod, Atlantic halibut) and endogenized crab-on-crab cannibalism on early benthic stages.
4. **Management Strategy Evaluation (MSE)**: Simulating alternative harvest control rules (HCRs), soft-shell seasonal closures, and marine protected area (MPA) buffer zones using the simple model as an operating model and the complex model for stress-testing.

---

## 8. Planned Manuscript Figures & Tables

### Tables
- **Table 1**: Comparative parameter and feature dictionary contrasting the Simple and Fully-featured model architectures.
- **Table 2**: Empirical modal bounds (`llbm`, `llbf`) and Hiatt growth regression coefficients.
- **Table 3**: Length-mass allometric parameters by sex and maturity status.
- **Table 4**: Carapace condition (Categories 1–5) definitions, transition hazards, and mortality parameters.
- **Table 5**: Summary of simulated fishery exploitation scenarios under Simple and Complex modes.

### Figures
1. **Figure 1**: Conceptual life cycle diagram of *Chionoecetes opilio* and configurable IBM architecture workflow.
2. **Figure 2**: Hiatt post-molt growth increments and empirical logistic maturity ogives.
3. **Figure 3**: Thermal degree-day accumulation curves, skip-molting phenology, and carapace condition wear progression.
4. **Figure 4**: Multi-panel simulation dashboard (population timeseries, stage breakdown, size distribution, spatial habitat map).
5. **Figure 5**: Comparative 10-year population trajectories under Baseline vs. Fully-Featured architectures.
6. **Figure 6**: Stock biomass timeseries ($B_{\text{total}}$, $\text{SSB}_F$, $\text{MMB}$, $B_{\text{legal}}$) under baseline vs. commercial harvest scenarios.
7. **Figure 7**: Spatial habitat density map overlaid on bottom temperature isotherms and habitat taxis trajectories.
8. **Figure 8**: Empirical stock-recruitment relationships with fitted Beverton-Holt and Ricker curves.
9. **Figure 9**: Macro-ecological population dynamics fits (`macro_population_dynamics.png`): Observed abundance contrasted with fitted Schaefer and Delay-Difference models.
10. **Figure 10**: Retrospective pattern analysis and Mohn's $\rho$ diagnostic (`macro_retrospective.png`): 4-panel diagnostic figure displaying peel spaghetti trajectories and $|\rho| \le 0.20$ bounds.
11. **Figure 11**: Stage-specific population dynamics and transition matrix diagnostics (`stage_specific_dynamics.png`): 4-panel figure presenting stage abundance, SSD, reproductive values ($v_1$), and elasticity matrix ($E_{ij}$).
12. **Figure 12**: Three-way macro-ecological model comparison for the commercial fishable stock (`fishable_macro_comparison.png`): Projections, residuals, variability ($CV, \sigma$), and AIC model rankings.
13. **Figure 13**: Interactive ABM exploration dashboard and frame sequence from the simulation video animation.

---

## 9. References & Empirical Literature

1. **Alunno-Bruscia, M., & Sainte-Marie, B.** (1998). Abdomen allometry, ovary development, and growth of female snow crab, *Chionoecetes opilio* (Brachyura, Majidae), in the northwestern Gulf of St. Lawrence. *Canadian Journal of Fisheries and Aquatic Sciences*, 55(2), 459–477. https://doi.org/10.1139/f97-243
2. **Beverton, R. J. H., & Holt, S. J.** (1957). *On the Dynamics of Exploited Fish Populations*. Fishery Investigations Series II, Vol. 19. Ministry of Agriculture, Fisheries and Food, London.
3. **Boudreau, S. A., Anderson, S. C., & Worm, B.** (2011). Top-down interactions and temperature control of snow crab abundance in the northwest Atlantic Ocean. *Marine Ecology Progress Series*, 429, 169–183. https://doi.org/10.3354/meps09081
4. **Caswell, H.** (2001). *Matrix Population Models: Construction, Analysis, and Interpretation*. 2nd Edition. Sinauer Associates, Sunderland, MA.
5. **Choi, J. S., & Zisserson, B. M.** (2012). An assessment of snow crab in the Scotian Shelf (Areas 20 to 24, 4X). *DFO Canadian Science Advisory Secretariat Research Document*, 2012/038.
6. **Comeau, M., Conan, G. Y., Maynou, F., Robichaud, G., Therriault, J. C., & Starr, M.** (1998). Growth, spatial distribution, and abundance of snow crab (*Chionoecetes opilio*) in the Baie des Chaleurs, southwestern Gulf of St. Lawrence, Canada. *Canadian Journal of Fisheries and Aquatic Sciences*, 55(1), 247–260. https://doi.org/10.1139/f97-220
7. **Conan, G. Y., & Comeau, M.** (1986). Functional maturity and terminal molt of male snow crab, *Chionoecetes opilio*. *Canadian Journal of Fisheries and Aquatic Sciences*, 43(9), 1710–1719. https://doi.org/10.1139/f86-214
8. **Datseris, G., Vahdati, A. R., & DuBois, T. C.** (2022). Agents.jl: a performant and feature-full agent-based modeling software of minimal code complexity. *Simulation*, 98(9), 803–813. https://doi.org/10.1177/00375497211068307
9. **Dawe, E. G., Mullowney, D. R., Moriyasu, M., & Wade, E.** (2012). Effects of temperature on size-at-terminal molt and morphometric maturity in snow crab (*Chionoecetes opilio*). *Fisheries Research*, 121–122, 61–70. https://doi.org/10.1016/j.fishres.2012.01.011
10. **Deriso, R. B.** (1980). Harvesting strategies and parameter estimation for an age-structured model. *Canadian Journal of Fisheries and Aquatic Sciences*, 37(2), 268–282. https://doi.org/10.1139/f80-034
11. **DFO (Department of Fisheries and Oceans Canada).** (2020). Assessment of Nova Scotia (4VWX) Snow Crab. *DFO Canadian Science Advisory Secretariat Science Advisory Report*, 2020/027.
12. **DFO (Department of Fisheries and Oceans Canada).** (2022). Assessment of Southern Gulf of St. Lawrence Snow Crab (*Chionoecetes opilio*). *DFO Canadian Science Advisory Secretariat Science Advisory Report*, 2022/019.
13. **Dionne, M., Sainte-Marie, B., Bourget, E., & Gilbert, D.** (2003). Distribution and habitat selection of benthic stages of snow crab, *Chionoecetes opilio*, in relation to water depth and temperature. *Marine Ecology Progress Series*, 259, 117–128. https://doi.org/10.3354/meps259117
14. **Drinkwater, K. F.** (2005). The response of Atlantic cod (*Gadus morhua*) to future climate change. *ICES Journal of Marine Science*, 62(7), 1327–1337. https://doi.org/10.1016/j.icesjms.2005.05.015
15. **Elner, R. W., & Beninger, P. G.** (1995). Multiple reproductive strategies in snow crab, *Chionoecetes opilio*: Physiological paths and behavioral plasticity. *Journal of Experimental Marine Biology and Ecology*, 193(1-2), 93–112. https://doi.org/10.1016/0022-0981(95)00113-5
16. **Ernst, B., Orensanz, J. M., & Armstrong, D. A.** (2005). Spatial dynamics of female snow crab (*Chionoecetes opilio*) in the eastern Bering Sea. *Canadian Journal of Fisheries and Aquatic Sciences*, 62(2), 250–268. https://doi.org/10.1139/f04-201
17. **Fedewa, E. J., Jackson, T. M., Daly, B. J., & Gardner, L.** (2020). Recent fluctuations in snow crab (*Chionoecetes opilio*) size-at-maturity and recruitment in the eastern Bering Sea. *Deep Sea Research Part II: Topical Studies in Oceanography*, 181–182, 104903. https://doi.org/10.1016/j.dsr2.2020.104903
18. **Foyle, T. P., O'Dor, R. K., & Elner, R. W.** (1989). Energetically defining the thermal limits of the snow crab, *Chionoecetes opilio*. *Journal of Experimental Biology*, 145(1), 371–393. https://doi.org/10.1242/jeb.145.1.371
19. **Grimm, V., Berger, U., Bastiansen, F., Eliassen, S., Ginot, V., Giske, J., Goss-Custard, J., Grand, T., Heinz, S. K., Huse, G., Huth, A., Jepsen, J. U., Jørgensen, C., Mooij, W. M., Müller, B., Pe'er, G., Piou, C., Railsback, S. F., Robbins, A. M., Robbins, M. M., Rossmanith, E., Rüger, N., Strand, E., Souissi, S., Stillman, R. A., Vabø, R., Visser, U., & DeAngelis, D. L.** (2006). A standard protocol for describing individual-based and agent-based models (ODD). *Ecological Modelling*, 198(1-2), 115–126. https://doi.org/10.1016/j.ecolmodel.2006.04.023
20. **Grimm, V., Railsback, S. F., Vincenot, C. E., Berger, U., Gallagher, C., DeAngelis, D. L., Edmonds, B., Ge, J., Giske, J., Groeneveld, J., Johnston, A. S., Milles, A., Nabe-Nielsen, J., Polhill, J. G., Radchuk, V., Rohwäder, M. S., Stillman, R. A., Thiele, J. C., & Ayllón, D.** (2020). The ODD protocol for describing agent-based and other simulation models: A second update to improve clarity, replication, and structural realism. *Journal of Artificial Societies and Social Simulation*, 23(2), 7. https://doi.org/10.18564/jasss.4259
21. **Hébert, M., Benhalima, K., Miron, G., & Moriyasu, M.** (2002). Molting and growth of male snow crab, *Chionoecetes opilio* (O. Fabricius, 1788) (Crustacea: Majidae) in the southern Gulf of St. Lawrence. *Crustaceana*, 75(5), 671–702. https://doi.org/10.1163/156854002760202688
22. **Hiatt, R. W.** (1948). The biology of the lined shore crab, *Pachygrapsus crassipes* Randall. *Pacific Science*, 2(3), 135–213.
23. **Hurtado-Ferro, F., Szuwalski, C. S., Valero, J. L., Anderson, S. C., Cunningham, C. J., Johnson, K. F., Licandeo, R., McGilliard, C. R., Monnahan, C. C., Rudd, M. B., Stawitz, C. C., Hicks, A. C., Sansó, B., & Punt, A. E.** (2015). Looking in the rear-view mirror: bias and retrospective patterns in integrated, age-structured stock assessment models. *ICES Journal of Marine Science*, 72(1), 99–110. https://doi.org/10.1093/icesjms/fsu198
24. **Kooijman, S. A. L. M.** (2010). *Dynamic Energy Budget Theory for Metabolic Organisation*. 3rd Edition. Cambridge University Press, Cambridge.
25. **Kuhn, S., & Choi, J. S.** (2011). Influence of temperature on embryo development and hatching phenology in the snow crab, *Chionoecetes opilio*. *Fisheries Oceanography*, 20(6), 461–472. https://doi.org/10.1111/j.1365-2419.2011.00597.x
26. **Lefkovitch, L. P.** (1965). The study of population growth in organisms grouped by stages. *Biometrics*, 21(1), 1–18. https://doi.org/10.2307/2528348
27. **Leslie, P. H.** (1945). On the use of matrices in certain population mathematics. *Biometrika*, 33(3), 183–212. https://doi.org/10.1093/biomet/33.3.183
28. **Marcello, L. A., Fairweather, P. G., & Choi, J. S.** (2012). Assessing the stability and resilience of snow crab (*Chionoecetes opilio*) ecosystem dynamics in the Northwest Atlantic. *Ecological Applications*, 22(8), 2110–2126. https://doi.org/10.1890/11-1934.1
29. **Mohn, R.** (1999). The retrospective problem in stock assessment: An amphidrome of check and balance. *ICES Journal of Marine Science*, 56(4), 473–488. https://doi.org/10.1006/jmsc.1999.0491
30. **Moriyasu, M., Wade, E., Sinclair, A., & Chiasson, Y.** (1998). Snow crab, *Chionoecetes opilio*, stock assessment in the southwestern Gulf of St. Lawrence by bottom trawl survey. *Canadian Technical Report of Fisheries and Aquatic Sciences*, 2240.
31. **Railsback, S. F., & Grimm, V.** (2019). *Agent-Based and Individual-Based Modeling: A Practical Introduction*. 2nd Edition. Princeton University Press, Princeton, NJ.
32. **Rondeau, A., & Sainte-Marie, B.** (2001). Variable mate-guarding time and sperm allocation by male snow crabs (*Chionoecetes opilio*) in response to competition. *Canadian Journal of Fisheries and Aquatic Sciences*, 58(6), 1243–1256. https://doi.org/10.1139/f01-073
33. **Sainte-Marie, B.** (1993). Reproductive cycle and fecundity of primiparous and multiparous female snow crab, *Chionoecetes opilio*, in the northwest Gulf of St. Lawrence. *Canadian Journal of Fisheries and Aquatic Sciences*, 50(10), 2147–2156. https://doi.org/10.1139/f93-240
34. **Sainte-Marie, B., Raymond, S., & Brêthes, J. C.** (1995). Growth and maturation of the benthic stages of male snow crab, *Chionoecetes opilio* (Brachyura: Majidae). *Canadian Journal of Fisheries and Aquatic Sciences*, 52(5), 903–924. https://doi.org/10.1139/f95-091
35. **Sainte-Marie, B., Sevigny, J. M., & Gignac, M.** (1997). Assessment of female snow crab (*Chionoecetes opilio*) sperm stores, and courtship behavior in the wild. *Canadian Journal of Fisheries and Aquatic Sciences*, 54(2), 239–248. https://doi.org/10.1139/f96-278
36. **Sainte-Marie, B., Sevigny, J. M., & Carpentier, M.** (2002). Interannual variability of sperm reserves and fecundity of primiparous females of the snow crab (*Chionoecetes opilio*) in relation to sex ratio. *Canadian Journal of Fisheries and Aquatic Sciences*, 59(12), 1932–1940. https://doi.org/10.1139/f02-157
37. **Sainte-Marie, G., & Sainte-Marie, B.** (1999). Growth, developmental schedule, and terminal molt of the snow crab (*Chionoecetes opilio*) in the western Gulf of St. Lawrence. *Canadian Journal of Fisheries and Aquatic Sciences*, 56(6), 940–953. https://doi.org/10.1139/f99-019
38. **Schaefer, M. B.** (1954). Some aspects of the dynamics of populations important to the management of the commercial marine fisheries. *Inter-American Tropical Tuna Commission Bulletin*, 1(2), 27–56.
39. **Schnute, J.** (1985). A general theory for analysis of catch and effort data. *Canadian Journal of Fisheries and Aquatic Sciences*, 42(3), 414–429. https://doi.org/10.1139/f85-056
40. **Skanes, K. R., Mullowney, D. R., & Dawe, E. G.** (2017). An assessment of Newfoundland and Labrador snow crab (*Chionoecetes opilio*) in 2015. *DFO Canadian Science Advisory Secretariat Research Document*, 2017/025.
41. **Szuwalski, C. S., Cheng, W., Foy, R., Hermann, A. J., Hollowed, A. B., Holsman, K. K., Lee, J., Zheng, J., & Ianelli, J.** (2021). Climate change and the future of snow crab in the Bering Sea. *Progress in Oceanography*, 194, 102555. https://doi.org/10.1016/j.pocean.2021.102555
42. **Szuwalski, C. S., Aydin, K., Fedewa, E. J., Halloran, E., & Litzow, M. A.** (2023). The collapse of eastern Bering Sea snow crab. *Science*, 382(6668), 306–310. https://doi.org/10.1126/science.adf6035
43. **Thompson, R. J., & Hawryluk, M.** (1990). Physiological energetics of the snow crab *Chionoecetes opilio*. In *Proceedings of the International Symposium on King and Snow Crabs*, pp. 283–291. Alaska Sea Grant College Program, Fairbanks, AK.
44. **Tremblay, M. J.** (1997). Snow crab (*Chionoecetes opilio*) distribution and abundance in relation to temperature and bathymetry on the Scotian Shelf. *Canadian Technical Report of Fisheries and Aquatic Sciences*, 2161.
45. **Visser, A. W.** (2008). Biomass and functional diversity in individual-based models of marine ecosystems. *Journal of Marine Systems*, 74(1-2), 488–499. https://doi.org/10.1016/j.jmarsys.2008.03.012
46. **Webb, J. B., Eckert, G. L., Shirley, T. C., & Tamone, S. L.** (2007). Changes in embryonic development and hatching in snow crab, *Chionoecetes opilio*, under temperature treatment. *Journal of Crustacean Biology*, 27(1), 107–115. https://doi.org/10.1651/S-2738.1
