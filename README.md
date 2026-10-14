# Individual and Agent Based Models (GeoIBM)

A species-parameterised individual-based model engine built on
[Agents.jl](https://juliadynamics.github.io/Agents.jl/stable/). The engine is
generic: species biology lives entirely in configuration, dispatched through the
`SpeciesBehavior` interface, so the execution loop is shared. Snow crab
(*Chionoecetes bairdi*) is the reference species and is calibrated in
`configs/snowcrab.toml`.

## Quick start

```powershell
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. test/runtests.jl
julia --project=. scripts/organism_ibm.jl --config=configs/default.toml --days=365
julia --project=. scripts/organism_ibm.jl --config=configs/snowcrab.toml --save
```

## Species dispatch, and two reporting invariants

`SpeciesBehavior` is the extension point for a second species, and the generic
layer holds three methods that used to defeat it. `stage_size_lower_bound`,
`growth_increment` and `build_stages` each built a `SpeciesBehaviorImpl()` and
called *that*, discarding the behaviour they were handed. With one species the
mistake is invisible, because every behaviour **is** a `SpeciesBehaviorImpl` and
dispatch already selects the right method — and an altered `SpeciesParams` does
give a different answer, which is what makes it easy to miss. The cost appears
only when someone writes a second species: it would be measured against snow crab
growth coefficients and the stage-size curve, and produce plausible, wrong animals.
All three now fail at the call site with the offending type named.

Two reporting invariants are worth stating because the data does not show them:

- **`catch_total` and `catch_legal` are the same series.** Not a shortcut: the
  fishery gate removes an animal only when `is_legal_male` passes, so no sublegal
  animal can be harvested. They are reported separately because stock-assessment
  convention separates them, and a caller assuming they were independent would
  double-count the removals. A test asserts the identity, so a configuration that
  admits sublegal harvest has to compute the two columns separately.
- **The fishery is off by default, and one number says so.**
  `fishery.fishing_probability_daily` is `0.0`, and that is the entire switch —
  `fishery_active(params)` is the predicate, and there is deliberately no
  `toggles.use_fishery` alongside it. The two used to express one thing, so the
  model could report the fishery as enabled while no animal could ever be caught.
  Set the probability to the rate your scenario implies.


## Architecture

Two pillars:

1. **Generic simulation core** — `GridSpace` in time, `Agents.jl` for the agent
   loop, aggregate cohorts for the pelagic phase, demographic caches for the
   reporting layer.
2. **Species parameterisation** — every biological rate, threshold, schedule and
   allometric coefficient is a field on one of the twelve parameter structs.
   There are no species constants in the code; `configs/default.toml` documents
   each one, and `configs/snowcrab.toml` is the species overlay.

```
GeoIBM/
├── Project.toml
├── configs/
│   ├── default.toml          # every declared parameter, at its struct default
│   └── snowcrab.toml         # snow crab overlay: biology plus operating scale
├── docs/
│   ├── GeoIBM.md        # detailed reference
│   └── snowcrab_ibm_research_paper.md
├── src/
│   ├── GeoIBM.jl        # module, includes, exports
│   ├── types.jl              # Organism, the twelve parameter structs, config plumbing
│   ├── species.jl            # the SpeciesBehavior implementation and stage tables
│   ├── organism_dynamics.jl  # allometry, bioenergetics, molting, mortality, cohorts
│   ├── simulation.jl         # grid construction, model_step!/agent_step!, config loading
│   ├── analysis.jl           # summaries, stock-recruitment, macro models, criteria
│   ├── cli.jl                # argument parsing, run reports, orchestration
│   ├── visualization.jl      # CairoMakie figures and animations
│   └── gui.jl                # GLMakie interactive dashboard
├── notebooks/
│   └── workflow.jl            # Pluto: assimilate the store, configure, run, analyse
├── scripts/
│   ├── organism_ibm.jl              # simulation CLI
│   ├── organism_orchestrator.jl     # calibration driver
│   └── fit_macro_models.jl          # post-hoc macro model fitting
└── test/runtests.jl
```

## The environmental store

An IBM needs a world: a sea floor and a thermal cycle. Those come from published products —
ETOPO 2022 bathymetry and WOA23 (or GLORYS12V1) hydrography — and they are assimilated once
into a GeoData store beside the other halves':

| artefact | product | where |
|---|---|---|
| `bathymetry.zarr` | ETOPO 2022, 15 arc-second, over ERDDAP | `../lakehouse/GeoIBM/` |
| `woa23_temperature_<mm>_0.25deg.zarr` | WOA23, keyless | `../lakehouse/GeoIBM/` |
| `woa23_salinity_<mm>_0.25deg.zarr` | WOA23, keyless | `../lakehouse/GeoIBM/` |

```julia
status  = GeoIBM.environmental_input_status(env)        # what is already there
result  = GeoIBM.assimilate_environmental_inputs(env)   # fetch what is missing
fields  = GeoIBM.read_assimilated_environment(env)      # what a run reads
```

It is idempotent: what is already assimilated is skipped, so a second run does no downloading.
`notebooks/workflow.jl` starts with this step and explains what each assumption costs.

| File | Responsibility |
|---|---|
| `types.jl` | `Organism` agent, `IBMParameters` and its twelve sections, TOML load/save, field-name validation |
| `species.jl` | Growth increments, maturity ogives, fecundity, molting, reproduction, the Leslie stage table |
| `organism_dynamics.jl` | Bioenergetics, mortality drivers, movement, the aggregate-cohort lifecycle |
| `simulation.jl` | `initialize_model`, `model_step!`, `agent_step!`, environment fields, `run_simulation` |
| `analysis.jl` | `summarize_population`, `evaluate_population_criteria`, stock-recruitment, Schaefer / delay-difference / generalized Leslie |

## Life history (crustacean example)

```
  [ Adult Females ] --mating--> [ Spermatheca (sperm stores) ]
         |                              |
         v                              v
  [ Brood Incubation ] <---------- [ Egg Production ]
    (~365 days external)              (empirical fecundity)
         |
         v
  [ Pelagic: egg -> zoea -> megalopa ]      aggregate cohorts
         |
         v  (individuation at juvenile_max_stage)
  [ Benthic recruits ] --> [ Developmental stages ] --> [ Maturity ]
                                                            |
                                        +-------------------+-------------------+
                                        v                                       v
                               [ Adult Females ]                      [ Adult Males ]
```

Feature toggles live on `IBMParameters.toggles`: `use_energetics`,
`use_molting`, `use_mortality`, `use_reproduction`,
`use_spermatheca`, `use_degree_days`, `use_thermal_stress`,
`use_food_dynamics`, `use_habitat_taxis`, `use_senescence`, `use_cannibalism`,
`use_movement`, `use_directed_movement`, `use_immigration`.

## Mathematical reference

**Growth.** Post-molt size is a log-linear function of pre-molt size with
maturation, condition and temperature terms:

```
ln(size_next) = intercept + slope * ln(size) + δ_maturity + c·condition
                + γ·(T - T_ref) + ε
```

`slope` is slightly above 1, so the size increment compounds with each ecdysis.
Post-molt size is therefore capped at the species' asymptotic width
(`species.max_size_male`, `species.max_size_female`), which is also the
saturating limit of the stage-size curve — a limit the growth rule cannot reach
without a ceiling.

**Stage-size curve.** The lower bound on carapace width per stage saturates at
the asymptotic width:

```
ln S(stage) = ln L + (intercept - ln L) · exp(-decay · (stage - anchor))
```

which equals `exp(intercept)` at the anchor stage and tends to `L`. The
parameters are fitted to the lower edge of each tabulated stage
(`species.stage_size_range_lower_*`).

**Maturity.** Logistic in log-size, with a temperature-dependent $L_{50}$:

```
P(mature | size, sex) = logistic(α + β·ln(size) + γ·(T - T_ref))
```

**Mortality.** Multiplicative drivers over a shared annual baseline converted to
a daily hazard, with a density term `1 + (N/K)^exponent`:

```
m = m_base + m_predation(size) + m_soft_shell(Δt) + m_starvation(E)
    + m_thermal(T) + m_senescence(age) + m_cannibalism(local density)
```

**Carrying capacity.** `N` is the total benthic stock, individuated plus
aggregate cohorts, and `K` (`environment.benthic_carrying_capacity`) must exceed
the seeded stock. At the reference scale the seeded stock is 6500 against
K = 10000, giving a multiplier of 1.42. `initialize_model` warns when the stock
already exceeds K, because that is a silent failure: at K = 300 against the same
seed the multiplier is 2050 and every hazard is a certainty on day one.

**Fecundity.** Two allometric branches, primiparous and multiparous, selected by
shell age (`species.primiparous_shell_age_max`). Parity is a property of
reproductive history, not of how much sperm a female is holding.

**Ecdysis.** Stages below `length(intermolt_days_larval)` develop on the pelagic
schedules; everything above develops on the benthic schedules. Both are stage
indexed and out-of-range stages clamp to the terminal entry.

## Operating scale and performance

`configs/snowcrab.toml` runs a **1000 × 1000 grid**, which at one cell per metre
is a 1 km × 1 km survey stratum. Three `Float64` fields of that size cost 23 MB,
and they are the only per-cell state.

The measured cost is about **21 ms per simulated day**, so a two-year run is
roughly **15 s** of dynamics plus whatever the observation columns add. What that
time is spent on, and what has been done about it:

| Cost | Note |
|---|---|
| Temperature field | Drawn in place with `randn!` and scaled by broadcast. Previously `rand(rng, Normal(0, sd), dims)` allocated a second 7.6 MB matrix every day — 5.4 GB over two years. Now 4× faster with no allocation. |
| Food field | Written as the affine map it is, `F = (1 - r·dt)F + r·f_max·dt`, one multiply-add per cell instead of a subtract, two multiplies and an add. |
| Per-organism steps | `model.params` is hoisted out of `agent_step!`. It is a symbol lookup into the properties dictionary on every access, and the function read it a dozen times per organism per day. |
| Neighbourhood queries | The cannibalism density query is gated on `use_cannibalism`. It was run for every organism on every day regardless. |
| Observation columns | `run_simulation` collects `adata` and `mdata` every step unless `when` is given. Each column is a pass over the population, so pass an interval for long runs. |

Two things worth knowing before scaling further:

* **A sparse grid is cheap.** At 1000 × 1000 with a few thousand organisms the
  cells are mostly empty, so `nearby_agents` returns almost nothing and the
  per-organism cost is low. The same organisms on a 30 × 30 grid were three
  times more expensive per organism, because each neighbourhood query returned
  hundreds of occupants. Grid size does not track population cost the way you
  would expect.
* **Pin the activation order if you store runs.** `initialize_model` now defaults
  to `Agents.Schedulers.Randomly()`, which draws the activation order from the
  model's RNG, so a fixed seed fixes the order by construction. The Agents.jl
  default, `Schedulers.fastest`, takes the agent container's own id order, which
  for a `Dict` container is whatever the hash table holds and is not derived from
  the RNG. That matters because organisms interact — mortality, mate search and
  cannibalism all read neighbours — so the activation order is part of the state
  transition. Measured cost of the change: none (19.3 s vs 18.8 s over 120 steps
  at 1500 organisms on the 1000 × 1000 grid, and two same-seed `Randomly` runs
  agreed on every observation column).

## Configuration

Three-tier precedence, later wins:

1. struct defaults on the twelve parameter structs;
2. `configs/default.toml`, which holds every declared parameter at its default
   and is checked against the structs in both directions by the test suite;
3. `configs/snowcrab.toml`, the species overlay; then CLI flags on top.

```powershell
julia --project=. scripts/organism_ibm.jl --temp_base=4.5 --days=90
```

A configuration key that is not a declared parameter is reported as a warning and
ignored rather than silently dropped, and the test suite fails if either shipped
file contains one. Adding a field to a parameter struct is all that is required
to make it configurable: the section map, the round-trip loader and the
serializer are all derived from `fieldnames`.

## CLI

Flag names are the TOML key names verbatim; there are no abbreviations, and an
unrecognized name is reported and ignored rather than guessed at. `--config`
selects the *overlay*, not the baseline: `default.toml` is always loaded first,
and the conventional `snowcrab.toml` sibling is overlaid on it.

| Option | Default | Description |
|---|---|---|
| `--config=<path>` | — | Overlay TOML; if it names the baseline, its `snowcrab.toml` sibling is used instead |
| `--days=<Int>` | `730` | Duration in days |
| `--n_agents=<Int>` | `300` | Seeded individuals |
| `--grid=<nx,ny>` | `100,100` | Grid dimensions |
| `--grid_nx`, `--grid_ny` | `100` | Grid dimensions, individually |
| `--seed=<Int>` | `1234` | RNG seed |
| `--temp_base=<Float>` | `5.0` | Mean bottom-water temperature (°C) |
| `--benthic_carrying_capacity=<Float>` | `12000.0` | Total benthic capacity, whole model |
| `--fishing_probability_daily=<Float>` | `0.0` | **The fishery switch.** Zero means no harvest |
| `--fishing_probability_daily=<Float>` | `0.0` | **The fishery switch.** Zero means no harvest |
| `--frequency_days=<Float>` | `365.0` | Output sampling interval |
| `--snapshot_day=<Float>` | `274.0` | Annual survey snapshot day of year |
| `--output_dir=<Path>` | `outputs` | Results directory |
| `--save=<Bool>` | `true` | Write annual indicators to CSV |
| `--plot=<Bool>` | `false` | Emit diagnostic figures |
| `--animate=<Bool>` | `false` | Record an animation |
| `--gui=<Bool>` | `false` | Launch the interactive dashboard |

Any parameter declared in a TOML section is settable, for example
`--maturity_l50_male=88.0`, `--rate_predation=0.1`, `--use_energetics=false`,
or a whole schedule as `--intermolt_days_schedule=60,75,90`.

## API

```julia
using GeoIBM

params = IBMParameters(
    fishery = FisheryParams(fishing_probability_daily = 0.005),
    environment = EnvironmentParams(temp_base = 4.5),
)

model = initialize_model(;
    n_agents = 2000,
    grid = (1000, 1000),
    seed = 42,
    params = params,
)

agent_df, model_df = run_simulation(model, 730; when = 7)
```

## Tests

```powershell
julia --project=. test/runtests.jl
```

The suite covers the empirical growth and maturity rules, the mortality
decomposition, the configuration round trip, the cohort lifecycle and
individuation boundary, the macro-model fits, and the reporting schemas. Two of
its checks are guards rather than behaviour tests and are worth knowing about:

* every declared parameter must have a use site in `src/` or `scripts/`, found by
  scanning each file separately — a concatenated subject gives order-dependent
  answers, and stripping declarations by rejoining their lines silently fails on
  CRLF files, which made this guard vacuous;
* an animal can never exceed its sex's asymptotic size, and the stage-size curve
  can never produce one.