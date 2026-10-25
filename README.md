# Individual and Agent Based Models (TethysBio)

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
TethysBio/
├── Project.toml
├── configs/
│   ├── default.toml          # every declared parameter, at its struct default
│   └── snowcrab.toml         # snow crab overlay: biology plus operating scale
├── docs/
│   ├── TethysBio.md             # detailed reference
│   └── deep_check_2026-10.md # the October documentation audit and its disposition
├── src/
│   ├── TethysBio.jl             # module, includes, exports
│   ├── types.jl              # Organism, the twelve parameter structs, config plumbing
│   ├── species.jl            # the SpeciesBehavior implementation and stage tables
│   ├── organism_dynamics.jl  # allometry, bioenergetics, molting, mortality, cohorts
│   ├── simulation.jl         # grid construction, model_step!/agent_step!, config loading
│   ├── analysis.jl           # summaries, stock-recruitment, macro models, criteria
│   ├── cli.jl                # argument parsing, run reports, orchestration
│   ├── visualization.jl      # CairoMakie figures and animations
│   ├── gui.jl                # GLMakie interactive dashboard
│   ├── assimilation.jl       # Tethys store assimilation and input status
│   └── notebook_support.jl   # dashboard helpers shared with the Pluto notebook
├── notebooks/
│   └── TethysBioNotes.jl        # Pluto: configure, run, visualise, analyse, calibrate
├── scripts/
│   ├── organism_ibm.jl             # simulation CLI
│   ├── organism_orchestrator.jl    # calibration driver
│   └── fit_macro_models.jl         # post-hoc macro model fitting
└── test/
    ├── runtests.jl                 # segment dispatcher (biology, macro, config,
    │                               #  simulation, vis, habitat)
    ├── test_allometry_biology.jl   # growth, maturity, mortality, fecundity
    ├── test_config_parameters.jl   # TOML round trip and parameter coverage
    ├── test_fixtures.jl            # shared test helpers
    ├── test_habitat_environment.jl # Tethys store and environmental fields
    ├── test_macro_dynamics.jl      # macro-model fits and reporting
    ├── test_simulation_agents.jl   # cohort lifecycle, seeding, individuation, step loop
    └── test_visualization_gui.jl   # figures and dashboard internals
```

## The environmental store

An IBM needs a world: a sea floor and a thermal cycle. Those come from published products —
ETOPO 2022 bathymetry and WOA23 (or GLORYS12V1) hydrography — and they are assimilated once
into a Tethys store beside the other halves':

| artefact | product | where |
|---|---|---|
| `bathymetry.zarr` | ETOPO 2022, 15 arc-second, over ERDDAP | `lakehouse_dir("TethysBio")` |
| `woa23_temperature_<mm>_0.25deg.zarr` | WOA23, keyless | `lakehouse_dir("TethysBio")` |
| `woa23_salinity_<mm>_0.25deg.zarr` | WOA23, keyless | `lakehouse_dir("TethysBio")` |

### Lakehouse Location Configuration

`TethysBio` resolves the shared data lakehouse via `Tethys.lakehouse_dir("TethysBio")`.
The lakehouse root is determined by the following precedence:
* `ENV["TethysBio_STORE_ROOT"]` or `ENV["GEODATA_LAKEHOUSE"]`.
* `LocalPreferences.toml` (`[Tethys] lakehouse_dir`).
* `~/.geodata/storage.toml` (`[storage] store_root`).
* Proximity fallback (`projects/lakehouse/TethysBio`).

To configure the lakehouse persistently for your environment:
```julia
using Tethys
Tethys.set_lakehouse_dir!("/path/to/shared/lakehouse")
```

```julia
status  = TethysBio.environmental_input_status(env)        # what is already there
result  = TethysBio.assimilate_environmental_inputs(env)   # fetch what is missing
fields  = TethysBio.read_assimilated_environment(env)      # what a run reads
```

It is idempotent: what is already assimilated is skipped, so a second run does no downloading.
`notebooks/TethysBioNotes.jl` starts with this step and explains what each assumption costs.

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

Feature toggles live on `IBMParameters.toggles`, and these are the ten that
exist: `use_energetics`, `use_thermal_stress`, `use_cannibalism`,
`use_senescence`, `use_spermatheca`, `use_movement`, `use_directed_movement`,
`use_degree_days`, `use_molting`, `use_reproduction`. The fishery is
deliberately not among them: harvest is switched by
`fishery.fishing_probability_daily` (zero means off), so the toggle set cannot
disagree with the catch statistics about whether a fishery is running.

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

**Carrying capacity.** `N` is the total benthic stock — individuated agents plus
aggregate cohorts not yet individuated — and `K`
(`environment.benthic_carrying_capacity`) must exceed it, because the density
term `1 + (N/K)^exponent` multiplies every mortality hazard. The shipped cohort
vectors seed one aggregate cohort of 2000 members at the first cohort stage and
no individuated agents (both sex vectors are zero), so a run starts at
N = 2000 — 1.04 against the snow crab `K` = 10000, with exponent 2.0 — and the
multiplier rises as the population grows. `initialize_model` warns when the
stock already exceeds `K`, because that is a silent failure: over-seed against a
small capacity (`n_agents = 4800`, `K = 300`) and the multiplier is 257, so
every hazard is a certainty on day one.

**Fecundity.** Two allometric branches, primiparous and multiparous, selected by
shell age (`species.primiparous_shell_age_max`). Parity is a property of
reproductive history, not of how much sperm a female is holding.

**Ecdysis.** Stages below `length(intermolt_days_larval)` develop on the pelagic
schedules; everything above develops on the benthic schedules. Both are stage
indexed and out-of-range stages clamp to the terminal entry.

## Operating scale and performance

`configs/snowcrab.toml` runs a **1000 × 1000 grid**, which at one cell per metre
is a 1 km × 1 km survey stratum. The shipped seeding holds one 2000-member
aggregate clutch at stage 1 and an empty individuated stock: day one runs
exactly 2000 benthic organisms, and the density multiplier is
$1 + (2000/10000)^2 = 1.04$.

Measured cost (benchmarked 2026-10-23; median of three replicates, 1500
organisms at the seeded distribution, warm-up excluded; script and replicate log
in `scratch/bench_perf.jl` / `scratch/bench_perf.log`):

| Grid (cells) | Dynamics only (ms/day) | With the CLI's default observation set (ms/day) |
|---|---|---|
| 1000 × 1000 | 40.3 | 49.9 |
| 200 × 200 | 30.7 | 29.3 |
| 100 × 100 | 36.5 | 28.5 |
| 30 × 30 | 22.6 | 20.9 |

Replicate spread is about ±25%, so read the table in halves, not decimals. An
end-to-end timing of the shipped snowcrab configuration measured 32.6 ms of
bare dynamics per simulated day: a two-year, 730-day run is roughly **24 s** of
dynamics, or roughly **36 s** with the CLI's default per-step observation
columns. What that time is spent on, and what has been done about it:

| Cost | Note |
|---|---|
| Temperature field | Drawn in place with `randn!` and scaled by broadcast. Previously `rand(rng, Normal(0, sd), dims)` allocated a second 7.6 MB matrix every day — 5.4 GB over two years. Now 4× faster with no allocation. |
| Food field | Written as the affine map it is, `F = (1 - r·dt)F + r·f_max·dt`, one multiply-add per cell instead of a subtract, two multiplies and an add. |
| Per-organism steps | `model.params` is hoisted out of `agent_step!`. It is a symbol lookup into the properties dictionary on every access, and the function read it a dozen times per organism per day. |
| Neighbourhood queries | The cannibalism density query is gated on `use_cannibalism`. It was run for every organism on every day regardless. |
| Observation columns | `run_simulation` collects `adata` and `mdata` every step unless `when` is given. Each column is a pass over the population, so pass an interval for long runs. |

Two things worth knowing before scaling further:

* **Neither grid extreme dominates.** At 1500 organisms the sweep moved both
  ways: the cheapest grid measured was 30 × 30 (22.6 ms/day bare) and the most
  expensive was 1000 × 1000 (40.3 ms/day). Per-cell environmental field work
  scales with grid area, while per-organism bookkeeping and neighbourhood
  queries grow with local occupancy, which is highest on coarse grids; the two
  drivers pull against each other, and the ±25% replicate noise brackets their
  ordering.
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
julia --project=. scripts/organism_ibm.jl --temp_offset=4.5 --days=90
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
| `--n_agents=<Int>` | sex-vector cohort sum | Seeded individuated agents. The default is `sum(initial_female_cohort) + sum(initial_male_cohort)` — `0` on the shipped baselines, whose life starts from the aggregate cohort instead. The aggregate cohort is seeded independently of this flag either way |
| `--grid=<nx,ny>` | `100,100` | Grid dimensions |
| `--grid_nx`, `--grid_ny` | `100` | Grid dimensions, individually |
| `--seed=<Int>` | `1234` | RNG seed |
| `--temp_offset=<Float>` | `0.0` | Baseline bottom-water temperature offset (°C) |
| `--temp_multiplier=<Float>` | `1.0` | Seasonal cycle amplitude multiplier |
| `--temp_base`, `--temp_amplitude` | — | Legacy aliases for `--temp_offset` / `--temp_multiplier` |
| `--benthic_carrying_capacity=<Float>` | `12000.0` | Total benthic capacity, whole model |
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
using TethysBio

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

The suite runs in six selectable segments (pass one or more segment names to run
those alone):

```powershell
julia --project=. test/runtests.jl              # all segments
julia --project=. test/runtests.jl biology      # allometry, growth, empirical biology
julia --project=. test/runtests.jl habitat      # habitat suitability, Tethys integration
julia --project=. test/runtests.jl macro        # macro models and Leslie matrices
julia --project=. test/runtests.jl config       # configuration, CLI parsing, parameters
julia --project=. test/runtests.jl simulation   # agents, energetics, mortality, dispersal
julia --project=. test/runtests.jl vis          # plots and GUI dashboard
```

Pass counts as of 2026-10-23 (`vis` below):

| Segment | Checks passed |
|---|---|
| `biology` | 140 |
| `habitat` | 165 |
| `macro` | 203 |
| `config` | 1492 |
| `simulation` | 481 |

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

The `vis` segment currently fails under Julia 1.13 in a headless environment
with a GLMakie autolimits error while the GUI dashboard renders. This is a
pre-existing environment interaction, not a regression in the plotting code or
the model; it is left as-is rather than masked, and the other five segments are
unaffected.