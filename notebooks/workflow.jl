### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# ╔═╡ 8a1c4e72-3b65-4d09-9f7a-2c6e0b4d1a38
# ╠═╡
using Pkg

# ╔═╡ 2f7b0d93-5c18-4a26-8b3e-7d1f9c4a6e02
# ╠═╡
md"""
# IABM — the individual-based model, and its environmental store

An individual- and agent-based simulation needs a world before it can run anything: a sea floor,
a bottom temperature, and a seasonal cycle of both. This notebook walks the whole model, and it
starts where a run starts — with the *environment*.

`IABM` simulates a population organism by organism: growth by moulting,
bioenergetics, sex-specific maturity, reproduction, mortality from named causes, directed
movement, and a fishery with a season. The biological rules are all configurable; what is *not*
configurable in any interesting sense is the world they happen in, and that is what this
notebook documents first.

### The notebook's order

1. **Assimilate the environmental inputs** — convert the published products into a GeoData
   store at `projects/lakehouse/IABM/`, skipping what is already there.
2. **Load the configuration** and see what it fixes, and why.
3. **Run the IBM** on the assimilated world, and inspect what it produced.
4. **Analyse and plot** — stage distributions, maturity ogives, stock and recruitment.
5. **The assumptions and references** — one list of every choice this model bakes in.

Each step is an ordinary package call, so a cell can be re-run on its own while you inspect what
it returned. That is what Pluto buys you over a script.
"""

# ╔═╡ 5c3e8a26-0b79-4f1d-8a64-3e2b7d0a5c38
# ╠═╡
md"""
## Running this notebook

Launch Pluto from an environment that already has it (the global environment, or one where you
have added `Pluto`) and open this file from the notebook list:

```
julia -e 'using Pluto; Pluto.run()'
```

The first three cells pin the working directory, activate **this package's environment**, and
load it, so `configs/snowcrab.toml` and the `../lakehouse/IABM` store paths resolve relative to
the package rather than to wherever the notebook was opened from.

`Pluto` is deliberately **not** a dependency of this package: a notebook is how you look at the
work, not something the run needs.
"""

# ╔═╡ 9d4f6b13-2a08-4c5e-8b7f-6a3d1e0c9b47
# ╠═╡
cd(joinpath(@__DIR__, ".."))

# ╔═╡ 3e8c1a54-7f26-4b0d-9e58-2a5f8c3b1d69
# ╠═╡
Pkg.activate(@__DIR__)

# ╔═╡ 7b5d2c98-1a34-4e7b-9c0f-8d6b4a2e0f15
# ╠═╡
using IABM

# ╔═╡ c1a9e642-8b07-4f3d-9e16-5c0b7a3d8e42
# ╠═╡
md"""
## Step 0 — assimilate the environmental inputs

An IBM is a *biological* model with a *physical* setting, and the setting comes from published
products. There are only two, and they are the only external inputs this package has:

| input | product | what it becomes |
|---|---|---|
| the sea floor | **ETOPO 2022** bathymetry, 15 arc-second, over ERDDAP | `bathymetry.zarr`, sampled to the model's own grid |
| the water column | **WOA23** (or GLORYS12V1) climatological hydrography | temperature and salinity, twelve monthly fields |

Everything else — the growth curve, the maturity ogive, the fecundity, the fishing season — is
either in the configuration or a parameter this model estimates. So the assimilation step has
exactly two things in it, and both are real downloaded data rather than a synthetic stand-in:
if the download fails, the run stops and says so, rather than quietly running on a different
sea floor.

### Where it goes

`projects/lakehouse/IABM/`, beside the other halves' stores:

```
lakehouse/
  GeoData/          shared physical products
  GeoOceanModel/    the ocean half's runs
  GeoLarvalTracking/   the larval half's runs
  IABM/             this package's environment
    bathymetry.zarr
    woa23_temperature_01_0.25deg.zarr     one per month
    woa23_salinity_01_0.25deg.zarr
    manifest.toml
```

The store is shared by every run of this configuration, which is what makes a second run cheap:
the WOA23 temperature field for a month is ~500 MB, and re-fetching it for each run that wanted
a different number of simulated days would be the run that made people abandon the cache.

### It is optional

Assimilation is idempotent. The store knows when to stop: a product already assimilated is
skipped, and a second call with no changes does no work at all. That is the whole point of the
manifest — a run must not silently rebuild a 6 GB store, and a changed input must not be
silently reused.
"""

# ╔═╡ 6f2a7d84-3b19-4e5c-8a07-9d4e0f6b2c58
# ╠═╡
md"""
### Asking before fetching

The cell below resolves what this configuration will read and reports what is already
assimilated. It does no downloading, so it is safe to run first: on a fresh install it shows
both inputs absent and tells you what is about to be fetched; on a machine that has run this
before, it shows both present and the next cell does nothing.
"""

# ╔═╡ 8c3b0e15-7d42-4f9e-8b26-2a5f1c9d0e73
# ╠═╡
configuration = IABM.load_ibm_configuration("configs/snowcrab.toml")

# ╔═╡ 4d6e9f28-0a58-4b1c-9e37-8f3b2c7e1a04
# ╠═╡
env = IABM.EnvironmentParams(;
    domain_lon        = Float64.(configuration[:domain_lon]),
    domain_lat        = Float64.(configuration[:domain_lat]),
    time_range        = configuration[:time_range],
    pelagic_depth     = configuration[:pelagic_depth],
    temperature_source = configuration[:temperature_source],
    temperature_month  = configuration[:temperature_month],
    bathymetry_source  = configuration[:bathymetry_source],
    use_geodata        = configuration[:use_geodata],
    geodata_store_dir  = "../lakehouse/IABM",
)

# ╔═╡ 2a5c8b39-1e07-4d3f-9e48-7c6b0d3a1f85
# ╠═╡
env_status = IABM.environmental_input_status(env)

# ╔═╡ 9e4d1f67-2b83-4a5c-8f09-3c8b5a2d7e16
# ╠═╡
assimilated = IABM.assimilate_environmental_inputs(env; verbose = true)

# ╔═╡ 5f8c2a74-0e19-4b7d-8a35-9d3e6b0c1f48
# ╠═╡
md"""
### The two facts the store reports

- **`present`** is what the run will reuse. It is `bathymetry` first because that is one ERDDAP
  call for a small file; the hydrography is twelve monthly downloads of ~500 MB each *temperature*,
  ~425 MB *salinity* and ~35 MB *oxygen*, which is the expensive part and the reason the store
  exists.
- **`absent`** is what the next cell will fetch. On a machine with a Copernicus account and the
  `temperature_source` set to `glorys12v1`, the hydrography comes from Copernicus Marine instead;
  `woa23` is keyless, which is why it is the default here.

`temperature_month = 0` means *the twelve-month seasonal cycle*, which is what a multi-year
simulation with a growing season needs. Setting it to a single month fetches only that month and
runs on a fixed climate — which is a faster run and a different model, because a fixed climate
has no season and therefore no seasonal moulting or maturity constraint.
"""

# ╔═╡ 3b7d4e91-6a28-4c0f-9e57-2f9b8c5e3a74
# ╠═╡
md"""
### Reading the store back

The reader is one call and returns exactly what the simulation consumes — depth, bottom and
pelagic temperature, and the twelve-month series — as fields on the configured grid. Nothing in
this half knows where the products were published: the store is the whole interface, and
`IABM.build_geodata_environmental_fields` is called by both the reader and the model.
"""

# ╔═╡ 7c9e5b02-4f61-4d8a-9b34-1e0f7c2b5a83
# ╠═╡
environment = IABM.read_assimilated_environment(env)

# ╔═╡ 1e6b8f35-9a07-4c5b-8d20-6b3a0e7c1f94
# ╠═╡
md"""
## What the configuration fixes, and why

`configs/snowcrab.toml` is the production configuration. It is read through
`load_ibm_configuration`, which merges the package defaults, the file, and any CLI overrides —
one path, so a run and a notebook can never disagree about a value.

**The domain.** `domain_lon = [-65, -63]`, `domain_lat = [43, 45]` — a 2° × 2° box on the
Scotian Shelf, with the snow-crab benthic band in mind. The grid is `grid_nx × grid_ny` over that
box, and every field below is on that grid.

**Habitat.** `use_geodata = true` is the switch that connects the IBM to the physical world at
all; with it off, the model runs on an analytic temperature and food field. That is the
distinction to be clear about: a run with `use_geodata = false` is a *synthetic* environment
used for testing the biology, and a run with it on is the same biology on a real sea floor with
a real seasonal cycle.

**Where it saves.** `output_dir` receives the run's results; the environment comes *from* the
store rather than being written into the run directory, so two runs of the same configuration
share one assimilated world and differ only in what they simulate.
"""

# ╔═╡ 4a2f8c93-1e35-4b6d-8e07-5c9b1d4a0f62
# ╠═╡
md"""
## Running the IBM

The default entry points, in the order a run uses them:

- `initialize_model(params, rng)` — the model object: the grid, the food field, the organisms.
- `run_simulation(model, params)` — the time loop.
- `analyze_*` / `plot_*` — the results.

The two configuration switches that matter most for cost are `days` and `dt`: a multi-year run
at a daily step is what the biology needs (degree-days accumulate per moult), and it is also
what makes a first pass slow. Reduce `days` to watch a single season, then raise it.
"""

# ╔═╡ 0c8b3f27-4a09-4d6e-8b15-7f4a9d2c1e86
# ╠═╡
parameter_structs = IABM.parameter_struct_defaults()

# ╔═╡ 8d1c5a39-2b64-4e0f-9c78-3a6f0b7d2e15
# ╠═╡
md"""
## Assumptions, in one list

Everything below is a choice this model makes, not a fact it discovers. Each is a field in the
configuration, so changing one is a line in the TOML rather than an edit to the code.

1. **The domain is the world.** Nothing outside `domain_lon`/`domain_lat` exists, and movement
   is bounded by the grid edges. A simulation cannot say anything about a stock that spends part
   of its year outside the box.
2. **The environment is a 2° box at 15 arc-second bathymetry** — ETOPO 2022, resampled onto the
   model's grid. Anything finer than a grid cell is averaged away, so a reef or a channel
   narrower than a cell does not exist.
3. **WOA23's seasonal cycle is the animals' thermal environment.** WOA23 is a climatology built
   from observations: it is an average over decades, so it has no weather and no anomaly. A run
   on WOA23 describes the *mean* world, and a marine heatwave is not representable in it.
4. **Temperature drives the biology through degree-days.** `use_degree_days` accumulates
   against `temp_base_degreedays` (−1.5 °C) and drives moulting; `growth_temperature_reference`
   and `metabolic_optimal_temperature` set the rest. The assumption is that temperature is the
   master variable, which is the standard snow-crabecology result and is not universally true.
5. **Maturity and fecundity are allometric, not behavioural.** A female is mature at a size
   (`maturity_l50_female`) and fecundity scales with carapace width through the
   St. Marie parameterization. No mate limitation and no behavioural choice are modelled.
6. **Mortality is a sum of named hazards** — predation, cannibalism, starvation, soft-shell,
   thermal, senescence, larval overage, fishing — each with its own rate. They add, so a doubling
   of one hazard is not compensated by the others, and `hazard_cap` exists because a sum of
   probabilities can exceed 1.
7. **Carrying capacity is logistic in aggregate abundance** with an exponent, not a
   per-capita resource curve. `benthic_carrying_capacity` is a stock level that must exceed the
   seeded cohort, or the mortality multiplier at time zero is enormous and everything dies on
   day one — a documented trap in `EnvironmentParams`.
8. **The fishery is a season with a size selection.** `fishing_season_start`/`_end`,
   `fishing_probability_daily`, and `legal_size_male` define it; there is no fleet behaviour, no
   effort dynamics, and no catch reported against a quota.
9. **Aggregate cohorts are a population-level shortcut.** `initial_aggregate_cohort` and the
   `AggregatePopulations` machinery model the bulk of the population as compartments rather
   than individuals, with individuation into the IBM by stage. That is the assumption that the
   compartment and the individual halves agree on the demographics of the joint, which is what
   `compare_macro_models` tests.
10. **A run is reproducible from its seed.** `seed` fixes the RNG; nothing in the model draws
    from wall-clock or unseeded state.

## References

1. **IBM / agent-based modelling** — DeAngelis, D. L. & Mooij, W. M. (2005). Individual-based
   modeling of ecological populations. *Ecological Modelling*, 186(2), 159–175.
2. **Individual-based snow crab ecology** — Sainte-Marie, B. et al. (1995–); the growth,
   maturity and fecundity parameterizations used here follow the Snow Crab Working Group series
   and St. Marie (1993) on fecundity in *Chionoecetes opilio*.
3. **Bioenergetics** — Kooijman, B. (2010). *Dynamic Energy Budget theory*. Cambridge.
   (The intake, assimilation, metabolic and energy-molt terms follow DEB's structure.)
4. **Degree-day phenology** — a standard developmental-rate model; see Kuhn & Choi (2011) for
   the snow-crab application on the Scotian Shelf.
5. **Stage-structured projection** — Leslie, P. H. (1945). On the use of matrices in certain
   population mathematics. *Biometrika*, 33(3), 183–212. (`fit_generalized_leslie_model`.)
6. **Stock and recruitment** — Schaefer, M. B. (1954). Some aspects of the dynamics of
   populations. *ICES*, 21(1), 291–302.
7. **Warm temperate physics of the domain** — the climatologies are WOA23 (Locarnini et al.
   and Zweng et al., 2018, NOAA NCEI) and ETOPO 2022 (NOAA NCEI), both public domain.
"""

# ╔═╡ 6a4b0d15-2c39-4e7f-8b04-1d5c9a3f0e87
# ╠═╡
md"""
_Published by Pluto — see [the Pluto documentation](https://featured.plutojl.org/basics/notebooks)
for how notebooks are structured and shared._
"""

# ╔═╡ Cell order:
# ╠═╡ 8a1c4e72-3b65-4d09-9f7a-2c6e0b4d1a38 => 1
# ╠═╡ 2f7b0d93-5c18-4a26-8b3e-7d1f9c4a6e02 => 2
# ╠═╡ 5c3e8a26-0b79-4f1d-8a64-3e2b7d0a5c38 => 3
# ╠═╡ 9d4f6b13-2a08-4c5e-8b7f-6a3d1e0c9b47 => 4
# ╠═╡ 3e8c1a54-7f26-4b0d-9e58-2a5f8c3b1d69 => 5
# ╠═╡ 7b5d2c98-1a34-4e7b-9c0f-8d6b4a2e0f15 => 6
# ╠═╡ c1a9e642-8b07-4f3d-9e16-5c0b7a3d8e42 => 7
# ╠═╡ 6f2a7d84-3b19-4e5c-8a07-9d4e0f6b2c58 => 8
# ╠═╡ 8c3b0e15-7d42-4f9e-8b26-2a5f1c9d0e73 => 9
# ╠═╡ 4d6e9f28-0a58-4b1c-9e37-8f3b2c7e1a04 => 10
# ╠═╡ 2a5c8b39-1e07-4d3f-9e48-7c6b0d3a1f85 => 11
# ╠═╡ 9e4d1f67-2b83-4a5c-8f09-3c8b5a2d7e16 => 12
# ╠═╡ 5f8c2a74-0e19-4b7d-8a35-9d3e6b0c1f48 => 13
# ╠═╡ 3b7d4e91-6a28-4c0f-9e57-2f9b8c5e3a74 => 14
# ╠═╡ 7c9e5b02-4f61-4d8a-9b34-1e0f7c2b5a83 => 15
# ╠═╡ 1e6b8f35-9a07-4c5b-8d20-6b3a0e7c1f94 => 16
# ╠═╡ 4a2f8c93-1e35-4b6d-8e07-5c9b1d4a0f62 => 17
# ╠═╡ 0c8b3f27-4a09-4d6e-8b15-7f4a9d2c1e86 => 18
# ╠═╡ 8d1c5a39-2b64-4e0f-9c78-3a6f0b7d2e15 => 19
# ╠═╡ 6a4b0d15-2c39-4e7f-8b04-1d5c9a3f0e87 => 20
