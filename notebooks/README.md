# Notebooks

A Pluto notebook for the individual-based model: assimilate the environmental store, load the
configuration, inspect the fields the animals live on, run the IBM, and read its results back.

| Notebook | Domain | Species | Inputs |
|---|---|---|---|
| [`TethysBioNotes.jl`](TethysBioNotes.jl) | a configurable Scotian Shelf / Laurentian Channel window centered on a chosen point | snow crab | config file + interactive steering; environmental store via `Tethys.lakehouse_dir` |

A **Pluto** notebook: a `.jl` file in Pluto's own format, with `md"""..."""` cells between the
code. It documents the run as it performs it — what each option means, which assumptions the
model bakes in, and where each biological relationship comes from.

## What it does

1. **Configuration & steering** — loads `configs/snowcrab.toml` through `load_ibm_configuration`
   and layers reactive widgets on top: domain center (an AOI picker), environmental offsets and
   multipliers, fishery season and rate, no-take proportion, process toggles, and an interactive
   initial stage-abundance distribution that feeds the `initial_*_cohort` configuration.
2. **Model execution** — constructs `IBMParameters` from the merged configuration, calls
   `initialize_model` with an AOI-derived bathymetric depth map, and drives `run_simulation`
   across a multi-year horizon. By default, the notebook sets `run_model = false` and loads
   previously computed results from `../scratch/saved_ibm_run.jld2` to protect existing runs from
   accidental overwrites; setting `run_model = true` explicitly triggers re-integration and updates the store.
3. **Dashboards** — spatial habitat/organism map, environmental distributions, stage structure,
   size-frequency densities by sex and maturity, maturity ogives, abundance trajectories,
   biomass/catch panels, and stock–recruitment scatter.
4. **Macro-ecology** — Schaefer surplus production, Deriso–Schnute delay-difference, and
   generalized Leslie/Lefkovitch fits to the annual series (`fit_population_dynamics`,
   `fit_generalized_leslie_model`, `compare_macro_models`), with a three-way comparison panel.
5. **Calibration** — `evaluate_population_criteria` scored against five biological criteria
   (persistence, bounds, size-structure stability, maturity-ogive fidelity, growth fidelity).

## Running it

Pluto is deliberately **not** a dependency of this package: a notebook is how you look at the
work, not something the run needs. Launched from the package directory, the first code cell
activates the parent project (`Pkg.activate(joinpath(@__DIR__, ".."))`), so the notebook is
pinned to this package's environment:

```
julia --project=. -e 'using Pluto; Pluto.run()'
```

Open `TethysBioNotes.jl` from the notebook list. Relative paths (`configs/snowcrab.toml`,
`../scratch/saved_ibm_run.jld2`) resolve against the notebook's location, and the environmental
store resolves through `Tethys.lakehouse_dir("TethysBio")`.

For a no-browser smoke run — verify every cell executes cleanly — there is a headless runner
that drives Pluto's session API directly:

```
julia --project=@v1.13 scratch/run_notebook.jl
```

It opens the notebook with `run_notebook_on_load = true`, prints one line per cell (cell
number, output MIME type, first line of code) and flags any cell whose output is an error,
a parse failure, or a stacktrace. Note this runs the notebook for real: with an empty
environmental store the first ingesting run may download published data products.

## Not Jupyter

This package uses Pluto only. Its notebooks are reactive, so re-running the configuration cell
re-derives everything downstream of it, and the cells below a plot can be re-run on their own
while you inspect what they returned.
