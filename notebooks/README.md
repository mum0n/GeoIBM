# Notebooks

A Pluto notebook for the individual-based model: assimilate the environmental store, load the
configuration, inspect the fields the animals live on, run the IBM, and read its results back.

| Notebook | Domain | Species | Inputs |
|---|---|---|---|
| [`workflow.jl`](workflow.jl) | the Scotian Shelf snow-crab box (2° × 2°) | snow crab | real: ETOPO 2022 bathymetry, WOA23 hydrography |

A **Pluto** notebook: a `.jl` file in Pluto's own format, with `md"""..."""` cells between the
code. It documents the run as it performs it — what each option means, which assumptions the
model bakes in, and where each biological relationship comes from.

## What it starts with

Step 0 is the assimilation step: the two published products this model needs are converted into
a GeoData store at `../lakehouse/IABM/`, and the run ingests from that store rather than from a
run-local cache. It is idempotent — what is already assimilated is skipped — so the second run
of the notebook does no downloading.

## Running it

Pluto is deliberately **not** a dependency of this package: a notebook is how you look at the
work, not something the run needs. Launched from the package directory, `Pkg.activate(@__DIR__)`
inside the first cells pins the notebook to this package's environment:

```
julia --project=. -e 'using Pluto; Pluto.run()'
```

Open `workflow.jl` from the notebook list. The first cells pin the working directory and
activate the package's environment, so `configs/snowcrab.toml` and the `../lakehouse/IABM` paths
resolve relative to the package.

## Not Jupyter

This package uses Pluto only. Its notebooks are reactive, so re-running the configuration cell
re-derives everything downstream of it, and the cells below a plot can be re-run on their own
while you inspect what they returned.
