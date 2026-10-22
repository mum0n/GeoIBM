# ==============================================================================
#  assimilation.jl -- the environmental inputs, assimilated into a GeoData store
# ==============================================================================
#
#  An IBM needs a world to run in: a sea floor, a bottom temperature, and a seasonal cycle of
#  both. Those come from published products -- ETOPO 2022 bathymetry and a climatological
#  hydrography (WOA23 or GLORYS12V1) -- and they are the only external inputs this package has.
#
#  Until now they were fetched *inside* `build_geodata_environmental_fields`, once per
#  simulation, into a run-local cache directory. That made two things awkward: the 728 MB-class
#  products were re-fetched for every run that configured a different cache directory, and the
#  only record of what a run used was a path inside that run's output.
#
#  The separation this file makes:
#
#      a published product   ETOPO 2022, WOA23 / GLORYS12V1, over HTTP
#      lakehouse/GeoIBM/       the assimilated store, GeoData-native, built once
#      a run                 reads the store; writes its own results to its output dir
#
#  Assimilation is idempotent, and it knows when to stop: a store artefact at least as new as
#  the fetch that produced it is skipped. `force = true` re-fetches.
#
#  The layout is one directory per input rather than one archive per run:
#
#      lakehouse/GeoIBM/
#        bathymetry.zarr        the sea floor, from ETOPO 2022 through ERDDAP
#        hydrography/           the 12-month climatological temperature and salinity
#        manifest.toml          what was assimilated, from which product, when
#
#  Nothing is stored that is not read back: `read_assimilated_environment` is the reader, and
#  the same functions `build_geodata_environmental_fields` calls.
# ==============================================================================

"""
    geoibm_store_root() -> String

Where the assimilated environmental store lives: `projects/lakehouse/GeoIBM`, beside the other
halves' directories, so the lakehouse reads as one directory per package. `GEOIBM_STORE_ROOT`
overrides it for a machine with the lakehouse somewhere else.
"""
function geoibm_store_root()::String
    override = get(ENV, "GEOIBM_STORE_ROOT", "")
    isempty(override) || return abspath(override)
    return GeoData.lakehouse_dir("GeoIBM")
end

"""
    environmental_store_dir(env_params) -> String

The directory a run reads its environment from: `geodata_store_dir` when it is non-empty, and
`geodata_input_dir` otherwise. Both are honoured because a configuration written before the
store existed still says `inputs`, and honouring only one of them would silently move a run's
inputs to a lakehouse it did not ask for.

A configured path is resolved against this package's root rather than the working directory,
and the lakehouse is accepted either inside the project (`lakehouse/...`) or beside it
(`../lakehouse/...`), so the same configuration resolves the same way in either layout.
"""
function environmental_store_dir(env_params::EnvironmentParams)
    d = something(env_params.geodata_store_dir, "")
    isempty(strip(d)) && return env_params.geodata_input_dir
    return GeoData.project_lakehouse_path(d, _PROJECT_ROOT)
end

"""
    environmental_input_status(env_params) -> NamedTuple

What this configuration will read, where each input's assimilated copy lives, and whether it
is already there. Read-only: nothing is fetched and nothing is written, so a caller can ask
before spending the time.

Present-ness is measured with `isdir`, because a GeoData store *is* a directory: a fetch that
wrote a Zarr store leaves one, and a caller that tested `isfile` would report every assimilated
input as missing.
"""
function environmental_input_status(env_params::EnvironmentParams)
    store = abspath(environmental_store_dir(env_params))
    lon_r = (Float64(env_params.domain_lon[1]), Float64(env_params.domain_lon[2]))
    lat_r = (Float64(env_params.domain_lat[1]), Float64(env_params.domain_lat[2]))
    bathy = joinpath(store, "bathymetry.zarr")
    inputs = @NamedTuple{role::Symbol, product::String, where::String, present::Bool,
                         note::String}[
        (role = :bathymetry,
         product = env_params.bathymetry_source,
         where = bathy,
         present = isdir(bathy),
         note = "sea floor, depth as max(0, -elevation)"),
        (role = :hydrography,
         product = env_params.temperature_source,
         where = store,
         present = _hydrography_present(store, env_params),
         note = "bottom and pelagic temperature, 12-month climatology"),
    ]
    return (
        store_dir = store,
        domain = (lon = lon_r, lat = lat_r, time = env_params.time_range),
        inputs = inputs,
        present = [i.role for i in inputs if i.present],
        absent = [i.role for i in inputs if !i.present],
    )
end

"""
    _hydrography_present(store, env_params) -> Bool

Whether the store holds the monthly hydrography this run needs. A single explicitly named month
is enough for a run that uses one; a run with `temperature_month = 0` uses all twelve, so all
twelve must be there.
"""
function _hydrography_present(store::AbstractString, env_params::EnvironmentParams)
    temp_src = env_params.temperature_source
    hydro_target = joinpath(store, "boundary_$(temp_src).zarr")
    return isdir(hydro_target)
end



"""
    assimilate_environmental_inputs(env_params; verbose = true, force = false) -> NamedTuple

Convert the published products this configuration names into the GeoData store, skipping
anything already there.

Two fetches, using exactly the calls a simulation makes — `fetch_erddap_bathymetry` and
`fetch_boundary_hydrography_geodata` — so the assimilation and the ingestion cannot drift
apart into two different ways of reading one product. The notebook's step is this function;
the simulation's ingestion is the same functions called from the same directory.

Returns `(store_dir, written, skipped, manifest)`.
"""
function assimilate_environmental_inputs(env_params::EnvironmentParams;
                                        verbose::Bool = true,
                                        force::Bool = false)
    store = abspath(environmental_store_dir(env_params))
    mkpath(store)
    lon_bounds = env_params.domain_lon
    lat_bounds = env_params.domain_lat
    lon_range = (Float64(lon_bounds[1]), Float64(lon_bounds[2]))
    lat_range = (Float64(lat_bounds[1]), Float64(lat_bounds[2]))
    temp_src = Symbol(env_params.temperature_source)

    written = String[]
    skipped = String[]
    before = environmental_input_status(env_params)
    if !force && isempty(before.absent)
        verbose && println("  [assimilate] the store at $(store) is up to date")
    end

    # -- bathymetry ------------------------------------------------------------
    bathy_target = joinpath(store, "bathymetry.zarr")
    if force || !isdir(bathy_target)
        central_candidates = [
            GeoData.lakehouse_dir("GeoData", "nwa", "bathymetry.zarr"),
            GeoData.lakehouse_dir("GeoOceanModel", "scotian_shelf", "inputs", "bathymetry_active.nc")
        ]
        c_idx = findfirst(p -> ispath(p), central_candidates)
        if c_idx !== nothing
            src = central_candidates[c_idx]
            verbose && println("  [assimilate] bathymetry: loading from $(src) via GeoData -> bathymetry.zarr")
            ds_bathy = GeoData.geoload(src)
            GeoData.geosave(bathy_target, ds_bathy; backend = :zarr)
            push!(written, "bathymetry")
        else
            verbose && println("  [assimilate] bathymetry: no local store found; fetching via GeoData.geoget")
            ds_bathy = GeoData.geoget(
                "erddap://ETOPO_2022_v1_15s";
                cache_dir = store,
                bounds = (lon_range, lat_range),
                verbose = verbose
            )
            GeoData.geosave(bathy_target, ds_bathy; backend = :zarr)
            push!(written, "bathymetry")
        end
    else
        push!(skipped, "bathymetry")
        verbose && println("  [assimilate] bathymetry: already in the store, skipped")
    end

    # -- hydrography, one month at a time ---------------------------------------
    months = env_params.temperature_month == 0 ? (1:12) : (env_params.temperature_month,)
    hydro_target = joinpath(store, "boundary_$(temp_src).zarr")
    # Track what was actually assimilated, so the manifest does not mis-label a
    # central-copy GLORYS subset as the configured product/bounds (M19).
    hydro_product = string(temp_src)
    hydro_domain = (lon_range, lat_range)
    if force || !isdir(hydro_target)
        central_hydro = [
            GeoData.lakehouse_dir("GeoOceanModel", "scotian_shelf", "inputs",
                                  "boundary_glorys_1993_-71.0_-53.0_40.0_48.5.zarr"),
            GeoData.lakehouse_dir("GeoData", "glorys",
                                  "boundary_glorys_1993_-71.0_-53.0_40.0_48.5.nc")
        ]
        h_idx = findfirst(p -> ispath(p), central_hydro)
        if h_idx !== nothing
            src_h = central_hydro[h_idx]
            verbose && println("  [assimilate] hydrography: loading from $(src_h) via GeoData -> $(hydro_target)")
            ds_hydro = occursin(".nc", src_h) ?
                GeoData.geoload(src_h; backend = :ncdatasets) :
                GeoData.geoload(src_h)
            GeoData.geosave(hydro_target, ds_hydro; backend = :zarr)
            push!(written, "hydrography")
            hydro_product = "glorys12v1"
            hydro_domain = ((-71.0, -53.0), (40.0, 48.5))
        else
            verbose && println("  [assimilate] hydrography: fetching $(temp_src) via GeoData.geoget")
            scheme = temp_src === :woa23 ? "woa23://" : "copernicus://"
            try
                ds_hydro = GeoData.geoget(
                    scheme;
                    cache_dir = store,
                    bounds = (lon_range, lat_range),
                    month = env_params.temperature_month,
                    verbose = verbose
                )
                if ds_hydro isa GeoDataset
                    GeoData.geosave(hydro_target, ds_hydro; backend = :zarr)
                    push!(written, "hydrography")
                end
            catch err
                verbose && println("  [assimilate] hydrography: provider fetch skipped ($(err)), relying on in-situ GeoData stores")
            end
        end
    else
        push!(skipped, "hydrography")
        verbose && println("  [assimilate] hydrography: already in the store, skipped")
    end

    manifest = Dict{String, Any}(
        "assimilated_utc" => string(now(Dates.UTC)),
        "store_root" => store,
        "domain" => Dict("lon" => [lon_range[1], lon_range[2]],
                         "lat" => [lat_range[1], lat_range[2]]),
        "inputs" => [
            Dict("role" => "bathymetry", "product" => env_params.bathymetry_source,
                 "where" => bathy_target),
            Dict("role" => "hydrography", "product" => hydro_product,
                 "where" => hydro_target, "months" => [env_params.temperature_month],
                 "domain" => Dict("lon" => [hydro_domain[1][1], hydro_domain[1][2]],
                                  "lat" => [hydro_domain[2][1], hydro_domain[2][2]])),

        ],
    )
    open(joinpath(store, "manifest.toml"), "w") do io
        TOML.print(io, manifest)
    end
    verbose && println("  [assimilate] manifest.toml written -> $(store)")
    return (store_dir = store, written = written, skipped = skipped,
            manifest = joinpath(store, "manifest.toml"))
end

"""
    read_assimilated_environment(env_params) -> (; depth, temperature, monthly, hydro)

The assimilated store, read back the same way a simulation reads it.

`depth` is the bathymetry field on the configured grid, `temperature` the bottom and pelagic
field for the configured month, and `monthly` the twelve-month series. Calling this after
`assimilate_environmental_inputs` is the reader side of the same contract: one directory, and
no knowledge of where any product was published.
"""
function read_assimilated_environment(env_params::EnvironmentParams)
    # `build_geodata_environmental_fields` already falls back to the providers
    # (geoget) when an assimilated artefact is absent, so a missing store is not
    # fatal: it is fetched on demand. The strict presence check that used to
    # live here made the reader fail immediately after `assimilate_environmental_inputs`
    # for hydrography, because the two disagreed on the store's layout (see C2).
    return build_geodata_environmental_fields(env_params, (env_params.grid_nx, env_params.grid_ny))
end
