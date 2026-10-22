"""
    src/notebook_support.jl

Modular helper and rendering functions for Pluto.jl notebooks (e.g. `GeoIBMNotes.jl`)
and interactive dashboard components.

Provides:
- Standardized HTML/JS widget builders for Pluto reactive bindings:
  - `aoi_coords_widget`: Leaflet domain locator and coordinate picker for the Scotian Shelf.
  - `parameter_steering_sliders_widget`: Two-column simulation sliders and biological toggles.
  - `simulation_transport_widget`: Execution transport bar (Start, Step, Pause, Stop, Rewind, interval, duration).
  - `steering_controls_widget`: Backward-compatible combined transport and parameter control panel.
  - `estimate_aoi_bathymetry`: Area bathymetry lookup and bilinear interpolation within ±500 m.
- Reusable, standalone Makie dashboard plotting functions matching the `gui.jl` panels:
  - `plot_dashboard_spatial`: Bottom temperature thermal field with agent particles and explicit grid bounds.
  - `plot_dashboard_abundance`: Dynamic log10 population trajectories with day/year timestamps.
  - `plot_dashboard_stages`: Instar stage abundance lollipop forest glyphs (`stem!`).
  - `plot_dashboard_size_density`: Gaussian kernel density spectra by sex & maturity.
  - `plot_dashboard_ogive`: Realized and theoretical maturity and fishery selectivity.
  - `plot_dashboard_macro`: Annual legal biomass, catch, and fitted macro-ecological models.
  - `plot_dashboard_stock_recruitment`: Realized spawner-recruit pairs with Ricker & Beverton-Holt fits.
"""

using Printf
using CairoMakie
using DataFrames
using Agents
using Statistics
using GeoData
using Interpolations


"""
    extract_ctrl_val(ctrl, key::Symbol, default)

Extract a setting value from `ctrl` whether it is a `Dict{String, Any}`,
`Dict{Symbol, Any}`, `NamedTuple`, or struct. Returns `default` if `ctrl` is
missing, `nothing`, or does not contain `key`.
"""
function extract_ctrl_val(ctrl, key::Symbol, default)
    if ctrl === nothing || ismissing(ctrl)
        return default
    elseif ctrl isa AbstractDict
        skey = string(key)
        haskey(ctrl, skey) && return ctrl[skey]
        haskey(ctrl, key) && return ctrl[key]
        return default
    elseif hasproperty(ctrl, key)
        return getproperty(ctrl, key)
    end
    return default
end

# ------------------------------------------------------------------------------
# 1. Interactive HTML Widget Builders
# ------------------------------------------------------------------------------

"""
    aoi_coords_widget(; default_lon = -63.5, default_lat = 44.5) -> String

Construct the HTML and embedded JavaScript string for the interactive area of
interest (AOI) coordinate selector on the Scotian Shelf & Gulf domain.
"""
function aoi_coords_widget(; default_lon::Float64 = -63.5, default_lat::Float64 = 44.5)
    lon_s = @sprintf("%.2f", default_lon)
    lat_s = @sprintf("%.2f", default_lat)
    return """
<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 820px; margin: 0 0 10px 0; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
    <span style="font-size: 13px; font-weight: 700; color: #1e293b;">
      Scotian Shelf & Gulf Domain Coordinate Selection
    </span>
    <span id="coord_display" style="font-size: 12px; font-family: monospace; color: #0f172a; background: #e2e8f0; padding: 3px 8px; border-radius: 4px; font-weight: 600;">
      Selected: $(lat_s)°N, $(lon_s)°W
    </span>
  </div>

  <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 14px; align-items: center;">
    <div style="position: relative; background: #0f172a; border-radius: 6px; border: 1px solid #334155; overflow: hidden; height: 210px;">
      <div id="aoi_map" style="width: 100%; height: 100%;"></div>
      <div style="position: absolute; bottom: 4px; left: 8px; z-index: 1000; font-size: 10px; color: #cbd5e1; background: rgba(15,23,42,0.8); padding: 2px 6px; border-radius: 4px; pointer-events: none;">
        Click or drag map marker to position Area of Interest (AOI)
      </div>
    </div>

    <div style="display: flex; flex-direction: column; gap: 10px;">
      <div>
        <div style="display: flex; justify-content: space-between; font-size: 11px; color: #475569; margin-bottom: 2px;">
          <span>Longitude (West):</span>
          <span id="lon_label" style="font-family: monospace; font-weight: 600;">$(lon_s)°</span>
        </div>
        <input type="range" id="slider_lon" min="-66.5" max="-57.5" step="0.1" value="$(lon_s)" style="width: 100%;">
      </div>

      <div>
        <div style="display: flex; justify-content: space-between; font-size: 11px; color: #475569; margin-bottom: 2px;">
          <span>Latitude (North):</span>
          <span id="lat_label" style="font-family: monospace; font-weight: 600;">$(lat_s)°</span>
        </div>
        <input type="range" id="slider_lat" min="42.0" max="48.5" step="0.1" value="$(lat_s)" style="width: 100%;">
      </div>

      <div style="font-size: 10px; color: #64748b; line-height: 1.3; background: #f8fafc; padding: 6px; border-radius: 4px; border: 1px solid #e2e8f0;">
        Basemap: Esri World Ocean (GEBCO / NOAA bathymetry). Domain bounds: [42.0°N – 48.5°N], [-66.5°W – -57.5°W].
      </div>
    </div>
  </div>

<script>
const container = currentScript.parentElement;
const s_lon = container.querySelector('#slider_lon');
const s_lat = container.querySelector('#slider_lat');
const l_lon = container.querySelector('#lon_label');
const l_lat = container.querySelector('#lat_label');
const c_disp = container.querySelector('#coord_display');
const mapDiv = container.querySelector('#aoi_map');

let currentVal = { lon: $(lon_s), lat: $(lat_s) };
container.value = currentVal;

function updateCoordinates(lon, lat, dispatch = true, updateMarker = true) {
  lon = Math.min(-57.5, Math.max(-66.5, Math.round(lon * 100) / 100));
  lat = Math.min(48.5, Math.max(42.0, Math.round(lat * 100) / 100));
  currentVal = { lon: lon, lat: lat };
  
  s_lon.value = lon;
  s_lat.value = lat;
  l_lon.innerText = lon.toFixed(2) + "°";
  l_lat.innerText = lat.toFixed(2) + "°";
  c_disp.innerText = `Selected: \${lat.toFixed(2)}°N, \${lon.toFixed(2)}°W`;
  
  if (updateMarker && window.aoi_marker) {
    window.aoi_marker.setLatLng([lat, lon]);
  }
  
  // When location changes, immediately pause the running simulation, update to new location and wait
  if (dispatch && window._geoibm_state) {
    let gs = window._geoibm_state;
    if (typeof gs.pauseSimulation === 'function') {
      gs.pauseSimulation();
    } else {
      gs.simState = "paused";
      if (gs.timer !== null) {
        clearInterval(gs.timer);
        gs.timer = null;
      }
    }
  }

  container.value = currentVal;
  if (dispatch) {
    container.dispatchEvent(new CustomEvent('input', { bubbles: true, composed: true }));
    if (window._geoibm_state) {
      let gs = window._geoibm_state;
      if (gs.slidersContainer) {
        gs.slidersContainer.dispatchEvent(new CustomEvent('input', { bubbles: true, composed: true }));
      }
      if (gs.transportContainer) {
        gs.transportContainer.dispatchEvent(new CustomEvent('input', { bubbles: true, composed: true }));
      }
    }
  }
}

s_lon.addEventListener('input', () => {
  updateCoordinates(parseFloat(s_lon.value), parseFloat(s_lat.value), true, true);
});

s_lon.addEventListener('change', () => {
  updateCoordinates(parseFloat(s_lon.value), parseFloat(s_lat.value), true, true);
});

s_lat.addEventListener('input', () => {
  updateCoordinates(parseFloat(s_lon.value), parseFloat(s_lat.value), true, true);
});

s_lat.addEventListener('change', () => {
  updateCoordinates(parseFloat(s_lon.value), parseFloat(s_lat.value), true, true);
});

function initMap() {
  if (typeof L === 'undefined') {
    // Dynamically inject Leaflet CSS and JS if not already loaded
    if (!document.getElementById('leaflet_css')) {
      const link = document.createElement('link');
      link.id = 'leaflet_css';
      link.rel = 'stylesheet';
      link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
      document.head.appendChild(link);
    }
    const script = document.createElement('script');
    script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
    script.onload = () => setupLeafletMap();
    document.head.appendChild(script);
  } else {
    setupLeafletMap();
  }
}

function setupLeafletMap() {
  if (mapDiv._leaflet_id) {
    return; // Already initialized on this element
  }

  const map = L.map(mapDiv, {
    zoomControl: false,
    attributionControl: false
  }).setView([$(lat_s), $(lon_s)], 6);

  // 1. GEBCO Global Bathymetry (WMS shaded relief & depth contours) - Open / Keyless
  const gebcoBathymetry = L.tileLayer.wms('https://wms.gebco.net/mapserv?', {
    layers: 'GEBCO_LATEST',
    format: 'image/png',
    transparent: true,
    version: '1.3.0',
    attribution: '&copy; GEBCO / IHO / IOC-UNESCO'
  });

  // 2. Esri World Ocean Base (GEBCO / NOAA bathymetric relief)
  const esriOcean = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 12,
    attribution: '&copy; Esri, GEBCO, NOAA'
  });

  // 3. NOAA / Esri National Geographic Topographic & Bathymetric Relief
  const natGeoWorld = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 12,
    attribution: '&copy; National Geographic, Esri'
  });

  // Add primary layer (GEBCO Bathymetry by default for reliable bathymetric contours)
  gebcoBathymetry.addTo(map);

  // Add interactive Layer Control for instant provider switching
  const baseMaps = {
    "GEBCO Bathymetry (WMS)": gebcoBathymetry,
    "Esri World Ocean": esriOcean,
    "NatGeo Topo / Bathy": natGeoWorld
  };
  L.control.layers(baseMaps, null, { position: 'topright' }).addTo(map);

  // Scotian Shelf & Laurentian Channel domain bounding box outline
  const domainBounds = [[42.0, -66.5], [48.5, -57.5]];
  L.rectangle(domainBounds, {
    color: "#38bdf8",
    weight: 1.5,
    fillOpacity: 0.05,
    dashArray: "4, 4"
  }).addTo(map);

  // Draggable Crosshair / Target Marker
  const customIcon = L.divIcon({
    className: 'aoi-marker',
    html: `<div style="width: 18px; height: 18px; border-radius: 50%; background: rgba(239, 68, 68, 0.85); border: 2px solid #ffffff; box-shadow: 0 0 6px rgba(0,0,0,0.4); transform: translate(-50%, -50%);"></div>`,
    iconSize: [0, 0]
  });

  const marker = L.marker([$(lat_s), $(lon_s)], {
    draggable: true,
    icon: customIcon
  }).addTo(map);
  window.aoi_marker = marker;

  marker.on('drag', function(e) {
    const latlng = e.target.getLatLng();
    updateCoordinates(latlng.lng, latlng.lat, false, false);
  });

  marker.on('dragend', function(e) {
    const latlng = e.target.getLatLng();
    updateCoordinates(latlng.lng, latlng.lat, true, false);
  });

  map.on('click', function(e) {
    updateCoordinates(e.latlng.lng, e.latlng.lat, true, true);
  });

  setTimeout(() => { map.invalidateSize(); }, 250);
}

initMap();
updateCoordinates($(lon_s), $(lat_s), false, false);
</script>
</div>
"""
end

"""
    estimate_aoi_bathymetry(lon::Real, lat::Real;
                            span_meters::Real = 1000.0,
                            n_points::Int = 25) -> (; depths, mean_depth, min_depth, max_depth, lons, lats)

Estimate seabed depths over a geographic area centred at `(lon, lat)` spanning
`± span_meters / 2` in both longitude and latitude directions.
At latitude `lat`, 1 degree of latitude corresponds to ~111,139 m, and 1 degree of
longitude corresponds to ~111,139 * cos(lat * π / 180) m.

Searches local store candidates via `GeoData.geoload` and continuous interpolation
via `GeoData.get_bathymetry_interpolator`. If offline or missing, defaults to the
regional shelf analytical bathymetry evaluated over the sampled grid offsets.

# Arguments
- `lon::Real`: Center longitude in decimal degrees.
- `lat::Real`: Center latitude in decimal degrees.
- `span_meters::Real`: Total width/height of the bounding square (default: 1000.0 m).
- `n_points::Int`: Number of sample points per axis (default: 25).

# Returns
NamedTuple with:
- `depths::Matrix{Float64}`: `(n_points, n_points)` matrix of seabed depths (m).
- `mean_depth::Float64`: Mean water depth over the bounding square (m).
- `min_depth::Float64`: Minimum water depth over the square (m).
- `max_depth::Float64`: Maximum water depth over the square (m).
- `lons::Vector{Float64}`: Sampled longitude coordinates.
- `lats::Vector{Float64}`: Sampled latitude coordinates.
"""
function estimate_aoi_bathymetry(
    lon::Real,
    lat::Real;
    span_meters::Real = 1000.0,
    n_points::Int = 25
)
    span_m = Float64(span_meters)
    half_m = span_m / 2.0
    lat_deg_len = 111139.0
    lon_deg_len = max(1000.0, 111139.0 * cos(deg2rad(Float64(lat))))

    dlat = half_m / lat_deg_len
    dlon = half_m / lon_deg_len

    lon_pts = range(lon - dlon, lon + dlon; length = n_points)
    lat_pts = range(lat - dlat, lat + dlat; length = n_points)

    candidate_paths = [
        GeoData.lakehouse_dir("GeoData", "bathymetry", "nwa_bathymetry.nc"),
        GeoData.lakehouse_dir("GeoOceanModel", "scotian_shelf", "inputs", "bathymetry_active.nc"),
        GeoData.lakehouse_dir("GeoData", "nwa", "bathymetry.zarr")
    ]
    c_idx = findfirst(p -> ispath(p), candidate_paths)

    depth_mat = zeros(Float64, n_points, n_points)
    if c_idx !== nothing
        try
            candidate_file = candidate_paths[c_idx]
            ds_bathy = occursin(".nc", candidate_file) ?
                GeoData.geoload(candidate_file; backend = :ncdatasets) :
                GeoData.geoload(candidate_file)
            bathy_itp = GeoData.get_bathymetry_interpolator(ds_bathy)
            for j in 1:n_points, i in 1:n_points
                val = bathy_itp(lon_pts[i], lat_pts[j])
                depth_mat[i, j] = isnan(val) ? 0.0 : max(0.0, -val)
            end
        catch
            for j in 1:n_points, i in 1:n_points
                depth_mat[i, j] = max(
                    25.0,
                    180.0 + 80.0 * sin(0.8 * lon_pts[i]) + 60.0 * cos(1.1 * lat_pts[j])
                )
            end
        end
    else
        for j in 1:n_points, i in 1:n_points
            depth_mat[i, j] = max(
                25.0,
                180.0 + 80.0 * sin(0.8 * lon_pts[i]) + 60.0 * cos(1.1 * lat_pts[j])
            )
        end
    end

    return (
        depths = depth_mat,
        mean_depth = mean(depth_mat),
        min_depth = minimum(depth_mat),
        max_depth = maximum(depth_mat),
        lons = collect(lon_pts),
        lats = collect(lat_pts)
    )
end

"""
    parameter_steering_sliders_widget(; default_years = 30, default_interval = 10) -> String

Construct the HTML and embedded JavaScript string for the two-column set of
simulation and environmental sliders, fishery thresholds, and biological process toggles.
Synchronizes with `window._geoibm_state` so that parameter adjustments seamlessly update
the model state and notify Pluto subscribers.
"""
function parameter_steering_sliders_widget(; default_years::Int = 30, default_interval::Int = 10)
    return """
<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px; margin: 12px 0; max-width: 820px;">
  
  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
    <!-- Column 1: Simulation & Environmental Controls -->
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 14px;">
      <div style="font-weight: 700; font-size: 13px; color: #1e293b; margin-bottom: 12px; border-bottom: 2px solid #cbd5e1; padding-bottom: 4px;">
        Simulation & Environmental Controls
      </div>

      <div style="margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px;">
          <span>Grid Resolution (cells):</span>
          <span id="val_grid" style="font-weight: 600; font-family: monospace;">50</span>
        </div>
        <input type="range" id="ctrl_grid" min="30" max="150" step="10" value="50" style="width: 100%;">
      </div>

      <div style="margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px;">
          <span>Base Bottom Temp Offset (°C):</span>
          <span id="val_temp_offset" style="font-weight: 600; font-family: monospace;">0.0</span>
        </div>
        <input type="range" id="ctrl_temp_offset" min="-5.0" max="5.0" step="0.5" value="0.0" style="width: 100%;">
      </div>

      <div style="margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px;">
          <span>Seasonal Temp Multiplier:</span>
          <span id="val_temp_mult" style="font-weight: 600; font-family: monospace;">1.0</span>
        </div>
        <input type="range" id="ctrl_temp_mult" min="0.0" max="3.0" step="0.1" value="1.0" style="width: 100%;">
      </div>

      <div style="margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px;">
          <span>Thermal Spatial Noise (SD):</span>
          <span id="val_temp_noise" style="font-weight: 600; font-family: monospace;">0.00 °C</span>
        </div>
        <input type="range" id="ctrl_temp_noise" min="0.0" max="1.5" step="0.05" value="0.0" style="width: 100%;">
      </div>

      <div style="margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px;">
          <span>Habitat Suitability Offset:</span>
          <span id="val_hab_offset" style="font-weight: 600; font-family: monospace;">0.00</span>
        </div>
        <input type="range" id="ctrl_hab_offset" min="-1.0" max="1.0" step="0.05" value="0.0" style="width: 100%;">
      </div>

      <div style="margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px;">
          <span>Habitat Multiplier:</span>
          <span id="val_hab_mult" style="font-weight: 600; font-family: monospace;">1.0×</span>
        </div>
        <input type="range" id="ctrl_hab_mult" min="0.0" max="3.0" step="0.1" value="1.0" style="width: 100%;">
      </div>

      <div style="margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px;">
          <span>Benthic Food Max:</span>
          <span id="val_food" style="font-weight: 600; font-family: monospace;">10.0</span>
        </div>
        <input type="range" id="ctrl_food" min="4.0" max="20.0" step="1.0" value="10.0" style="width: 100%;">
      </div>

      <div style="margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px;">
          <span>Carrying Capacity K:</span>
          <span id="val_k" style="font-weight: 600; font-family: monospace;">6000</span>
        </div>
        <input type="range" id="ctrl_k" min="1000" max="20000" step="1000" value="6000" style="width: 100%;">
      </div>

      <div>
        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 12px;">
          <span style="font-weight: 600;">Spatial Map Background Layer:</span>
          <select id="ctrl_layer" style="padding: 2px 6px; border: 1px solid #cbd5e1; border-radius: 4px; font-size: 12px;">
            <option value="habitat_quality" selected>Habitat Quality (HSI)</option>
            <option value="temperature">Bottom Temperature</option>
            <option value="depth">Seabed Bathymetric Depth (m)</option>
          </select>
        </div>
      </div>
    </div>

    <!-- Column 2: Fishery Sliders & Biological Process Toggles -->
    <div style="display: flex; flex-direction: column; gap: 14px;">
      <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 14px;">
        <div style="font-weight: 700; font-size: 13px; color: #1e293b; margin-bottom: 12px; border-bottom: 2px solid #cbd5e1; padding-bottom: 4px;">
          Commercial Fishery Selectivity
        </div>

        <div style="margin-bottom: 10px;">
          <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px;">
            <span>Fishing Start (Day):</span>
            <span id="val_fish_start" style="font-weight: 600; font-family: monospace;">90</span>
          </div>
          <input type="range" id="ctrl_fish_start" min="1" max="365" step="1" value="90" style="width: 100%;">
        </div>

        <div style="margin-bottom: 10px;">
          <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px;">
            <span>Fishing End (Day):</span>
            <span id="val_fish_end" style="font-weight: 600; font-family: monospace;">150</span>
          </div>
          <input type="range" id="ctrl_fish_end" min="1" max="365" step="1" value="150" style="width: 100%;">
        </div>

        <div style="margin-bottom: 10px;">
          <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px;">
            <span>Daily Harvest Probability:</span>
            <span id="val_fish_rate" style="font-weight: 600; font-family: monospace;">0.0020</span>
          </div>
          <input type="range" id="ctrl_fish_rate" min="0.0" max="0.01" step="0.0005" value="0.002" style="width: 100%;">
        </div>

        <div>
          <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px;">
            <span>Legal Male Size Threshold:</span>
            <span id="val_legal_size" style="font-weight: 600; font-family: monospace;">95.0 mm</span>
          </div>
          <input type="range" id="ctrl_legal_size" min="70.0" max="105.0" step="1.0" value="95.0" style="width: 100%;">
        </div>
      </div>

      <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 14px; flex-grow: 1;">
        <div style="font-weight: 700; font-size: 13px; color: #1e293b; margin-bottom: 12px; border-bottom: 2px solid #cbd5e1; padding-bottom: 4px;">
          Biological Process Toggles
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 12px;">
          <label style="display: flex; align-items: center; gap: 6px; cursor: pointer;">
            <input type="checkbox" id="tog_energetics" checked>
            <span>Bioenergetics</span>
          </label>
          <label style="display: flex; align-items: center; gap: 6px; cursor: pointer;">
            <input type="checkbox" id="tog_thermal" checked>
            <span>Thermal Stress</span>
          </label>
          <label style="display: flex; align-items: center; gap: 6px; cursor: pointer;">
            <input type="checkbox" id="tog_cannibalism">
            <span>Cannibalism</span>
          </label>
          <label style="display: flex; align-items: center; gap: 6px; cursor: pointer;">
            <input type="checkbox" id="tog_senescence" checked>
            <span>Senescence</span>
          </label>
          <label style="display: flex; align-items: center; gap: 6px; cursor: pointer;">
            <input type="checkbox" id="tog_movement" checked>
            <span>Taxis Movement</span>
          </label>
          <label style="display: flex; align-items: center; gap: 6px; cursor: pointer;">
            <input type="checkbox" id="tog_reproduction" checked>
            <span>Reproduction</span>
          </label>
        </div>
      </div>
    </div>
  </div>

<script>
const container = currentScript.parentElement;

const c_grid = container.querySelector('#ctrl_grid');
const c_toffset = container.querySelector('#ctrl_temp_offset');
const c_tmult = container.querySelector('#ctrl_temp_mult');
const c_tnoise = container.querySelector('#ctrl_temp_noise');
const c_hab_offset = container.querySelector('#ctrl_hab_offset');
const c_hab_mult = container.querySelector('#ctrl_hab_mult');
const c_food = container.querySelector('#ctrl_food');
const c_k = container.querySelector('#ctrl_k');
const c_layer = container.querySelector('#ctrl_layer');
const c_fstart = container.querySelector('#ctrl_fish_start');
const c_fend = container.querySelector('#ctrl_fish_end');
const c_frate = container.querySelector('#ctrl_fish_rate');
const c_lsize = container.querySelector('#ctrl_legal_size');

const t_energetics = container.querySelector('#tog_energetics');
const t_thermal = container.querySelector('#tog_thermal');
const t_cannibalism = container.querySelector('#tog_cannibalism');
const t_senescence = container.querySelector('#tog_senescence');
const t_movement = container.querySelector('#tog_movement');
const t_reproduction = container.querySelector('#tog_reproduction');

if (!window._geoibm_state) {
  window._geoibm_state = {
    simState: "stopped",
    currentSimDay: 0,
    timer: null,
    tick: 0,
    update_interval: $(default_interval),
    sim_years: $(default_years),
    activeContainer: null,
    slidersContainer: null,
    transportContainer: null
  };
}

let s = window._geoibm_state;
s.slidersContainer = container;

function getSlidersState() {
  const toff = parseFloat(c_toffset.value);
  const tmult = parseFloat(c_tmult.value);
  const tnoise = parseFloat(c_tnoise ? c_tnoise.value : 0.0);
  const hab_off = parseFloat(c_hab_offset.value);
  const hab_m = parseFloat(c_hab_mult.value);
  return {
    sim_status: s.simState,
    current_day: s.currentSimDay,
    tick: ++s.tick,
    update_interval: s.update_interval || $(default_interval),
    sim_years: s.sim_years || $(default_years),
    grid_res: parseInt(c_grid.value),
    temp_offset: toff,
    temp_multiplier: tmult,
    temp_noise_sd: tnoise,
    temp_base: toff,
    temp_amplitude: tmult,
    habitat_offset: hab_off,
    habitat_multiplier: hab_m,
    background_layer: c_layer.value,
    food_max: parseFloat(c_food.value),
    benthic_carrying_capacity: parseFloat(c_k.value),
    fishing_season_start: parseFloat(c_fstart.value),
    fishing_season_end: parseFloat(c_fend.value),
    fishing_rate: parseFloat(c_frate.value),
    legal_size: parseFloat(c_lsize.value),
    use_energetics: t_energetics.checked,
    use_thermal_stress: t_thermal.checked,
    use_cannibalism: t_cannibalism.checked,
    use_senescence: t_senescence.checked,
    use_movement: t_movement.checked,
    use_reproduction: t_reproduction.checked
  };
}

function updateSliders() {
  container.querySelector('#val_grid').innerText = c_grid.value;
  container.querySelector('#val_temp_offset').innerText = (parseFloat(c_toffset.value) >= 0 ? "+" : "") + parseFloat(c_toffset.value).toFixed(1);
  container.querySelector('#val_temp_mult').innerText = parseFloat(c_tmult.value).toFixed(1) + "×";
  if (c_tnoise) {
    container.querySelector('#val_temp_noise').innerText = parseFloat(c_tnoise.value).toFixed(2) + " °C";
  }
  container.querySelector('#val_hab_offset').innerText = (parseFloat(c_hab_offset.value) >= 0 ? "+" : "") + parseFloat(c_hab_offset.value).toFixed(2);
  container.querySelector('#val_hab_mult').innerText = parseFloat(c_hab_mult.value).toFixed(1) + "×";
  container.querySelector('#val_food').innerText = parseFloat(c_food.value).toFixed(1);
  container.querySelector('#val_k').innerText = c_k.value;
  container.querySelector('#val_fish_start').innerText = c_fstart.value;
  container.querySelector('#val_fish_end').innerText = c_fend.value;
  container.querySelector('#val_fish_rate').innerText = parseFloat(c_frate.value).toFixed(4);
  container.querySelector('#val_legal_size').innerText = parseFloat(c_lsize.value).toFixed(1) + " mm";

  container.value = getSlidersState();
  container.dispatchEvent(new CustomEvent('input', { bubbles: true, composed: true }));
  if (s.transportContainer) {
    s.transportContainer.value = getSlidersState();
    s.transportContainer.dispatchEvent(new CustomEvent('input', { bubbles: true, composed: true }));
  }
}

[c_grid, c_toffset, c_tmult, c_tnoise, c_hab_offset, c_hab_mult, c_food, c_k, c_fstart, c_fend, c_frate, c_lsize].forEach(el => {
  if (el) el.addEventListener('input', updateSliders);
});
[c_layer, t_energetics, t_thermal, t_cannibalism, t_senescence, t_movement, t_reproduction].forEach(el => {
  el.addEventListener('change', updateSliders);
});

container.value = getSlidersState();
</script>
</div>
"""
end

"""
    simulation_transport_widget(; default_interval = 10, default_years = 30) -> String

Construct the standalone HTML and embedded JavaScript string for the transport controls
(Start, Step, Pause, Stop/Reset, Rewind, Update Interval, and Duration in Years).
Positioned below Model Initialization to steer the simulation execution loop.
"""
function simulation_transport_widget(; default_interval::Int = 10, default_years::Int = 30)
    return """
<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px 18px; margin: 12px 0; max-width: 820px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
    <span style="font-weight: 700; font-size: 14px; color: #0f172a;">Simulation Execution & Sequence Controls</span>
    <span id="status_pill" style="font-size: 12px; font-weight: 600; padding: 3px 10px; border-radius: 12px; background: #e2e8f0; color: #475569;">
      Ready (Pending Start)
    </span>
  </div>
  
  <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
    <button type="button" id="btn_start" style="background: #16a34a; color: white; border: none; border-radius: 4px; padding: 6px 14px; font-weight: 600; cursor: pointer; font-size: 13px;">
      ▶ Start
    </button>
    <button type="button" id="btn_step" style="background: #0284c7; color: white; border: none; border-radius: 4px; padding: 6px 14px; font-weight: 600; cursor: pointer; font-size: 13px;">
      ▶❚ Step
    </button>
    <button type="button" id="btn_pause" style="background: #d97706; color: white; border: none; border-radius: 4px; padding: 6px 14px; font-weight: 600; cursor: pointer; font-size: 13px;">
      ⏸ Pause
    </button>
    <button type="button" id="btn_stop" style="background: #dc2626; color: white; border: none; border-radius: 4px; padding: 6px 14px; font-weight: 600; cursor: pointer; font-size: 13px;">
      ⏹ Stop / Reset
    </button>
    
    <div style="display: flex; align-items: center; gap: 6px; margin-left: 8px; font-size: 12px; color: #334155;">
      <span>Rewind / Target Day:</span>
      <input type="number" id="ctrl_rewind_day" min="0" max="36500" value="0" style="width: 70px; padding: 3px 6px; border: 1px solid #cbd5e1; border-radius: 4px; font-family: monospace;">
      <button type="button" id="btn_rewind" style="background: #3b82f6; color: white; border: none; border-radius: 4px; padding: 4px 10px; font-weight: 500; cursor: pointer; font-size: 12px;">
        Rewind
      </button>
    </div>

    <div style="display: flex; align-items: center; gap: 6px; margin-left: 8px; font-size: 12px; color: #334155;">
      <span>Update Step (days):</span>
      <input type="number" id="ctrl_update_interval" min="1" max="180" value="$(default_interval)" style="width: 55px; padding: 3px 6px; border: 1px solid #cbd5e1; border-radius: 4px; font-family: monospace;">
    </div>

    <div style="display: flex; align-items: center; gap: 6px; margin-left: 8px; font-size: 12px; color: #334155;">
      <span>Duration (Years):</span>
      <input type="number" id="ctrl_years" min="1" max="100" step="1" value="$(default_years)" style="width: 55px; padding: 3px 6px; border: 1px solid #cbd5e1; border-radius: 4px; font-family: monospace;">
    </div>
  </div>

<script>
const container = currentScript.parentElement;

const c_years = container.querySelector('#ctrl_years');
const inp_interval = container.querySelector('#ctrl_update_interval');
const btn_start = container.querySelector('#btn_start');
const btn_step = container.querySelector('#btn_step');
const btn_pause = container.querySelector('#btn_pause');
const btn_stop = container.querySelector('#btn_stop');
const btn_rewind = container.querySelector('#btn_rewind');
const inp_rewind = container.querySelector('#ctrl_rewind_day');
const status_pill = container.querySelector('#status_pill');

if (!window._geoibm_state) {
  window._geoibm_state = {
    simState: "stopped",
    currentSimDay: 0,
    timer: null,
    tick: 0,
    update_interval: $(default_interval),
    sim_years: $(default_years),
    activeContainer: null,
    slidersContainer: null,
    transportContainer: null
  };
}

let s = window._geoibm_state;
s.transportContainer = container;

if (s.currentSimDay > 0) {
  inp_rewind.value = s.currentSimDay;
}

function getTransportState() {
  const years = Math.max(1, parseInt(c_years.value) || $(default_years));
  const interval = Math.max(1, parseInt(inp_interval.value) || $(default_interval));
  s.sim_years = years;
  s.update_interval = interval;
  
  let baseState = {};
  if (s.slidersContainer && s.slidersContainer.value) {
    baseState = s.slidersContainer.value;
  }
  return Object.assign({}, baseState, {
    sim_status: s.simState,
    current_day: s.currentSimDay,
    tick: ++s.tick,
    update_interval: interval,
    sim_years: years
  });
}

function updateBadge() {
  if (!status_pill) return;
  if (s.simState === "running") {
    status_pill.innerText = `Running (Day \${s.currentSimDay})`;
    status_pill.style.background = "#dcfce7";
    status_pill.style.color = "#15803d";
  } else if (s.simState === "paused") {
    status_pill.innerText = `Paused at Day \${s.currentSimDay}`;
    status_pill.style.background = "#fef3c7";
    status_pill.style.color = "#b45309";
  } else {
    status_pill.innerText = `Stopped / Ready (Day \${s.currentSimDay})`;
    status_pill.style.background = "#f1f5f9";
    status_pill.style.color = "#475569";
  }
}

function dispatchTransportChange() {
  const state = getTransportState();
  container.value = state;
  container.dispatchEvent(new CustomEvent('input', { bubbles: true, composed: true }));
  if (s.slidersContainer) {
    s.slidersContainer.value = state;
    s.slidersContainer.dispatchEvent(new CustomEvent('input', { bubbles: true, composed: true }));
  }
}

function clearTimer() {
  if (s.timer !== null) {
    clearInterval(s.timer);
    s.timer = null;
  }
}

function advanceStep(force = false) {
  const maxDay = (parseInt(c_years.value) || $(default_years)) * 365;
  const interval = Math.max(1, parseInt(inp_interval.value) || $(default_interval));
  if (s.currentSimDay >= maxDay && !force) {
    s.simState = "paused";
    clearTimer();
    updateBadge();
    dispatchTransportChange();
    return;
  }
  s.currentSimDay = Math.min(maxDay, s.currentSimDay + interval);
  inp_rewind.value = s.currentSimDay;
  updateBadge();
  dispatchTransportChange();
}

btn_start.onclick = () => {
  s.simState = "running";
  updateBadge();
  clearTimer();
  advanceStep();
  s.timer = setInterval(() => {
    if (s.simState === "running") {
      advanceStep();
    } else {
      clearTimer();
    }
  }, 1000);
};

function pauseSimulation() {
  s.simState = "paused";
  clearTimer();
  updateBadge();
  dispatchTransportChange();
}

s.pauseSimulation = pauseSimulation;

btn_step.onclick = () => {
  clearTimer();
  s.simState = "paused";
  advanceStep(true);
};

btn_pause.onclick = () => {
  pauseSimulation();
};

btn_stop.onclick = () => {
  s.simState = "stopped";
  clearTimer();
  s.currentSimDay = 0;
  inp_rewind.value = 0;
  updateBadge();
  dispatchTransportChange();
};

btn_rewind.onclick = () => {
  clearTimer();
  const target = Math.max(0, parseInt(inp_rewind.value) || 0);
  s.currentSimDay = target;
  s.simState = target === 0 ? "stopped" : "paused";
  updateBadge();
  dispatchTransportChange();
};

c_years.addEventListener('change', () => {
  s.sim_years = Math.max(1, parseInt(c_years.value) || $(default_years));
  dispatchTransportChange();
});

inp_interval.addEventListener('change', () => {
  s.update_interval = Math.max(1, parseInt(inp_interval.value) || $(default_interval));
  dispatchTransportChange();
});

container.value = getTransportState();
updateBadge();
</script>
</div>
"""
end

"""
    steering_controls_widget(; default_interval = 10, default_years = 30) -> String

Backward-compatible combined steering control panel combining execution bar
and parameter sliders. Defaults to 10 days step and 30 years duration.
"""
function steering_controls_widget(; default_interval::Int = 10, default_years::Int = 30)
    return """
<div>
  $(simulation_transport_widget(default_interval = default_interval, default_years = default_years))
  $(parameter_steering_sliders_widget(default_interval = default_interval, default_years = default_years))
</div>
"""
end


# ------------------------------------------------------------------------------
# 2. Makie Dashboard Plotting Functions (Shared Visual Style)
# ------------------------------------------------------------------------------

"""
    plot_bottom_temperature_cycle(lon::Real, lat::Real;
                                  temp_offset::Real = 0.0,
                                  temp_multiplier::Real = 1.0,
                                  phase_day::Real = 80.0) -> Figure

Render the annual bottom temperature cycle (°C) for a chosen geographic location
(longitude, latitude), accounting for depth-attenuated seasonal amplitude,
optional climatological offset, and seasonal amplitude multiplier.
"""
function plot_bottom_temperature_cycle(
    lon::Real,
    lat::Real;
    temp_offset::Real = 0.0,
    temp_multiplier::Real = 1.0,
    phase_day::Real = 80.0
)
    # Regional shelf bathymetric and thermal climatology formulation
    depth = max(25.0, 180.0 + 80.0 * sin(0.8 * lon) + 60.0 * cos(1.1 * lat))
    mean_t = 2.0 + 2.5 * exp(-((depth - 150.0)/60.0)^2) + 0.3 * (45.0 - lat) + temp_offset
    
    # Seasonal amplitude decays with water depth (shallower banks experience larger seasonal swings)
    base_amp = max(0.5, 5.0 * exp(-depth / 100.0))
    amp = base_amp * temp_multiplier

    days = 1:365
    temps = [mean_t + amp * sin(2.0 * π * (d - phase_day) / 365.0) for d in days]

    fig = Figure(size = (740, 260))
    ax = Axis(
        fig[1, 1],
        title = @sprintf("Annual Bottom Temperature Cycle (%.2f°N, %.2f°W | Depth: %.0f m)",
                         lat, -lon, depth),
        xlabel = "Day of Year",
        ylabel = "Bottom Temp (°C)",
        xticks = (
            [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        )
    )
    
    # Highlight optimal snowcrab thermal envelope (0 to 4 °C)
    hspan!(ax, 0.0, 4.0; color = (:teal, 0.12), label = "Preferred Habitat Envelope (0–4°C)")
    hlines!(ax, [mean_t]; color = :gray50, linestyle = :dash, label = @sprintf("Mean: %.1f°C", mean_t))
    lines!(ax, collect(days), temps; color = :crimson, linewidth = 2.5, label = "Seasonal Cycle")

    axislegend(ax, position = :rt, framevisible = true, backgroundcolor = (:white, 0.85))
    return fig
end

"""
    plot_dashboard_spatial(model, params; sim_days=0, horizon_days=730, sim_status="STOPPED", background_layer=:habitat_quality) -> Figure

Render the spatial habitat raster (Habitat Quality HSI with `:viridis` colormap, bottom temperature
with `:thermal`, or seabed depth with `:deep`) overlaid with organism particles colored via `organism_color` and sized via `organism_size`.
"""
function plot_dashboard_spatial(
    model,
    params;
    sim_days::Real = 0,
    horizon_days::Real = 730,
    sim_status::AbstractString = "STOPPED",
    background_layer::Symbol = :habitat_quality
)
    day_i = round(Int, sim_days)
    yr_f = round(sim_days / 365.0, digits = 1)
    sub_title = "Day $(day_i) (Yr $(yr_f)) / $(round(Int, horizon_days)) [$(uppercase(sim_status))]"

    grid_dim = size(model.temperature_field)
    nx, ny = grid_dim[1], grid_dim[2]

    fig = Figure(size = (740, 560))
    ax = Axis(
        fig[1, 1],
        title = "Spatial Habitat & Organism Distribution (Day $(day_i) / Year $(yr_f))\n$(sub_title)",
        xlabel = "Grid X (cells)",
        ylabel = "Grid Y (cells)",
        limits = (0.5, nx + 0.5, 0.5, ny + 0.5),
        autolimitaspect = 1.0,
        xticks = LinearTicks(5),
        yticks = LinearTicks(5)
    )

    env_layer, colormap, cb_label = if background_layer === :habitat_quality &&
                                       haskey(abmproperties(model), :habitat_quality_field)
        (model.habitat_quality_field, :viridis, "Habitat Quality (HSI)")
    elseif background_layer === :depth &&
           haskey(abmproperties(model), :depth_field)
        (model.depth_field, :deep, "Seabed Depth (m)")
    else
        (model.temperature_field, :thermal, "Bottom Temperature (°C)")
    end

    living_orgs = collect(allagents(model))
    xs = Float32[Float32(a.pos[1]) for a in living_orgs]
    ys = Float32[Float32(a.pos[2]) for a in living_orgs]
    szs = Float32[Float32(max(4.0, a.size * 0.15)) for a in living_orgs]
    cols = String[organism_color(a, params) for a in living_orgs]

    hm = setup_spatial_panel!(ax, env_layer, xs, ys, szs, cols; colormap = colormap)
    Colorbar(fig[1, 2], hm, label = cb_label, ticks = LinearTicks(4))

    if isempty(living_orgs)
        Label(fig[1, 1], "No living organisms present",
              color = :gray, halign = :center, valign = :center)
    end
    return fig
end

"""
    plot_dashboard_abundance(agent_df; sim_days=0) -> Figure

Render logarithmic population trajectories for Total, Mature Females, Mature Males,
and Legal Fishable Males matching `ax_ts_n` in the native Makie GUI.
"""
function plot_dashboard_abundance(agent_df::DataFrame; sim_days::Real = 0)
    day_i = round(Int, sim_days)
    yr_f = round(sim_days / 365.0, digits = 1)
    fig = Figure(size = (780, 360))
    ax = Axis(
        fig[1, 1],
        title = "Abundance Trajectories (Day $(day_i) / Year $(yr_f))",
        xlabel = "Time (days)",
        ylabel = "log10(count)"
    )
    
    if nrow(agent_df) > 0
        t_hist = agent_df.time
        n_tot = Float64.(agent_df.length_stage)
        n_mf = (:count_is_mature_female in propertynames(agent_df)) ?
            Float64.(agent_df.count_is_mature_female) : zeros(Float64, nrow(agent_df))
        n_mm = (:count_is_mature_male in propertynames(agent_df)) ?
            Float64.(agent_df.count_is_mature_male) : zeros(Float64, nrow(agent_df))
        n_leg = (:count_is_legal_male in propertynames(agent_df)) ?
            Float64.(agent_df.count_is_legal_male) : zeros(Float64, nrow(agent_df))
        
        pts_tot = [Point2f(Float32(t_hist[i]), Float32(log10_count(n_tot[i]))) for i in eachindex(t_hist)]
        pts_mf  = [Point2f(Float32(t_hist[i]), Float32(log10_count(n_mf[i]))) for i in eachindex(t_hist)]
        pts_mm  = [Point2f(Float32(t_hist[i]), Float32(log10_count(n_mm[i]))) for i in eachindex(t_hist)]
        pts_leg = [Point2f(Float32(t_hist[i]), Float32(log10_count(n_leg[i]))) for i in eachindex(t_hist)]
        
        setup_abundance_panel!(ax, pts_tot, pts_mf, pts_mm, pts_leg)
    else
        Label(fig[1, 1], "Awaiting simulation progression (Day $(day_i))",
              color = :gray50, halign = :center, valign = :center)
    end
    return fig
end

"""
    plot_dashboard_stages(model, params; sim_days=0) -> Figure

Render instar stage abundance lollipop stems (`stem!`), split 50/50 for unsexed
egg/aggregate stages and by real counts for individuated stages matching `ax_stage` in `gui.jl`.
"""
function plot_dashboard_stages(model, params; sim_days::Real = 0)
    day_i = round(Int, sim_days)
    yr_f = round(sim_days / 365.0, digits = 1)
    living_pop = collect(allagents(model))
    
    n_agg = params.organism.juvenile_max_stage - 1
    n_male_st = params.organism.max_stage_male - params.organism.juvenile_max_stage + 1
    n_fem_st = params.organism.max_stage_female - params.organism.juvenile_max_stage + 1
    max_stage = params.organism.max_stage_male
    
    larval_cnt = larval_stage_count(SpeciesBehaviorImpl())
    st_labels = [cohort_stage_label(s, params, larval_cnt) for s in 0:max_stage]
    first_agent_cat = 2 + n_agg
    n_cats = length(st_labels)
    
    fig = Figure(size = (800, 380))
    ax = Axis(
        fig[1, 1],
        title = "Stage Abundance (Day $(day_i) / Year $(yr_f))",
        xlabel = "Stage",
        ylabel = "log10(number)",
        xticks = (collect(1:n_cats), st_labels),
        xticklabelrotation = pi / 2
    )
    
    stage_base = params.organism.juvenile_max_stage
    st_male = zeros(Float64, n_male_st)
    st_fem = zeros(Float64, n_fem_st)
    local n_eggs_brooded = 0.0
    for o in living_pop
        k = o.stage - stage_base + 1
        if o.sex === male
            1 <= k <= length(st_male) && (st_male[k] += 1.0)
        elseif o.sex === female
            1 <= k <= length(st_fem) && (st_fem[k] += 1.0)
        end
        n_eggs_brooded += Float64(o.egg_count)
    end
    
    st_agg = zeros(Float64, n_agg)
    agg_pop = get(abmproperties(model), :aggregate_populations, nothing)
    if agg_pop !== nothing
        for g in agg_pop.groups
            k = g.stage
            1 <= k <= length(st_agg) && (st_agg[k] += Float64(g.abundance))
        end
    end
    
    half_w = 0.21
    y_egg_half = log10_count(n_eggs_brooded / 2.0)
    y_agg_half = [log10_count(x / 2.0) for x in st_agg]
    y_male = [log10_count(x) for x in st_male]
    y_fem = [log10_count(x) for x in st_fem]
    
    base_agg = Float64.(2:(1 + n_agg))
    base_agent = Float64.(first_agent_cat:(first_agent_cat + max(n_male_st, n_fem_st) - 1))
    
    setup_stages_panel!(
        ax, half_w, base_agg, base_agent, n_male_st, n_fem_st,
        [y_egg_half], y_agg_half, y_male, y_fem
    )
    return fig
end

"""
    plot_dashboard_size_density(model, params; sim_days=0) -> Figure

Render probability densities of carapace width for immature and mature components of both
sexes using Gaussian kernel estimation (`kde_density`, `SIZE_DENSITY_BANDWIDTH_MM = 3.0`)
alongside the legal size reference line.
"""
function plot_dashboard_size_density(model, params; sim_days::Real = 0)
    day_i = round(Int, sim_days)
    yr_f = round(sim_days / 365.0, digits = 1)
    fig = Figure(size = (780, 380))
    ax = Axis(
        fig[1, 1],
        title = "Size Frequency by Sex and Maturity (Day $(day_i) / Year $(yr_f))",
        xlabel = "Size (mm)",
        ylabel = "Density",
        limits = (0.0, 165.0, nothing, nothing)
    )
    
    orgs = collect(allagents(model))
    size_grid = collect(0.0:0.5:165.0)
    
    cw_im = Float64[Float64(a.size) for a in orgs if a.sex === male && !a.is_mature]
    cw_mm = Float64[Float64(a.size) for a in orgs if a.sex === male && a.is_mature]
    cw_if = Float64[Float64(a.size) for a in orgs if a.sex === female && !a.is_mature]
    cw_mf = Float64[Float64(a.size) for a in orgs if a.sex === female && a.is_mature]
    
    dens_im = kde_density(cw_im, size_grid; bandwidth = SIZE_DENSITY_BANDWIDTH_MM, scale = :probability)
    dens_mm = kde_density(cw_mm, size_grid; bandwidth = SIZE_DENSITY_BANDWIDTH_MM, scale = :probability)
    dens_if = kde_density(cw_if, size_grid; bandwidth = SIZE_DENSITY_BANDWIDTH_MM, scale = :probability)
    dens_mf = kde_density(cw_mf, size_grid; bandwidth = SIZE_DENSITY_BANDWIDTH_MM, scale = :probability)
    
    pts_im = [Point2f(Float32(size_grid[i]), Float32(dens_im[i])) for i in eachindex(size_grid)]
    pts_mm = [Point2f(Float32(size_grid[i]), Float32(dens_mm[i])) for i in eachindex(size_grid)]
    pts_if = [Point2f(Float32(size_grid[i]), Float32(dens_if[i])) for i in eachindex(size_grid)]
    pts_mf = [Point2f(Float32(size_grid[i]), Float32(dens_mf[i])) for i in eachindex(size_grid)]
    
    setup_size_density_panel!(
        ax, pts_im, pts_mm, pts_if, pts_mf, params.fishery.legal_size_male
    )
    return fig
end

"""
    plot_dashboard_ogive(model, params; sim_days=0) -> Figure

Render theoretical maturity ogives (`maturity`), fishery retention selectivity (`selectivity`),
and realized cross-sectional maturity proportions from living agents.
"""
function plot_dashboard_ogive(model, params; sim_days::Real = 0)
    day_i = round(Int, sim_days)
    yr_f = round(sim_days / 365.0, digits = 1)
    fig = Figure(size = (800, 380))
    ax = Axis(
        fig[1, 1],
        title = "Maturity Ogive & Fishery Selectivity (Day $(day_i) / Year $(yr_f))",
        xlabel = "Size (mm)",
        ylabel = "P(Mature) / Selectivity",
        limits = (0.0, 165.0, 0.0, 1.05)
    )
    
    living_now = collect(allagents(model))
    bins = 20.0:10.0:150.0
    n_bins = length(bins) - 1
    m_pts = Point2f[]
    f_pts = Point2f[]
    for k in 1:n_bins
        b_low = bins[k]
        b_high = bins[k + 1]
        mid = (b_low + b_high) / 2.0
        m_in_bin = filter(a -> a.sex === male && b_low <= a.size < b_high, living_now)
        if !isempty(m_in_bin)
            push!(m_pts, Point2f(mid, count(a -> a.is_mature, m_in_bin) / length(m_in_bin)))
        end
        f_in_bin = filter(a -> a.sex === female && b_low <= a.size < b_high, living_now)
        if !isempty(f_in_bin)
            push!(f_pts, Point2f(mid, count(a -> a.is_mature, f_in_bin) / length(f_in_bin)))
        end
    end
    
    setup_ogive_panel!(ax, params.fishery.legal_size_male, m_pts, f_pts)
    return fig
end

"""
    plot_dashboard_macro(agent_df, model_df, annual_df; sim_days=0) -> Figure

Render legal fishable biomass and annual catch time series overlaid with calibrated
macro-ecological models (Schaefer, Delay-Difference, Leslie) when annual data is available.
"""
function plot_dashboard_macro(
    agent_df::DataFrame,
    model_df::Union{DataFrame, Nothing},
    annual_df::DataFrame;
    sim_days::Real = 0
)
    day_i = round(Int, sim_days)
    yr_f = round(sim_days / 365.0, digits = 1)
    fig = Figure(size = (800, 380))
    ax = Axis(
        fig[1, 1],
        title = "Biomass, Catch & Macro-Models (Day $(day_i) / Year $(yr_f))",
        xlabel = "Simulation Time (days)",
        ylabel = "Biomass (kg)"
    )
    
    pts_b = Point2f[]
    if nrow(agent_df) > 0 && :sum_mass_if_legal_male in propertynames(agent_df)
        pts_b = [Point2f(Float32(agent_df.time[i]), Float32(agent_df.sum_mass_if_legal_male[i] / 1000.0))
                 for i in 1:nrow(agent_df)]
    end
    
    pts_c = Point2f[]
    if model_df !== nothing && nrow(model_df) > 0 && :total_catch_biomass_kg in propertynames(model_df)
        pts_c = [Point2f(Float32(model_df.time[i]), Float32(model_df.total_catch_biomass_kg[i]))
                 for i in 1:nrow(model_df)]
    end
    
    pts_obs = Point2f[]
    pts_sch = Point2f[]
    pts_dd  = Point2f[]
    pts_les = Point2f[]
    l_obs = "Annual Surveyed Biomass"
    l_sch = "Schaefer Fit"
    l_dd  = "Delay-Diff Fit"
    l_les = "Leslie Fit"
    
    if nrow(annual_df) >= 3
        try
            stage_mat, catch_mat, yrs = extract_annual_stage_matrix(annual_df)
            les_fit = fit_generalized_leslie_model(stage_mat, catch_mat; years = yrs)
            m_comp = compare_macro_models(annual_df, les_fit; lag = 1)
            
            times_ann = Float64.(annual_df.time)
            T_ann = length(times_ann)
            
            measured_b = (:fishable_male_biomass_kg in propertynames(annual_df)) ?
                Float64.(annual_df.fishable_male_biomass_kg) : Float64.(m_comp.biomass_obs_kg)
            pts_obs = [Point2f(Float32(times_ann[i]), Float32(measured_b[i])) for i in 1:T_ann]
            
            mw_kg = (:mean_weight_legal_kg in propertynames(annual_df)) ?
                Float64.(annual_df.mean_weight_legal_kg) : fill(m_comp.mean_component_weight_kg, T_ann)
            
            pts_sch = [Point2f(Float32(times_ann[i]), Float32(m_comp.numbers_schaefer[i] * mw_kg[i]))
                       for i in 1:T_ann]
            pts_dd  = [Point2f(Float32(times_ann[i]), Float32(m_comp.numbers_delay_diff[i] * mw_kg[i]))
                       for i in 1:T_ann]
            pts_les = [Point2f(Float32(times_ann[i]), Float32(m_comp.numbers_leslie[i] * mw_kg[i]))
                       for i in 1:T_ann]
            
            l_sch = m_comp.label_schaefer
            l_dd  = m_comp.label_delay_diff
            l_les = m_comp.label_leslie
        catch
        end
    end
    
    setup_macro_panel!(
        ax, pts_b, pts_c, pts_obs, pts_sch, pts_dd, pts_les;
        label_obs = l_obs, label_sch = l_sch, label_dd = l_dd, label_les = l_les,
        label_catch = "Cumulative Catch Biomass (kg)")
    return fig
end

"""
    plot_dashboard_stock_recruitment(annual_df; sim_days=0) -> Figure

Render log10 spawner-recruit pairs with parametric Ricker and Beverton-Holt fitted curves
matching `ax_sr` and `update_sr_fit!` in `gui.jl`.
"""
function plot_dashboard_stock_recruitment(annual_df::DataFrame; sim_days::Real = 0)
    day_i = round(Int, sim_days)
    yr_f = round(sim_days / 365.0, digits = 1)
    fig = Figure(size = (800, 420))
    ax = Axis(
        fig[1, 1],
        title = "Stock-Recruitment: Male Component (Day $(day_i) / Year $(yr_f))",
        xlabel = "log10(Annual Eggs / 2)",
        ylabel = "log10(Males at Recruitment Stage)"
    )
    
    sr_pts = Point2f[]
    spawners_raw = Float64[]
    recruits_raw = Float64[]
    
    eggs_col = (:annual_egg_production in propertynames(annual_df)) ?
        :annual_egg_production : :total_egg_production
    rec_col = (:recruits_male in propertynames(annual_df)) ?
        :recruits_male : ((:recruits in propertynames(annual_df)) ? :recruits : nothing)
    
    if nrow(annual_df) >= 2 && rec_col !== nothing && eggs_col in propertynames(annual_df)
        for yr in 1:(nrow(annual_df) - 1)
            s_val = Float64(annual_df[!, eggs_col][yr]) / 2.0
            r_val = Float64(annual_df[!, rec_col][yr + 1])
            if s_val > 0.0 && r_val > 0.0
                push!(spawners_raw, s_val)
                push!(recruits_raw, r_val)
                push!(sr_pts, Point2f(Float32(log10_count(s_val)), Float32(log10_count(r_val))))
            end
        end
    end
    
    rk_pts = Point2f[]
    bh_pts = Point2f[]
    
    if !isempty(sr_pts)
        fit_sr = fit_stock_recruitment(spawners_raw, recruits_raw)
        if fit_sr.is_fitted
            s_lo, s_hi = extrema(spawners_raw)
            span = max(s_hi - s_lo, s_hi * 0.05, eps(Float64))
            grid_s = range(max(s_lo - 0.1 * span, 0.0), s_hi + 0.5 * span; length = 128)
            
            rk = fit_sr.ricker
            bh = fit_sr.beverton_holt
            
            rk_pts = [Point2f(Float32(log10_count(s)), Float32(log10_count(rk.predict(s))))
                      for s in grid_s if s > 0.0 && rk.predict(s) > 0.0]
            bh_pts = [Point2f(Float32(log10_count(s)), Float32(log10_count(bh.predict(s))))
                      for s in grid_s if s > 0.0 && bh.predict(s) > 0.0]
            
            a_r = round(rk.alpha; sigdigits = 3)
            b_r = round(rk.beta; sigdigits = 3)
            a_b = round(bh.alpha; sigdigits = 3)
            b_b = round(bh.beta; sigdigits = 8)
            
            fit_text = "Ricker: R = $(a_r)·S·exp(-$(b_r)·S)  (R²=$(round(rk.R2, digits=3)))\n" *
                       "Beverton–Holt: R = $(a_b)·S / (1 + $(b_b)·S)  (R²=$(round(bh.R2, digits=3)))\n" *
                       "Preferred: $(fit_sr.best_model === :ricker ? "Ricker" : "Beverton–Holt")"
            Label(fig[1, 1], fit_text, halign = :left, valign = :top,
                  fontsize = 9, color = :black, tellwidth = false, tellheight = false)
        end
        setup_stock_recruitment_panel!(ax, sr_pts, rk_pts, bh_pts)
    else
        Label(fig[1, 1], "Awaiting multi-year annual recruitment data (Day $(day_i))",
              color = :gray50, halign = :center, valign = :center)
    end
    return fig
end

"""
    plot_dashboard_environmental_distributions(model; sim_days::Real = 0) -> Figure

Render empirical frequency distributions (histograms with kernel density envelopes)
for seabed bathymetric depth, ambient bottom temperature, and realized Habitat
Suitability Index (HSI) across all grid cells at the current simulation step.
"""
function plot_dashboard_environmental_distributions(model; sim_days::Real = 0)
    day_i = round(Int, sim_days)
    yr_f = round(sim_days / 365.0, digits = 1)

    mprops = abmproperties(model)
    d_field = haskey(mprops, :depth_field) ? vec(model.depth_field) : Float64[]
    t_field = haskey(mprops, :temperature_field) ? vec(model.temperature_field) : Float64[]
    h_field = haskey(mprops, :habitat_quality_field) ? vec(model.habitat_quality_field) : Float64[]

    d_vals = filter(!isnan, d_field)
    t_vals = filter(!isnan, t_field)
    h_vals = filter(!isnan, h_field)

    fig = Figure(size = (840, 270))

    # Panel 1: Bathymetric Depth
    if !isempty(d_vals)
        d_min, d_max = extrema(d_vals)
        d_mean = mean(d_vals)
        d_span = d_max - d_min
        ax_d = Axis(
            fig[1, 1],
            title = @sprintf("Seabed Depth\n(μ = %.1f m, Δ = %.1f m)", d_mean, d_span),
            xlabel = "Depth (m)",
            ylabel = "Cell Frequency",
            xticks = LinearTicks(4)
        )
        if d_span < 1e-4
            vlines!(ax_d, [d_min], color = :navy, linewidth = 2.5)
            xlims!(ax_d, max(0.0, d_min - 5.0), d_min + 5.0)
        else
            hist!(ax_d, d_vals, bins = 16, color = (:navy, 0.45), strokecolor = :navy, strokewidth = 1.0)
            pad_d = max(0.5, 0.08 * d_span)
            xlims!(ax_d, max(0.0, d_min - pad_d), d_max + pad_d)
        end
    else
        ax_d = Axis(fig[1, 1], title = "Seabed Depth", xlabel = "Depth (m)", ylabel = "Cell Frequency")
        Label(fig[1, 1], "No Depth Data", color = :gray, halign = :center, valign = :center)
    end

    # Panel 2: Ambient Bottom Temperature
    if !isempty(t_vals)
        t_min, t_max = extrema(t_vals)
        t_mean = mean(t_vals)
        t_span = t_max - t_min
        ax_t = Axis(
            fig[1, 2],
            title = @sprintf("Bottom Temperature\n(μ = %.2f °C, Δ = %.2f °C)", t_mean, t_span),
            xlabel = "Temperature (°C)",
            ylabel = "Cell Frequency",
            xticks = LinearTicks(4)
        )
        if t_span < 1e-4
            vlines!(ax_t, [t_min], color = :firebrick, linewidth = 2.5)
            xlims!(ax_t, t_min - 0.5, t_min + 0.5)
        else
            hist!(ax_t, t_vals, bins = 16, color = (:firebrick, 0.45), strokecolor = :firebrick, strokewidth = 1.0)
            pad_t = max(0.05, 0.08 * t_span)
            xlims!(ax_t, t_min - pad_t, t_max + pad_t)
        end
    else
        ax_t = Axis(fig[1, 2], title = "Bottom Temperature", xlabel = "Temperature (°C)", ylabel = "Cell Frequency")
        Label(fig[1, 2], "No Temperature Data", color = :gray, halign = :center, valign = :center)
    end

    # Panel 3: Habitat Suitability Index (HSI)
    if !isempty(h_vals)
        h_min, h_max = extrema(h_vals)
        h_mean = mean(h_vals)
        h_span = h_max - h_min
        ax_h = Axis(
            fig[1, 3],
            title = @sprintf("Habitat Suitability (HSI)\n(μ = %.3f, range [%.3f, %.3f])", h_mean, h_min, h_max),
            xlabel = "HSI [0, 1]",
            ylabel = "Cell Frequency",
            limits = (-0.02, 1.02, nothing, nothing),
            xticks = LinearTicks(4)
        )
        if h_span < 1e-4
            vlines!(ax_h, [h_min], color = :teal, linewidth = 2.5)
        else
            # Ensure bins resolve fine structure across realized HSI span
            nbins = clamp(round(Int, 25 * max(0.2, h_span)), 15, 25)
            hist!(ax_h, h_vals, bins = nbins,
                  color = (:teal, 0.45), strokecolor = :teal, strokewidth = 1.0)
            pad_h = max(0.02, 0.08 * h_span)
            xlims!(ax_h, max(0.0, h_min - pad_h), min(1.0, h_max + pad_h))
        end
    else
        ax_h = Axis(fig[1, 3], title = "Habitat Suitability (HSI)", xlabel = "HSI [0, 1]", ylabel = "Cell Frequency")
        Label(fig[1, 3], "No HSI Data", color = :gray, halign = :center, valign = :center)
    end

    return fig
end
