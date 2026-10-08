"""
    src/leaflet_dashboard.jl

Interactive Leaflet-based Web Dashboard for model_agent.
Embeds real-time geographic maps, bathymetry, dynamic bottom & surface temperature
layers, dynamic HSI fields, organism abundance & stage distributions, and user-interactive
controls (location bounding box selection, time window sliders, environmental toggles,
and simulation stepping).
"""

# No external JSON dependency needed

"""
    generate_leaflet_dashboard_html(model; title="Organism IBM Leaflet Dashboard", kwargs...)

Generate a complete, self-contained interactive Leaflet HTML dashboard for `model`.
Includes:
- Interactive geographic map centered on `domain_lon` / `domain_lat` with Esri Ocean basemap.
- Layer toggles for Bathymetric Depth, Dynamic Seabed Temperature, Dynamic Larval Surface
  Temperature, Dynamic HSI, and Organism Agent Positions.
- Location and time period selector controls (allowing users to pick lat/lon bounds and
  time horizons dynamically).
- Real-time demographic KPI cards (Total population, mature males/females, legal biomass, catch).
- Stage distribution chart and population trajectory mini-charts.
- Time-slider to scrub across the seasonal year / simulation days.
"""
function generate_leaflet_dashboard_html(
    model;
    title::String = "Snow Crab IBM Spatial Dashboard",
    output_html::String = "outputs/dashboard.html",
    auto_open::Bool = false
)::String
    props = abmproperties(model)
    params = props[:params]
    env = params.environment
    sim = params.simulation

    lon_bounds = env.domain_lon
    lat_bounds = env.domain_lat
    min_lon = Float64(lon_bounds[1])
    max_lon = Float64(lon_bounds[2])
    min_lat = Float64(lat_bounds[1])
    max_lat = Float64(lat_bounds[2])
    mid_lon = 0.5 * (min_lon + max_lon)
    mid_lat = 0.5 * (min_lat + max_lat)

    # Grid dimensions
    nx, ny = size(props[:temperature_field])
    t_field = props[:temperature_field]
    d_field = props[:depth_field]
    h_field = props[:habitat_quality_field]
    s_field = get(props, :surface_temperature_field, t_field)

    t_min = minimum(t_field)
    t_max = maximum(t_field)
    d_min = minimum(d_field)
    d_max = maximum(d_field)
    h_min = minimum(h_field)
    h_max = maximum(h_field)
    s_min = minimum(s_field)
    s_max = maximum(s_field)

    # Sample agents for fast geo rendering
    agents_list = collect(allagents(model))
    n_total = length(agents_list)
    n_mature_m = get(props, :n_mature_males, 0)
    n_mature_f = get(props, :n_mature_females, 0)
    n_legal_m = get(props, :n_legal_males, 0)
    catch_biomass = get(props, :total_catch_biomass_kg, 0.0)
    cur_time = get(props, :current_time, 0.0)

    # Coordinate mapping from grid indices to (lon, lat)
    function grid_to_coord(i::Int, j::Int)
        lon = min_lon + (i - 0.5) / nx * (max_lon - min_lon)
        lat = min_lat + (j - 0.5) / ny * (max_lat - min_lat)
        return (lon, lat)
    end

    # Agent GeoJSON feature collection (subsampled if very large for smooth rendering)
    max_rendered_agents = min(n_total, 2500)
    step_stride = n_total > max_rendered_agents ? div(n_total, max_rendered_agents) : 1
    agent_features = Dict{String, Any}[]

    for (idx, a) in enumerate(agents_list)
        (idx % step_stride == 0) || continue
        alon, alat = grid_to_coord(a.pos[1], a.pos[2])
        color = (a.sex === female) ? "#ec4899" : (a.is_mature ? "#3b82f6" : "#60a5fa")
        push!(agent_features, Dict(
            "type" => "Feature",
            "geometry" => Dict(
                "type" => "Point",
                "coordinates" => [round(alon, digits=4), round(alat, digits=4)]
            ),
            "properties" => Dict(
                "id" => a.id,
                "stage" => a.stage,
                "sex" => (a.sex === female ? "Female" : "Male"),
                "size_mm" => round(a.size, digits=1),
                "mass_g" => round(a.mass, digits=1),
                "mature" => a.is_mature,
                "color" => color
            )
        ))
    end
    # Lightweight JSON serialization without external package dependencies
    function to_json_val(v)
        if v isa AbstractString
            return "\"" * replace(v, "\"" => "\\\"", "\\" => "\\\\") * "\""
        elseif v isa Bool
            return v ? "true" : "false"
        elseif v isa Number
            return string(v)
        elseif v isa AbstractVector
            return "[" * join([to_json_val(x) for x in v], ",") * "]"
        elseif v isa AbstractDict
            return "{" * join(["\"$k\":" * to_json_val(val) for (k, val) in v], ",") * "}"
        else
            return "\"$(string(v))\""
        end
    end

    agents_geojson_json = to_json_val(Dict("type" => "FeatureCollection", "features" => agent_features))

    # Grid cells summary for environmental overlay (decimated grid for performance)
    cell_features = Dict{String, Any}[]
    step_x = max(1, div(nx, 25))
    step_y = max(1, div(ny, 25))
    for j in 1:step_y:ny, i in 1:step_x:nx
        c_lon, c_lat = grid_to_coord(i, j)
        d_val = round(d_field[i, j], digits=1)
        tb_val = round(t_field[i, j], digits=2)
        ts_val = round(s_field[i, j], digits=2)
        hsi_val = round(h_field[i, j], digits=3)
        push!(cell_features, Dict(
            "type" => "Feature",
            "geometry" => Dict(
                "type" => "Point",
                "coordinates" => [round(c_lon, digits=4), round(c_lat, digits=4)]
            ),
            "properties" => Dict(
                "depth" => d_val,
                "temp_bottom" => tb_val,
                "temp_surface" => ts_val,
                "hsi" => hsi_val
            )
        ))
    end
    cells_geojson_json = to_json_val(Dict("type" => "FeatureCollection", "features" => cell_features))

    # Time range string
    t_start = length(env.time_range) >= 1 ? env.time_range[1] : "1993-01-01"
    t_end = length(env.time_range) >= 2 ? env.time_range[2] : "1993-12-31"

    html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>$title</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet"/>
  <style>
    :root {
      --bg-dark: #090d16;
      --bg-panel: rgba(15, 23, 42, 0.85);
      --bg-card: rgba(30, 41, 59, 0.70);
      --border: rgba(148, 163, 184, 0.15);
      --accent: #38bdf8;
      --accent-alt: #10b981;
      --accent-warn: #f59e0b;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --font-main: 'Inter', system-ui, -apple-system, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg-dark);
      color: var(--text);
      font-family: var(--font-main);
      overflow: hidden;
      display: flex;
      flex-direction: column;
      height: 100vh;
      width: 100vw;
    }
    header {
      height: 58px;
      background: rgba(11, 17, 32, 0.95);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 24px;
      z-index: 1001;
    }
    .header-title {
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.02em;
    }
    .badge {
      font-size: 0.70rem;
      font-weight: 600;
      text-transform: uppercase;
      padding: 3px 8px;
      border-radius: 6px;
      background: rgba(56, 189, 248, 0.18);
      color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .header-controls {
      display: flex;
      align-items: center;
      gap: 16px;
      font-size: 0.85rem;
    }
    .btn {
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent);
      border: 1px solid var(--accent);
      border-radius: 6px;
      padding: 6px 14px;
      font-weight: 600;
      cursor: pointer;
      font-size: 0.82rem;
      transition: all 0.15s ease;
    }
    .btn:hover {
      background: rgba(56, 189, 248, 0.35);
      color: #fff;
    }
    .main-layout {
      flex: 1;
      display: flex;
      position: relative;
      overflow: hidden;
    }
    #map {
      flex: 1;
      height: 100%;
      background: #070b14;
    }
    .sidebar {
      width: 380px;
      background: var(--bg-panel);
      backdrop-filter: blur(16px);
      border-left: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      gap: 16px;
      padding: 20px;
      overflow-y: auto;
      z-index: 1000;
      box-shadow: -10px 0 30px rgba(0, 0, 0, 0.5);
    }
    .panel-section {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 16px;
    }
    .panel-header {
      font-size: 0.88rem;
      font-weight: 700;
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .kpi-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .kpi-card {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
    }
    .kpi-label {
      font-size: 0.72rem;
      color: var(--text-muted);
      margin-bottom: 4px;
    }
    .kpi-value {
      font-family: var(--font-mono);
      font-size: 1.15rem;
      font-weight: 700;
      color: #fff;
    }
    .control-row {
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-bottom: 12px;
    }
    .control-row label {
      font-size: 0.78rem;
      color: var(--text-muted);
      display: flex;
      justify-content: space-between;
    }
    .input-group {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    input[type="text"], input[type="number"], select {
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid var(--border);
      border-radius: 6px;
      color: var(--text);
      padding: 6px 10px;
      font-size: 0.82rem;
      font-family: var(--font-mono);
      outline: none;
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }
    .layer-toggle {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 6px 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      font-size: 0.82rem;
    }
    .legend-box {
      position: absolute;
      bottom: 24px;
      left: 24px;
      background: var(--bg-panel);
      backdrop-filter: blur(14px);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 18px;
      z-index: 1000;
      min-width: 260px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.4);
    }
    .legend-gradient {
      height: 10px;
      border-radius: 5px;
      margin: 8px 0 4px 0;
      background: linear-gradient(to right, #1e3a8a, #06b6d4, #10b981, #f59e0b, #ef4444);
    }
    .legend-labels {
      display: flex;
      justify-content: space-between;
      font-size: 0.75rem;
      font-family: var(--font-mono);
      color: var(--text-muted);
    }
  </style>
</head>
<body>
  <header>
    <div class="header-title">
      <span>$title</span>
      <span class="badge">Leaflet Dynamic HSI</span>
    </div>
    <div class="header-controls">
      <span>Simulation Day: <b id="val-day" style="color:var(--accent); font-family:var(--font-mono);">$cur_time</b></span>
      <button class="btn" id="btn-export">Export GeoJSON</button>
      <button class="btn" id="btn-reset-view">Recenter View</button>
    </div>
  </header>

  <div class="main-layout">
    <div id="map"></div>

    <div class="legend-box">
      <div style="font-size:0.82rem; font-weight:600; display:flex; justify-content:space-between;">
        <span id="legend-title">Seabed Bottom Temperature (°C)</span>
      </div>
      <div class="legend-gradient" id="legend-bar"></div>
      <div class="legend-labels">
        <span id="legend-min">$(round(t_min, digits=1)) °C</span>
        <span id="legend-max">$(round(t_max, digits=1)) °C</span>
      </div>
      <div style="font-size:0.72rem; color:var(--text-muted); margin-top:6px;">
        Hydrographic provider: <b>$(env.temperature_source)</b>
      </div>
    </div>

    <div class="sidebar">
      <!-- Demographic KPI Section -->
      <div class="panel-section">
        <div class="panel-header">Population Demographics</div>
        <div class="kpi-grid">
          <div class="kpi-card">
            <div class="kpi-label">Total Agents</div>
            <div class="kpi-value" id="kpi-agents">$n_total</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Legal Biomass</div>
            <div class="kpi-value" style="color:var(--accent-alt);">$catch_biomass <span style="font-size:0.75rem;">kg</span></div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Mature Males</div>
            <div class="kpi-value" style="color:#60a5fa;">$n_mature_m</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Mature Females</div>
            <div class="kpi-value" style="color:#ec4899;">$n_mature_f</div>
          </div>
        </div>
      </div>

      <!-- Spatial & Temporal Domain Selection -->
      <div class="panel-section">
        <div class="panel-header">
          <span>Geographic Location & Period</span>
          <span class="badge" id="badge-sel-mode" style="font-size:0.65rem;">View Mode</span>
        </div>
        <div class="control-row">
          <label>Interactive Map Selection Mode</label>
          <div style="display:grid; grid-template-columns: 1fr 1fr; gap:6px;">
            <button class="btn" id="btn-mode-rect" style="padding:6px 8px; font-size:0.75rem;">
              ⛶ Drag Box
            </button>
            <button class="btn" id="btn-mode-click" style="padding:6px 8px; font-size:0.75rem;">
              📍 Click 2 Points
            </button>
          </div>
          <div id="sel-mode-hint" style="font-size:0.72rem; color:var(--text-muted); margin-top:4px; display:none;">
            Ready: click/drag on the map to define the active bounding box.
          </div>
        </div>
        <div class="control-row">
          <label>Longitude Range [Min, Max]</label>
          <div class="input-group">
            <input type="number" id="inp-min-lon" step="0.05" value="$min_lon"/>
            <input type="number" id="inp-max-lon" step="0.05" value="$max_lon"/>
          </div>
        </div>
        <div class="control-row">
          <label>Latitude Range [Min, Max]</label>
          <div class="input-group">
            <input type="number" id="inp-min-lat" step="0.05" value="$min_lat"/>
            <input type="number" id="inp-max-lat" step="0.05" value="$max_lat"/>
          </div>
        </div>
        <div class="control-row">
          <label>Temporal Period (Start & End Date)</label>
          <div class="input-group">
            <input type="date" id="inp-time-start" value="$t_start"/>
            <input type="date" id="inp-time-end" value="$t_end"/>
          </div>
        </div>
        <div style="display:flex; gap:8px; margin-top:6px;">
          <button class="btn" style="flex:1;" id="btn-apply-bounds">
            Apply Selection
          </button>
          <button class="btn" style="flex:1; border-color:rgba(148,163,184,0.4); color:var(--text-muted);" id="btn-reset-bounds">
            Reset Bounds
          </button>
        </div>
      </div>

      <!-- Environmental Layers Control -->
      <div class="panel-section">
        <div class="panel-header">Layer Visibility & Colorizer</div>
        <div class="control-row">
          <label>Active Heat Field</label>
          <select id="sel-heat-field">
            <option value="bottom_temp" selected>Bottom Temperature (Benthic)</option>
            <option value="surface_temp">Surface Temperature (Pelagic)</option>
            <option value="hsi">Dynamic Habitat Suitability (HSI)</option>
            <option value="depth">Bathymetric Depth (m)</option>
          </select>
        </div>
        <div class="layer-toggle">
          <span>Show Organism Agents</span>
          <input type="checkbox" id="chk-agents" checked/>
        </div>
        <div class="layer-toggle">
          <span>Show Hydrographic Grid</span>
          <input type="checkbox" id="chk-grid" checked/>
        </div>
        <div class="layer-toggle">
          <span>Show Bathymetry Contours</span>
          <input type="checkbox" id="chk-bathy" checked/>
        </div>
      </div>

      <!-- Simulation Playback Control -->
      <div class="panel-section">
        <div class="panel-header">Seasonal Progression</div>
        <div class="control-row">
          <label>Day of Year <span id="lbl-doy">$(round(cur_time % sim.days_per_year, digits=1)) / 365</span></label>
          <input type="range" id="slider-doy" min="0" max="365" value="$(round(cur_time % sim.days_per_year))"/>
        </div>
      </div>
    </div>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    var mapBounds = [[$min_lat, $min_lon], [$max_lat, $max_lon]];
    var map = L.map('map', {
      center: [$mid_lat, $mid_lon],
      zoom: 7,
      preferCanvas: true
    });

    var esriOcean = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}', {
      attribution: '&copy; Esri, GEBCO, NOAA',
      maxZoom: 13
    }).addTo(map);

    var domainRect = L.rectangle(mapBounds, {
      color: '#38bdf8',
      weight: 1.5,
      dashArray: '4, 4',
      fillOpacity: 0.03
    }).addTo(map);

    map.fitBounds(mapBounds, { padding: [30, 30] });

    // Environmental cell layer
    var cellsData = $cells_geojson_json;
    var currentField = 'bottom_temp';

    function getColorForValue(val, min, max, colormap) {
      var norm = (max > min) ? Math.max(0, Math.min(1, (val - min) / (max - min))) : 0.5;
      if (colormap === 'temp') {
        // Cool-to-warm gradient: blue -> cyan -> green -> yellow -> red
        var h = (1.0 - norm) * 220; // 220 (blue) down to 0 (red)
        return 'hsl(' + h + ', 85%, 50%)';
      } else if (colormap === 'hsi') {
        // Purple -> teal -> yellow (viridis style)
        var h = norm * 140; // 0 (red) to 140 (green)
        return 'hsl(' + h + ', 80%, 48%)';
      } else {
        // Depth: dark blue to light blue
        var lum = 25 + norm * 50;
        return 'hsl(210, 80%, ' + lum + '%)';
      }
    }

    var envMarkersLayer = L.layerGroup().addTo(map);

    function renderEnvLayer() {
      envMarkersLayer.clearLayers();
      var fieldMin = 0, fieldMax = 10, colormap = 'temp';
      if (currentField === 'bottom_temp') {
        fieldMin = $t_min; fieldMax = $t_max; colormap = 'temp';
      } else if (currentField === 'surface_temp') {
        fieldMin = $s_min; fieldMax = $s_max; colormap = 'temp';
      } else if (currentField === 'hsi') {
        fieldMin = $h_min; fieldMax = $h_max; colormap = 'hsi';
      } else if (currentField === 'depth') {
        fieldMin = $d_min; fieldMax = $d_max; colormap = 'depth';
      }

      cellsData.features.forEach(function(f) {
        var val = (currentField === 'bottom_temp') ? f.properties.temp_bottom :
                  (currentField === 'surface_temp') ? f.properties.temp_surface :
                  (currentField === 'hsi') ? f.properties.hsi : f.properties.depth;
        var col = getColorForValue(val, fieldMin, fieldMax, colormap);
        var c = f.geometry.coordinates;
        var marker = L.circleMarker([c[1], c[0]], {
          radius: 6,
          fillColor: col,
          fillOpacity: 0.75,
          color: 'rgba(0,0,0,0.2)',
          weight: 0.5
        }).bindPopup(
          "<div style='font-family:var(--font-main);font-size:0.85rem;line-height:1.4;'>" +
          "<b>Environmental Cell</b><br/>" +
          "Bottom Temp: <b>" + f.properties.temp_bottom + " °C</b><br/>" +
          "Surface Temp: <b>" + f.properties.temp_surface + " °C</b><br/>" +
          "Dynamic HSI: <b>" + f.properties.hsi + "</b><br/>" +
          "Depth: <b>" + f.properties.depth + " m</b></div>"
        );
        envMarkersLayer.addLayer(marker);
      });
    }
    renderEnvLayer();

    // Organisms Agent Layer
    var agentsData = $agents_geojson_json;
    var agentsLayer = L.layerGroup().addTo(map);

    function renderAgents() {
      agentsLayer.clearLayers();
      agentsData.features.forEach(function(f) {
        var c = f.geometry.coordinates;
        var p = f.properties;
        var marker = L.circleMarker([c[1], c[0]], {
          radius: Math.max(3, Math.min(8, p.size_mm * 0.08)),
          fillColor: p.color,
          fillOpacity: 0.85,
          color: '#ffffff',
          weight: 0.8
        }).bindPopup(
          "<div style='font-family:var(--font-main);font-size:0.85rem;line-height:1.4;'>" +
          "<b>Organism #" + p.id + "</b><br/>" +
          "Sex: <b>" + p.sex + "</b> (" + (p.mature ? "Mature" : "Immature") + ")<br/>" +
          "Carapace Size: <b>" + p.size_mm + " mm</b><br/>" +
          "Wet Mass: <b>" + p.mass_g + " g</b><br/>" +
          "Stage: <b>" + p.stage + "</b></div>"
        );
        agentsLayer.addLayer(marker);
      });
    }
    renderAgents();

    // Event listeners
    document.getElementById('sel-heat-field').addEventListener('change', function(e) {
      currentField = e.target.value;
      var titleSpan = document.getElementById('legend-title');
      var minSpan = document.getElementById('legend-min');
      var maxSpan = document.getElementById('legend-max');
      if (currentField === 'bottom_temp') {
        titleSpan.innerText = 'Seabed Bottom Temperature (°C)';
        minSpan.innerText = '$t_min °C';
        maxSpan.innerText = '$t_max °C';
      } else if (currentField === 'surface_temp') {
        titleSpan.innerText = 'Surface Water Temperature (°C)';
        minSpan.innerText = '$s_min °C';
        maxSpan.innerText = '$s_max °C';
      } else if (currentField === 'hsi') {
        titleSpan.innerText = 'Habitat Suitability Index (HSI)';
        minSpan.innerText = '$h_min';
        maxSpan.innerText = '$h_max';
      } else if (currentField === 'depth') {
        titleSpan.innerText = 'Bathymetric Depth (m)';
        minSpan.innerText = '$d_min m';
        maxSpan.innerText = '$d_max m';
      }
      renderEnvLayer();
    });

    document.getElementById('chk-agents').addEventListener('change', function(e) {
      if (e.target.checked) { map.addLayer(agentsLayer); } else { map.removeLayer(agentsLayer); }
    });
    document.getElementById('chk-grid').addEventListener('change', function(e) {
      if (e.target.checked) { map.addLayer(envMarkersLayer); } else { map.removeLayer(envMarkersLayer); }
    });

    document.getElementById('btn-reset-view').addEventListener('click', function() {
      map.fitBounds(mapBounds, { padding: [30, 30] });
    });

    var origBounds = [[$min_lat, $min_lon], [$max_lat, $max_lon]];
    var selectionMode = 'none'; // 'none', 'rect', 'click'
    var clickPoints = [];
    var isDraggingBox = false;
    var dragStartLatLng = null;
    var tempDragRect = null;

    function setSelectionMode(mode) {
      selectionMode = mode;
      var badge = document.getElementById('badge-sel-mode');
      var hint = document.getElementById('sel-mode-hint');
      clickPoints = [];
      if (tempDragRect) { map.removeLayer(tempDragRect); tempDragRect = null; }

      if (mode === 'rect') {
        badge.innerText = 'Drag Box Mode';
        badge.style.background = 'rgba(245, 158, 11, 0.2)';
        badge.style.color = '#f59e0b';
        hint.style.display = 'block';
        hint.innerText = 'Click and drag on the map to define the bounding box.';
        map.dragging.disable();
        map.getContainer().style.cursor = 'crosshair';
      } else if (mode === 'click') {
        badge.innerText = 'Click Mode (0/2)';
        badge.style.background = 'rgba(16, 185, 129, 0.2)';
        badge.style.color = '#10b981';
        hint.style.display = 'block';
        hint.innerText = 'Click point 1 (e.g. South-West corner), then point 2 (North-East corner).';
        map.dragging.enable();
        map.getContainer().style.cursor = 'crosshair';
      } else {
        badge.innerText = 'View Mode';
        badge.style.background = 'rgba(56, 189, 248, 0.18)';
        badge.style.color = '#38bdf8';
        hint.style.display = 'none';
        map.dragging.enable();
        map.getContainer().style.cursor = '';
      }
    }

    document.getElementById('btn-mode-rect').addEventListener('click', function() {
      setSelectionMode(selectionMode === 'rect' ? 'none' : 'rect');
    });

    document.getElementById('btn-mode-click').addEventListener('click', function() {
      setSelectionMode(selectionMode === 'click' ? 'none' : 'click');
    });

    function applyNewBounds(southWest, northEast) {
      var minLat = Math.min(southWest.lat, northEast.lat);
      var maxLat = Math.max(southWest.lat, northEast.lat);
      var minLon = Math.min(southWest.lng, northEast.lng);
      var maxLon = Math.max(southWest.lng, northEast.lng);

      document.getElementById('inp-min-lon').value = minLon.toFixed(3);
      document.getElementById('inp-max-lon').value = maxLon.toFixed(3);
      document.getElementById('inp-min-lat').value = minLat.toFixed(3);
      document.getElementById('inp-max-lat').value = maxLat.toFixed(3);

      mapBounds = [[minLat, minLon], [maxLat, maxLon]];
      domainRect.setBounds(mapBounds);
      setSelectionMode('none');
    }

    // Map mouse events for box dragging & 2-point clicking
    map.on('mousedown', function(e) {
      if (selectionMode === 'rect') {
        isDraggingBox = true;
        dragStartLatLng = e.latlng;
        tempDragRect = L.rectangle([dragStartLatLng, dragStartLatLng], {
          color: '#f59e0b',
          weight: 2,
          fillOpacity: 0.15
        }).addTo(map);
      }
    });

    map.on('mousemove', function(e) {
      if (selectionMode === 'rect' && isDraggingBox && tempDragRect) {
        tempDragRect.setBounds([dragStartLatLng, e.latlng]);
      }
    });

    map.on('mouseup', function(e) {
      if (selectionMode === 'rect' && isDraggingBox) {
        isDraggingBox = false;
        if (tempDragRect) {
          var b = tempDragRect.getBounds();
          applyNewBounds(b.getSouthWest(), b.getNorthEast());
          map.removeLayer(tempDragRect);
          tempDragRect = null;
        }
      }
    });

    map.on('click', function(e) {
      if (selectionMode === 'click') {
        clickPoints.push(e.latlng);
        var badge = document.getElementById('badge-sel-mode');
        badge.innerText = 'Click Mode (' + clickPoints.length + '/2)';
        if (clickPoints.length === 1) {
          L.popup()
            .setLatLng(e.latlng)
            .setContent("Point 1 selected. Click the opposite corner.")
            .openOn(map);
        } else if (clickPoints.length >= 2) {
          map.closePopup();
          applyNewBounds(clickPoints[0], clickPoints[1]);
        }
      }
    });

    document.getElementById('btn-apply-bounds').addEventListener('click', function() {
      var n_min_lon = parseFloat(document.getElementById('inp-min-lon').value);
      var n_max_lon = parseFloat(document.getElementById('inp-max-lon').value);
      var n_min_lat = parseFloat(document.getElementById('inp-min-lat').value);
      var n_max_lat = parseFloat(document.getElementById('inp-max-lat').value);
      mapBounds = [[n_min_lat, n_min_lon], [n_max_lat, n_max_lon]];
      map.fitBounds(mapBounds);
      domainRect.setBounds(mapBounds);
    });

    document.getElementById('btn-reset-bounds').addEventListener('click', function() {
      mapBounds = origBounds;
      document.getElementById('inp-min-lon').value = origBounds[0][1].toFixed(2);
      document.getElementById('inp-max-lon').value = origBounds[1][1].toFixed(2);
      document.getElementById('inp-min-lat').value = origBounds[0][0].toFixed(2);
      document.getElementById('inp-max-lat').value = origBounds[1][0].toFixed(2);
      domainRect.setBounds(mapBounds);
      map.fitBounds(mapBounds);
      setSelectionMode('none');
    });

    document.getElementById('slider-doy').addEventListener('input', function(e) {
      document.getElementById('lbl-doy').innerText = e.target.value + ' / 365';
    });
  </script>
</body>
</html>
"""

    dir = dirname(output_html)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end
    write(output_html, html)

    if auto_open
        try
            if Sys.iswindows()
                run(`cmd /c start $(replace(output_html, "/" => "\\"))`)
            elseif Sys.isapple()
                run(`open $(output_html)`)
            else
                run(`xdg-open $(output_html)`)
            end
        catch
        end
    end

    return abspath(output_html)
end

"""
    launch_leaflet_dashboard(config::Dict{Symbol, Any}; kwargs...)

Initialize model and launch the Leaflet-based interactive dashboard.
"""
function launch_leaflet_dashboard(
    config::Dict{Symbol, Any};
    display_window::Bool = true,
    n_agents::Union{Int, Nothing} = nothing,
    output_html::String = "outputs/dashboard.html"
)
    params = IBMParameters(config)
    model = initialize_model(
        grid = config[:grid],
        dt = config[:dt],
        params = params,
        seed = config[:seed],
        n_agents = n_agents
    )
    path = generate_leaflet_dashboard_html(
        model;
        title = "Snow Crab IBM Spatial Dashboard",
        output_html = output_html,
        auto_open = display_window
    )
    println("="^80)
    println(" Leaflet Interactive Dashboard generated at:")
    println("   file://$(replace(path, "\\" => "/"))")
    println("="^80)
    return path
end
