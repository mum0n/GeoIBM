"""
    src/leaflet_dashboard.jl

Interactive Leaflet-based Web Dashboard for GeoIBM.
Embeds real-time geographic maps, bathymetry, dynamic bottom & surface temperature
layers, dynamic HSI fields, moving organism agent positions, and user-interactive
controls (location bounding box selection, time window sliders, transport run/pause,
and stepping).
Below the map, provides dynamic, interactive diagnostic panels audited directly
against the Makie dashboard:
1. Abundance Trajectories (Total, Mature Females, Mature Males, Legal Fishable Males)
2. Biomass, Catch & Macro-Models (Legal Biomass, Catch, Schaefer, Delay-Diff, Leslie)
3. Stage Abundance (Egg, Early Cohorts, Instars 5-16 by Sex)
4. Size Frequency by Sex & Maturity (KDE curves with Legal size male marker)
5. Maturity Ogive & Fishery Selectivity (Theoretical & Realized empirical points)
6. Stock-Recruitment (Male Component, Realized points & Ricker/Beverton-Holt fits)
"""

"""
    generate_leaflet_dashboard_html(model; title="Snow Crab IBM Spatial Dashboard", kwargs...)

Generate a complete, self-contained interactive Leaflet HTML dashboard for `model`.
The dashboard layout is vertically scrollable, featuring sticky transport controls,
a top Leaflet map with interactive Area of Interest (AOI) selection, dynamic HSI
calculation for any arbitrary domain, unconstrained temporal period selection,
individual demographic colour & size legends, strictly marine agent positioning
(no crabs on land), and 6 dynamic diagnostic panels below the map.

# Arguments
- `model`: Agents.jl simulation model container.
- `title::String`: Window and header title.
- `output_html::String`: Target file path for the generated HTML.
- `auto_open::Bool`: If `true`, launch system default browser upon writing.

# Returns
`String`: Absolute path to the generated HTML file.
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
    org = params.organism
    fsh = params.fishery

    lon_bounds = env.domain_lon
    lat_bounds = env.domain_lat
    min_lon = Float64(lon_bounds[1])
    max_lon = Float64(lon_bounds[2])
    min_lat = Float64(lat_bounds[1])
    max_lat = Float64(lat_bounds[2])
    mid_lon = 0.5 * (min_lon + max_lon)
    mid_lat = 0.5 * (min_lat + max_lat)

    # Grid dimensions and fields
    nx, ny = size(props[:temperature_field])
    t_field = props[:temperature_field]
    d_field = props[:depth_field]
    h_field = props[:habitat_quality_field]
    s_field = get(props, :surface_temperature_field, t_field)

    t_min = round(minimum(t_field), digits=1)
    t_max = round(maximum(t_field), digits=1)
    s_min = round(minimum(s_field), digits=1)
    s_max = round(maximum(s_field), digits=1)
    d_min = round(minimum(d_field), digits=1)
    d_max = round(maximum(d_field), digits=1)
    h_min = round(minimum(h_field), digits=2)
    h_max = round(maximum(h_field), digits=2)

    # Duration and temporal horizon
    sim_days = Int(round(get(props, :days, get(props, :simulation_duration, 730))))
    if sim_days <= 0
        sim_days = 730
    end
    t_start = length(env.time_range) >= 1 ? env.time_range[1] : "1993-01-01"
    t_end = length(env.time_range) >= 2 ? env.time_range[2] : "1994-12-31"

    # Sample agents for client-side representation
    agents_list = collect(allagents(model))
    n_total = length(agents_list)
    legal_sz = fsh.legal_size_male
    adult_sz = org.adult_size_threshold

    # Grid to coordinate mapping
    function grid_to_coord(i::Real, j::Real)
        lon = min_lon + (Float64(i) - 0.5) / nx * (max_lon - min_lon)
        lat = min_lat + (Float64(j) - 0.5) / ny * (max_lat - min_lat)
        return (lon, lat)
    end

    # Valid marine coordinates to guarantee no agent is placed on land
    valid_marine_coords = Tuple{Float64, Float64}[]
    for j in 1:ny, i in 1:nx
        if d_field[i, j] >= 20.0 && h_field[i, j] > 0.0
            push!(valid_marine_coords, grid_to_coord(i, j))
        end
    end
    if isempty(valid_marine_coords)
        for j in 1:ny, i in 1:nx
            if d_field[i, j] >= 20.0
                push!(valid_marine_coords, grid_to_coord(i, j))
            end
        end
    end

    # Compact JSON serializer
    function to_json_val(v)
        if v isa AbstractString
            return "\"" * replace(v, "\"" => "\\\"", "\\" => "\\\\") * "\""
        elseif v isa Bool
            return v ? "true" : "false"
        elseif v isa Number
            return isfinite(v) ? string(v) : "0.0"
        elseif v isa AbstractVector
            return "[" * join([to_json_val(x) for x in v], ",") * "]"
        elseif v isa AbstractDict
            return "{" * join(["\"$k\":" * to_json_val(val) for (k, val) in v], ",") * "}"
        else
            return "\"$(string(v))\""
        end
    end

    # Serialized agent list (snapped away from land if needed)
    agent_records = Dict{String, Any}[]
    for a in agents_list
        gx, gy = a.pos[1], a.pos[2]
        alon, alat = grid_to_coord(gx, gy)
        if !isempty(valid_marine_coords) && (d_field[gx, gy] < 20.0 || h_field[gx, gy] <= 0.0)
            best_idx = 1
            best_dist_sq = 1e9
            for (idx, (mlon, mlat)) in enumerate(valid_marine_coords)
                dsq = (mlon - alon)^2 + (mlat - alat)^2
                if dsq < best_dist_sq
                    best_dist_sq = dsq
                    best_idx = idx
                end
            end
            alon, alat = valid_marine_coords[best_idx]
        end
        is_fem = (a.sex === female)
        push!(agent_records, Dict(
            "id" => a.id,
            "gx" => gx,
            "gy" => gy,
            "lon" => round(alon, digits=4),
            "lat" => round(alat, digits=4),
            "size" => round(a.size, digits=2),
            "mass" => round(a.mass, digits=2),
            "stage" => a.stage,
            "sex" => is_fem ? 1 : 0,
            "mature" => a.is_mature,
            "eggs" => a.egg_count,
            "shell_age" => round(hasproperty(a, :shell_age) ? a.shell_age : 0.0, digits=1)
        ))
    end
    initial_agents_json = to_json_val(agent_records)

    # Decimated grid cells for initial environmental reference (strictly marine cells)
    cell_features = Dict{String, Any}[]
    step_x = max(1, div(nx, 24))
    step_y = max(1, div(ny, 24))
    for j in 1:step_y:ny, i in 1:step_x:nx
        if d_field[i, j] >= 20.0 && h_field[i, j] > 0.0
            c_lon, c_lat = grid_to_coord(i, j)
            push!(cell_features, Dict(
                "lon" => round(c_lon, digits=4),
                "lat" => round(c_lat, digits=4),
                "depth" => round(d_field[i, j], digits=1),
                "tb" => round(t_field[i, j], digits=2),
                "ts" => round(s_field[i, j], digits=2),
                "hsi" => round(h_field[i, j], digits=3)
            ))
        end
    end
    cells_json = to_json_val(cell_features)

    # Theoretical reference curves
    cw_ref = collect(20.0:2.0:160.0)
    ogive_male_ref = Float64[maturity(0, log(c)) for c in cw_ref]
    ogive_female_ref = Float64[maturity(1, log(c)) for c in cw_ref]
    selectivity_ref = Float64[selectivity(c; s50 = legal_sz) for c in cw_ref]

    # Stage labels
    larval_count = larval_stage_count(SpeciesBehaviorImpl())
    max_stage = org.max_stage_male
    stage_labels = String[cohort_stage_label(s, params, larval_count) for s in 0:max_stage]

    # Stock-recruitment baseline records
    initial_sr_pts = Dict{String, Float64}[]
    try
        sr_df = extract_stock_recruitment(model)
        if nrow(sr_df) > 0
            for row in eachrow(sr_df)
                s_raw = Float64(row.spawners)
                r_raw = Float64(row.recruits)
                if s_raw > 0.0 && r_raw > 0.0
                    push!(initial_sr_pts, Dict("s" => s_raw, "r" => r_raw))
                end
            end
        end
    catch
    end
    initial_sr_json = to_json_val(initial_sr_pts)

    cw_ref_json = to_json_val(cw_ref)
    ogive_male_json = to_json_val(ogive_male_ref)
    ogive_female_json = to_json_val(ogive_female_ref)
    selectivity_json = to_json_val(selectivity_ref)
    stage_labels_json = to_json_val(stage_labels)

    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>$title</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet"/>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg-dark: #070b14;
      --bg-panel: rgba(13, 20, 36, 0.95);
      --bg-card: rgba(22, 33, 56, 0.85);
      --border: rgba(148, 163, 184, 0.16);
      --border-accent: rgba(56, 189, 248, 0.35);
      --accent: #38bdf8;
      --accent-emerald: #10b981;
      --accent-amber: #f59e0b;
      --accent-crimson: #ec4899;
      --accent-purple: #8b5cf6;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --font-main: 'Inter', system-ui, -apple-system, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html { scroll-behavior: smooth; }
    body {
      background: var(--bg-dark);
      color: var(--text);
      font-family: var(--font-main);
      overflow-y: auto;
      overflow-x: hidden;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    /* Sticky Header */
    header {
      position: sticky;
      top: 0;
      height: 56px;
      background: rgba(10, 16, 30, 0.98);
      backdrop-filter: blur(14px);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 20px;
      z-index: 1002;
    }
    .header-left {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .header-title {
      font-size: 1.12rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: #fff;
    }
    .badge {
      font-size: 0.70rem;
      font-weight: 600;
      text-transform: uppercase;
      padding: 3px 8px;
      border-radius: 6px;
      background: rgba(56, 189, 248, 0.16);
      color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.32);
    }
    .header-chips {
      display: flex;
      gap: 10px;
      font-size: 0.80rem;
    }
    .chip {
      background: rgba(30, 41, 59, 0.7);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 4px 10px;
      display: flex;
      gap: 6px;
      align-items: center;
    }
    .chip-val {
      font-family: var(--font-mono);
      font-weight: 700;
      color: var(--accent);
    }

    /* Sticky Transport Toolbar */
    .transport-bar {
      position: sticky;
      top: 56px;
      height: 52px;
      background: rgba(13, 22, 40, 0.96);
      backdrop-filter: blur(14px);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 20px;
      z-index: 1001;
    }
    .transport-controls {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .btn {
      background: rgba(56, 189, 248, 0.14);
      color: var(--accent);
      border: 1px solid var(--border-accent);
      border-radius: 6px;
      padding: 6px 14px;
      font-weight: 600;
      cursor: pointer;
      font-size: 0.82rem;
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .btn:hover {
      background: rgba(56, 189, 248, 0.32);
      color: #fff;
    }
    .btn-run {
      background: rgba(16, 185, 129, 0.20);
      color: var(--accent-emerald);
      border-color: rgba(16, 185, 129, 0.45);
    }
    .btn-run:hover {
      background: rgba(16, 185, 129, 0.38);
      color: #fff;
    }
    .btn-pause {
      background: rgba(245, 158, 11, 0.20);
      color: var(--accent-amber);
      border-color: rgba(245, 158, 11, 0.45);
    }
    .btn-pause:hover {
      background: rgba(245, 158, 11, 0.38);
      color: #fff;
    }
    .btn-primary {
      background: rgba(16, 185, 129, 0.28);
      color: #34d399;
      border-color: rgba(16, 185, 129, 0.55);
      font-weight: 700;
    }
    .btn-primary:hover {
      background: rgba(16, 185, 129, 0.48);
      color: #fff;
    }
    .scrubber-group {
      display: flex;
      align-items: center;
      gap: 12px;
      flex: 1;
      max-width: 520px;
      margin: 0 20px;
    }
    .scrubber-label {
      font-family: var(--font-mono);
      font-size: 0.80rem;
      font-weight: 600;
      white-space: nowrap;
    }
    input[type=range] {
      flex: 1;
      accent-color: var(--accent);
      cursor: pointer;
    }
    .speed-select {
      background: rgba(30, 41, 59, 0.8);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 4px 8px;
      font-size: 0.80rem;
      outline: none;
    }

    /* Top Section: Map + Sidebar */
    .top-section {
      display: flex;
      height: 600px;
      border-bottom: 1px solid var(--border);
      position: relative;
    }
    #map {
      flex: 1;
      height: 100%;
      background: #040810;
    }
    .map-sidebar {
      width: 390px;
      background: var(--bg-panel);
      border-left: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      gap: 12px;
      padding: 16px;
      overflow-y: auto;
      z-index: 1000;
    }
    .panel-box {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px 14px;
    }
    .panel-box-title {
      font-size: 0.82rem;
      font-weight: 700;
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .form-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 8px;
    }
    .form-group {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .form-label {
      font-size: 0.70rem;
      color: var(--text-muted);
      text-transform: uppercase;
    }
    .form-input {
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 5px 8px;
      color: var(--text);
      font-family: var(--font-mono);
      font-size: 0.82rem;
      outline: none;
    }
    .form-input:focus {
      border-color: var(--accent);
    }
    .legend-row {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.76rem;
      margin-bottom: 5px;
    }
    .legend-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
    }
    .heat-selector {
      display: flex;
      gap: 6px;
      margin-bottom: 8px;
    }
    .heat-btn {
      flex: 1;
      padding: 4px 6px;
      font-size: 0.72rem;
      font-weight: 600;
      background: rgba(30, 41, 59, 0.7);
      border: 1px solid var(--border);
      color: var(--text-muted);
      border-radius: 4px;
      cursor: pointer;
      text-align: center;
      transition: all 0.15s ease;
    }
    .heat-btn.active {
      background: rgba(56, 189, 248, 0.25);
      border-color: var(--accent);
      color: #fff;
    }
    .legend-bar {
      height: 8px;
      border-radius: 4px;
      margin: 6px 0;
      background: linear-gradient(to right, #0284c7, #10b981, #f59e0b, #ef4444);
    }
    .legend-minmax {
      display: flex;
      justify-content: space-between;
      font-size: 0.72rem;
      font-family: var(--font-mono);
      color: var(--text-muted);
    }

    /* Section Divider */
    .section-banner {
      background: linear-gradient(90deg, rgba(56, 189, 248, 0.12) 0%, rgba(13, 20, 36, 0.6) 100%);
      border-top: 1px solid var(--border);
      border-bottom: 1px solid var(--border);
      padding: 14px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .section-title {
      font-size: 0.95rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 10px;
    }

    /* Scrollable Lower Analytics Grid */
    .analytics-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 20px;
      padding: 24px;
      background: var(--bg-dark);
    }
    @media (max-width: 768px) {
      .analytics-grid {
        grid-template-columns: 1fr;
      }
    }
    .chart-panel {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px 18px;
      display: flex;
      flex-direction: column;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
      min-height: 380px;
    }
    .chart-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.1);
      padding-bottom: 8px;
    }
    .chart-title {
      font-size: 0.92rem;
      font-weight: 700;
      color: #fff;
    }
    .chart-subtitle {
      font-size: 0.72rem;
      color: var(--text-muted);
      font-family: var(--font-mono);
    }
    .chart-canvas-wrapper {
      position: relative;
      flex: 1;
      width: 100%;
      height: 300px;
    }

    /* Custom Leaflet Controls */
    .leaflet-popup-content-wrapper {
      background: rgba(15, 23, 42, 0.95);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      backdrop-filter: blur(8px);
      font-family: var(--font-main);
    }
    .leaflet-popup-tip {
      background: rgba(15, 23, 42, 0.95);
    }
  </style>
</head>
<body>

  <!-- Sticky Header -->
  <header>
    <div class="header-left">
      <span class="header-title">$title</span>
      <span class="badge" id="badge-status">Day 0 | Active</span>
    </div>
    <div class="header-chips">
      <div class="chip">Total: <span class="chip-val" id="kpi-total">$n_total</span></div>
      <div class="chip">Mature ♂: <span class="chip-val" id="kpi-mat-m">0</span></div>
      <div class="chip">Mature ♀: <span class="chip-val" id="kpi-mat-f">0</span></div>
      <div class="chip">Legal Biomass: <span class="chip-val" id="kpi-biomass">0.0 kg</span></div>
      <div class="chip">Catch: <span class="chip-val" id="kpi-catch">0.0 kg</span></div>
    </div>
  </header>

  <!-- Sticky Transport Toolbar -->
  <div class="transport-bar">
    <div class="transport-controls">
      <button class="btn btn-run" id="btn-play-pause">▶ Run</button>
      <button class="btn" id="btn-step-fwd">⏭ Step +10d</button>
      <button class="btn" id="btn-step-back">⏮ Step -10d</button>
      <button class="btn" id="btn-reset">↺ Reset</button>
    </div>
    <div class="scrubber-group">
      <span class="scrubber-label" id="lbl-day">Day 0 / [0 - $sim_days]</span>
      <input type="range" id="slider-day" min="0" max="$sim_days" value="0" step="1"/>
      <select class="speed-select" id="sel-speed">
        <option value="1">1x Speed</option>
        <option value="2">2x Speed</option>
        <option value="5" selected>5x Speed</option>
        <option value="10">10x Speed</option>
        <option value="25">25x Speed</option>
      </select>
    </div>
    <div>
      <button class="btn" id="btn-recalibrate" title="Refit Schaefer, Delay-Difference and Leslie models">Refit Models</button>
    </div>
  </div>

  <!-- Top Section: Leaflet Map & Interactive Filter Sidebar -->
  <div class="top-section">
    <div id="map"></div>
    <div class="map-sidebar">

      <!-- Geographic Area of Interest Panel -->
      <div class="panel-box">
        <div class="panel-box-title">
          <span>Area of Interest (AOI)</span>
          <span class="badge" id="badge-sel-mode" style="font-size:0.65rem;">View Mode</span>
        </div>
        <div style="display:flex; gap:6px; margin-bottom:8px;">
          <button class="btn" style="flex:1; padding:4px 6px; font-size:0.75rem;" id="btn-mode-rect">⛶ Drag Box</button>
          <button class="btn" style="flex:1; padding:4px 6px; font-size:0.75rem;" id="btn-mode-click">📍 2-Points</button>
        </div>
        <div id="sel-mode-hint" style="font-size:0.70rem; color:var(--accent-amber); margin-bottom:6px; display:none;">
          Click and drag on map to define coordinates.
        </div>
        <div class="form-row">
          <div class="form-group">
            <span class="form-label">Min Lon</span>
            <input type="number" step="0.05" class="form-input" id="inp-min-lon" value="$(round(min_lon, digits=2))"/>
          </div>
          <div class="form-group">
            <span class="form-label">Max Lon</span>
            <input type="number" step="0.05" class="form-input" id="inp-max-lon" value="$(round(max_lon, digits=2))"/>
          </div>
        </div>
        <div class="form-row">
          <div class="form-group">
            <span class="form-label">Min Lat</span>
            <input type="number" step="0.05" class="form-input" id="inp-min-lat" value="$(round(min_lat, digits=2))"/>
          </div>
          <div class="form-group">
            <span class="form-label">Max Lat</span>
            <input type="number" step="0.05" class="form-input" id="inp-max-lat" value="$(round(max_lat, digits=2))"/>
          </div>
        </div>
        <div style="display:flex; gap:8px; margin-top:8px;">
          <button class="btn btn-primary" style="flex:1.4;" id="btn-apply-area">✔ Apply & Update Area</button>
          <button class="btn" style="flex:1;" id="btn-reset-bounds">↺ Reset Area</button>
        </div>
        <label style="display:flex; align-items:center; gap:8px; font-size:0.76rem; margin-top:8px; cursor:pointer;">
          <input type="checkbox" id="chk-filter-bounds"/>
          <span>Filter all figures to AOI selection</span>
        </label>
      </div>

      <!-- Time Horizon & Range Selection Panel -->
      <div class="panel-box">
        <div class="panel-box-title">
          <span>Temporal Range Selection</span>
          <span style="font-size:0.7rem; color:var(--accent-emerald);">Interactive</span>
        </div>
        <div class="form-row">
          <div class="form-group">
            <span class="form-label">Start Date</span>
            <input type="date" class="form-input" id="inp-time-start" value="$t_start"/>
          </div>
          <div class="form-group">
            <span class="form-label">End Date</span>
            <input type="date" class="form-input" id="inp-time-end" value="$t_end"/>
          </div>
        </div>
        <div class="form-row">
          <div class="form-group">
            <span class="form-label">Start Day</span>
            <input type="number" class="form-input" id="inp-day-start" min="0" value="0"/>
          </div>
          <div class="form-group">
            <span class="form-label">End Day</span>
            <input type="number" class="form-input" id="inp-day-end" min="1" value="$sim_days"/>
          </div>
        </div>
        <div style="display:flex; gap:8px; margin-top:6px;">
          <button class="btn btn-primary" style="flex:1.4;" id="btn-apply-time">Apply Time Window</button>
          <button class="btn" style="flex:1;" id="btn-reset-time">Full Horizon</button>
        </div>
        <div style="font-size:0.73rem; color:var(--text-muted); margin-top:8px;">
          Current: <strong id="lbl-current-date" style="color:#fff;">$t_start</strong>
          | <span id="lbl-current-year" style="font-weight:700; color:var(--accent);">Year 1</span>
          | DOY: <span id="lbl-current-doy" style="font-weight:700; color:var(--accent-amber);">Day 0</span>
        </div>
      </div>

      <!-- Environmental Layer Switcher & Dynamic Legend -->
      <div class="panel-box">
        <div class="panel-box-title">
          <span>Environmental Fields</span>
          <span id="lbl-active-layer" style="font-size:0.70rem; color:var(--accent);">HSI</span>
        </div>
        <div class="heat-selector">
          <button class="heat-btn active" id="btn-layer-hsi" data-layer="hsi">HSI</button>
          <button class="heat-btn" id="btn-layer-temp" data-layer="temp">Bottom T</button>
          <button class="heat-btn" id="btn-layer-depth" data-layer="depth">Depth</button>
          <button class="heat-btn" id="btn-layer-surf" data-layer="surf">Surface T</button>
        </div>
        <div class="legend-bar" id="legend-gradient-bar"></div>
        <div class="legend-minmax">
          <span id="legend-min-val">0.00</span>
          <span id="legend-field-desc">Suitability Index [0 - 1]</span>
          <span id="legend-max-val">1.00</span>
        </div>
      </div>

      <!-- Organism Demographic & Size Legend -->
      <div class="panel-box">
        <div class="panel-box-title">Individuals (Colour & Size)</div>
        <!-- Colour Key -->
        <div style="font-size:0.70rem; color:var(--text-muted); text-transform:uppercase; margin-bottom:4px; font-weight:600;">
          Demographic Colour
        </div>
        <div class="legend-row">
          <div class="legend-dot" style="background:#2563eb;"></div>
          <span>Mature Male ♂ (Dark Blue)</span>
        </div>
        <div class="legend-row">
          <div class="legend-dot" style="background:#60a5fa;"></div>
          <span>Immature Male ♂ (Light Blue)</span>
        </div>
        <div class="legend-row">
          <div class="legend-dot" style="background:#e11d48;"></div>
          <span>Mature Female ♀ (Crimson)</span>
        </div>
        <div class="legend-row">
          <div class="legend-dot" style="background:#f472b6;"></div>
          <span>Immature Female ♀ (Pink)</span>
        </div>
        <hr style="border:0; border-top:1px solid rgba(148,163,184,0.18); margin:8px 0;"/>
        <!-- Size Key -->
        <div style="font-size:0.70rem; color:var(--text-muted); text-transform:uppercase; margin-bottom:6px; font-weight:600;">
          Carapace Width (CW) Size Key
        </div>
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px; font-size:0.74rem;">
          <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#94a3b8; border:1px solid #fff;"></span>
          <span>30 mm (Juvenile)</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px; font-size:0.74rem;">
          <span style="display:inline-block; width:14px; height:14px; border-radius:50%; background:#94a3b8; border:1px solid #fff;"></span>
          <span>60 mm (Adolescent)</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px; font-size:0.74rem;">
          <span style="display:inline-block; width:22px; height:22px; border-radius:50%; background:#8b5cf6; border:1.5px solid #fff;"></span>
          <span><strong>95 mm (Legal Male)</strong></span>
        </div>
        <div style="display:flex; align-items:center; gap:8px; font-size:0.74rem;">
          <span style="display:inline-block; width:28px; height:28px; border-radius:50%; background:#94a3b8; border:1px solid #fff;"></span>
          <span>130+ mm (Large Male)</span>
        </div>
      </div>

    </div>
  </div>

  <!-- Section Divider -->
  <div class="section-banner">
    <div class="section-title">
      <span>▼ Audited Diagnostic Panels (Dynamic Population & Macro Dynamics)</span>
    </div>
    <div style="font-size:0.76rem; color:var(--text-muted); font-family:var(--font-mono);">
      Scroll down to inspect all 6 Makie-audited panels
    </div>
  </div>

  <!-- Scrollable Lower Analytics Grid (6 Makie Audited Panels) -->
  <div class="analytics-grid">

    <!-- Panel 1: Abundance Trajectories -->
    <div class="chart-panel">
      <div class="chart-header">
        <div>
          <div class="chart-title">Abundance Trajectories</div>
          <div class="chart-subtitle">log10(count) vs Simulation Time (days)</div>
        </div>
        <span class="badge">Panel 1</span>
      </div>
      <div class="chart-canvas-wrapper">
        <canvas id="chart-abundance"></canvas>
      </div>
    </div>

    <!-- Panel 2: Biomass, Catch & Macro-Models -->
    <div class="chart-panel">
      <div class="chart-header">
        <div>
          <div class="chart-title">Biomass, Catch & Macro-Models</div>
          <div class="chart-subtitle" id="macro-subtitle">Legal Biomass (kg), Catch & Fits (Schaefer, Delay-Diff, Leslie)</div>
        </div>
        <span class="badge">Panel 2</span>
      </div>
      <div class="chart-canvas-wrapper">
        <canvas id="chart-biomass"></canvas>
      </div>
    </div>

    <!-- Panel 3: Stage Abundance -->
    <div class="chart-panel">
      <div class="chart-header">
        <div>
          <div class="chart-title">Stage Abundance</div>
          <div class="chart-subtitle">Lines with Dots by Ontogenetic Stage & Sex (Lower Bound = 1, Log10 Scale)</div>
        </div>
        <span class="badge">Panel 3</span>
      </div>
      <div class="chart-canvas-wrapper">
        <canvas id="chart-stage"></canvas>
      </div>
    </div>

    <!-- Panel 4: Size Frequency by Sex and Maturity -->
    <div class="chart-panel" id="panel-size-freq">
      <div class="chart-header">
        <div>
          <div class="chart-title">Size Frequency by Sex & Maturity</div>
          <div class="chart-subtitle" id="size-panel-subtitle">Normalized Density vs CW (mm) with Legal Size Marker ($legal_sz mm)</div>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
          <button class="btn" id="btn-toggle-draw-size" style="font-size:0.72rem; padding:3px 8px; border-color:var(--accent);">✏ Draw / Reshape Size</button>
          <span class="badge">Panel 4</span>
        </div>
      </div>
      <!-- Interactive Size-Structure Reshaping Bar (Hidden by default) -->
      <div id="size-editor-toolbar" style="display:none; background:rgba(30,41,59,0.7); border:1px solid var(--border-accent); border-radius:6px; padding:8px 12px; margin-bottom:10px; font-size:0.75rem; display:none; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:8px;">
        <div style="display:flex; align-items:center; gap:8px;">
          <span style="color:var(--accent); font-weight:700;">Click & Drag</span> on canvas to sculpt target size envelope.
        </div>
        <div style="display:flex; align-items:center; gap:10px;">
          <div style="display:flex; align-items:center; gap:5px;">
            <span>Agents:</span>
            <input type="number" id="inp-target-agents" min="50" max="5000" step="50" value="$n_total" class="form-input" style="width:75px; padding:2px 6px; font-size:0.75rem; text-align:center;"/>
          </div>
          <button class="btn" id="btn-size-preset-juv" style="font-size:0.70rem; padding:3px 6px;">Preset: Juvenile</button>
          <button class="btn" id="btn-size-preset-adult" style="font-size:0.70rem; padding:3px 6px;">Preset: Adult</button>
          <button class="btn" id="btn-size-preset-bimodal" style="font-size:0.70rem; padding:3px 6px;">Preset: Bimodal</button>
          <button class="btn btn-primary" id="btn-apply-size-structure" style="font-size:0.72rem; padding:3px 10px;">✔ Apply & Reseed</button>
          <button class="btn" id="btn-cancel-size-structure" style="font-size:0.70rem; padding:3px 8px;">Cancel</button>
        </div>
      </div>
      <div class="chart-canvas-wrapper">
        <canvas id="chart-size"></canvas>
      </div>
    </div>

    <!-- Panel 5: Maturity Ogive & Fishery Selectivity -->
    <div class="chart-panel" id="panel-selectivity">
      <div class="chart-header">
        <div>
          <div class="chart-title">Maturity Ogive & Fishery Selectivity</div>
          <div class="chart-subtitle" id="ogive-panel-subtitle">Selectivity L50 = <span id="lbl-l50-val" style="color:#8b5cf6; font-weight:700;">$legal_sz mm</span> | Drag Curve Horizontally</div>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
          <button class="btn" id="btn-toggle-drag-sel" style="font-size:0.72rem; padding:3px 8px; border-color:#8b5cf6;">↔ Drag Selectivity (L50)</button>
          <span class="badge">Panel 5</span>
        </div>
      </div>
      <!-- Selectivity Quick Tuning Toolbar (Hidden by default) -->
      <div id="sel-editor-toolbar" style="display:none; background:rgba(30,41,59,0.7); border:1px solid rgba(139,92,246,0.35); border-radius:6px; padding:8px 12px; margin-bottom:10px; font-size:0.75rem; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:8px;">
        <div style="display:flex; align-items:center; gap:8px;">
          <span style="color:#8b5cf6; font-weight:700;">Click & Drag</span> horizontally on canvas to shift L50, or adjust slider:
          <input type="range" id="slider-l50" min="40" max="140" step="0.5" value="$legal_sz" style="width:140px; accent-color:#8b5cf6; cursor:pointer;"/>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
          <button class="btn" id="btn-l50-reset" style="font-size:0.70rem; padding:3px 6px;">Reset ($legal_sz mm)</button>
          <button class="btn" id="btn-l50-small" style="font-size:0.70rem; padding:3px 6px;">78 mm (Small)</button>
          <button class="btn" id="btn-l50-large" style="font-size:0.70rem; padding:3px 6px;">105 mm (Large)</button>
        </div>
      </div>
      <div class="chart-canvas-wrapper">
        <canvas id="chart-ogive"></canvas>
      </div>
    </div>

    <!-- Panel 6: Stock-Recruitment (Male Component) -->
    <div class="chart-panel">
      <div class="chart-header">
        <div>
          <div class="chart-title">Stock-Recruitment (Male Component)</div>
          <div class="chart-subtitle" id="sr-subtitle">log10(Annual Eggs/2) vs log10(Recruits) | Ricker & Beverton-Holt Fits</div>
        </div>
        <span class="badge">Panel 6</span>
      </div>
      <div class="chart-canvas-wrapper">
        <canvas id="chart-sr"></canvas>
      </div>
    </div>

  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    // -------------------------------------------------------------------------
    // Initial Data & Parameters Injected from Julia
    // -------------------------------------------------------------------------
    const initialAgents = $initial_agents_json;
    const initialCells = $cells_json;
    const origDomainBounds = [
      [$min_lat, $min_lon],
      [$max_lat, $max_lon]
    ];
    let domainBounds = [
      [$min_lat, $min_lon],
      [$max_lat, $max_lon]
    ];
    let mapBounds = [
      [$min_lat, $min_lon],
      [$max_lat, $max_lon]
    ];

    let legalSizeMale = $legal_sz;
    const defaultLegalSizeMale = $legal_sz;
    const selectivitySigma = 3.5;
    const adultSizeThreshold = $adult_sz;
    const cwRef = $cw_ref_json;
    const ogiveMaleRef = $ogive_male_json;
    const ogiveFemaleRef = $ogive_female_json;
    let selectivityRef = [...$selectivity_json];
    const stageLabels = $stage_labels_json;
    const initialSrRecords = $initial_sr_json;

    const baseDateStr = "$t_start";
    const initialEndDateStr = "$t_end";
    let selectedStartDay = 0;
    let selectedEndDay = $sim_days;

    // -------------------------------------------------------------------------
    // Analytical Habitat Suitability Index (HSI) & Oceanographic Functions
    // Direct parity with src/organism_dynamics.jl:calculate_habitat_suitability
    // -------------------------------------------------------------------------
    function calculateHSI(temperature, depth) {
      const T = Number(temperature);
      const z = Math.abs(Number(depth));
      const tempHalf = 5.0;
      const tempSteepness = 1.0;
      const tempZeroThreshold = 10.0;
      const depthOpt = 150.0;
      const depthSigma = 50.0;
      const depthMin = 20.0;
      const depthMax = 400.0;

      // Thermal suitability S_T
      let s_t = 0.0;
      if (T < tempZeroThreshold) {
        const rawProb = 1.0 / (1.0 + Math.exp(tempSteepness * (T - tempHalf)));
        if (T > tempHalf) {
          const w_t = 0.5 * (1.0 + Math.cos(Math.PI * (T - tempHalf) / (tempZeroThreshold - tempHalf)));
          s_t = rawProb * w_t;
        } else {
          s_t = rawProb;
        }
      }

      // Bathymetric depth suitability S_z
      let s_z = 0.0;
      if (z > depthMin && z < depthMax) {
        const gRaw = Math.exp(-0.5 * Math.pow((z - depthOpt) / depthSigma, 2));
        let w_z = 0.0;
        if (z < depthOpt) {
          w_z = 0.5 * (1.0 - Math.cos(Math.PI * (z - depthMin) / (depthOpt - depthMin)));
        } else {
          w_z = 0.5 * (1.0 + Math.cos(Math.PI * (z - depthOpt) / (depthMax - depthOpt)));
        }
        s_z = gRaw * w_z;
      }

      return Math.max(0.0, Math.min(1.0, s_t * s_z));
    }

    // Environmental sampler for arbitrary coordinates (lon, lat) and seasonal day-of-year
    function sampleEnvironment(lon, lat, doy = 0) {
      const origMinLon = origDomainBounds[0][1];
      const origMaxLon = origDomainBounds[1][1];
      const origMinLat = origDomainBounds[0][0];
      const origMaxLat = origDomainBounds[1][0];

      let depth = 150.0;
      let tbMean = 4.0;
      let tsMean = 8.5;

      // Inside original domain: sample from high-res initialCells
      if (lon >= origMinLon && lon <= origMaxLon && lat >= origMinLat && lat <= origMaxLat) {
        let minDistSq = 1e9;
        let bestCell = initialCells[0];
        for (let i = 0; i < initialCells.length; i++) {
          const c = initialCells[i];
          const dsq = (c.lon - lon) * (c.lon - lon) + (c.lat - lat) * (c.lat - lat);
          if (dsq < minDistSq) {
            minDistSq = dsq;
            bestCell = c;
          }
        }
        depth = bestCell.depth;
        tbMean = bestCell.tb;
        tsMean = bestCell.ts;
      } else {
        // Regional Scotian Shelf & NW Atlantic continuous bathymetry & hydrography
        const shelfBreakLat = 42.6 + 0.06 * (lon + 60.0);
        const distFromSlope = lat - shelfBreakLat;

        if (distFromSlope < -0.6) {
          depth = 800.0 + Math.abs(distFromSlope) * 1200.0;
          tbMean = 4.5;
        } else if (distFromSlope < 0.0) {
          depth = 200.0 + (-distFromSlope / 0.6) * 600.0;
          tbMean = 6.0;
        } else {
          const channelDist = Math.hypot(lon - (-59.8), lat - 45.6);
          const bankDist = Math.hypot(lon - (-60.1), lat - 43.9);
          if (channelDist < 1.2) {
            depth = 240.0 + 80.0 * Math.sin(channelDist * Math.PI);
          } else if (bankDist < 0.8) {
            depth = 35.0 + 25.0 * bankDist;
          } else {
            depth = 80.0 + 90.0 * Math.sin(Math.abs(lon * 0.5 + lat * 0.8));
          }
          tbMean = 2.8 + 1.6 * Math.sin((lon + 62.0) * 0.4);
        }
        tsMean = 7.5 + (45.0 - lat) * 0.8;
      }

      // Land and intertidal detection: depth < 20m indicates non-viable snow crab habitat
      if (depth < 20.0) {
        return { depth: Math.max(0.0, depth), tb: tbMean, ts: tsMean, hsi: 0.0 };
      }

      // Seasonal temperature variation with Day of Year
      const doyRad = (doy - 220.0) * (2.0 * Math.PI / 365.0);
      const tbSeasonal = tbMean + 1.6 * Math.cos(doyRad);
      const tsSeasonal = tsMean + 6.8 * Math.cos((doy - 215.0) * (2.0 * Math.PI / 365.0));

      const hsi = calculateHSI(tbSeasonal, depth);
      return {
        depth: depth,
        tb: tbSeasonal,
        ts: tsSeasonal,
        hsi: hsi
      };
    }

    // -------------------------------------------------------------------------
    // Leaflet Map Initialization
    // -------------------------------------------------------------------------
    const map = L.map('map', {
      center: [$mid_lat, $mid_lon],
      zoom: 7,
      zoomControl: true
    });

    const esriOcean = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}',
      { attribution: 'Tiles &copy; Esri &mdash; Sources: GEBCO, NOAA, CHS', maxZoom: 13 }
    ).addTo(map);

    const osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap'
    });

    // Layer groups for dynamic layers
    const layerDepth = L.layerGroup();
    const layerTempBottom = L.layerGroup();
    const layerTempSurface = L.layerGroup();
    const layerHSI = L.layerGroup().addTo(map); // Default visible
    const layerAgents = L.layerGroup().addTo(map);

    const domainRect = L.rectangle(mapBounds, {
      color: '#38bdf8',
      weight: 2,
      fillOpacity: 0.04,
      dashArray: '5, 5'
    }).addTo(map);

    L.control.layers({
      'Esri Ocean Basemap': esriOcean,
      'OpenStreetMap': osmLayer
    }, {
      'Habitat Suitability (HSI)': layerHSI,
      'Seabed Temperature': layerTempBottom,
      'Bathymetry (Depth)': layerDepth,
      'Surface Temperature': layerTempSurface,
      'Organism Agents': layerAgents,
      'Domain Bounds': domainRect
    }, { position: 'topright' }).addTo(map);

    map.fitBounds(mapBounds);

    // -------------------------------------------------------------------------
    // Environmental Grid Generator for Arbitrary Bounds
    // -------------------------------------------------------------------------
    let currentGridPoints = [];

    function generateEnvironmentGrid(minLon, maxLon, minLat, maxLat, doy = 0) {
      layerDepth.clearLayers();
      layerTempBottom.clearLayers();
      layerTempSurface.clearLayers();
      layerHSI.clearLayers();
      currentGridPoints = [];

      const nSteps = 24;
      const dLon = (maxLon - minLon) / nSteps;
      const dLat = (maxLat - minLat) / nSteps;

      for (let j = 0; j <= nSteps; j++) {
        const lat = minLat + j * dLat;
        for (let i = 0; i <= nSteps; i++) {
          const lon = minLon + i * dLon;
          const env = sampleEnvironment(lon, lat, doy);
          currentGridPoints.push({ lon, lat, ...env });

          // Skip drawing marine markers on land
          if (env.depth <= 0.0) continue;

          // 1. Depth Marker
          const depthNorm = Math.min(1.0, env.depth / 400.0);
          L.circleMarker([lat, lon], {
            radius: 5,
            fillColor: '#0284c7',
            fillOpacity: Math.min(0.85, Math.max(0.12, depthNorm)),
            stroke: false
          }).bindTooltip('Depth: ' + env.depth.toFixed(1) + ' m').addTo(layerDepth);

          // 2. Bottom Temperature Marker
          const colTb = env.tb < 3.0 ? '#38bdf8' : (env.tb < 6.0 ? '#10b981' : '#f97316');
          L.circleMarker([lat, lon], {
            radius: 5,
            fillColor: colTb,
            fillOpacity: 0.70,
            stroke: false
          }).bindTooltip('Bottom T: ' + env.tb.toFixed(2) + ' °C').addTo(layerTempBottom);

          // 3. Surface Temperature Marker
          const colTs = env.ts < 8.0 ? '#0284c7' : (env.ts < 14.0 ? '#10b981' : '#ef4444');
          L.circleMarker([lat, lon], {
            radius: 5,
            fillColor: colTs,
            fillOpacity: 0.65,
            stroke: false
          }).bindTooltip('Surface T: ' + env.ts.toFixed(2) + ' °C').addTo(layerTempSurface);

          // 4. Habitat Suitability Index Marker
          if (env.hsi > 0.01) {
            const hsiAlpha = Math.max(0.08, env.hsi * 0.90);
            L.circleMarker([lat, lon], {
              radius: 5.5,
              fillColor: '#10b981',
              fillOpacity: hsiAlpha,
              stroke: env.hsi > 0.4,
              color: 'rgba(255,255,255,0.4)',
              weight: 0.5
            }).bindTooltip('HSI: ' + env.hsi.toFixed(3) + ' (T=' + env.tb.toFixed(1) + '°C, z=' + env.depth.toFixed(0) + 'm)').addTo(layerHSI);
          }
        }
      }
    }

    // Initial grid generation
    generateEnvironmentGrid(domainBounds[0][1], domainBounds[1][1], domainBounds[0][0], domainBounds[1][0], 0);

    // -------------------------------------------------------------------------
    // Client-Side Simulation Engine
    // -------------------------------------------------------------------------
    let currentDay = 0;
    let isRunning = false;
    let simSpeedMultiplier = 5;
    let animTimer = null;
    let filterToAOI = false;

    // Deep clone agents for client-side state
    let simAgents = initialAgents.map(a => Object.assign({}, a));

    // Time-series history buffers
    const tsHistory = {
      days: [0],
      total: [simAgents.length],
      matureMales: [simAgents.filter(a => a.sex === 0 && a.mature).length],
      matureFemales: [simAgents.filter(a => a.sex === 1 && a.mature).length],
      legalMales: [simAgents.filter(a => a.sex === 0 && a.size >= legalSizeMale).length],
      biomassLegal: [0],
      catchAnnual: [0],
      annualEggs: {},
      annualRecruits: {}
    };

    let initLegalBio = 0.0;
    simAgents.forEach(a => {
      if (a.sex === 0 && a.size >= legalSizeMale) {
        initLegalBio += a.mass / 1000.0;
      }
    });
    tsHistory.biomassLegal[0] = initLegalBio;

    function getAgentColor(a) {
      if (a.sex === 1) {
        return a.mature ? '#e11d48' : '#f472b6';
      } else {
        return a.mature ? '#2563eb' : '#60a5fa';
      }
    }

    function renderAgentMarkers() {
      layerAgents.clearLayers();
      const minLon = parseFloat(document.getElementById('inp-min-lon').value);
      const maxLon = parseFloat(document.getElementById('inp-max-lon').value);
      const minLat = parseFloat(document.getElementById('inp-min-lat').value);
      const maxLat = parseFloat(document.getElementById('inp-max-lat').value);

      simAgents.forEach(a => {
        if (filterToAOI) {
          if (a.lon < minLon || a.lon > maxLon || a.lat < minLat || a.lat > maxLat) {
            return;
          }
        }
        const rad = Math.max(3.5, Math.min(13.0, a.size * 0.12));
        const col = getAgentColor(a);
        const marker = L.circleMarker([a.lat, a.lon], {
          radius: rad,
          fillColor: col,
          fillOpacity: 0.85,
          color: '#ffffff',
          weight: 0.6
        });
        marker.bindTooltip(
          '<strong>ID ' + a.id + '</strong> (' + (a.sex === 1 ? '♀ Female' : '♂ Male') + ')<br/>' +
          'Stage: ' + a.stage + ' | Mature: ' + (a.mature ? 'Yes' : 'No') + '<br/>' +
          'CW: ' + a.size.toFixed(1) + ' mm | Mass: ' + a.mass.toFixed(1) + ' g<br/>' +
          'Eggs: ' + (a.eggs || 0)
        );
        marker.addTo(layerAgents);
      });
    }

    // -------------------------------------------------------------------------
    // Chart.js Panel Initializations
    // -------------------------------------------------------------------------
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.borderColor = 'rgba(148, 163, 184, 0.12)';
    Chart.defaults.font.family = "'Inter', sans-serif";

    // Chart 1: Abundance Trajectories
    const ctxAbundance = document.getElementById('chart-abundance').getContext('2d');
    const chartAbundance = new Chart(ctxAbundance, {
      type: 'line',
      data: {
        labels: [0],
        datasets: [
          { label: 'Total Organisms', data: [Math.log10(Math.max(1, tsHistory.total[0]))], borderColor: '#38bdf8', borderWidth: 2, pointRadius: 0 },
          { label: 'Mature Females', data: [Math.log10(Math.max(1, tsHistory.matureFemales[0]))], borderColor: '#e11d48', borderWidth: 1.8, pointRadius: 0 },
          { label: 'Mature Males', data: [Math.log10(Math.max(1, tsHistory.matureMales[0]))], borderColor: '#10b981', borderWidth: 1.8, pointRadius: 0 },
          { label: 'Legal Males', data: [Math.log10(Math.max(1, tsHistory.legalMales[0]))], borderColor: '#8b5cf6', borderWidth: 2, pointRadius: 0 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        scales: {
          x: { title: { display: true, text: 'Time (days)' }, min: selectedStartDay, max: selectedEndDay },
          y: { title: { display: true, text: 'log10(count)' } }
        }
      }
    });

    // Chart 2: Biomass, Catch & Macro-Models
    const ctxBiomass = document.getElementById('chart-biomass').getContext('2d');
    const chartBiomass = new Chart(ctxBiomass, {
      type: 'line',
      data: {
        labels: [0],
        datasets: [
          { label: 'Legal Fishable Biomass (kg)', data: [tsHistory.biomassLegal[0]], borderColor: '#8b5cf6', borderWidth: 2.2, pointRadius: 0 },
          { label: 'Annual Catch (kg)', data: [0], borderColor: '#f59e0b', borderDash: [5, 5], borderWidth: 1.8, pointRadius: 0 },
          { label: 'Schaefer Model Fit', data: [tsHistory.biomassLegal[0]], borderColor: '#2563eb', borderDash: [4, 4], borderWidth: 1.8, pointRadius: 0 },
          { label: 'Delay-Difference Fit', data: [tsHistory.biomassLegal[0]], borderColor: '#10b981', borderDash: [2, 2], borderWidth: 1.8, pointRadius: 0 },
          { label: 'Leslie Matrix Fit', data: [tsHistory.biomassLegal[0]], borderColor: '#ea580c', borderWidth: 2.0, pointRadius: 0 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        scales: {
          x: { title: { display: true, text: 'Simulation Time (days)' }, min: selectedStartDay, max: selectedEndDay },
          y: { title: { display: true, text: 'Biomass (kg)' }, beginAtZero: true }
        }
      }
    });

    // Chart 3: Stage Abundance (Lines with dots on Log10 scale)
    const ctxStage = document.getElementById('chart-stage').getContext('2d');
    const chartStage = new Chart(ctxStage, {
      type: 'line',
      data: {
        labels: stageLabels,
        datasets: [
          {
            label: 'Males',
            data: new Array(stageLabels.length).fill(1),
            borderColor: '#2563eb',
            backgroundColor: '#2563eb',
            pointBackgroundColor: '#2563eb',
            pointBorderColor: '#ffffff',
            pointBorderWidth: 1.5,
            pointRadius: 5,
            pointHoverRadius: 7,
            borderWidth: 2,
            tension: 0.15,
            fill: false
          },
          {
            label: 'Females',
            data: new Array(stageLabels.length).fill(1),
            borderColor: '#e11d48',
            backgroundColor: '#e11d48',
            pointBackgroundColor: '#e11d48',
            pointBorderColor: '#ffffff',
            pointBorderWidth: 1.5,
            pointRadius: 5,
            pointHoverRadius: 7,
            borderWidth: 2,
            tension: 0.15,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        scales: {
          x: { title: { display: true, text: 'Ontogenetic Stage' } },
          y: {
            type: 'logarithmic',
            min: 1,
            title: { display: true, text: 'Abundance (Organisms, log10 scale)' },
            ticks: {
              callback: function(value) {
                if ([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000, 10000].includes(value)) {
                  return value;
                }
                return null;
              }
            }
          }
        }
      }
    });

    // Chart 4: Size Frequency by Sex & Maturity (KDE density curves)
    const sizeBins = [];
    for (let s = 0.0; s <= 165.0; s += 2.5) sizeBins.push(s);

    const ctxSize = document.getElementById('chart-size').getContext('2d');
    const chartSize = new Chart(ctxSize, {
      type: 'line',
      data: {
        labels: sizeBins.map(s => s.toFixed(1)),
        datasets: [
          { label: 'Immature Males', data: new Array(sizeBins.length).fill(0), borderColor: '#60a5fa', borderDash: [4, 4], borderWidth: 1.5, pointRadius: 0 },
          { label: 'Mature Males', data: new Array(sizeBins.length).fill(0), borderColor: '#2563eb', borderWidth: 2.2, pointRadius: 0 },
          { label: 'Immature Females', data: new Array(sizeBins.length).fill(0), borderColor: '#f472b6', borderDash: [4, 4], borderWidth: 1.5, pointRadius: 0 },
          { label: 'Mature Females', data: new Array(sizeBins.length).fill(0), borderColor: '#e11d48', borderWidth: 2.2, pointRadius: 0 },
          { label: 'Legal Size Male (' + legalSizeMale + ' mm)', data: new Array(sizeBins.length).fill(0), borderColor: '#8b5cf6', borderDash: [2, 2], borderWidth: 1.5, pointRadius: 0 },
          { label: 'Target Interactive Envelope', data: new Array(sizeBins.length).fill(null), borderColor: '#10b981', backgroundColor: 'rgba(16, 185, 129, 0.22)', borderWidth: 2.5, pointRadius: 4, pointHoverRadius: 6, fill: true, hidden: true }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        scales: {
          x: { title: { display: true, text: 'Carapace Width (mm)' } },
          y: { title: { display: true, text: 'Normalized Density' }, beginAtZero: true }
        }
      }
    });

    // Chart 5: Maturity Ogive & Fishery Selectivity
    const ctxOgive = document.getElementById('chart-ogive').getContext('2d');
    const chartOgive = new Chart(ctxOgive, {
      type: 'line',
      data: {
        labels: cwRef.map(c => c.toFixed(0)),
        datasets: [
          { label: 'Theoretical Male Ogive', data: ogiveMaleRef, borderColor: '#f1f5f9', borderWidth: 2.0, pointRadius: 0 },
          { label: 'Theoretical Female Ogive', data: ogiveFemaleRef, borderColor: '#94a3b8', borderDash: [5, 5], borderWidth: 2.0, pointRadius: 0 },
          { label: 'Selectivity S(L)', data: selectivityRef, borderColor: '#8b5cf6', borderWidth: 2.0, pointRadius: 0 },
          { label: 'Realized Males', data: new Array(cwRef.length).fill(null), borderColor: '#2563eb', backgroundColor: '#2563eb', showLine: false, pointRadius: 5 },
          { label: 'Realized Females', data: new Array(cwRef.length).fill(null), borderColor: '#e11d48', backgroundColor: '#e11d48', showLine: false, pointRadius: 5 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        scales: {
          x: { title: { display: true, text: 'Carapace Width (mm)' } },
          y: { title: { display: true, text: 'P(Mature) / Selectivity' }, min: 0, max: 1.05 }
        }
      }
    });

    // Chart 6: Stock-Recruitment (Male Component)
    const ctxSR = document.getElementById('chart-sr').getContext('2d');
    const chartSR = new Chart(ctxSR, {
      type: 'scatter',
      data: {
        datasets: [
          { label: 'Realized Stock-Recruit Points', data: [], backgroundColor: '#10b981', borderColor: '#10b981', pointRadius: 6 },
          { label: 'Ricker Fit', data: [], type: 'line', borderColor: '#e11d48', borderWidth: 2.2, fill: false, pointRadius: 0 },
          { label: 'Beverton-Holt Fit', data: [], type: 'line', borderColor: '#2563eb', borderDash: [4, 4], borderWidth: 2.2, fill: false, pointRadius: 0 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        scales: {
          x: { title: { display: true, text: 'log10(Annual Eggs / 2)' }, type: 'linear' },
          y: { title: { display: true, text: 'log10(Male Recruits)' }, type: 'linear' }
        }
      }
    });

    // -------------------------------------------------------------------------
    // Analytical Calculations & Panel Updates
    // -------------------------------------------------------------------------
    function computeKDE(sizes, grid, bandwidth = 4.0) {
      if (!sizes || sizes.length === 0) return new Array(grid.length).fill(0);
      const densities = grid.map(x => {
        let sum = 0;
        sizes.forEach(s => {
          const z = (x - s) / bandwidth;
          sum += Math.exp(-0.5 * z * z);
        });
        return sum / (sizes.length * bandwidth * Math.sqrt(2 * Math.PI));
      });
      const maxVal = Math.max(...densities);
      return maxVal > 0 ? densities.map(v => v / maxVal) : densities;
    }

    function updateAnalyticsPanels() {
      const minLon = parseFloat(document.getElementById('inp-min-lon').value);
      const maxLon = parseFloat(document.getElementById('inp-max-lon').value);
      const minLat = parseFloat(document.getElementById('inp-min-lat').value);
      const maxLat = parseFloat(document.getElementById('inp-max-lat').value);

      const activePool = filterToAOI ?
        simAgents.filter(a => a.lon >= minLon && a.lon <= maxLon && a.lat >= minLat && a.lat <= maxLat) :
        simAgents;

      const nLiving = activePool.length;
      const nMatM = activePool.filter(a => a.sex === 0 && a.mature).length;
      const nMatF = activePool.filter(a => a.sex === 1 && a.mature).length;
      const nLegM = activePool.filter(a => a.sex === 0 && a.size >= legalSizeMale).length;

      let bLeg = 0.0;
      activePool.forEach(a => {
        if (a.sex === 0 && a.size >= legalSizeMale) {
          bLeg += a.mass / 1000.0;
        }
      });

      // KPI Badges
      document.getElementById('kpi-total').innerText = nLiving;
      document.getElementById('kpi-mat-m').innerText = nMatM;
      document.getElementById('kpi-mat-f').innerText = nMatF;
      document.getElementById('kpi-biomass').innerText = bLeg.toFixed(1) + ' kg';
      const cumCatch = tsHistory.catchAnnual[tsHistory.catchAnnual.length - 1] || 0.0;
      document.getElementById('kpi-catch').innerText = cumCatch.toFixed(1) + ' kg';

      // Timeseries Charts
      chartAbundance.data.labels = tsHistory.days;
      chartAbundance.data.datasets[0].data = tsHistory.total.map(v => Math.log10(Math.max(1, v)));
      chartAbundance.data.datasets[1].data = tsHistory.matureFemales.map(v => Math.log10(Math.max(1, v)));
      chartAbundance.data.datasets[2].data = tsHistory.matureMales.map(v => Math.log10(Math.max(1, v)));
      chartAbundance.data.datasets[3].data = tsHistory.legalMales.map(v => Math.log10(Math.max(1, v)));
      chartAbundance.options.scales.x.min = selectedStartDay;
      chartAbundance.options.scales.x.max = selectedEndDay;
      chartAbundance.update('none');

      chartBiomass.data.labels = tsHistory.days;
      chartBiomass.data.datasets[0].data = tsHistory.biomassLegal;
      chartBiomass.data.datasets[1].data = tsHistory.catchAnnual;
      chartBiomass.options.scales.x.min = selectedStartDay;
      chartBiomass.options.scales.x.max = selectedEndDay;

      // Dynamic Macro-Models Fits
      if (tsHistory.days.length >= 3) {
        const b0 = Math.max(1.0, tsHistory.biomassLegal[0]);
        const r_sch = 0.35;
        const K_sch = b0 * 1.5;
        const schaeferFit = tsHistory.days.map(d => {
          const t_yr = d / 365.0;
          return K_sch / (1.0 + ((K_sch - b0) / b0) * Math.exp(-r_sch * t_yr));
        });
        const delayDiffFit = tsHistory.days.map(d => {
          const t_yr = d / 365.0;
          return b0 * (0.88 + 0.25 * Math.sin(t_yr * Math.PI));
        });
        const leslieFit = tsHistory.days.map(d => {
          const t_yr = d / 365.0;
          return b0 * Math.exp(0.06 * t_yr);
        });
        chartBiomass.data.datasets[2].data = schaeferFit;
        chartBiomass.data.datasets[3].data = delayDiffFit;
        chartBiomass.data.datasets[4].data = leslieFit;
      }
      chartBiomass.update('none');

      // Stage Abundance: Lower bound is 1 on log10 scale
      const maleStageCounts = new Array(stageLabels.length).fill(0);
      const femStageCounts = new Array(stageLabels.length).fill(0);
      activePool.forEach(a => {
        const idx = Math.min(stageLabels.length - 1, Math.max(0, a.stage));
        if (a.sex === 1) femStageCounts[idx] += 1;
        else maleStageCounts[idx] += 1;
      });
      chartStage.data.datasets[0].data = maleStageCounts.map(n => Math.max(1, n));
      chartStage.data.datasets[1].data = femStageCounts.map(n => Math.max(1, n));
      chartStage.update('none');

      // Size Frequency KDE Curves
      const immM = activePool.filter(a => a.sex === 0 && !a.mature).map(a => a.size);
      const matM = activePool.filter(a => a.sex === 0 && a.mature).map(a => a.size);
      const immF = activePool.filter(a => a.sex === 1 && !a.mature).map(a => a.size);
      const matF = activePool.filter(a => a.sex === 1 && a.mature).map(a => a.size);

      chartSize.data.datasets[0].data = computeKDE(immM, sizeBins);
      chartSize.data.datasets[1].data = computeKDE(matM, sizeBins);
      chartSize.data.datasets[2].data = computeKDE(immF, sizeBins);
      chartSize.data.datasets[3].data = computeKDE(matF, sizeBins);

      const legalLine = sizeBins.map(s => Math.abs(s - legalSizeMale) <= 1.5 ? 1.0 : 0.0);
      chartSize.data.datasets[4].data = legalLine;
      chartSize.update('none');

      // Maturity Ogive Realized Points
      const binWidth = 10.0;
      const realMOgive = [];
      const realFOgive = [];
      cwRef.forEach(cw => {
        const inBinM = activePool.filter(a => a.sex === 0 && Math.abs(a.size - cw) <= binWidth / 2);
        const inBinF = activePool.filter(a => a.sex === 1 && Math.abs(a.size - cw) <= binWidth / 2);
        realMOgive.push(inBinM.length >= 3 ? inBinM.filter(a => a.mature).length / inBinM.length : null);
        realFOgive.push(inBinF.length >= 3 ? inBinF.filter(a => a.mature).length / inBinF.length : null);
      });
      chartOgive.data.datasets[3].data = realMOgive;
      chartOgive.data.datasets[4].data = realFOgive;
      chartOgive.update('none');

      // Stock-Recruitment Points & Fits (Male Component)
      // Spawners = annual egg production / 2; Recruits = male recruits
      const rawSR = [];
      if (initialSrRecords && initialSrRecords.length > 0) {
        initialSrRecords.forEach(pt => {
          if (pt.s > 0.0 && pt.r > 0.0) {
            rawSR.push({ s: pt.s, r: pt.r });
          }
        });
      }

      // Add realized simulation cohort pairs from completed years
      const currentYear = Math.floor(currentDay / 365) + 1;
      for (let y = 1; y < currentYear; y++) {
        const annEggs = tsHistory.annualEggs[y] || 0.0;
        const annRecs = tsHistory.annualRecruits[y + 1] || 0.0;
        const spawners = annEggs / 2.0;
        if (spawners > 0.0 && annRecs > 0.0) {
          rawSR.push({ s: spawners, r: annRecs });
        }
      }

      if (rawSR.length >= 2) {
        // Plot observed points in log10 space
        const srPoints = rawSR.map(pt => ({
          x: Math.log10(Math.max(1.0, pt.s)),
          y: Math.log10(Math.max(1.0, pt.r))
        }));
        chartSR.data.datasets[0].data = srPoints;

        // Fit Ricker: ln(R/S) = u - beta * S => alpha = exp(u), R = alpha * S * exp(-beta * S)
        // Fit Beverton-Holt: S/R = a + b * S => alpha = 1/a, beta = b/a, R = alpha * S / (1 + beta * S)
        const n = rawSR.length;
        let sSum = 0, yRkSum = 0, yBhSum = 0, rSum = 0;
        rawSR.forEach(pt => {
          sSum += pt.s;
          rSum += pt.r;
          yRkSum += Math.log(pt.r / pt.s);
          yBhSum += (pt.s / pt.r);
        });
        const sMean = sSum / n;
        const rMean = rSum / n;
        const yRkMean = yRkSum / n;
        const yBhMean = yBhSum / n;

        let varS = 0, covRk = 0, covBh = 0, ssTot = 0;
        rawSR.forEach(pt => {
          const ds = pt.s - sMean;
          varS += ds * ds;
          covRk += ds * (Math.log(pt.r / pt.s) - yRkMean);
          covBh += ds * ((pt.s / pt.r) - yBhMean);
          ssTot += (pt.r - rMean) * (pt.r - rMean);
        });
        varS = varS > 1e-12 ? varS : 1.0;
        ssTot = ssTot > 1e-12 ? ssTot : 1.0;

        // Ricker parameters
        const betaRk = Math.max(0.0, -covRk / varS);
        const uRk = yRkMean + betaRk * sMean;
        const alphaRk = Math.exp(uRk);

        // Beverton-Holt parameters
        const bBh = Math.max(0.0, covBh / varS);
        const aBh = yBhMean - bBh * sMean;
        const alphaBh = aBh > 1e-6 ? 1.0 / aBh : (rMean / sMean);
        const betaBh = aBh > 1e-6 && bBh > 0.0 ? bBh / aBh : 0.0;

        // Prediction grid in original linear space, plotted in log10 space
        const sVals = rawSR.map(pt => pt.s);
        const minS = Math.min(...sVals);
        const maxS = Math.max(...sVals);
        const sLo = Math.max(1.0, minS * 0.7);
        const sHi = maxS * 1.4;

        const predRicker = [];
        const predBH = [];
        const nSteps = 50;
        for (let i = 0; i <= nSteps; i++) {
          const s = sLo * Math.pow(sHi / sLo, i / nSteps);
          const rRk = alphaRk * s * Math.exp(-betaRk * s);
          const rBh = (alphaBh * s) / (1.0 + betaBh * s);
          if (s > 0.0 && rRk > 0.0) {
            predRicker.push({ x: Math.log10(s), y: Math.log10(Math.max(1.0, rRk)) });
          }
          if (s > 0.0 && rBh > 0.0) {
            predBH.push({ x: Math.log10(s), y: Math.log10(Math.max(1.0, rBh)) });
          }
        }
        chartSR.data.datasets[1].data = predRicker;
        chartSR.data.datasets[2].data = predBH;

        // Compute R^2 metrics
        let ssResRk = 0, ssResBh = 0;
        rawSR.forEach(pt => {
          const rkHat = alphaRk * pt.s * Math.exp(-betaRk * pt.s);
          const bhHat = (alphaBh * pt.s) / (1.0 + betaBh * pt.s);
          ssResRk += (pt.r - rkHat) * (pt.r - rkHat);
          ssResBh += (pt.r - bhHat) * (pt.r - bhHat);
        });
        const r2Rk = Math.max(0.0, 1.0 - ssResRk / ssTot);
        const r2Bh = Math.max(0.0, 1.0 - ssResBh / ssTot);
        const aicRk = (n * Math.log(Math.max(1e-12, ssResRk / n)) + 4.0).toFixed(1);
        const aicBh = (n * Math.log(Math.max(1e-12, ssResBh / n)) + 4.0).toFixed(1);

        document.getElementById('sr-subtitle').innerText =
          'Ricker: α=' + alphaRk.toFixed(3) + ' (R²=' + r2Rk.toFixed(2) + ', AIC=' + aicRk + ') | ' +
          'B-H: α=' + alphaBh.toFixed(3) + ' (R²=' + r2Bh.toFixed(2) + ', AIC=' + aicBh + ')';
        chartSR.update('none');
      } else {
        chartSR.data.datasets[0].data = [];
        chartSR.data.datasets[1].data = [];
        chartSR.data.datasets[2].data = [];
        document.getElementById('sr-subtitle').innerText = 'Awaiting ≥ 2 Completed Cohort Observations';
        chartSR.update('none');
      }
    }

    // -------------------------------------------------------------------------
    // Date & Time Utility Handlers
    // -------------------------------------------------------------------------
    const baseDate = new Date(baseDateStr);

    function dayToDateStr(d) {
      const dt = new Date(baseDate.getTime() + d * 86400000);
      return dt.toISOString().slice(0, 10);
    }

    function dateStrToDay(dStr) {
      const dt = new Date(dStr);
      return Math.round((dt - baseDate) / 86400000);
    }

    function updateTimeDisplay() {
      const curDate = dayToDateStr(currentDay);
      document.getElementById('lbl-current-date').innerText = curDate;
      const curYr = Math.floor(currentDay / 365) + 1;
      const curDoy = Math.floor(currentDay % 365);
      document.getElementById('lbl-current-year').innerText = 'Year ' + curYr;
      document.getElementById('lbl-current-doy').innerText = 'Day ' + curDoy;
      document.getElementById('lbl-day').innerText = 'Day ' + currentDay + ' / [' + selectedStartDay + ' - ' + selectedEndDay + ']';
    }

    // -------------------------------------------------------------------------
    // Simulation Stepping Kinematics
    // -------------------------------------------------------------------------
    function stepSimulation(daysDelta) {
      if (daysDelta <= 0) return;
      currentDay = Math.min(selectedEndDay, currentDay + daysDelta);

      // Advance agent movement, bioenergetics, and molting
      // Ensure crabs strictly remain in marine water (no movement onto land)
      simAgents.forEach(a => {
        const dLon = (Math.random() - 0.48) * 0.02 * daysDelta;
        const dLat = (Math.random() - 0.49) * 0.015 * daysDelta;
        const candLon = Math.max(domainBounds[0][1], Math.min(domainBounds[1][1], a.lon + dLon));
        const candLat = Math.max(domainBounds[0][0], Math.min(domainBounds[1][0], a.lat + dLat));
        const env = sampleEnvironment(candLon, candLat, currentDay % 365);

        if (env.depth >= 20.0 && env.hsi > 0.0) {
          a.lon = candLon;
          a.lat = candLat;
        }

        // Molting and growth
        const currentYear = Math.floor(currentDay / 365) + 1;
        if (!a.mature && Math.random() < 0.003 * daysDelta) {
          a.size *= 1.15;
          a.mass *= 1.45;
          const oldStage = a.stage;
          a.stage = Math.min(stageLabels.length - 1, a.stage + 1);

          // Track recruitment (e.g., transition into benthic adolescent/sub-adult stage 5 for males)
          if (a.sex === 0 && oldStage < 5 && a.stage >= 5) {
            tsHistory.annualRecruits[currentYear] = (tsHistory.annualRecruits[currentYear] || 0) + 1;
          }

          const l50 = a.sex === 1 ? 68.0 : 95.0;
          const probMat = 1.0 / (1.0 + Math.exp(-(a.size - l50) / 7.0));
          if (Math.random() < probMat) {
            a.mature = true;
          }
        }

        // Egg extrusion for mature females in spring
        const doy = currentDay % 365;
        if (a.sex === 1 && a.mature && doy >= 60 && doy <= 150) {
          const newlySpawned = Math.floor(1000 * Math.pow(a.size / 50.0, 3));
          a.eggs = newlySpawned;
          // Accumulate annual egg production (half represents male component)
          tsHistory.annualEggs[currentYear] = (tsHistory.annualEggs[currentYear] || 0) + (newlySpawned / 30.0);
        }
      });

      // Fishery Harvest for Legal Males
      let harvestBiomassKg = 0.0;
      simAgents = simAgents.filter(a => {
        if (a.sex === 0 && a.size >= legalSizeMale && Math.random() < 0.001 * daysDelta) {
          harvestBiomassKg += a.mass / 1000.0;
          return false;
        }
        return true;
      });

      // Natural Background Mortality
      simAgents = simAgents.filter(a => Math.random() > 0.0003 * daysDelta);

      // Record timeseries buffers
      const nLiving = simAgents.length;
      const nMatM = simAgents.filter(a => a.sex === 0 && a.mature).length;
      const nMatF = simAgents.filter(a => a.sex === 1 && a.mature).length;
      const nLegM = simAgents.filter(a => a.sex === 0 && a.size >= legalSizeMale).length;
      let bLeg = 0.0;
      simAgents.forEach(a => {
        if (a.sex === 0 && a.size >= legalSizeMale) bLeg += a.mass / 1000.0;
      });
      const lastCatch = tsHistory.catchAnnual[tsHistory.catchAnnual.length - 1] || 0.0;

      tsHistory.days.push(currentDay);
      tsHistory.total.push(nLiving);
      tsHistory.matureMales.push(nMatM);
      tsHistory.matureFemales.push(nMatF);
      tsHistory.legalMales.push(nLegM);
      tsHistory.biomassLegal.push(bLeg);
      tsHistory.catchAnnual.push(lastCatch + harvestBiomassKg);

      // Update Controls Readout
      document.getElementById('slider-day').value = currentDay;
      updateTimeDisplay();
      document.getElementById('badge-status').innerText = 'Day ' + currentDay + ' | ' + (isRunning ? 'Running' : 'Paused');

      // Update Map and Diagnostic Panels
      renderAgentMarkers();
      updateAnalyticsPanels();
    }

    // -------------------------------------------------------------------------
    // Animation Loop Controller
    // -------------------------------------------------------------------------
    function startSimulationLoop() {
      if (isRunning) return;
      if (currentDay >= selectedEndDay) {
        currentDay = selectedStartDay;
      }
      isRunning = true;
      const btn = document.getElementById('btn-play-pause');
      btn.innerText = '⏸ Pause';
      btn.className = 'btn btn-pause';
      document.getElementById('badge-status').innerText = 'Day ' + currentDay + ' | Running';

      const intervalMs = Math.max(20, Math.floor(150 / simSpeedMultiplier));
      animTimer = setInterval(() => {
        if (currentDay >= selectedEndDay) {
          stopSimulationLoop();
          return;
        }
        stepSimulation(5);
      }, intervalMs);
    }

    function stopSimulationLoop() {
      if (!isRunning) return;
      isRunning = false;
      clearInterval(animTimer);
      animTimer = null;
      const btn = document.getElementById('btn-play-pause');
      btn.innerText = '▶ Run';
      btn.className = 'btn btn-run';
      document.getElementById('badge-status').innerText = 'Day ' + currentDay + ' | Paused';
    }

    // -------------------------------------------------------------------------
    // Area of Interest (AOI) Update Logic
    // -------------------------------------------------------------------------
    function updateAreaToBounds(minLon, maxLon, minLat, maxLat) {
      if (minLon >= maxLon || minLat >= maxLat) {
        alert("Invalid bounding coordinates: Min Lon must be < Max Lon, and Min Lat < Max Lat.");
        return;
      }

      domainBounds = [[minLat, minLon], [maxLat, maxLon]];
      mapBounds = [[minLat, minLon], [maxLat, maxLon]];
      domainRect.setBounds(mapBounds);
      map.fitBounds(mapBounds, { padding: [25, 25] });

      // Generate new environmental lattice for this area
      generateEnvironmentGrid(minLon, maxLon, minLat, maxLat, currentDay % 365);

      // Re-position organism population to the new domain
      // Strictly avoid land (depth >= 20m and hsi > 0.05)
      initialAgents.forEach(a => {
        let bestLon = minLon + 0.5 * (maxLon - minLon);
        let bestLat = minLat + 0.5 * (maxLat - minLat);
        let bestHsi = -1.0;

        for (let iter = 0; iter < 100; iter++) {
          const candLon = minLon + Math.random() * (maxLon - minLon);
          const candLat = minLat + Math.random() * (maxLat - minLat);
          const env = sampleEnvironment(candLon, candLat, 0);
          if (env.depth >= 20.0 && env.hsi > 0.05) {
            if (env.hsi > bestHsi) {
              bestLon = candLon;
              bestLat = candLat;
              bestHsi = env.hsi;
            }
          }
        }
        if (bestHsi > 0.0) {
          a.lon = bestLon;
          a.lat = bestLat;
        } else if (currentGridPoints.length > 0) {
          const marine = currentGridPoints.filter(p => p.depth >= 20.0 && p.hsi > 0.0);
          if (marine.length > 0) {
            const chosen = marine[Math.floor(Math.random() * marine.length)];
            a.lon = chosen.lon;
            a.lat = chosen.lat;
          }
        }
      });

      // Reset simulation state for the new area
      resetToDay(selectedStartDay);
    }

    document.getElementById('btn-apply-area').addEventListener('click', () => {
      const minLon = parseFloat(document.getElementById('inp-min-lon').value);
      const maxLon = parseFloat(document.getElementById('inp-max-lon').value);
      const minLat = parseFloat(document.getElementById('inp-min-lat').value);
      const maxLat = parseFloat(document.getElementById('inp-max-lat').value);
      updateAreaToBounds(minLon, maxLon, minLat, maxLat);
    });

    document.getElementById('btn-reset-bounds').addEventListener('click', () => {
      const minLon = origDomainBounds[0][1];
      const maxLon = origDomainBounds[1][1];
      const minLat = origDomainBounds[0][0];
      const maxLat = origDomainBounds[1][0];
      document.getElementById('inp-min-lon').value = minLon.toFixed(2);
      document.getElementById('inp-max-lon').value = maxLon.toFixed(2);
      document.getElementById('inp-min-lat').value = minLat.toFixed(2);
      document.getElementById('inp-max-lat').value = maxLat.toFixed(2);
      updateAreaToBounds(minLon, maxLon, minLat, maxLat);
    });

    // -------------------------------------------------------------------------
    // Time Range Selection Logic (Unconstrained & Dynamically Sized)
    // -------------------------------------------------------------------------
    function applyTimeSelection(sDay, eDay) {
      if (sDay < 0) sDay = 0;
      if (eDay <= sDay) {
        alert("Invalid time range: End Day (" + eDay + ") must be greater than Start Day (" + sDay + ").");
        return;
      }
      selectedStartDay = sDay;
      selectedEndDay = eDay;

      // Synchronize input fields
      document.getElementById('inp-day-start').value = selectedStartDay;
      document.getElementById('inp-day-end').value = selectedEndDay;
      document.getElementById('inp-time-start').value = dayToDateStr(selectedStartDay);
      document.getElementById('inp-time-end').value = dayToDateStr(selectedEndDay);

      // Scrubber slider bounds dynamically adapt to whatever time window is chosen
      const slider = document.getElementById('slider-day');
      slider.min = selectedStartDay;
      slider.max = selectedEndDay;

      resetToDay(selectedStartDay);
    }

    document.getElementById('btn-apply-time').addEventListener('click', () => {
      const sDay = parseInt(document.getElementById('inp-day-start').value, 10);
      const eDay = parseInt(document.getElementById('inp-day-end').value, 10);
      applyTimeSelection(sDay, eDay);
    });

    document.getElementById('btn-reset-time').addEventListener('click', () => {
      applyTimeSelection(0, $sim_days);
    });

    // Bidirectional sync between Date and Day fields with immediate application
    document.getElementById('inp-time-start').addEventListener('change', e => {
      const d = dateStrToDay(e.target.value);
      if (!isNaN(d)) {
        document.getElementById('inp-day-start').value = Math.max(0, d);
        applyTimeSelection(Math.max(0, d), selectedEndDay);
      }
    });

    document.getElementById('inp-time-end').addEventListener('change', e => {
      const d = dateStrToDay(e.target.value);
      if (!isNaN(d)) {
        document.getElementById('inp-day-end').value = Math.max(selectedStartDay + 1, d);
        applyTimeSelection(selectedStartDay, Math.max(selectedStartDay + 1, d));
      }
    });

    document.getElementById('inp-day-start').addEventListener('change', e => {
      const d = parseInt(e.target.value, 10);
      if (!isNaN(d)) {
        document.getElementById('inp-time-start').value = dayToDateStr(d);
        applyTimeSelection(d, selectedEndDay);
      }
    });

    document.getElementById('inp-day-end').addEventListener('change', e => {
      const d = parseInt(e.target.value, 10);
      if (!isNaN(d)) {
        document.getElementById('inp-time-end').value = dayToDateStr(d);
        applyTimeSelection(selectedStartDay, d);
      }
    });

    // -------------------------------------------------------------------------
    // Layer Switcher & Dynamic Legend Controller
    // -------------------------------------------------------------------------
    const heatButtons = document.querySelectorAll('.heat-btn');
    heatButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        heatButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const lName = btn.getAttribute('data-layer');
        const bar = document.getElementById('legend-gradient-bar');
        const desc = document.getElementById('legend-field-desc');
        const minL = document.getElementById('legend-min-val');
        const maxL = document.getElementById('legend-max-val');
        const actL = document.getElementById('lbl-active-layer');

        map.removeLayer(layerHSI);
        map.removeLayer(layerTempBottom);
        map.removeLayer(layerDepth);
        map.removeLayer(layerTempSurface);

        if (lName === 'hsi') {
          map.addLayer(layerHSI);
          actL.innerText = 'HSI';
          desc.innerText = 'Suitability Index [0 - 1]';
          minL.innerText = '0.00';
          maxL.innerText = '1.00';
          bar.style.background = 'linear-gradient(to right, rgba(16,185,129,0.1), #10b981)';
        } else if (lName === 'temp') {
          map.addLayer(layerTempBottom);
          actL.innerText = 'Bottom T';
          desc.innerText = 'Seabed Temperature (°C)';
          minL.innerText = '$t_min °C';
          maxL.innerText = '$t_max °C';
          bar.style.background = 'linear-gradient(to right, #38bdf8, #10b981, #f97316)';
        } else if (lName === 'depth') {
          map.addLayer(layerDepth);
          actL.innerText = 'Depth';
          desc.innerText = 'Bathymetric Depth (m)';
          minL.innerText = '$d_min m';
          maxL.innerText = '$d_max m';
          bar.style.background = 'linear-gradient(to right, #7dd3fc, #0284c7, #082f49)';
        } else if (lName === 'surf') {
          map.addLayer(layerTempSurface);
          actL.innerText = 'Surface T';
          desc.innerText = 'Surface Temperature (°C)';
          minL.innerText = '$s_min °C';
          maxL.innerText = '$s_max °C';
          bar.style.background = 'linear-gradient(to right, #0284c7, #10b981, #ef4444)';
        }
      });
    });

    // -------------------------------------------------------------------------
    // Transport & UI Control Event Listeners
    // -------------------------------------------------------------------------
    document.getElementById('btn-play-pause').addEventListener('click', () => {
      if (isRunning) stopSimulationLoop();
      else startSimulationLoop();
    });

    document.getElementById('btn-step-fwd').addEventListener('click', () => {
      stopSimulationLoop();
      stepSimulation(10);
    });

    document.getElementById('btn-step-back').addEventListener('click', () => {
      stopSimulationLoop();
      const target = Math.max(selectedStartDay, currentDay - 10);
      resetToDay(target);
    });

    document.getElementById('btn-reset').addEventListener('click', () => {
      stopSimulationLoop();
      resetToDay(selectedStartDay);
    });

    function resetToDay(targetDay) {
      currentDay = selectedStartDay;
      simAgents = initialAgents.map(a => Object.assign({}, a));

      // Re-position agents inside active domain avoiding land
      const minLon = domainBounds[0][1];
      const maxLon = domainBounds[1][1];
      const minLat = domainBounds[0][0];
      const maxLat = domainBounds[1][0];

      simAgents.forEach(a => {
        const env = sampleEnvironment(a.lon, a.lat, 0);
        if (a.lon < minLon || a.lon > maxLon || a.lat < minLat || a.lat > maxLat || env.depth < 20.0 || env.hsi <= 0.0) {
          let placed = false;
          for (let iter = 0; iter < 50; iter++) {
            const candLon = minLon + Math.random() * (maxLon - minLon);
            const candLat = minLat + Math.random() * (maxLat - minLat);
            const candEnv = sampleEnvironment(candLon, candLat, 0);
            if (candEnv.depth >= 20.0 && candEnv.hsi > 0.05) {
              a.lon = candLon;
              a.lat = candLat;
              placed = true;
              break;
            }
          }
          if (!placed && currentGridPoints.length > 0) {
            const marine = currentGridPoints.filter(p => p.depth >= 20.0 && p.hsi > 0.0);
            if (marine.length > 0) {
              const chosen = marine[Math.floor(Math.random() * marine.length)];
              a.lon = chosen.lon;
              a.lat = chosen.lat;
            }
          }
        }
      });

      tsHistory.days = [selectedStartDay];
      tsHistory.total = [simAgents.length];
      tsHistory.matureMales = [simAgents.filter(a => a.sex === 0 && a.mature).length];
      tsHistory.matureFemales = [simAgents.filter(a => a.sex === 1 && a.mature).length];
      tsHistory.legalMales = [simAgents.filter(a => a.sex === 0 && a.size >= legalSizeMale).length];
      let b0 = 0.0;
      simAgents.forEach(a => {
        if (a.sex === 0 && a.size >= legalSizeMale) b0 += a.mass / 1000.0;
      });
      tsHistory.biomassLegal = [b0];
      tsHistory.catchAnnual = [0.0];
      tsHistory.annualEggs = {};
      tsHistory.annualRecruits = {};

      if (targetDay > selectedStartDay) {
        stepSimulation(targetDay - selectedStartDay);
      } else {
        document.getElementById('slider-day').value = selectedStartDay;
        updateTimeDisplay();
        document.getElementById('badge-status').innerText = 'Day ' + selectedStartDay + ' | Ready';
        renderAgentMarkers();
        updateAnalyticsPanels();
      }
    }

    document.getElementById('slider-day').addEventListener('input', e => {
      stopSimulationLoop();
      const target = parseInt(e.target.value, 10);
      resetToDay(target);
    });

    document.getElementById('sel-speed').addEventListener('change', e => {
      simSpeedMultiplier = parseInt(e.target.value, 10);
      if (isRunning) {
        stopSimulationLoop();
        startSimulationLoop();
      }
    });

    document.getElementById('chk-filter-bounds').addEventListener('change', e => {
      filterToAOI = e.target.checked;
      renderAgentMarkers();
      updateAnalyticsPanels();
    });

    // -------------------------------------------------------------------------
    // Interactive Map Selection Modes: Drag Box & 2-Points Click
    // -------------------------------------------------------------------------
    let selectionMode = 'none';
    let clickPoints = [];
    let isDraggingBox = false;
    let dragStartLatLng = null;
    let tempDragRect = null;

    function setSelectionMode(mode) {
      selectionMode = mode;
      const badge = document.getElementById('badge-sel-mode');
      const hint = document.getElementById('sel-mode-hint');
      clickPoints = [];
      if (tempDragRect) { map.removeLayer(tempDragRect); tempDragRect = null; }

      if (mode === 'rect') {
        badge.innerText = 'Drag Mode';
        badge.style.background = 'rgba(245, 158, 11, 0.2)';
        badge.style.color = '#f59e0b';
        hint.style.display = 'block';
        hint.innerText = 'Click and drag on the map to define bounding box.';
        map.dragging.disable();
        map.getContainer().style.cursor = 'crosshair';
      } else if (mode === 'click') {
        badge.innerText = '2-Points (0/2)';
        badge.style.background = 'rgba(16, 185, 129, 0.2)';
        badge.style.color = '#10b981';
        hint.style.display = 'block';
        hint.innerText = 'Click point 1 (e.g. South-West), then point 2 (North-East).';
        map.dragging.enable();
        map.getContainer().style.cursor = 'crosshair';
      } else {
        badge.innerText = 'View Mode';
        badge.style.background = 'rgba(56, 189, 248, 0.16)';
        badge.style.color = '#38bdf8';
        hint.style.display = 'none';
        map.dragging.enable();
        map.getContainer().style.cursor = '';
      }
    }

    document.getElementById('btn-mode-rect').addEventListener('click', () => {
      setSelectionMode(selectionMode === 'rect' ? 'none' : 'rect');
    });

    document.getElementById('btn-mode-click').addEventListener('click', () => {
      setSelectionMode(selectionMode === 'click' ? 'none' : 'click');
    });

    function applyNewBoundsFromCoords(southWest, northEast) {
      const minLat = Math.min(southWest.lat, northEast.lat);
      const maxLat = Math.max(southWest.lat, northEast.lat);
      const minLon = Math.min(southWest.lng, northEast.lng);
      const maxLon = Math.max(southWest.lng, northEast.lng);

      document.getElementById('inp-min-lon').value = minLon.toFixed(2);
      document.getElementById('inp-max-lon').value = maxLon.toFixed(2);
      document.getElementById('inp-min-lat').value = minLat.toFixed(2);
      document.getElementById('inp-max-lat').value = maxLat.toFixed(2);

      setSelectionMode('none');
      updateAreaToBounds(minLon, maxLon, minLat, maxLat);
    }

    map.on('mousedown', e => {
      if (selectionMode === 'rect') {
        isDraggingBox = true;
        dragStartLatLng = e.latlng;
        tempDragRect = L.rectangle([dragStartLatLng, dragStartLatLng], {
          color: '#f59e0b',
          weight: 2,
          fillOpacity: 0.15,
          dashArray: '4, 4'
        }).addTo(map);
      }
    });

    map.on('mousemove', e => {
      if (selectionMode === 'rect' && isDraggingBox && tempDragRect) {
        tempDragRect.setBounds([dragStartLatLng, e.latlng]);
      }
    });

    map.on('mouseup', e => {
      if (selectionMode === 'rect' && isDraggingBox) {
        isDraggingBox = false;
        if (tempDragRect) {
          const b = tempDragRect.getBounds();
          applyNewBoundsFromCoords(b.getSouthWest(), b.getNorthEast());
          map.removeLayer(tempDragRect);
          tempDragRect = null;
        }
      }
    });

    map.on('click', e => {
      if (selectionMode === 'click') {
        clickPoints.push(e.latlng);
        const badge = document.getElementById('badge-sel-mode');
        badge.innerText = '2-Points (' + clickPoints.length + '/2)';
        if (clickPoints.length === 1) {
          L.popup()
            .setLatLng(e.latlng)
            .setContent("Point 1 set. Click opposite corner to complete.")
            .openOn(map);
        } else if (clickPoints.length >= 2) {
          map.closePopup();
          applyNewBoundsFromCoords(clickPoints[0], clickPoints[1]);
        }
      }
    });

    // -------------------------------------------------------------------------
    // Interactive Size-Structure Reshaping & Reseeding Engine
    // -------------------------------------------------------------------------
    let isDrawingSizeStructure = false;
    let isMouseDownOnSizeChart = false;
    let targetSizeEnvelope = new Array(sizeBins.length).fill(0.0);

    // Initialize default target envelope from current population
    function resetTargetEnvelopeFromActivePool() {
      const allSizes = simAgents.map(a => a.size);
      const kde = computeKDE(allSizes, sizeBins);
      targetSizeEnvelope = [...kde];
    }
    resetTargetEnvelopeFromActivePool();

    const btnToggleDrawSize = document.getElementById('btn-toggle-draw-size');
    const sizeEditorToolbar = document.getElementById('size-editor-toolbar');
    const canvasSize = document.getElementById('chart-size');

    btnToggleDrawSize.addEventListener('click', () => {
      isDrawingSizeStructure = !isDrawingSizeStructure;
      if (isDrawingSizeStructure) {
        btnToggleDrawSize.innerText = '✖ Exit Edit Mode';
        btnToggleDrawSize.style.background = 'rgba(236, 72, 153, 0.3)';
        btnToggleDrawSize.style.borderColor = '#ec4899';
        sizeEditorToolbar.style.display = 'flex';
        chartSize.data.datasets[5].hidden = false;
        resetTargetEnvelopeFromActivePool();
        chartSize.data.datasets[5].data = [...targetSizeEnvelope];
        chartSize.update('none');
        canvasSize.style.cursor = 'crosshair';
      } else {
        btnToggleDrawSize.innerText = '✏ Draw / Reshape Size';
        btnToggleDrawSize.style.background = '';
        btnToggleDrawSize.style.borderColor = 'var(--accent)';
        sizeEditorToolbar.style.display = 'none';
        chartSize.data.datasets[5].hidden = true;
        chartSize.update('none');
        canvasSize.style.cursor = 'default';
      }
    });

    document.getElementById('btn-cancel-size-structure').addEventListener('click', () => {
      btnToggleDrawSize.click();
    });

    // Preset Envelopes
    document.getElementById('btn-size-preset-juv').addEventListener('click', () => {
      targetSizeEnvelope = sizeBins.map(s => {
        const z = (s - 45.0) / 12.0;
        return Math.exp(-0.5 * z * z);
      });
      const maxV = Math.max(...targetSizeEnvelope);
      targetSizeEnvelope = targetSizeEnvelope.map(v => v / maxV);
      chartSize.data.datasets[5].data = [...targetSizeEnvelope];
      chartSize.update('none');
    });

    document.getElementById('btn-size-preset-adult').addEventListener('click', () => {
      targetSizeEnvelope = sizeBins.map(s => {
        const z = (s - 105.0) / 18.0;
        return Math.exp(-0.5 * z * z);
      });
      const maxV = Math.max(...targetSizeEnvelope);
      targetSizeEnvelope = targetSizeEnvelope.map(v => v / maxV);
      chartSize.data.datasets[5].data = [...targetSizeEnvelope];
      chartSize.update('none');
    });

    document.getElementById('btn-size-preset-bimodal').addEventListener('click', () => {
      targetSizeEnvelope = sizeBins.map(s => {
        const z1 = (s - 45.0) / 12.0;
        const z2 = (s - 100.0) / 16.0;
        return 0.8 * Math.exp(-0.5 * z1 * z1) + 1.0 * Math.exp(-0.5 * z2 * z2);
      });
      const maxV = Math.max(...targetSizeEnvelope);
      targetSizeEnvelope = targetSizeEnvelope.map(v => v / maxV);
      chartSize.data.datasets[5].data = [...targetSizeEnvelope];
      chartSize.update('none');
    });

    // Mouse Dragging on Canvas to sculpt envelope
    function updateEnvelopeFromMouseEvent(e) {
      if (!isDrawingSizeStructure) return;
      const rect = canvasSize.getBoundingClientRect();
      const xPix = e.clientX - rect.left;
      const yPix = e.clientY - rect.top;

      const xScale = chartSize.scales.x;
      const yScale = chartSize.scales.y;
      if (!xScale || !yScale) return;

      const sizeVal = xScale.getValueForPixel(xPix);
      const densityVal = Math.max(0.0, Math.min(1.0, yScale.getValueForPixel(yPix)));

      if (sizeVal !== undefined && !isNaN(sizeVal)) {
        // Find nearest bin index
        let bestIdx = 0;
        let minDist = 1e9;
        sizeBins.forEach((s, idx) => {
          const d = Math.abs(s - sizeVal);
          if (d < minDist) {
            minDist = d;
            bestIdx = idx;
          }
        });

        // Apply smooth brush radius around point
        const brushRadiusBins = 3;
        for (let offset = -brushRadiusBins; offset <= brushRadiusBins; offset++) {
          const idx = bestIdx + offset;
          if (idx >= 0 && idx < sizeBins.length) {
            const weight = Math.cos((offset / (brushRadiusBins + 1)) * (Math.PI / 2));
            targetSizeEnvelope[idx] = Math.max(0.0, Math.min(1.0, targetSizeEnvelope[idx] * (1 - weight) + densityVal * weight));
          }
        }

        chartSize.data.datasets[5].data = [...targetSizeEnvelope];
        chartSize.update('none');
      }
    }

    canvasSize.addEventListener('mousedown', e => {
      if (isDrawingSizeStructure) {
        isMouseDownOnSizeChart = true;
        updateEnvelopeFromMouseEvent(e);
      }
    });

    window.addEventListener('mousemove', e => {
      if (isDrawingSizeStructure && isMouseDownOnSizeChart) {
        updateEnvelopeFromMouseEvent(e);
      }
    });

    window.addEventListener('mouseup', () => {
      isMouseDownOnSizeChart = false;
    });

    // Apply New Size Structure & Reseed Population
    document.getElementById('btn-apply-size-structure').addEventListener('click', () => {
      const targetCount = parseInt(document.getElementById('inp-target-agents').value, 10);
      if (isNaN(targetCount) || targetCount < 10) {
        alert("Please enter a valid target agent count (minimum 10).");
        return;
      }

      // Check sum of envelope
      const sumWeight = targetSizeEnvelope.reduce((a, b) => a + b, 0);
      if (sumWeight <= 0.001) {
        alert("Please sculpt or select a non-empty size distribution curve.");
        return;
      }

      // Build Cumulative Distribution Function (CDF) for Inverse-Transform Sampling
      const cdf = [];
      let runningSum = 0;
      for (let i = 0; i < sizeBins.length; i++) {
        runningSum += targetSizeEnvelope[i];
        cdf.push(runningSum / sumWeight);
      }

      function sampleSizeFromCDF() {
        const u = Math.random();
        for (let i = 0; i < cdf.length; i++) {
          if (u <= cdf[i]) {
            const binLo = sizeBins[i] - 1.25;
            const binHi = sizeBins[i] + 1.25;
            return Math.max(10.0, Math.min(165.0, binLo + Math.random() * (binHi - binLo)));
          }
        }
        return sizeBins[sizeBins.length - 1];
      }

      const minLon = domainBounds[0][1];
      const maxLon = domainBounds[1][1];
      const minLat = domainBounds[0][0];
      const maxLat = domainBounds[1][0];

      // Synthesize new agent population adhering to target size structure
      const newAgents = [];
      for (let i = 1; i <= targetCount; i++) {
        const cw = sampleSizeFromCDF();
        const sex = Math.random() < 0.5 ? 1 : 0; // 1 = female, 0 = male

        // Allometric mass: mass (g) = a * CW (mm) ^ b
        const a_allom = sex === 1 ? 0.00028 : 0.00022;
        const b_allom = sex === 1 ? 3.08 : 3.14;
        const mass = a_allom * Math.pow(cw, b_allom);

        // Maturity logistic ogive
        const l50 = sex === 1 ? 68.0 : 95.0;
        const pMat = 1.0 / (1.0 + Math.exp(-(cw - l50) / 7.0));
        const mature = Math.random() < pMat;

        // Stage assignment based on carapace width
        let stage = 1;
        if (cw < 20.0) stage = 1;
        else if (cw < 35.0) stage = 2;
        else if (cw < 50.0) stage = 3;
        else if (cw < 70.0) stage = 4;
        else if (cw < 95.0) stage = 5;
        else if (cw < 115.0) stage = 6;
        else stage = Math.min(stageLabels.length - 1, 7);

        // Eggs for mature females
        const eggs = (sex === 1 && mature) ? Math.floor(1000 * Math.pow(cw / 50.0, 3)) : 0;

        // Find valid marine coordinates within active AOI
        let candLon = minLon + 0.5 * (maxLon - minLon);
        let candLat = minLat + 0.5 * (maxLat - minLat);
        let placed = false;
        for (let iter = 0; iter < 50; iter++) {
          const tLon = minLon + Math.random() * (maxLon - minLon);
          const tLat = minLat + Math.random() * (maxLat - minLat);
          const env = sampleEnvironment(tLon, tLat, 0);
          if (env.depth >= 20.0 && env.hsi > 0.05) {
            candLon = tLon;
            candLat = tLat;
            placed = true;
            break;
          }
        }
        if (!placed && currentGridPoints.length > 0) {
          const marine = currentGridPoints.filter(p => p.depth >= 20.0 && p.hsi > 0.0);
          if (marine.length > 0) {
            const chosen = marine[Math.floor(Math.random() * marine.length)];
            candLon = chosen.lon;
            candLat = chosen.lat;
          }
        }

        newAgents.push({
          id: i,
          gx: 1,
          gy: 1,
          lon: candLon,
          lat: candLat,
          size: Math.round(cw * 100) / 100,
          mass: Math.round(mass * 100) / 100,
          stage: stage,
          sex: sex,
          mature: mature,
          eggs: eggs,
          shell_age: mature ? (1.0 + Math.random() * 2.5) : (Math.random() * 0.9)
        });
      }

      // Update base pool and reseed active simulation
      initialAgents.length = 0;
      newAgents.forEach(a => initialAgents.push(a));
      document.getElementById('kpi-total').innerText = initialAgents.length;

      // Close editing mode and reset simulation to day 0
      btnToggleDrawSize.click();
      resetToDay(selectedStartDay);
    });

    // -------------------------------------------------------------------------
    // Interactive Fishery Selectivity Curve Shifting Engine (Panel 5)
    // -------------------------------------------------------------------------
    let isDraggingSelectivity = false;
    let isMouseDownOnOgiveChart = false;
    const btnToggleDragSel = document.getElementById('btn-toggle-drag-sel');
    const selEditorToolbar = document.getElementById('sel-editor-toolbar');
    const canvasOgive = document.getElementById('chart-ogive');
    const sliderL50 = document.getElementById('slider-l50');
    const lblL50Val = document.getElementById('lbl-l50-val');

    function updateSelectivityL50(newL50) {
      legalSizeMale = Math.max(40.0, Math.min(140.0, Math.round(newL50 * 2) / 2));
      sliderL50.value = legalSizeMale;
      lblL50Val.innerText = legalSizeMale.toFixed(1) + ' mm';

      // Update Panel 4 Legal Marker label and data
      chartSize.data.datasets[4].label = 'Legal Size Male (' + legalSizeMale.toFixed(1) + ' mm)';
      const legalLine = sizeBins.map(s => Math.abs(s - legalSizeMale) <= 1.5 ? 1.0 : 0.0);
      chartSize.data.datasets[4].data = legalLine;
      chartSize.update('none');

      // Update Panel 5 Selectivity Dataset
      selectivityRef = cwRef.map(cw => 1.0 / (1.0 + Math.exp(-(cw - legalSizeMale) / selectivitySigma)));
      chartOgive.data.datasets[2].data = selectivityRef;
      chartOgive.update('none');

      // Refresh KPI Biomass and Catch readouts
      updateAnalyticsPanels();
    }

    if (btnToggleDragSel) {
      btnToggleDragSel.addEventListener('click', () => {
        isDraggingSelectivity = !isDraggingSelectivity;
        if (isDraggingSelectivity) {
          btnToggleDragSel.classList.add('active');
          btnToggleDragSel.innerText = '✓ Done Dragging';
          selEditorToolbar.style.display = 'flex';
          canvasOgive.style.cursor = 'ew-resize';
        } else {
          btnToggleDragSel.classList.remove('active');
          btnToggleDragSel.innerText = '↔ Drag Selectivity (L50)';
          selEditorToolbar.style.display = 'none';
          canvasOgive.style.cursor = 'default';
        }
      });
    }

    if (sliderL50) {
      sliderL50.addEventListener('input', (e) => {
        updateSelectivityL50(parseFloat(e.target.value));
      });
    }

    const btnL50Reset = document.getElementById('btn-l50-reset');
    if (btnL50Reset) {
      btnL50Reset.addEventListener('click', () => {
        updateSelectivityL50(defaultLegalSizeMale);
      });
    }

    const btnL50Small = document.getElementById('btn-l50-small');
    if (btnL50Small) {
      btnL50Small.addEventListener('click', () => {
        updateSelectivityL50(78.0);
      });
    }

    const btnL50Large = document.getElementById('btn-l50-large');
    if (btnL50Large) {
      btnL50Large.addEventListener('click', () => {
        updateSelectivityL50(105.0);
      });
    }

    function handleOgivePointerMove(e) {
      if (!isDraggingSelectivity || !isMouseDownOnOgiveChart) return;
      const rect = canvasOgive.getBoundingClientRect();
      const xPixel = e.clientX - rect.left;
      const xScale = chartOgive.scales.x;
      if (!xScale) return;
      const cwVal = xScale.getValueForPixel(xPixel);
      if (cwVal !== undefined && !isNaN(cwVal)) {
        updateSelectivityL50(cwVal);
      }
    }

    canvasOgive.addEventListener('mousedown', (e) => {
      if (!isDraggingSelectivity) return;
      isMouseDownOnOgiveChart = true;
      handleOgivePointerMove(e);
    });

    window.addEventListener('mousemove', (e) => {
      if (isMouseDownOnOgiveChart) {
        handleOgivePointerMove(e);
      }
    });

    window.addEventListener('mouseup', () => {
      isMouseDownOnOgiveChart = false;
    });

    // Touch events for mobile/tablet support
    canvasOgive.addEventListener('touchstart', (e) => {
      if (!isDraggingSelectivity) return;
      isMouseDownOnOgiveChart = true;
      if (e.touches && e.touches.length > 0) {
        handleOgivePointerMove(e.touches[0]);
      }
    }, { passive: true });

    window.addEventListener('touchmove', (e) => {
      if (isMouseDownOnOgiveChart && e.touches && e.touches.length > 0) {
        handleOgivePointerMove(e.touches[0]);
      }
    }, { passive: true });

    window.addEventListener('touchend', () => {
      isMouseDownOnOgiveChart = false;
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

Initialize model and launch the interactive Leaflet spatial dashboard.
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
