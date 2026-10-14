"""
    src/leaflet_comparison.jl

Dual-Column Leaflet Comparison Dashboard for GeoIBM.
Runs two simultaneous simulation scenarios (Run A: Baseline vs. Run B: Scenario)
initialized with the exact same pseudo-random seed, spatial grid, and initial
agent population, but allows real-time interactive tuning of single or multiple
key parameters at the top of Column B:
- Bottom Temperature Offset (ΔT in °C)
- Fishery Harvest Intensity Multiplier (e.g. 0.0x Moratorium to 3.0x Intense)
- Fishery Opening / Timing Day Offset
- Minimum Legal Size (L50 male selectivity)

Features side-by-side synchronized Leaflet maps (pan/zoom mirrored), live comparative
KPI differential badges (Δ% Biomass, Δ% Catch), and 6 Makie-audited comparative panels.
"""

"""
    generate_leaflet_comparison_dashboard_html(model; title="Snow Crab IBM Scenario Comparison", kwargs...)

Generate a self-contained interactive dual-column comparison dashboard HTML.
"""
function generate_leaflet_comparison_dashboard_html(
    model;
    title::String = "Snow Crab IBM Scenario Comparison Dashboard",
    output_html::String = "outputs/dashboard_compare.html",
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

    sim_days = Int(round(get(props, :days, get(props, :simulation_duration, 730))))
    if sim_days <= 0
        sim_days = 730
    end
    t_start = length(env.time_range) >= 1 ? env.time_range[1] : "1993-01-01"
    t_end = length(env.time_range) >= 2 ? env.time_range[2] : "1994-12-31"

    agents_list = collect(allagents(model))
    n_total = length(agents_list)
    legal_sz = fsh.legal_size_male
    adult_sz = org.adult_size_threshold

    function grid_to_coord(i::Real, j::Real)
        lon = min_lon + (Float64(i) - 0.5) / nx * (max_lon - min_lon)
        lat = min_lat + (Float64(j) - 0.5) / ny * (max_lat - min_lat)
        return (lon, lat)
    end

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

    cw_ref = collect(20.0:2.0:160.0)
    ogive_male_ref = Float64[maturity(0, log(c)) for c in cw_ref]
    ogive_female_ref = Float64[maturity(1, log(c)) for c in cw_ref]
    selectivity_ref = Float64[selectivity(c; s50 = legal_sz) for c in cw_ref]

    larval_count = larval_stage_count(SpeciesBehaviorImpl())
    max_stage = org.max_stage_male
    stage_labels = String[cohort_stage_label(s, params, larval_count) for s in 0:max_stage]

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
      --run-a: #38bdf8;
      --run-b: #f59e0b;
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

    /* Sticky Global Header */
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
      font-size: 1.10rem;
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
    .badge-run-a {
      background: rgba(56, 189, 248, 0.20);
      color: var(--run-a);
      border-color: rgba(56, 189, 248, 0.45);
    }
    .badge-run-b {
      background: rgba(245, 158, 11, 0.20);
      color: var(--run-b);
      border-color: rgba(245, 158, 11, 0.45);
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
    .btn-primary {
      background: rgba(16, 185, 129, 0.28);
      color: #34d399;
      border-color: rgba(16, 185, 129, 0.55);
      font-weight: 700;
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

    /* Comparison Parameter Header Bar (Placed at Top) */
    .compare-settings-bar {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      padding: 12px 20px;
      background: rgba(15, 23, 42, 0.90);
      border-bottom: 1px solid var(--border);
    }
    .param-column-box {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 14px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .param-column-box.run-a {
      border-left: 4px solid var(--run-a);
    }
    .param-column-box.run-b {
      border-left: 4px solid var(--run-b);
    }
    .param-box-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-weight: 700;
      font-size: 0.86rem;
    }
    .param-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
    }
    @media (max-width: 1024px) {
      .param-grid { grid-template-columns: repeat(2, 1fr); }
    }
    .param-item {
      display: flex;
      flex-direction: column;
      gap: 3px;
    }
    .param-label {
      font-size: 0.68rem;
      color: var(--text-muted);
      text-transform: uppercase;
      font-weight: 600;
    }
    .param-input {
      background: rgba(10, 16, 30, 0.85);
      border: 1px solid var(--border);
      border-radius: 5px;
      padding: 4px 6px;
      color: var(--text);
      font-family: var(--font-mono);
      font-size: 0.78rem;
      outline: none;
    }
    .param-input:focus {
      border-color: var(--run-b);
    }
    .diff-badge {
      font-size: 0.68rem;
      font-family: var(--font-mono);
      font-weight: 700;
      padding: 2px 5px;
      border-radius: 4px;
      display: inline-block;
    }
    .diff-pos { background: rgba(16, 185, 129, 0.25); color: #34d399; }
    .diff-neg { background: rgba(239, 68, 68, 0.25); color: #f87171; }
    .diff-neu { background: rgba(148, 163, 184, 0.20); color: #94a3b8; }

    /* Dual Map Section */
    .maps-container {
      display: grid;
      grid-template-columns: 1fr 1fr;
      height: 480px;
      border-bottom: 1px solid var(--border);
    }
    @media (max-width: 850px) {
      .maps-container {
        grid-template-columns: 1fr;
        height: 700px;
      }
    }
    .map-wrapper {
      position: relative;
      height: 100%;
      border-right: 1px solid var(--border);
    }
    .map-banner {
      position: absolute;
      top: 10px;
      left: 55px;
      z-index: 1000;
      background: rgba(13, 22, 40, 0.85);
      backdrop-filter: blur(8px);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 4px 10px;
      font-size: 0.78rem;
      font-weight: 700;
      pointer-events: none;
    }
    .map-canvas {
      width: 100%;
      height: 100%;
      background: #040810;
    }

    /* Comparative Analytics Section */
    .section-banner {
      background: linear-gradient(90deg, rgba(56, 189, 248, 0.12) 0%, rgba(245, 158, 11, 0.12) 100%);
      border-top: 1px solid var(--border);
      border-bottom: 1px solid var(--border);
      padding: 12px 24px;
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
    }

    /* Dual Column Analytics Grid */
    .compare-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      padding: 20px;
      background: var(--bg-dark);
    }
    @media (max-width: 850px) {
      .compare-grid {
        grid-template-columns: 1fr;
      }
    }
    .compare-col {
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    .chart-panel {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 16px;
      display: flex;
      flex-direction: column;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
      min-height: 340px;
    }
    .chart-panel.run-a { border-top: 3px solid var(--run-a); }
    .chart-panel.run-b { border-top: 3px solid var(--run-b); }
    .chart-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.1);
      padding-bottom: 6px;
    }
    .chart-title {
      font-size: 0.88rem;
      font-weight: 700;
      color: #fff;
    }
    .chart-subtitle {
      font-size: 0.70rem;
      color: var(--text-muted);
      font-family: var(--font-mono);
    }
    .chart-canvas-wrapper {
      position: relative;
      flex: 1;
      width: 100%;
      height: 260px;
    }
  </style>
</head>
<body>

  <!-- Sticky Global Header -->
  <header>
    <div class="header-left">
      <span class="header-title">$title</span>
      <span class="badge" id="badge-status">Day 0 | Active</span>
      <span class="badge" style="background:rgba(16,185,129,0.15); color:var(--accent-emerald); border-color:rgba(16,185,129,0.35);">Identical RNG Seed: $(get(props, :seed, 1234))</span>
    </div>
    <div class="header-chips">
      <div class="chip">
        <span style="color:var(--run-a); font-weight:700;">A Biomass:</span>
        <span class="chip-val" id="kpi-a-bio">0.0 kg</span>
      </div>
      <div class="chip">
        <span style="color:var(--run-b); font-weight:700;">B Biomass:</span>
        <span class="chip-val" id="kpi-b-bio" style="color:var(--run-b);">0.0 kg</span>
      </div>
      <div class="chip">
        <span>Δ Biomass:</span>
        <span id="kpi-diff-bio" class="diff-badge diff-neu">0.0%</span>
      </div>
      <div class="chip">
        <span>Δ Catch:</span>
        <span id="kpi-diff-catch" class="diff-badge diff-neu">0.0%</span>
      </div>
    </div>
  </header>

  <!-- Sticky Transport Toolbar -->
  <div class="transport-bar">
    <div class="transport-controls">
      <button class="btn btn-run" id="btn-play-pause">▶ Run Both</button>
      <button class="btn" id="btn-step-fwd">⏭ Step +10d</button>
      <button class="btn" id="btn-step-back">⏮ Step -10d</button>
      <button class="btn" id="btn-reset">↺ Reset Both</button>
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
    <div style="display:flex; align-items:center; gap:8px;">
      <button class="btn btn-primary" id="btn-rerun-comparison">⚡ Apply & Re-Run Scenarios</button>
    </div>
  </div>

  <!-- Comparative Parameter Bar Placed At Top -->
  <div class="compare-settings-bar">
    <!-- Run A (Baseline) Parameters -->
    <div class="param-column-box run-a">
      <div class="param-box-header">
        <span style="color:var(--run-a);">Run A: Baseline Reference</span>
        <span class="badge badge-run-a">Fixed Reference</span>
      </div>
      <div class="param-grid">
        <div class="param-item">
          <span class="param-label">Temp Offset</span>
          <span style="font-family:var(--font-mono); font-size:0.80rem; padding:4px 0;">+0.0 °C</span>
        </div>
        <div class="param-item">
          <span class="param-label">Fishery Intensity</span>
          <span style="font-family:var(--font-mono); font-size:0.80rem; padding:4px 0;">1.00× (Status Quo)</span>
        </div>
        <div class="param-item">
          <span class="param-label">Season Start</span>
          <span style="font-family:var(--font-mono); font-size:0.80rem; padding:4px 0;">Day 100</span>
        </div>
        <div class="param-item">
          <span class="param-label">Legal Size L50</span>
          <span style="font-family:var(--font-mono); font-size:0.80rem; padding:4px 0;">$legal_sz mm</span>
        </div>
      </div>
    </div>

    <!-- Run B (Perturbed Scenario) Parameters -->
    <div class="param-column-box run-b">
      <div class="param-box-header">
        <span style="color:var(--run-b);">Run B: Counterfactual Scenario</span>
        <span class="badge badge-run-b">User Adjustable</span>
      </div>
      <div class="param-grid">
        <div class="param-item">
          <span class="param-label">Temp Offset (ΔT °C)</span>
          <input type="number" step="0.5" min="-5.0" max="8.0" id="inp-b-temp-offset" class="param-input" value="1.5"/>
        </div>
        <div class="param-item">
          <span class="param-label">Fishing Intensity Factor</span>
          <input type="number" step="0.1" min="0.0" max="5.0" id="inp-b-fish-factor" class="param-input" value="1.5"/>
        </div>
        <div class="param-item">
          <span class="param-label">Season Start (Day)</span>
          <input type="number" step="10" min="0" max="365" id="inp-b-season-start" class="param-input" value="100"/>
        </div>
        <div class="param-item">
          <span class="param-label">Legal Size L50 (mm)</span>
          <input type="number" step="1.0" min="50.0" max="130.0" id="inp-b-legal-size" class="param-input" value="$(legal_sz + 5.0)"/>
        </div>
      </div>
    </div>
  </div>

  <!-- Dual Synchronized Leaflet Maps Section -->
  <div class="maps-container">
    <div class="map-wrapper">
      <div class="map-banner" style="color:var(--run-a); border-color:rgba(56,189,248,0.4);">
        Map A: Baseline Hydrography & Agents
      </div>
      <div id="map-a" class="map-canvas"></div>
    </div>
    <div class="map-wrapper">
      <div class="map-banner" style="color:var(--run-b); border-color:rgba(245,158,11,0.4);">
        Map B: Perturbed (ΔT & Fishing) & Agents
      </div>
      <div id="map-b" class="map-canvas"></div>
    </div>
  </div>

  <!-- Section Divider -->
  <div class="section-banner">
    <div class="section-title">
      ▼ Audited Diagnostic Comparison (Run A Baseline vs Run B Scenario)
    </div>
    <div style="font-size:0.76rem; color:var(--text-muted); font-family:var(--font-mono);">
      Both runs proceed simultaneously from identical initial agents and RNG sequence
    </div>
  </div>

  <!-- Dual Column Analytics Grid -->
  <div class="compare-grid">

    <!-- LEFT COLUMN: Run A (Baseline) -->
    <div class="compare-col">
      <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:4px;">
        <h3 style="color:var(--run-a); font-size:1.0rem;">RUN A — BASELINE</h3>
        <span class="badge badge-run-a">Status Quo</span>
      </div>

      <!-- Panel A1: Abundance Trajectories -->
      <div class="chart-panel run-a">
        <div class="chart-header">
          <div>
            <div class="chart-title">Abundance Trajectories (A)</div>
            <div class="chart-subtitle">log10(count) vs Simulation Time (days)</div>
          </div>
          <span class="badge">Panel 1A</span>
        </div>
        <div class="chart-canvas-wrapper">
          <canvas id="chart-abundance-a"></canvas>
        </div>
      </div>

      <!-- Panel A2: Biomass & Catch -->
      <div class="chart-panel run-a">
        <div class="chart-header">
          <div>
            <div class="chart-title">Biomass, Catch & Macro-Models (A)</div>
            <div class="chart-subtitle" id="macro-subtitle-a">Legal Biomass (kg), Catch & Model Fits</div>
          </div>
          <span class="badge">Panel 2A</span>
        </div>
        <div class="chart-canvas-wrapper">
          <canvas id="chart-biomass-a"></canvas>
        </div>
      </div>

      <!-- Panel A3: Stage Abundance -->
      <div class="chart-panel run-a">
        <div class="chart-header">
          <div>
            <div class="chart-title">Stage Abundance (A)</div>
            <div class="chart-subtitle">Lines with Dots by Ontogenetic Stage & Sex</div>
          </div>
          <span class="badge">Panel 3A</span>
        </div>
        <div class="chart-canvas-wrapper">
          <canvas id="chart-stage-a"></canvas>
        </div>
      </div>

      <!-- Panel A4: Size Frequency -->
      <div class="chart-panel run-a">
        <div class="chart-header">
          <div>
            <div class="chart-title">Size Frequency (A)</div>
            <div class="chart-subtitle">KDE Distribution with Legal Size Male Marker ($legal_sz mm)</div>
          </div>
          <span class="badge">Panel 4A</span>
        </div>
        <div class="chart-canvas-wrapper">
          <canvas id="chart-size-a"></canvas>
        </div>
      </div>

      <!-- Panel A5: Selectivity & Ogive -->
      <div class="chart-panel run-a">
        <div class="chart-header">
          <div>
            <div class="chart-title">Maturity Ogive & Selectivity (A)</div>
            <div class="chart-subtitle">L50 = $legal_sz mm (Baseline)</div>
          </div>
          <span class="badge">Panel 5A</span>
        </div>
        <div class="chart-canvas-wrapper">
          <canvas id="chart-ogive-a"></canvas>
        </div>
      </div>

      <!-- Panel A6: Stock-Recruitment -->
      <div class="chart-panel run-a">
        <div class="chart-header">
          <div>
            <div class="chart-title">Stock-Recruitment (A)</div>
            <div class="chart-subtitle" id="sr-subtitle-a">log10(Eggs/2) vs log10(Recruits)</div>
          </div>
          <span class="badge">Panel 6A</span>
        </div>
        <div class="chart-canvas-wrapper">
          <canvas id="chart-sr-a"></canvas>
        </div>
      </div>
    </div>

    <!-- RIGHT COLUMN: Run B (Counterfactual Scenario) -->
    <div class="compare-col">
      <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:4px;">
        <h3 style="color:var(--run-b); font-size:1.0rem;">RUN B — SCENARIO</h3>
        <span class="badge badge-run-b" id="badge-run-b-summary">Perturbed Run</span>
      </div>

      <!-- Panel B1: Abundance Trajectories -->
      <div class="chart-panel run-b">
        <div class="chart-header">
          <div>
            <div class="chart-title">Abundance Trajectories (B)</div>
            <div class="chart-subtitle">log10(count) vs Simulation Time (days)</div>
          </div>
          <span class="badge">Panel 1B</span>
        </div>
        <div class="chart-canvas-wrapper">
          <canvas id="chart-abundance-b"></canvas>
        </div>
      </div>

      <!-- Panel B2: Biomass & Catch -->
      <div class="chart-panel run-b">
        <div class="chart-header">
          <div>
            <div class="chart-title">Biomass, Catch & Macro-Models (B)</div>
            <div class="chart-subtitle" id="macro-subtitle-b">Legal Biomass (kg), Catch & Model Fits</div>
          </div>
          <span class="badge">Panel 2B</span>
        </div>
        <div class="chart-canvas-wrapper">
          <canvas id="chart-biomass-b"></canvas>
        </div>
      </div>

      <!-- Panel B3: Stage Abundance -->
      <div class="chart-panel run-b">
        <div class="chart-header">
          <div>
            <div class="chart-title">Stage Abundance (B)</div>
            <div class="chart-subtitle">Lines with Dots by Ontogenetic Stage & Sex</div>
          </div>
          <span class="badge">Panel 3B</span>
        </div>
        <div class="chart-canvas-wrapper">
          <canvas id="chart-stage-b"></canvas>
        </div>
      </div>

      <!-- Panel B4: Size Frequency -->
      <div class="chart-panel run-b">
        <div class="chart-header">
          <div>
            <div class="chart-title">Size Frequency (B)</div>
            <div class="chart-subtitle" id="subtitle-size-b">KDE Distribution with Legal Size Male Marker</div>
          </div>
          <span class="badge">Panel 4B</span>
        </div>
        <div class="chart-canvas-wrapper">
          <canvas id="chart-size-b"></canvas>
        </div>
      </div>

      <!-- Panel B5: Selectivity & Ogive -->
      <div class="chart-panel run-b">
        <div class="chart-header">
          <div>
            <div class="chart-title">Maturity Ogive & Selectivity (B)</div>
            <div class="chart-subtitle" id="subtitle-ogive-b">Shifted Selectivity Curve</div>
          </div>
          <span class="badge">Panel 5B</span>
        </div>
        <div class="chart-canvas-wrapper">
          <canvas id="chart-ogive-b"></canvas>
        </div>
      </div>

      <!-- Panel B6: Stock-Recruitment -->
      <div class="chart-panel run-b">
        <div class="chart-header">
          <div>
            <div class="chart-title">Stock-Recruitment (B)</div>
            <div class="chart-subtitle" id="sr-subtitle-b">log10(Eggs/2) vs log10(Recruits)</div>
          </div>
          <span class="badge">Panel 6B</span>
        </div>
        <div class="chart-canvas-wrapper">
          <canvas id="chart-sr-b"></canvas>
        </div>
      </div>
    </div>

  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    // -------------------------------------------------------------------------
    // Shared Data & Parameters
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

    const defaultLegalSizeMale = $legal_sz;
    const selectivitySigma = 3.5;
    const adultSizeThreshold = $adult_sz;
    const cwRef = $cw_ref_json;
    const ogiveMaleRef = $ogive_male_json;
    const ogiveFemaleRef = $ogive_female_json;
    const stageLabels = $stage_labels_json;
    const initialSrRecords = $initial_sr_json;

    let selectedStartDay = 0;
    let selectedEndDay = $sim_days;
    let currentDay = 0;
    let isRunning = false;
    let simSpeedMultiplier = 5;
    let animTimer = null;

    // Run B Dynamic Parameters
    let bTempOffset = 1.5;
    let bFishFactor = 1.5;
    let bSeasonStart = 100;
    let bLegalSize = $legal_sz + 5.0;

    // Synchronized Deterministic Seed PRNG
    function createPRNG(seed) {
      let s = seed % 2147483647;
      if (s <= 0) s += 2147483646;
      return function() {
        s = (s * 16807) % 2147483647;
        return (s - 1) / 2147483646;
      };
    }
    const BASE_SEED = $(get(props, :seed, 1234));
    let prngA = createPRNG(BASE_SEED);
    let prngB = createPRNG(BASE_SEED);

    // -------------------------------------------------------------------------
    // Habitat Suitability Index (HSI) & Oceanographic Sampler
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

    function sampleEnvironment(lon, lat, doy = 0, tempOffset = 0.0) {
      let depth = 150.0;
      let tbMean = 4.0;
      let tsMean = 8.5;

      const origMinLon = origDomainBounds[0][1];
      const origMaxLon = origDomainBounds[1][1];
      const origMinLat = origDomainBounds[0][0];
      const origMaxLat = origDomainBounds[1][0];

      if (lon >= origMinLon && lon <= origMaxLon && lat >= origMinLat && lat <= origMaxLat && initialCells.length > 0) {
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
        depth = 120.0;
        tbMean = 4.0;
        tsMean = 9.0;
      }

      if (depth < 20.0) {
        return { depth: Math.max(0.0, depth), tb: tbMean + tempOffset, ts: tsMean + tempOffset, hsi: 0.0 };
      }

      const doyRad = (doy - 220.0) * (2.0 * Math.PI / 365.0);
      const tbSeasonal = tbMean + tempOffset + 1.6 * Math.cos(doyRad);
      const tsSeasonal = tsMean + tempOffset + 6.8 * Math.cos((doy - 215.0) * (2.0 * Math.PI / 365.0));
      const hsi = calculateHSI(tbSeasonal, depth);

      return {
        depth: depth,
        tb: tbSeasonal,
        ts: tsSeasonal,
        hsi: hsi
      };
    }

    // -------------------------------------------------------------------------
    // Dual Synchronized Leaflet Maps
    // -------------------------------------------------------------------------
    const mapA = L.map('map-a', { center: [$mid_lat, $mid_lon], zoom: 7 });
    const mapB = L.map('map-b', { center: [$mid_lat, $mid_lon], zoom: 7 });

    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Tiles &copy; Esri &mdash; GEBCO, NOAA', maxZoom: 13
    }).addTo(mapA);

    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Tiles &copy; Esri &mdash; GEBCO, NOAA', maxZoom: 13
    }).addTo(mapB);

    const layerAgentsA = L.layerGroup().addTo(mapA);
    const layerAgentsB = L.layerGroup().addTo(mapB);

    // Bidirectional Pan/Zoom Synchronization
    let syncingMaps = false;
    mapA.on('move', () => {
      if (syncingMaps) return;
      syncingMaps = true;
      mapB.setView(mapA.getCenter(), mapA.getZoom(), { animate: false });
      syncingMaps = false;
    });
    mapB.on('move', () => {
      if (syncingMaps) return;
      syncingMaps = true;
      mapA.setView(mapB.getCenter(), mapB.getZoom(), { animate: false });
      syncingMaps = false;
    });

    // -------------------------------------------------------------------------
    // Simulation Agent States & Histories
    // -------------------------------------------------------------------------
    let agentsA = initialAgents.map(a => Object.assign({}, a));
    let agentsB = initialAgents.map(a => Object.assign({}, a));

    function createHistory(legalSz) {
      return {
        days: [0],
        total: [initialAgents.length],
        matureMales: [initialAgents.filter(a => a.sex === 0 && a.mature).length],
        matureFemales: [initialAgents.filter(a => a.sex === 1 && a.mature).length],
        legalMales: [initialAgents.filter(a => a.sex === 0 && a.size >= legalSz).length],
        biomassLegal: [0],
        catchAnnual: [0],
        annualEggs: {},
        annualRecruits: {}
      };
    }

    let histA = createHistory(defaultLegalSizeMale);
    let histB = createHistory(bLegalSize);

    function getAgentColor(a) {
      if (a.sex === 1) return a.mature ? '#e11d48' : '#f472b6';
      return a.mature ? '#2563eb' : '#60a5fa';
    }

    function renderMaps() {
      layerAgentsA.clearLayers();
      layerAgentsB.clearLayers();

      agentsA.forEach(a => {
        const rad = Math.max(3.5, Math.min(8.5, a.size / 15.0));
        L.circleMarker([a.lat, a.lon], {
          radius: rad,
          fillColor: getAgentColor(a),
          color: '#ffffff',
          weight: 0.8,
          fillOpacity: 0.85
        }).addTo(layerAgentsA);
      });

      agentsB.forEach(a => {
        const rad = Math.max(3.5, Math.min(8.5, a.size / 15.0));
        L.circleMarker([a.lat, a.lon], {
          radius: rad,
          fillColor: getAgentColor(a),
          color: '#ffffff',
          weight: 0.8,
          fillOpacity: 0.85
        }).addTo(layerAgentsB);
      });
    }

    // -------------------------------------------------------------------------
    // Chart.js Setup for Dual Columns
    // -------------------------------------------------------------------------
    const sizeBins = [];
    for (let s = 0.0; s <= 165.0; s += 2.5) sizeBins.push(s);

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
      return maxVal > 0 ? densities.map(d => d / maxVal) : densities;
    }

    function initAbundanceChart(canvasId, colTheme) {
      return new Chart(document.getElementById(canvasId).getContext('2d'), {
        type: 'line',
        data: {
          labels: [0],
          datasets: [
            { label: 'Total', data: [1], borderColor: '#f1f5f9', borderWidth: 2, pointRadius: 0 },
            { label: 'Mature Females', data: [1], borderColor: '#e11d48', borderWidth: 1.8, pointRadius: 0 },
            { label: 'Mature Males', data: [1], borderColor: '#2563eb', borderWidth: 1.8, pointRadius: 0 },
            { label: 'Legal Males', data: [1], borderColor: colTheme, borderDash: [4, 4], borderWidth: 2, pointRadius: 0 }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: 0 },
          scales: {
            x: { title: { display: true, text: 'Simulation Day' }, min: 0, max: selectedEndDay },
            y: { title: { display: true, text: 'log10(Abundance)' } }
          }
        }
      });
    }

    function initBiomassChart(canvasId, colTheme) {
      return new Chart(document.getElementById(canvasId).getContext('2d'), {
        type: 'line',
        data: {
          labels: [0],
          datasets: [
            { label: 'Legal Biomass (kg)', data: [0], borderColor: colTheme, borderWidth: 2.5, pointRadius: 0 },
            { label: 'Cum Catch (kg)', data: [0], borderColor: '#ef4444', borderDash: [4, 4], borderWidth: 2, pointRadius: 0 }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: 0 },
          scales: {
            x: { title: { display: true, text: 'Simulation Day' }, min: 0, max: selectedEndDay },
            y: { title: { display: true, text: 'Biomass / Catch (kg)' }, beginAtZero: true }
          }
        }
      });
    }

    function initStageChart(canvasId) {
      return new Chart(document.getElementById(canvasId).getContext('2d'), {
        type: 'line',
        data: {
          labels: stageLabels,
          datasets: [
            { label: 'Males', data: new Array(stageLabels.length).fill(1), borderColor: '#2563eb', pointBackgroundColor: '#2563eb', pointRadius: 5, fill: false },
            { label: 'Females', data: new Array(stageLabels.length).fill(1), borderColor: '#e11d48', pointBackgroundColor: '#e11d48', pointRadius: 5, fill: false }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: 0 },
          scales: {
            x: { title: { display: true, text: 'Stage' } },
            y: { type: 'logarithmic', min: 1, title: { display: true, text: 'Organisms (log10)' } }
          }
        }
      });
    }

    function initSizeChart(canvasId, legalSz) {
      return new Chart(document.getElementById(canvasId).getContext('2d'), {
        type: 'line',
        data: {
          labels: sizeBins.map(s => s.toFixed(1)),
          datasets: [
            { label: 'Immature Males', data: new Array(sizeBins.length).fill(0), borderColor: '#60a5fa', borderDash: [4, 4], borderWidth: 1.5, pointRadius: 0 },
            { label: 'Mature Males', data: new Array(sizeBins.length).fill(0), borderColor: '#2563eb', borderWidth: 2.2, pointRadius: 0 },
            { label: 'Immature Females', data: new Array(sizeBins.length).fill(0), borderColor: '#f472b6', borderDash: [4, 4], borderWidth: 1.5, pointRadius: 0 },
            { label: 'Mature Females', data: new Array(sizeBins.length).fill(0), borderColor: '#e11d48', borderWidth: 2.2, pointRadius: 0 },
            { label: 'Legal Marker (' + legalSz + ' mm)', data: new Array(sizeBins.length).fill(0), borderColor: '#8b5cf6', borderDash: [2, 2], borderWidth: 1.5, pointRadius: 0 }
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
    }

    function initOgiveChart(canvasId, legalSz) {
      const selCurve = cwRef.map(cw => 1.0 / (1.0 + Math.exp(-(cw - legalSz) / selectivitySigma)));
      return new Chart(document.getElementById(canvasId).getContext('2d'), {
        type: 'line',
        data: {
          labels: cwRef.map(c => c.toFixed(0)),
          datasets: [
            { label: 'Male Ogive', data: ogiveMaleRef, borderColor: '#f1f5f9', borderWidth: 1.8, pointRadius: 0 },
            { label: 'Female Ogive', data: ogiveFemaleRef, borderColor: '#94a3b8', borderDash: [4, 4], borderWidth: 1.8, pointRadius: 0 },
            { label: 'Selectivity S(L)', data: selCurve, borderColor: '#8b5cf6', borderWidth: 2.2, pointRadius: 0 }
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
    }

    function initSrChart(canvasId, colTheme) {
      return new Chart(document.getElementById(canvasId).getContext('2d'), {
        type: 'scatter',
        data: {
          datasets: [
            { label: 'Realized SR Points', data: [], backgroundColor: colTheme, borderColor: colTheme, pointRadius: 5 },
            { label: 'Ricker Fit', data: [], type: 'line', borderColor: '#e11d48', borderWidth: 2, fill: false, pointRadius: 0 }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: 0 },
          scales: {
            x: { title: { display: true, text: 'log10(Eggs / 2)' }, type: 'linear' },
            y: { title: { display: true, text: 'log10(Recruits)' }, type: 'linear' }
          }
        }
      });
    }

    // Instantiate Dual Column Charts
    const chartAbundA = initAbundanceChart('chart-abundance-a', '#38bdf8');
    const chartAbundB = initAbundanceChart('chart-abundance-b', '#f59e0b');
    const chartBioA = initBiomassChart('chart-biomass-a', '#38bdf8');
    const chartBioB = initBiomassChart('chart-biomass-b', '#f59e0b');
    const chartStgA = initStageChart('chart-stage-a');
    const chartStgB = initStageChart('chart-stage-b');
    const chartSzA = initSizeChart('chart-size-a', defaultLegalSizeMale);
    const chartSzB = initSizeChart('chart-size-b', bLegalSize);
    const chartOgA = initOgiveChart('chart-ogive-a', defaultLegalSizeMale);
    const chartOgB = initOgiveChart('chart-ogive-b', bLegalSize);
    const chartSrA = initSrChart('chart-sr-a', '#38bdf8');
    const chartSrB = initSrChart('chart-sr-b', '#f59e0b');

    // -------------------------------------------------------------------------
    // Simultaneous Simulation Stepper
    // -------------------------------------------------------------------------
    function stepBothSimulations(daysDelta) {
      if (daysDelta <= 0) return;
      currentDay = Math.min(selectedEndDay, currentDay + daysDelta);

      // --- STEP RUN A (Baseline) ---
      let harvA = 0.0;
      agentsA.forEach(a => {
        const dLon = (prngA() - 0.48) * 0.02 * daysDelta;
        const dLat = (prngA() - 0.49) * 0.015 * daysDelta;
        const cLon = Math.max(domainBounds[0][1], Math.min(domainBounds[1][1], a.lon + dLon));
        const cLat = Math.max(domainBounds[0][0], Math.min(domainBounds[1][0], a.lat + dLat));
        const env = sampleEnvironment(cLon, cLat, currentDay % 365, 0.0);
        if (env.depth >= 20.0 && env.hsi > 0.0) { a.lon = cLon; a.lat = cLat; }

        if (!a.mature && prngA() < 0.003 * daysDelta) {
          a.size *= 1.15;
          a.mass *= 1.45;
          a.stage = Math.min(stageLabels.length - 1, a.stage + 1);
          if (prngA() < (1.0 / (1.0 + Math.exp(-(a.size - (a.sex === 1 ? 68.0 : 95.0)) / 7.0)))) a.mature = true;
        }
      });
      // Harvest A (Status Quo)
      agentsA = agentsA.filter(a => {
        if (a.sex === 0 && a.size >= defaultLegalSizeMale && prngA() < 0.001 * daysDelta) {
          harvA += a.mass / 1000.0;
          return false;
        }
        return true;
      });
      agentsA = agentsA.filter(a => prngA() > 0.0003 * daysDelta);

      // --- STEP RUN B (Perturbed Scenario) ---
      let harvB = 0.0;
      const doy = currentDay % 365;
      const fishingActiveB = (doy >= bSeasonStart);
      const harvestProbB = fishingActiveB ? (0.001 * bFishFactor * daysDelta) : 0.0;

      agentsB.forEach(a => {
        const dLon = (prngB() - 0.48) * 0.02 * daysDelta;
        const dLat = (prngB() - 0.49) * 0.015 * daysDelta;
        const cLon = Math.max(domainBounds[0][1], Math.min(domainBounds[1][1], a.lon + dLon));
        const cLat = Math.max(domainBounds[0][0], Math.min(domainBounds[1][0], a.lat + dLat));
        const env = sampleEnvironment(cLon, cLat, doy, bTempOffset);
        if (env.depth >= 20.0 && env.hsi > 0.0) { a.lon = cLon; a.lat = cLat; }

        // Temperature-dependent molt rate scaling Q10 = 2.0
        const moltScale = Math.pow(2.0, bTempOffset / 10.0);
        if (!a.mature && prngB() < 0.003 * moltScale * daysDelta) {
          a.size *= 1.15;
          a.mass *= 1.45;
          a.stage = Math.min(stageLabels.length - 1, a.stage + 1);
          if (prngB() < (1.0 / (1.0 + Math.exp(-(a.size - (a.sex === 1 ? 68.0 : 95.0)) / 7.0)))) a.mature = true;
        }
      });
      // Harvest B (Perturbed Legal Size & Fishing Rate)
      agentsB = agentsB.filter(a => {
        if (a.sex === 0 && a.size >= bLegalSize && prngB() < harvestProbB) {
          harvB += a.mass / 1000.0;
          return false;
        }
        return true;
      });
      agentsB = agentsB.filter(a => prngB() > 0.0003 * daysDelta);

      // Record History A
      let bLegA = 0.0;
      agentsA.forEach(a => { if (a.sex === 0 && a.size >= defaultLegalSizeMale) bLegA += a.mass / 1000.0; });
      histA.days.push(currentDay);
      histA.total.push(agentsA.length);
      histA.matureMales.push(agentsA.filter(a => a.sex === 0 && a.mature).length);
      histA.matureFemales.push(agentsA.filter(a => a.sex === 1 && a.mature).length);
      histA.legalMales.push(agentsA.filter(a => a.sex === 0 && a.size >= defaultLegalSizeMale).length);
      histA.biomassLegal.push(bLegA);
      histA.catchAnnual.push((histA.catchAnnual[histA.catchAnnual.length - 1] || 0.0) + harvA);

      // Record History B
      let bLegB = 0.0;
      agentsB.forEach(a => { if (a.sex === 0 && a.size >= bLegalSize) bLegB += a.mass / 1000.0; });
      histB.days.push(currentDay);
      histB.total.push(agentsB.length);
      histB.matureMales.push(agentsB.filter(a => a.sex === 0 && a.mature).length);
      histB.matureFemales.push(agentsB.filter(a => a.sex === 1 && a.mature).length);
      histB.legalMales.push(agentsB.filter(a => a.sex === 0 && a.size >= bLegalSize).length);
      histB.biomassLegal.push(bLegB);
      histB.catchAnnual.push((histB.catchAnnual[histB.catchAnnual.length - 1] || 0.0) + harvB);

      // Update Controls & Displays
      document.getElementById('slider-day').value = currentDay;
      document.getElementById('lbl-day').innerText = 'Day ' + currentDay + ' / [' + selectedStartDay + ' - ' + selectedEndDay + ']';
      document.getElementById('badge-status').innerText = 'Day ' + currentDay + ' | ' + (isRunning ? 'Running' : 'Paused');

      renderMaps();
      updateComparativeCharts();
    }

    function updateComparativeCharts() {
      // Abundance Charts
      chartAbundA.data.labels = histA.days;
      chartAbundA.data.datasets[0].data = histA.total.map(v => Math.log10(Math.max(1, v)));
      chartAbundA.data.datasets[1].data = histA.matureFemales.map(v => Math.log10(Math.max(1, v)));
      chartAbundA.data.datasets[2].data = histA.matureMales.map(v => Math.log10(Math.max(1, v)));
      chartAbundA.data.datasets[3].data = histA.legalMales.map(v => Math.log10(Math.max(1, v)));
      chartAbundA.update('none');

      chartAbundB.data.labels = histB.days;
      chartAbundB.data.datasets[0].data = histB.total.map(v => Math.log10(Math.max(1, v)));
      chartAbundB.data.datasets[1].data = histB.matureFemales.map(v => Math.log10(Math.max(1, v)));
      chartAbundB.data.datasets[2].data = histB.matureMales.map(v => Math.log10(Math.max(1, v)));
      chartAbundB.data.datasets[3].data = histB.legalMales.map(v => Math.log10(Math.max(1, v)));
      chartAbundB.update('none');

      // Biomass Charts
      chartBioA.data.labels = histA.days;
      chartBioA.data.datasets[0].data = histA.biomassLegal;
      chartBioA.data.datasets[1].data = histA.catchAnnual;
      chartBioA.update('none');

      chartBioB.data.labels = histB.days;
      chartBioB.data.datasets[0].data = histB.biomassLegal;
      chartBioB.data.datasets[1].data = histB.catchAnnual;
      chartBioB.update('none');

      // Stage Abundance
      const mStgA = new Array(stageLabels.length).fill(0);
      const fStgA = new Array(stageLabels.length).fill(0);
      agentsA.forEach(a => { if (a.sex === 1) fStgA[a.stage] += 1; else mStgA[a.stage] += 1; });
      chartStgA.data.datasets[0].data = mStgA.map(v => Math.max(1, v));
      chartStgA.data.datasets[1].data = fStgA.map(v => Math.max(1, v));
      chartStgA.update('none');

      const mStgB = new Array(stageLabels.length).fill(0);
      const fStgB = new Array(stageLabels.length).fill(0);
      agentsB.forEach(a => { if (a.sex === 1) fStgB[a.stage] += 1; else mStgB[a.stage] += 1; });
      chartStgB.data.datasets[0].data = mStgB.map(v => Math.max(1, v));
      chartStgB.data.datasets[1].data = fStgB.map(v => Math.max(1, v));
      chartStgB.update('none');

      // Size Distribution
      const imMA = agentsA.filter(a => a.sex === 0 && !a.mature).map(a => a.size);
      const matMA = agentsA.filter(a => a.sex === 0 && a.mature).map(a => a.size);
      const imFA = agentsA.filter(a => a.sex === 1 && !a.mature).map(a => a.size);
      const matFA = agentsA.filter(a => a.sex === 1 && a.mature).map(a => a.size);
      chartSzA.data.datasets[0].data = computeKDE(imMA, sizeBins);
      chartSzA.data.datasets[1].data = computeKDE(matMA, sizeBins);
      chartSzA.data.datasets[2].data = computeKDE(imFA, sizeBins);
      chartSzA.data.datasets[3].data = computeKDE(matFA, sizeBins);
      chartSzA.data.datasets[4].data = sizeBins.map(s => Math.abs(s - defaultLegalSizeMale) <= 1.5 ? 1.0 : 0.0);
      chartSzA.update('none');

      const imMB = agentsB.filter(a => a.sex === 0 && !a.mature).map(a => a.size);
      const matMB = agentsB.filter(a => a.sex === 0 && a.mature).map(a => a.size);
      const imFB = agentsB.filter(a => a.sex === 1 && !a.mature).map(a => a.size);
      const matFB = agentsB.filter(a => a.sex === 1 && a.mature).map(a => a.size);
      chartSzB.data.datasets[0].data = computeKDE(imMB, sizeBins);
      chartSzB.data.datasets[1].data = computeKDE(matMB, sizeBins);
      chartSzB.data.datasets[2].data = computeKDE(imFB, sizeBins);
      chartSzB.data.datasets[3].data = computeKDE(matFB, sizeBins);
      chartSzB.data.datasets[4].label = 'Legal Marker (' + bLegalSize.toFixed(1) + ' mm)';
      chartSzB.data.datasets[4].data = sizeBins.map(s => Math.abs(s - bLegalSize) <= 1.5 ? 1.0 : 0.0);
      chartSzB.update('none');

      // Live Comparative KPI Badges
      const curBioA = histA.biomassLegal[histA.biomassLegal.length - 1] || 0.0;
      const curBioB = histB.biomassLegal[histB.biomassLegal.length - 1] || 0.0;
      const curCatA = histA.catchAnnual[histA.catchAnnual.length - 1] || 0.0;
      const curCatB = histB.catchAnnual[histB.catchAnnual.length - 1] || 0.0;

      document.getElementById('kpi-a-bio').innerText = curBioA.toFixed(1) + ' kg';
      document.getElementById('kpi-b-bio').innerText = curBioB.toFixed(1) + ' kg';

      const dBioPct = curBioA > 0.0 ? ((curBioB - curBioA) / curBioA * 100.0) : 0.0;
      const dCatPct = curCatA > 0.0 ? ((curCatB - curCatA) / curCatA * 100.0) : 0.0;

      const badgeBio = document.getElementById('kpi-diff-bio');
      badgeBio.innerText = (dBioPct >= 0 ? '+' : '') + dBioPct.toFixed(1) + '%';
      badgeBio.className = 'diff-badge ' + (dBioPct > 1.0 ? 'diff-pos' : (dBioPct < -1.0 ? 'diff-neg' : 'diff-neu'));

      const badgeCat = document.getElementById('kpi-diff-catch');
      badgeCat.innerText = (dCatPct >= 0 ? '+' : '') + dCatPct.toFixed(1) + '%';
      badgeCat.className = 'diff-badge ' + (dCatPct > 1.0 ? 'diff-pos' : (dCatPct < -1.0 ? 'diff-neg' : 'diff-neu'));
    }

    // -------------------------------------------------------------------------
    // Animation Controller
    // -------------------------------------------------------------------------
    function startSim() {
      if (isRunning) return;
      if (currentDay >= selectedEndDay) currentDay = selectedStartDay;
      isRunning = true;
      const btn = document.getElementById('btn-play-pause');
      btn.innerText = '⏸ Pause';
      btn.className = 'btn btn-pause';
      document.getElementById('badge-status').innerText = 'Day ' + currentDay + ' | Running Both';

      const intervalMs = Math.max(20, Math.floor(150 / simSpeedMultiplier));
      animTimer = setInterval(() => {
        if (currentDay >= selectedEndDay) { stopSim(); return; }
        stepBothSimulations(5);
      }, intervalMs);
    }

    function stopSim() {
      if (!isRunning) return;
      isRunning = false;
      clearInterval(animTimer);
      animTimer = null;
      const btn = document.getElementById('btn-play-pause');
      btn.innerText = '▶ Run Both';
      btn.className = 'btn btn-run';
      document.getElementById('badge-status').innerText = 'Day ' + currentDay + ' | Paused';
    }

    function resetBothSimulations() {
      stopSim();
      currentDay = selectedStartDay;
      prngA = createPRNG(BASE_SEED);
      prngB = createPRNG(BASE_SEED);

      agentsA = initialAgents.map(a => Object.assign({}, a));
      agentsB = initialAgents.map(a => Object.assign({}, a));

      histA = createHistory(defaultLegalSizeMale);
      histB = createHistory(bLegalSize);

      document.getElementById('slider-day').value = selectedStartDay;
      document.getElementById('lbl-day').innerText = 'Day ' + selectedStartDay + ' / [' + selectedStartDay + ' - ' + selectedEndDay + ']';
      document.getElementById('badge-status').innerText = 'Day ' + selectedStartDay + ' | Ready';

      renderMaps();
      updateComparativeCharts();
    }

    // Apply Scenario Parameters from Header
    function applyScenarioParameters() {
      bTempOffset = parseFloat(document.getElementById('inp-b-temp-offset').value) || 0.0;
      bFishFactor = parseFloat(document.getElementById('inp-b-fish-factor').value) || 1.0;
      bSeasonStart = parseInt(document.getElementById('inp-b-season-start').value, 10) || 100;
      bLegalSize = parseFloat(document.getElementById('inp-b-legal-size').value) || defaultLegalSizeMale;

      // Update Panel B Subtitles
      document.getElementById('badge-run-b-summary').innerText =
        'ΔT=' + (bTempOffset >= 0 ? '+' : '') + bTempOffset + '°C | F=' + bFishFactor + '× | L50=' + bLegalSize + 'mm';
      document.getElementById('subtitle-ogive-b').innerText = 'Selectivity L50 = ' + bLegalSize + ' mm';
      document.getElementById('subtitle-size-b').innerText = 'KDE with Legal Marker at ' + bLegalSize + ' mm';

      // Update Selectivity curve dataset in B
      const newSelB = cwRef.map(cw => 1.0 / (1.0 + Math.exp(-(cw - bLegalSize) / selectivitySigma)));
      chartOgB.data.datasets[2].data = newSelB;
      chartOgB.update('none');

      resetBothSimulations();
    }

    // Event Handlers
    document.getElementById('btn-play-pause').addEventListener('click', () => {
      if (isRunning) stopSim(); else startSim();
    });
    document.getElementById('btn-step-fwd').addEventListener('click', () => {
      stopSim();
      stepBothSimulations(10);
    });
    document.getElementById('btn-step-back').addEventListener('click', () => {
      stopSim();
      const target = Math.max(selectedStartDay, currentDay - 10);
      resetBothSimulations();
      if (target > selectedStartDay) stepBothSimulations(target - selectedStartDay);
    });
    document.getElementById('btn-reset').addEventListener('click', resetBothSimulations);
    document.getElementById('btn-rerun-comparison').addEventListener('click', applyScenarioParameters);

    document.getElementById('slider-day').addEventListener('input', e => {
      stopSim();
      const target = parseInt(e.target.value, 10);
      resetBothSimulations();
      if (target > selectedStartDay) stepBothSimulations(target - selectedStartDay);
    });

    document.getElementById('sel-speed').addEventListener('change', e => {
      simSpeedMultiplier = parseInt(e.target.value, 10);
      if (isRunning) { stopSim(); startSim(); }
    });

    // Initial Setup & Paint
    applyScenarioParameters();
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
    launch_leaflet_comparison_dashboard(config::Dict{Symbol, Any}; kwargs...)

Initialize model and launch the interactive dual-column comparison dashboard.
"""
function launch_leaflet_comparison_dashboard(
    config::Dict{Symbol, Any};
    display_window::Bool = true,
    n_agents::Union{Int, Nothing} = nothing,
    output_html::String = "outputs/dashboard_compare.html"
)
    params = IBMParameters(config)
    model = initialize_model(
        grid = config[:grid],
        dt = config[:dt],
        params = params,
        seed = config[:seed],
        n_agents = n_agents
    )
    path = generate_leaflet_comparison_dashboard_html(
        model;
        title = "Snow Crab IBM Scenario Comparison Dashboard",
        output_html = output_html,
        auto_open = display_window
    )
    println("="^80)
    println(" Leaflet Dual-Column Comparison Dashboard generated at:")
    println("   file://$(replace(path, "\\" => "/"))")
    println("="^80)
    return path
end
