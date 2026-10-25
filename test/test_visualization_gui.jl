# Segment: Visualization Tools, Plots, and GUI Dashboard
include("test_fixtures.jl")

@testset "TethysBio - Visualization and Dashboard" begin
    @testset "Visualization Tools & Plotting" begin
        model = initialize_model(n_agents = 40, grid = (15, 15), seed = 42)
        adata = [(:stage, length), (:size, safe_mean), (is_mature_male, count),
                 (is_ovigerous, count)]
        mdata = [:total_eggs_spawned, :total_recruits_settled]

        agent_df, model_df = run_simulation(model, 30; adata = adata, mdata = mdata)

        # 1. Population timeseries
        p_pop = plot_population_timeseries(agent_df; model_df = model_df)
        @test p_pop isa Figure

        # 2. Stage distribution
        p_inst = plot_stage_distribution(model)
        @test plot_stage_distribution(model) isa Figure
        @test p_inst isa Figure

        # 3. Size distribution by sex
        p_size = plot_size_by_sex(model)
        @test p_size isa Figure

        # 4. Stage allometry
        p_allo = plot_stage_allometry(model)
        @test plot_stage_allometry(model) isa Figure
        @test p_allo isa Figure

        # 5. Reproduction dynamics
        p_rep = plot_reproduction_dynamics(agent_df, model_df)
        @test p_rep isa Figure

        # 6. Spatial habitat
        p_spat = plot_spatial_habitat(model)
        @test p_spat isa Figure

        # 7. Maturity ogives
        p_mat = plot_maturity_ogives(model)
        @test p_mat isa Figure

        # 8. Executive simulation dashboard
        p_dash = plot_simulation_dashboard(model, agent_df; model_df = model_df)
        @test p_dash isa Figure

        # 9. Stock-recruitment diagnostic figure
        p_sr = plot_stock_recruitment(model, agent_df; model_df = model_df)
        @test p_sr isa Figure

        # 10. Batch export to temporary directory
        mktempdir() do tmp_dir
            exported = export_all_plots(
                model,
                agent_df,
                tmp_dir;
                model_df = model_df,
                ext = "png"
            )
            @test length(exported) == 10
            @test any(endswith(f, "abm_spatial_summary.png") for f in exported)
            for f in exported
                @test isfile(f)
                @test filesize(f) > 0
            end
        end
    end

    @testset "Agents.jl Visualization and Animation Tools" begin
        # 1. Attribute mappings
        test_juv_m = Organism(
            1, (5, 5), juvenile, male, 5, 100.0, 15.0, 1.5, 20.0, 0.0, 50.0,
            false, 0, 0.0, 0
        )
        test_juv_f = Organism(
            2, (5, 5), juvenile, female, 5, 100.0, 15.0, 1.5, 20.0, 0.0, 50.0,
            false, 0, 0.0, 0
        )
        test_ovig = Organism(
            3, (5, 5), adult, female, 10, 800.0, 75.0, 120.0, 40.0, 0.0, 150.0,
            false, 50000, 365.0, 1
        )
        test_legal = Organism(
            4, (5, 5), adult, male, 12, 1200.0, 105.0, 450.0, 60.0, 0.0, 200.0,
            false, 0, 0.0, 0
        )
        test_subleg = Organism(
            5, (5, 5), adult, male, 10, 900.0, 80.0, 180.0, 40.0, 0.0, 160.0,
            false, 0, 0.0, 0
        )
        test_larva = Organism(
            6, (5, 5), zoea, undifferentiated, 1, 10.0, 1.5, 0.01, 5.0, 0.0, 10.0,
            false, 0, 0.0, 0
        )

        @test organism_color(test_juv_m, IBMParameters()) == "#56B4E9"
        @test organism_color(test_juv_f, IBMParameters()) == "#CC79A7"
        @test organism_color(test_ovig, IBMParameters()) == "#D55E00"
        @test organism_color(test_legal, IBMParameters()) == "#003B73"
        @test organism_color(test_subleg, IBMParameters()) == "#0072B2"
        @test organism_color(test_larva, IBMParameters()) == "#E69F00"
        # The colour ramp follows the configured legal size, not a hardcoded one
        loose = IBMParameters(fishery = FisheryParams(legal_size_male = 80.0))
        @test organism_color(test_subleg, loose) == "#003B73"

        @test organism_marker(test_juv_m) === :rect
        @test organism_marker(test_juv_f) === :circle
        @test organism_marker(test_larva) === :diamond

        @test organism_size(test_larva) >= 4.0
        @test organism_size(test_legal) > organism_size(test_juv_m)

        # 2. Environmental heatarray extraction
        model_test = initialize_model(n_agents = 10, grid = (15, 15), seed = 99)
        t_mat = organism_heatarray(model_test, :temperature_field)
        @test size(t_mat) == (15, 15)
        f_mat = organism_heatarray(model_test, :food_availability_field)
        @test size(f_mat) == (15, 15)

        # 3. Spatial summary plot
        out_summary = joinpath(tempdir(), "test_abm_summary_$(rand(1000:9999)).png")
        fig = plot_abm_summary(model_test; output_path = out_summary)
        @test isfile(out_summary)
        @test filesize(out_summary) > 0
        rm(out_summary; force = true)

        # 4. Animation video generation
        out_anim = joinpath(tempdir(), "test_anim_$(rand(1000:9999)).mp4")
        anim_path = animate_organism_simulation(
            model_test;
            filepath = out_anim,
            frames = 3,
            dt = 1,
            framerate = 5
        )
        @test isfile(anim_path)
        @test filesize(anim_path) > 0
        rm(anim_path; force = true)
    end

    @testset "Interactive GUI Dashboard" begin
        fig, state = launch_organism_gui(
            n_agents = 25,
            grid = (15, 15),
            dt = 1.0,
            seed = 42,
            display_window = false
        )
        @test isa(fig, Figure)
        @test isa(state, IBMGuiState)
        @test state.current_time[] == 0.0
        @test length(state.obs_agent_x[]) == 25

        # Test stepping GUI state
        step_gui_simulation!(state, 10.0)
        @test state.current_time[] == 10.0
        @test length(state.t_history) >= 2
        @test state.t_history[end] == 10.0
        @test length(state.obs_agent_x[]) > 0
    end

    @testset "Dashboard count axes are log10" begin
        # `log10(0)` is `-Inf` and an axis cannot place it. Zero is a real value
        # here — no eggs brooded yet, a cohort that has already individuated — so
        # the floor is what keeps the figure finite.
        @test COUNT_PLOT_FLOOR == 1.0
        @test log10_count(0) == 0.0
        @test log10_count(0.4) == 0.0            # sub-unit counts floor
        @test log10_count(1) == 0.0
        @test log10_count(10) ≈ 1.0
        @test log10_count(1_000_000) ≈ 6.0
        @test all(isfinite, (log10_count(v) for v in (0, 0.5, 1, 3, 1e7, -4)))
        @test issorted([log10_count(v) for v in [0, 1, 5, 50, 500, 5000]])

        fig, state = launch_organism_gui(
            n_agents = 30,
            grid = (12, 12),
            dt = 1.0,
            seed = 7,
            display_window = false
        )
        step_gui_simulation!(state, 400.0)

        # Every count series is the log of the raw buffer it was built from, and
        # the buffers still hold raw counts so a linear re-plot stays possible.
        @test state.obs_ts_total[][end][2] ≈ log10_count(state.n_total_history[end])
        @test state.obs_ts_legal[][end][2] ≈ log10_count(state.n_legal_history[end])
        @test state.n_total_history[end] == round(Int, state.n_total_history[end])
        @test state.n_total_history[end] == length(state.obs_agent_x[])

        for series in (state.obs_ts_total[], state.obs_ts_females[], state.obs_ts_males[],
            state.obs_ts_legal[], state.obs_ts_egg[],
            state.obs_ts_early_agents[], state.obs_ts_early_agents_f[],
            state.obs_ts_early_agents_u[],
            (o[] for o in state.obs_ts_early_stage)...)
            @test !isempty(series)
            @test all(p -> isfinite(p[2]), series)   # no -Inf, no NaN
            @test issorted([p[1] for p in series])  # time-ordered
        end
        # Biomass and catch are kilograms, not counts, and stay linear. The
        # plotted points are Float32, so compare with a tolerance.
        @test state.obs_ts_biomass[][end][2] ≈ state.biomass_legal_history[end]
        @test state.obs_ts_catch[][end][2] ≈ state.catch_history[end]

        # Stock-recruit has two count axes; both transformed.
        @test all(p -> isfinite(p[1]) && isfinite(p[2]), state.obs_sr_pts[])
        @test all(p -> p[1] ≈ log10_count(10.0^p[1]), state.obs_sr_pts[])
    end

    @testset "Stage abundance runs eggs then aggregate cohorts against time" begin
        @test GUI_EARLY_STAGES == FIRST_COHORT_STAGE:3

        fig, state = launch_organism_gui(
            n_agents = 40,
            grid = (14, 14),
            dt = 1.0,
            seed = 11,
            display_window = false
        )
        step_gui_simulation!(state, 500.0)

        # One time series per `GUI_EARLY_STAGES`, plus eggs, all sharing one x grid.
        @test length(state.obs_ts_early_stage) == length(GUI_EARLY_STAGES)
        @test all(o -> length(o[]) == length(state.t_history), state.obs_ts_early_stage)
        @test length(state.obs_ts_egg[]) == length(state.t_history)
        @test all(s -> length(s) == length(state.t_history), state.n_early_stage_history)
        @test length(state.n_early_agents_history) == length(state.t_history)

        # Each plotted value must be the log of the raw buffer beside it, so the
        # panel cannot drift from the numbers it reports.
        for k in eachindex(state.obs_ts_early_stage)
            pts = state.obs_ts_early_stage[k][]
            raw = state.n_early_stage_history[k]
            @test all(i -> isapprox(pts[i][2], log10_count(raw[i]); atol = 1e-4),
                eachindex(raw))
        end
        @test all(i -> isapprox(state.obs_ts_egg[][i][2], log10_count(state.n_egg_history[i]);
            atol = 1e-4), eachindex(state.n_egg_history))

        # Eggs and stage 0 are disjoint pools: a clutch only becomes a stage-0
        # cohort at `hatch_day`. If eggs were being counted as stage 0 this would
        # double-count the same animals.
        @test all(>=(0.0), state.n_egg_history)
        @test all(series -> all(>=(0.0), series), state.n_early_stage_history)
        @test all(h -> all(>=(0.0), h), state.n_early_agents_history)

        # The aggregate cohorts the model actually seeds must be visible on the
        # figure, otherwise the early stages are a flat line at the floor.
        @test any(>(1.0), state.n_early_stage_history[1])


        # Sex is not discriminated below individuation: `AggregateGroup` has no
        # `sex` field, so those series are single grey lines, and the sexed series
        # come only from `Organism` agents.
        @test !hasfield(AggregateGroup, :sex)
        m, f, u = state.n_early_agents_history[end]
        @test m + f + u == state.n_total_history[end]
        @test u == 0
        @test any(h -> h[1] > 0.0, state.n_early_agents_history)

        # Reset must clear the new buffers, or the figure redraws a stale run
        # stitched onto the fresh one.
        n_before = length(state.t_history)
        empty!(state.t_history)
        empty!(state.n_egg_history)
        for series in state.n_early_stage_history
            empty!(series)
        end
        empty!(state.n_early_agents_history)
        step_gui_simulation!(state, 5.0)
        @test length(state.t_history) == 1
        @test all(series -> length(series) == 1, state.n_early_stage_history)
        @test length(state.n_egg_history) == 1
        @test length(state.n_early_agents_history) == 1
        @test length(state.obs_ts_early_stage) == length(GUI_EARLY_STAGES)
        @test n_before > 0
    end

    @testset "History buffers stay aligned with the time axis" begin
        # The observable point vectors are rebuilt with `eachindex(t_history)` and
        # index every buffer positionally, so a buffer that is one longer or shorter
        # misaligns the series against the x axis rather than raising.
        fig, state = launch_organism_gui(n_agents = 20, grid = (12, 12), dt = 1.0,
            seed = 4, display_window = false)

        buffers = s -> [
            length(s.t_history), length(s.n_total_history), length(s.n_females_history),
            length(s.n_males_history), length(s.n_legal_history),
            length(s.biomass_legal_history), length(s.catch_history),
            length(s.mean_weight_legal_history), length(s.mean_weight_male_history),
            length(s.mean_weight_female_history), length(s.mean_weight_total_history),
            length(s.n_egg_history), length(s.n_early_agents_history),
            (length(b) for b in s.n_early_stage_history)...,
        ]

        # `launch_organism_gui` ends with a 0-day initial paint; that must not append
        # a second point at t = 0.
        @test length(state.t_history) == 1
        step_gui_simulation!(state, 0.0)
        @test length(state.t_history) == 1        # repaint only
        @test all(==(1), buffers(state))
        step_gui_simulation!(state, 10.0)
        @test state.t_history == [0.0, 10.0]      # no duplicated origin
        @test all(==(2), buffers(state))
        for _ in 1:8
            step_gui_simulation!(state, 5.0)
        end
        @test all(==(length(state.t_history)), buffers(state))

        # And the plotted series match the axis.
        n = length(state.t_history)
        @test all(o -> length(o[]) == n,
            [state.obs_ts_total, state.obs_ts_females, state.obs_ts_males,
                state.obs_ts_legal, state.obs_ts_egg, state.obs_ts_early_agents,
                state.obs_ts_early_agents_f, state.obs_ts_early_agents_u])
        @test all(o -> length(o[]) == n, state.obs_ts_early_stage)
    end

    @testset "Stage figure covers every stage and splits sex where distinguished" begin
        p = IBMParameters()
        juv = p.organism.juvenile_max_stage
        max_male = p.organism.max_stage_male
        max_female = p.organism.max_stage_female
        # Stage 0 is the brooded egg, which is not a cohort, so the aggregate series
        # covers stages 1 .. juv-1.
        n_agg = juv - 1                    # aggregate stages 1 .. juv-1
        n_male = max_male - juv + 1       # agent stages juv .. max_male
        n_female = max_female - juv + 1

        fig, state = launch_organism_gui(n_agents = 120, grid = (18, 18), dt = 1.0,
            seed = 3, display_window = false)
        for _ in 1:60
            step_gui_simulation!(state, 5.0)
        end

        # One category per stage from 0 to the highest any agent reaches, plus the
        # brooded clutch. The bound is `max_stage_*`, not the species stage tables:
        # those stop at 12 and 11 and bound the Leslie projection, while agents molt
        # on past them, so sizing from the tables would drop the largest cohorts.
        @test max_male >= p.species.leslie_max_male_stage
        @test max_female >= p.species.leslie_max_female_stage
        @test length(state.obs_stage_agg[]) == n_agg
        @test length(state.obs_stage_male[]) == n_male
        @test length(state.obs_stage_female[]) == n_female
        # Eggs + every stage, contiguous with no gap: one leading category for the
        # brooded clutch, then one per cohort stage and one per agent stage. The
        # `2 +` was wrong — it counted the leading category twice.
        @test 1 + n_agg + n_male == 1 + max_male

        # The tallies must agree with the agents themselves, indexed stage-wise.
        living = collect(allagents(state.model))
        male_counts = zeros(Int, n_male)
        female_counts = zeros(Int, n_female)
        for o in living
            k = o.stage - juv + 1
            o.sex === male && (1 <= k <= n_male) && (male_counts[k] += 1)
            o.sex === female && (1 <= k <= n_female) && (female_counts[k] += 1)
        end
        @test state.obs_stage_male[] == Float64.(male_counts)
        @test state.obs_stage_female[] == Float64.(female_counts)

        # Sex is split only where it is distinguished: `AggregateGroup` has no `sex`
        # field, so every stage below the individuation boundary is undifferentiated
        # and has no male/female pair.
        @test !hasfield(AggregateGroup, :sex)
        @test n_agg == juv - 1
        # Females stop two stages below males under the snow crab configuration, so
        # the two terminal categories carry a male bar only. If that ever changes the
        # figure must grow a third series rather than silently dropping a sex.
        @test n_female <= n_male

        # Aggregate cohorts accumulate into the per-stage series, and re-deriving
        # them from the groups must match.
        agg = Agents.abmproperties(state.model)[:aggregate_populations]
        expected = zeros(Float64, n_agg)
        for g in agg.groups
            k = g.stage
            1 <= k <= n_agg && (expected[k] += g.abundance)
        end
        @test state.obs_stage_agg[] == expected
        @test all(>=(0.0), state.obs_stage_agg[])
    end

    @testset "Calendar readout: year, month, julian day, duration" begin
        # `floor((day % 365) / 30.416) + 1` put the last day of every month into
        # the next one. The month now comes from real calendar month lengths.
        @test calendar_month(1, 365.0) == 1
        @test calendar_month(31, 365.0) == 1
        @test calendar_month(32, 365.0) == 2
        @test calendar_month(59, 365.0) == 2
        @test calendar_month(60, 365.0) == 3
        @test calendar_month(273, 365.0) == 9
        @test calendar_month(334, 365.0) == 11   # 334 is the last day of November
        @test calendar_month(335, 365.0) == 12   # 335 is the first of December
        @test calendar_month(365, 365.0) == 12
        # Every day of the year lands in exactly one month, in order.
        months = [calendar_month(d, 365.0) for d in 1:365]
        @test issorted(months)
        @test all(1 .<= months .<= 12)
        @test calendar_month(366, 365.0) == 12
        # In a leap year February gets its extra day, so later boundaries hold.
        @test calendar_month(60, 366.0) == 2
        @test calendar_month(61, 366.0) == 3

        fig, state = launch_organism_gui(
            n_agents = 20, grid = (10, 10), dt = 1.0, seed = 3, display_window = false
        )
        for _ in 1:90
            step_gui_simulation!(state, 5.0)
        end
        dpy = abmproperties(state.model)[:params].simulation.days_per_year
        # Year is an integer count, Day is 1-based within the year and wraps,
        # Duration is the elapsed model time under its own name.
        m = match(r"^Year: (\d+)\s+Month: (\d+)\s+Day: (\d+)\s+Duration: (\d+) d$",
            state.status_text[])
        @test m !== nothing
        if m !== nothing
            year_n = parse(Int, m.captures[1])
            month_n = parse(Int, m.captures[2])
            day_n = parse(Int, m.captures[3])
            dur = parse(Int, m.captures[4])
            t = state.t_history[end]
            @test year_n == floor(Int, t / dpy)
            @test 1 <= day_n <= round(Int, dpy)
            @test month_n == calendar_month(day_n, dpy)
            @test dur == round(Int, t)
            @test day_n != dur || t < dpy   # "Day" is not the elapsed total
        end
    end

    @testset "Stock-recruit panel carries a Ricker fit" begin
        # Fitted in linear space (the panel draws log10), from the untransformed
        # pairs, which therefore have to be kept alongside the plotted points.
        s = [1.0e5, 2.0e5, 4.0e5, 8.0e5, 1.6e6]
        r = [2.0e4, 3.5e4, 5.0e4, 5.5e4, 4.0e4]
        fit = fit_stock_recruitment(s, r)
        @test fit.is_fitted
        @test fit.n_obs == 5
        @test fit.ricker.alpha > 0
        # Predictor is the Ricker form, and peaks near S = 1/beta.
        @test fit.ricker.predict(1.0e5) > 0
        @test isfinite(fit.ricker.predict(1.0e5))
        argmax_s = reduce((a, b) -> fit.ricker.predict(a) >= fit.ricker.predict(b) ? a : b,
            range(1.0e4, 1.0e7; length = 2000))
        @test isapprox(argmax_s, fit.ricker.S_max; rtol = 0.05)

        # Below three usable points the fitter reports `is_fitted = false` and
        # returns a through-origin fallback; drawing that would put a confident
        # curve through two numbers.
        @test !fit_stock_recruitment([1.0e5], [2.0e4]).is_fitted
        @test !fit_stock_recruitment([1.0e5, 2.0e5], [2.0e4, 3.0e4]).is_fitted
        # A point with zero recruits is not usable and must not count.
        @test fit_stock_recruitment([1.0e5, 2.0e5, 0.0], [2.0e4, 3.0e4, 4.0e4]).n_obs == 2

        fig, state = launch_organism_gui(
            n_agents = 150, grid = (24, 24), dt = 1.0, seed = 5, display_window = false
        )
        for _ in 1:240
            step_gui_simulation!(state, 5.0)
        end
        # The raw pairs are retained untransformed even though the panel logs them.
        @test length(state.sr_raw) == length(state.obs_sr_pts[])
        @test all(t -> t[1] >= 0 && t[2] >= 0, state.sr_raw)
        @test all(i -> state.obs_sr_pts[][i][1] ≈ Float32(log10_count(state.sr_raw[i][1])),
            eachindex(state.sr_raw))
        if length(state.sr_raw) >= 3
            @test !isempty(state.obs_sr_fit_curve[])
            @test all(p -> isfinite(p[1]) && isfinite(p[2]), state.obs_sr_fit_curve[])
            # Parameters are in the corner as text, rounded.
            @test occursin("Ricker", state.obs_sr_fit_text[])
            @test occursin("R²", state.obs_sr_fit_text[])
        else
            @test isempty(state.obs_sr_fit_curve[])
            @test isempty(state.obs_sr_fit_text[])
        end
    end

    @testset "Size frequency: four kernel density curves on a linear axis" begin
        fig, state = launch_organism_gui(
            n_agents = 40, grid = (14, 14), dt = 1.0, seed = 7, display_window = false
        )
        step_gui_simulation!(state, 200.0)
living = collect(allagents(state.model))
        # The four plotted series are immature and mature, per sex, split by the
        # agent's own `is_mature` flag rather than an independent size threshold.
        # Mature and immature are therefore a true partition of each sex: between
        # them they account for every individual of that sex, which is what lets the
        # panel drop the whole-sex curves.
        for (imm_obs, mat_obs, sx) in (
            (state.obs_cw_immature_male, state.obs_cw_mature_male, male),
            (state.obs_cw_immature_female, state.obs_cw_mature_female, female))
            mat = [a for a in living if a.sex === sx && a.is_mature]
            imm = [a for a in living if a.sex === sx && !a.is_mature]
            @test length(mat) + length(imm) == count(a -> a.sex === sx, living)
            @test mat_obs[] == Float32[Float32(a.size) for a in mat]
            @test imm_obs[] == Float32[Float32(a.size) for a in imm]
        end

        # The panel is a kernel density estimate on a linear axis. The binned
        # histogram that once stood in for it is gone, so assert it stays gone: it
        # was a divergent second way to compute the same curve.
        @test !isdefined(TethysBio, :size_frequency)
        @test !isdefined(TethysBio, :GUI_SIZE_BIN_WIDTH_MM)
        # A shared bandwidth, or the small samples get smoothed far more finely than
        # the large ones and draw as spikes beside them.
        @test SIZE_DENSITY_BANDWIDTH_MM > 0

        # Both scalings are available and each is what it says: `:probability`
        # integrates to one so a linear axis can compare unequal groups, `:count`
        # integrates to the group size so a subset reads as lying inside its parent.
        grid = collect(0.0:0.5:165.0)
        step = grid[2] - grid[1]
        for n in (1, 15, 200)
            s = n == 1 ? [70.0] : collect(range(20.0, 120.0; length = n))
            @assert isapprox(sum(kde_density(s, grid; scale = :count)) * step, n;
                rtol = 0.02)
            @assert isapprox(sum(kde_density(s, grid; scale = :probability)) * step,
                1.0; rtol = 0.02)
        end
        @test_throws ErrorException kde_density([70.0], grid; scale = :nonsense)
        @test_throws ErrorException kde_density([70.0], grid; bandwidth = 0.0)
        # Empty and all-non-finite samples draw nothing rather than a phantom
        # cohort at the origin.
        @test all(iszero, kde_density(Float64[], grid))
        @test all(iszero, kde_density([NaN, Inf, -Inf], grid))
        # Identical input must give an identical curve — the case the panel relies
        # on when a group has no members outside its mature subset.
        a = collect(80.0:0.5:130.0)
        @test kde_density(a, grid) == kde_density(copy(a), grid)
        # The four series feeding the panel are plain vectors of sizes in
        # millimetres, which is what a KDE consumes: no transform, no bin edges.
        for obs in (state.obs_cw_immature_male, state.obs_cw_mature_male,
            state.obs_cw_immature_female, state.obs_cw_mature_female)
            @test eltype(obs[]) === Float32
            @test all(s -> isfinite(s) && s >= 0, obs[])
        end
    end

    @testset "GUI annual frame matches the production schema" begin
        # The GUI kept its own hand-written annual column list, which lacked
        # `day`, `total_biomass_kg` and `annual_egg_production` and called the
        # sample counter `year`. It must now be a superset of the shared schema.
        gui_schema = create_empty_annual_data()
        shared = first.(ANNUAL_PRODUCTION_SCHEMA)
        for col in shared
            @test col in propertynames(gui_schema)
        end
        @test :year ∉ propertynames(gui_schema)
        @test :recruits in propertynames(gui_schema)
        @test :cumulative_catch in propertynames(gui_schema)

        # Stepping past an annual snapshot must produce a row in that schema.
        fig, state = launch_organism_gui(
            n_agents = 15,
            grid = (10, 10),
            dt = 1.0,
            seed = 21,
            display_window = false
        )
        step_gui_simulation!(state, 300.0)
        @test nrow(state.annual_data) >= 1
        for col in shared
            @test col in propertynames(state.annual_data)
        end
        @test state.annual_data.sample_index[1] == 1
        @test state.annual_data.total_biomass_kg[1] > 0.0
    end

end
