"""
Test suite for the GeoIBM package.
"""

using Test
using Random
using Statistics
using DataFrames
using Agents
using LinearAlgebra
using TOML

# Activate package
using DrWatson
@quickactivate "GeoIBM"
using GeoIBM

@testset "GeoIBM Tests" begin

    @testset "Empirical Bounds and Allometry" begin
        # Monotonicity of lower bounds across stages
        @test llbm(7) < llbm(8) < llbm(9) < llbm(10)
        @test llbf(7) < llbf(8) < llbf(9) < llbf(10)

        # Bounds order: lower bound < upper bound
        @test get_stage_size(9, :male; type=:lb) < get_stage_size(9, :male; type=:ub)
        @test get_stage_size(9, :female; type=:lb) < get_stage_size(9, :female; type=:ub)

        mid_m = get_stage_size(6, :male; type=:mid)
        @test llbm(6) < mid_m < llbm(7)

        # Growth schedule dataframe. The stage range is derived from `max_stage_*`, so
        # this no longer asserts a hand-written row count that the schedule
        # outgrew silently.
        sched = growth_schedule()
        p_sched = IBMParameters()
        @test isa(sched, DataFrame)
        @test nrow(sched) ==
              max(p_sched.organism.max_stage_male, p_sched.organism.max_stage_female) - 1
        # The lower-bound curve saturates at `max_size_male`, so at high stages the
        # per-stage lower and the next stage's lower coincide and a strict
        # inequality no longer holds. Compare against the saturation limit instead,
        # which is the property that actually matters.
        sat_m = p_sched.species.max_size_male
        @test all(sched.size_male_lb .<= sat_m)
        @test all(sched.size_female_lb .<= p_sched.species.max_size_female)
        @test issorted(sched.size_male_lb)
        @test issorted(sched.size_female_lb)
        # And strictly increasing where the curve has not yet flattened.
        @test all(diff(sched.size_male_lb) .>= 0)
    end

    @testset "Empirical Growth Increments and Maturity" begin
        rng = MersenneTwister(42)

        # Hiatt growth increment: size must increase
        cw_orig = 50.0
        cw_next_imm = predict_postgrowth_size(cw_orig; sex=0, mat=0, rng=rng)
        cw_next_mat = predict_postgrowth_size(cw_orig; sex=0, mat=1, rng=rng)

        @test cw_next_imm > cw_orig
        @test cw_next_mat > cw_orig
        # Terminal molt increment is reduced due to reproduction allocation
        @test cw_next_mat < cw_next_imm

        # Maturity probability bounds and monotonicity
        # Small juveniles should have near-zero maturity probability
        @test maturity(0, log(20.0); rng=rng) < 0.05
        @test maturity(1, log(15.0); rng=rng) < 0.05

        # Large individuals should have near 1.0 maturity probability
        @test maturity(0, log(140.0); rng=rng) > 0.90
        @test maturity(1, log(80.0); rng=rng) > 0.90

        # Fecundity estimates must be biologically plausible (10^3 to 10^6 eggs)
        fec_prim = fecundity(60.0; fecundity_parameterization="moncton", is_primiparous=true, rng=rng)
        fec_mult = fecundity(60.0; fecundity_parameterization="moncton", is_primiparous=false, rng=rng)
        fec_sm   = fecundity(60.0; fecundity_parameterization="st.marie", rng=rng)

        @test 1000.0 < fec_prim < 500000.0
        @test 1000.0 < fec_mult < 500000.0
        @test 1000.0 < fec_sm < 500000.0

        # Selectivity ogive
        @test selectivity(50.0; s50=95.0, s50_sigma=5.0) < 0.01
        @test selectivity(95.0; s50=95.0, s50_sigma=5.0) ≈ 0.50
        @test selectivity(130.0; s50=95.0, s50_sigma=5.0) > 0.99
    end

    @testset "Organism Struct and Stage Tracking" begin
        rng = MersenneTwister(123)

        # Initialize model with 20 agents
        model = initialize_model(n_agents=20, grid=(20, 20), seed=123)
        @test nagents(model) == 20

        # Verify all agents have valid stage and size. The seeded range is the whole
        # benthic span, not the old `6:9` window: seeding stages 6-9 only meant the
        # maturity test `stg >= 10` could never be true, so nothing was ever mature
        # or legal at t=0.
        benthic_lo = model.params.organism.juvenile_max_stage
        for agent in allagents(model)
            max_stage = agent.sex === female ? model.params.organism.max_stage_female :
                        model.params.organism.max_stage_male
            @test benthic_lo <= agent.stage <= max_stage
            @test agent.size > 0.0
            @test agent.age_days >= 0.0
        end

        # Test molt progression and stage increment. An *immature* individual is picked
        # explicitly: `perform_molt!` returns early for a mature one, and maturity
        # is now evaluated every step rather than only at ecdysis, so `first(...)`
        # is no longer reliably juvenile.
        test_org = nothing
        for a in allagents(model)
            if !a.is_mature
                test_org = a
                break
            end
        end
        @test test_org !== nothing
        init_stage = test_org.stage
        init_size = test_org.size

        perform_molt!(test_org, 10.0; rng=rng)
        @test test_org.stage == init_stage + 1
        @test test_org.size > init_size
        @test test_org.last_growth_event_time == 10.0

        # Terminal molt check: mature organisms should not molt
        test_org.is_mature = true
        @test check_molt_conditions(test_org, 5.0) == false
    end

    @testset "Simulation Step and Population Summary" begin
        model = initialize_model(n_agents=30, grid=(15, 15), seed=456)
        init_count = nagents(model)
        @test init_count == 30

        # Run 5 daily simulation steps
        for _ in 1:5
            agent_step!(first(allagents(model)), model)
            model_step!(model)
        end

        # Summary statistics calculation.
        # The seeded count is not an upper bound: aggregate juvenile cohorts can
        # individuate into new agents, so assert instead that the summary agrees
        # with the live agent set and that its demographic partition is exact.
        summary = summarize_population(model)
        @test summary.total_organisms == nagents(model)
        @test summary.total_organisms >= 0
        @test summary.n_males + summary.n_females == summary.total_organisms
        @test summary.mean_size > 0.0
        @test summary.total_biomass_kg > 0.0
        @test summary.mean_mass > 0.0
        @test summary.total_ssb_kg >= 0.0
        @test summary.female_ssb_kg + summary.male_mature_biomass_kg == summary.total_ssb_kg
        @test summary.total_biomass_kg >= summary.total_ssb_kg
        @test !isempty(summary.size_by_stage)
    end

    @testset "Reproduction, Egg Generation, and Recruitment" begin
        # Initialize model in spring mating season (day 40)
        model = initialize_model(n_agents=0, grid=(10, 10), seed=789)
        model.current_time = 40.0

        # Add receptive adult female and adult male nearby
        w_f = calculate_mass(65.0, female; mat=1)
        w_m = calculate_mass(110.0, male; mat=1)
        f_org = add_agent!(
            (5, 5), model, 10, female, 1500.0, 65.0, w_f, 50.0,
            0.0, 1000.0, true, 0, 0.0, 0, true
        )
        m_org = add_agent!(
            (5, 6), model, 12, male, 1800.0, 110.0, w_m, 80.0,
            0.0, 1000.0, false, 0, 0.0, 0, true
        )

        # Female executes agent_step! -> copulates and extrudes clutch
        agent_step!(f_org, model)

        @test f_org.egg_count > 5000
        @test f_org.hatch_day > model.current_time
        @test model.total_eggs_spawned > 5000
        @test f_org.is_mating_ready == false

        # Advance time to hatch day
        model.current_time = f_org.hatch_day + 1.0
        n_groups_before = length(model.aggregate_populations.groups)

        agent_step!(f_org, model)

        @test f_org.egg_count == 0
        @test f_org.is_mating_ready == true
        @test length(model.aggregate_populations.groups) > n_groups_before
        # A clutch becomes a cohort at stage 1 on hatching: stage 0 is the brooded egg,
        # which is carried by the female rather than being a cohort.
        @test model.aggregate_populations.groups[end].stage == 1
        @test model.aggregate_populations.groups[end].abundance > 5000
    end

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

    @testset "Length-Weight Allometry, Stock Biomass, and Molt Seasonality" begin
        # 1. Length-Weight allometric scaling: W = alpha * Size^beta
        w_f_imm = calculate_mass(50.0, female; mat = 0)
        w_f_mat = calculate_mass(50.0, female; mat = 1)
        @test 0.0003440 * 50.0^2.980 ≈ w_f_imm
        @test 0.0004500 * 50.0^2.920 ≈ w_f_mat

        w_m_imm = calculate_mass(95.0, male; mat = 0)
        w_m_mat = calculate_mass(95.0, male; mat = 1)
        @test 0.0002665 * 95.0^3.098 ≈ w_m_imm
        @test 0.0001830 * 95.0^3.160 ≈ w_m_mat
        @test w_m_mat > 300.0  # ~335 g for 95mm mature male

        # 2. Degree-day accumulation and molt conditions
        p = IBMParameters()
        # Stage and both thresholds are derived, not written down. An `Organism`
        # exists only from `juvenile_max_stage`, so a literal like 6 became a
        # cohort stage after the renumbering and the organism was invalid without
        # anything complaining.
        stage_u = p.organism.juvenile_max_stage
        beh = default_species_behavior()
        req_dd_u = required_degree_days(beh, stage_u)
        min_days_u = min_intermolt_days(beh, stage_u)
        org_juv_dd = Organism(
            1, (5, 5), stage_u, male, beh, 400.0, 28.0,
            calculate_mass(28.0, male; mat = 0),
            min_days_u, 0.0, 0.5 * req_dd_u, false, 0, 0.0, 0, false
        )
        # Degree days = half the requirement -> cannot molt yet
        @test check_molt_conditions(org_juv_dd, 4.0, 50.0; params = p) == false

        # Past the requirement, but not yet past the intermolt duration
        org_juv_dd.degree_days = 1.5 * req_dd_u
        org_juv_dd.last_growth_event_time = 0.0
        @test check_molt_conditions(org_juv_dd, 4.0, 0.5 * min_days_u; params = p) == false

        # Both requirements met -> eligible to molt
        @test check_molt_conditions(org_juv_dd, 4.0, 1.5 * min_days_u; params = p) == true

        # Perform molt and verify degree-day reset and mass recalculation
        init_wt = org_juv_dd.mass
        perform_molt!(org_juv_dd, 1.5 * min_days_u; rng = MersenneTwister(42))
        @test org_juv_dd.degree_days == 0.0
        @test org_juv_dd.mass > init_wt
        @test org_juv_dd.stage == stage_u + 1

        # 3. Molt seasonality, for stages at or above `molt_season_min_stage`
        # Both quantities derived, so the block still works at whatever stage the
        # individual happens to be at.
        org_juv_dd.degree_days = 10 * req_dd_u
        org_juv_dd.energy_reserve = 100.0
        org_juv_dd.last_growth_event_time = -5 * min_days_u
        @test org_juv_dd.stage >= p.phenology.molt_season_min_stage
        inside = p.phenology.molt_season_start + 1
        outside = p.phenology.molt_season_end + 20
        @test check_molt_conditions(org_juv_dd, 4.0, inside; params = p) == true

        # Outside the window -> blocked by seasonality
        @test check_molt_conditions(org_juv_dd, 4.0, outside; params = p) == false

        # 4. Stock Biomass calculation
        model = initialize_model(n_agents = 50, grid = (20, 20), seed = 999)
        summary = summarize_population(model)
        @test summary.total_biomass_kg > 0.0
        @test summary.mean_mass > 0.0
        @test summary.female_ssb_kg >= 0.0
        @test summary.male_mature_biomass_kg >= 0.0
        @test summary.legal_biomass_kg >= 0.0
        @test summary.total_biomass_kg >= summary.total_ssb_kg
        @test summary.total_ssb_kg == (
            summary.female_ssb_kg + summary.male_mature_biomass_kg
        )
    end

    @testset "Stock-Recruitment Relationship Analysis & Visualization" begin
        # 1. Extraction from model
        model = initialize_model(n_agents = 20, grid = (10, 10), seed = 111)
        sr_empty = extract_stock_recruitment(model)
        @test sr_empty isa DataFrame
        @test nrow(sr_empty) == 0

        # Simulate clutch record
        push!(abmproperties(model)[:clutch_recruitment_records], (
            hatch_time = 365.0,
            mother_size = 68.0,
            mother_mass = 120.0,
            egg_count = 50000,
            recruits_settled = 18,
            density_feedback = 1.2
        ))
        sr_pop = extract_stock_recruitment(model)
        @test nrow(sr_pop) == 1
        @test sr_pop.spawners[1] == 50.0
        @test sr_pop.recruits[1] == 18
        @test sr_pop.recruits_per_spawner[1] == 18 / 50.0

        # 2. Fitting parametric models
        # Synthetic Beverton-Holt test data: R = 2.0 * S / (1 + 0.02 * S)
        s_vals = [10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 150.0]
        r_vals = [(2.0 * s) / (1.0 + 0.02 * s) for s in s_vals]
        fits = fit_stock_recruitment(s_vals, r_vals)
        @test fits.is_fitted == true
        @test fits.beverton_holt.R2 > 0.99
        @test fits.beverton_holt.alpha ≈ 2.0 atol = 0.05
        @test fits.beverton_holt.beta ≈ 0.02 atol = 0.005
        @test fits.beverton_holt.R_max ≈ 100.0 atol = 5.0
        @test fits.beverton_holt.S_50 ≈ 50.0 atol = 5.0

        # Synthetic Ricker test data: R = 2.0 * S * exp(-0.015 * S)
        r_ricker = [2.0 * s * exp(-0.015 * s) for s in s_vals]
        fits_rk = fit_stock_recruitment(s_vals, r_ricker)
        @test fits_rk.is_fitted == true
        @test fits_rk.ricker.R2 > 0.99
        @test fits_rk.ricker.alpha ≈ 2.0 atol = 0.05
        @test fits_rk.ricker.beta ≈ 0.015 atol = 0.005

        # 3. Textual summary
        summary_str = summarize_stock_recruitment(sr_pop; fits = fits)
        @test occursin("STOCK-RECRUITMENT ANALYSIS REPORT", summary_str)
        @test occursin("Beverton-Holt Model", summary_str)

        # 4. Multi-panel visualization
        sr_df = DataFrame(
            spawners = s_vals,
            recruits = round.(Int, r_vals),
            recruits_per_spawner = r_vals ./ s_vals,
            sample_index = 1:length(s_vals)
        )
        p_sr = plot_stock_recruitment(sr_df; fits = fits)
        @test p_sr isa Figure
    end

    @testset "Probabilistic Rates & Macro-Ecological Population Dynamic Models" begin
        rng = Random.Xoshiro(42)

        # 1. Biological stochasticity in growth increment and maturity
        cw_log = log(50.0)
        incs = [growth_increment(cw_log; sex=0, mat=0, rng=rng) for _ in 1:50]
        @test var(incs) > 0.0005
        @test length(unique(incs)) == 50

        # Maturity probability
        probs = [maturity(0, log(91.8); rng=rng) for _ in 1:30]
        @test all(0.3 .< probs .< 0.7)
        @test var(probs) > 0.0

        # 2. Molting probability and stochastic timing
        test_params = IBMParameters(
            molt_season_start = 15.0,
            molt_season_end = 140.0,
            dt = 1.0
        )
        # Stage 6 is benthic instar 1, which now needs 182.5 days and 912.5 degree
        # days. The timing and degree-day figures below clear both, so the test
        # exercises the probability rather than the ineligibility gate — there is a
        # separate test for that.
        org_juv = Organism(
            1, (1, 1), juvenile, male, 6, 200.0, 30.0, 10.0, 60.0,
            0.0, 1500.0, false, 0, 0.0, 0
        )
        p_m = molt_probability(org_juv, 4.0, 400.0; params = test_params)
        @test p_m > 0.0
        @test p_m <= 1.0

        # Deterministic check vs probabilistic trial
        @test check_molt_conditions(org_juv, 4.0, 400.0; params = test_params) == true
        molt_trials = [
            check_molt_conditions(org_juv, 4.0, 400.0; params = test_params, rng = rng)
            for _ in 1:100
        ]
        @test any(molt_trials)
        @test !all(molt_trials)

        # Ineligible when intermolt duration is insufficient
        @test molt_probability(org_juv, 4.0, 50.0; params = test_params) == 0.0

        # 3. Predicates for legal fishable individuals
        org_legal = Organism(
            2, (2, 2), adult, male, 12, 1200.0, 102.0, 650.0, 30.0,
            0.0, 0.0, true, 0, 0.0, 0
        )
        org_sublegal = Organism(
            3, (3, 3), adult, male, 10, 800.0, 88.0, 320.0, 25.0,
            0.0, 0.0, true, 0, 0.0, 0
        )
        org_female = Organism(
            4, (4, 4), adult, female, 10, 800.0, 96.0, 280.0, 20.0,
            0.0, 0.0, true, 0, 0.0, 0
        )
        @test is_legal_male(org_legal, 95.0, 68.0) == true
        @test is_legal(org_legal, 95.0, 68.0) == true
        @test is_legal_male(org_sublegal, 95.0, 68.0) == false
        @test is_legal_male(org_female, 95.0, 68.0) == false
        # The thresholds are mandatory arguments, so a configured legal size is
        # honoured rather than a hardcoded default silently applied.
        @test is_legal_male(org_sublegal, 80.0, 68.0) == true
        # A large individual that never matured is still legal by size. The
        # `is_mature` flag must be passed explicitly: the LifeStage forwarding
        # constructor defaults it to true for the adult stage.
        large_immature = Organism(
            7, (7, 7), adult, male, 11, 600.0, 110.0, 700.0, 40.0,
            0.0, 0.0, false, 0, 0.0, 0, false
        )
        @test is_legal_male(large_immature, 95.0, 68.0) == true
        @test is_mature_male(large_immature) == false
        @test is_adult(large_immature) == false  # large_immature has is_mature=false
        @test is_mature_female(large_immature) == false
        @test mass_if_mature_male(large_immature) == 0.0
        @test mass_if_legal_male(large_immature, 95.0, 68.0) == 700.0

        # 4. Schaefer logistic model fitting on synthetic data
        t_years = 1:10
        n_synth = zeros(Float64, 10)
        c_synth = zeros(Float64, 10)
        n_synth[1] = 120.0
        for t in 1:9
            c_synth[t] = (t > 4) ? 15.0 : 0.0
            prod_val = 0.45 * n_synth[t] * (1.0 - n_synth[t] / 400.0)
            n_synth[t + 1] = max(10.0, n_synth[t] + prod_val - c_synth[t])
        end
        sch_fit = fit_schaefer(t_years, n_synth, c_synth; label = "Synthetic Test")
        @test sch_fit.is_fitted == true
        @test sch_fit.R2 > 0.90
        @test sch_fit.r ≈ 0.45 atol = 0.15
        @test sch_fit.K ≈ 400.0 atol = 80.0
        @test sch_fit.MSY > 0.0

        # 5. Delay-Difference model fitting on synthetic data
        n_dd = zeros(Float64, 10)
        n_dd[1] = 150.0
        n_dd[2] = 160.0
        for t in 2:9
            rec = (0.25 * n_dd[t]) / (1.0 + 0.001 * n_dd[t])
            n_dd[t + 1] = 0.82 * (n_dd[t] - c_synth[t]) + rec
        end
        dd_fit = fit_delay_difference(
            t_years, n_dd, c_synth;
            lag = 1, label = "Synthetic DD"
        )
        @test dd_fit.is_fitted == true
        @test dd_fit.R2 > 0.90
        @test dd_fit.s ≈ 0.82 atol = 0.10

        # 6. File persistence and loading (.csv)
        tmp_csv = joinpath(tempdir(), "test_ann_prod.csv")
        ann_df = DataFrame(
            sample_index = collect(1:10),
            time = Float64.(collect(1:10) .* 365.0),
            total_numbers = n_synth,
            legal_numbers = n_synth .* 0.35,
            catch_total = c_synth,
            catch_legal = c_synth .* 0.8,
            surplus_production_total = zeros(10),
            surplus_production_legal = zeros(10)
        )
        write_dataframe_csv(tmp_csv, ann_df)
        @test isfile(tmp_csv)
        loaded_csv = read_dataframe_csv(tmp_csv)
        @test nrow(loaded_csv) == 10
        @test loaded_csv.total_numbers[1] ≈ 120.0

        # Full fit_population_dynamics on loaded data
        macro_fits = fit_population_dynamics(loaded_csv; lag = 1)
        @test macro_fits.whole_population.schaefer.is_fitted == true
        @test macro_fits.fishable_stock.schaefer.is_fitted == true

        # Textual summary and 4-panel plotting
        rep = summarize_population_dynamics(macro_fits)
        @test occursin("MACRO-ECOLOGICAL DYNAMIC MODEL ESTIMATION REPORT", rep)
        @test occursin("WHOLE POPULATION", rep)
        @test occursin("FISHABLE STOCK", rep)

        p_macro = plot_population_dynamics_fits(macro_fits)
        @test p_macro isa Figure

        # 7. Retrospective pattern analysis and Mohn's rho
        peeled_vals = [105.0, 115.0, 120.0]
        base_vals = [100.0, 100.0, 100.0]
        # (105-100)/100 = 0.05, (115-100)/100 = 0.15, (120-100)/100 = 0.20 -> mean = 0.13333...
        @test mohn_rho(peeled_vals, base_vals) ≈ (0.05 + 0.15 + 0.20) / 3.0 atol = 1e-6
        @test mohn_rho(Float64[], Float64[]) == 0.0
        @test_throws ErrorException mohn_rho([1.0], [1.0, 2.0])

        retro_res = retrospective_analysis(loaded_csv; n_peels = 4, model_type = :schaefer)
        @test retro_res.n_peels == 4
        @test length(retro_res.peel_indices) == 4
        @test retro_res.whole_population.mohn_rho isa Float64
        @test retro_res.fishable_stock.mohn_rho isa Float64
        @test length(retro_res.whole_population.rel_errors) == 4

        # Retrospective textual summary
        retro_rep = summarize_retrospective(retro_res)
        @test occursin("MOHN'S RHO DIAGNOSTIC REPORT", retro_rep)
        @test occursin("WHOLE POPULATION", retro_rep)
        @test occursin("FISHABLE STOCK", retro_rep)
        @test occursin("Mohn's Rho", retro_rep)

        # Retrospective 4-panel diagnostic plot
        p_retro = plot_retrospective_analysis(retro_res)
        @test p_retro isa Figure

        rm(tmp_csv, force = true)
    end

    @testset "Generalized Leslie Matrix & Macro Model Comparison" begin
        rng = MersenneTwister(999)

        # 1. Empirical Stages definition. Names carry the stage number the model now
        # uses, so they are derived from the stage tables rather than written
        # out — these were `Stage 4/5/6` and are `Stage 5/6/7` after the
        # renumbering, which is exactly the kind of drift that produced the
        # failure.
        stages = build_stages()
        ps = IBMParameters().species
        pooled = ps.stage_size_range_stage1
        male_first = ps.stage_size_range_stage_male[1]
        @test length(stages) == length(pooled) +
              length(ps.stage_size_range_stage_male) + length(ps.stage_size_range_stage_female)
        @test stages[1].name == "Stage $(Int(pooled[1])) (Recruit)"
        @test stages[1].sex === undifferentiated
        @test stages[2].name == "Stage $(Int(pooled[2])) (Juvenile)"
        @test stages[2].sex === undifferentiated
        @test stages[3].name == "Stage $(Int(male_first)) Male"
        @test stages[3].sex === male
fem4 = length(ps.stage_size_range_stage1) +
                       length(ps.stage_size_range_stage_male) + 4
        @test stages[fem4].name ==
              "Stage $(Int(ps.stage_size_range_stage_female[4])) Female"
        @test stages[fem4].sex === female

        # Legal males: size >= `leslie_legal_size_male`, *or* stage at or beyond the
        # `leslie_first_legal_stage_male` backstop. Both thresholds come from the
        # configuration; asserting either alone misses the other arm.
        legal_at = ps.leslie_legal_size_male
        legal_stage_at = ps.leslie_first_legal_stage_male
        for st in stages
            if st.sex === male
                @test st.is_legal_male ==
                      (st.size_mean >= legal_at || st.stage >= legal_stage_at)
            else
                @test st.is_legal_male == false
            end
        end

        # Reproductive maturity flags, from the configured first reproductive
        # female stage.
        repro_at = ps.leslie_first_repro_stage_female
        for st in stages
            if st.sex === female
                @test st.is_reproductive == (st.stage >= repro_at)
            else
                @test st.is_reproductive == false
            end
        end

        # 2. Gaussian Kernel Density Estimation (KDE)
        sample_cw = randn(rng, 500) .* 5.0 .+ 60.0
        grid, pdf = gaussian_kde(sample_cw)
        @test length(grid) == 200
        @test length(pdf) == 200
        @test all(pdf .>= 0.0)
        # Verify integral ≈ 1.0 using trapezoidal rule
        dx = grid[2] - grid[1]
        @test sum(pdf) * dx ≈ 1.0 atol = 0.05
        # Peak density should be near sample mean (~60.0)
        mode_cw = grid[argmax(pdf)]
        @test isapprox(mode_cw, 60.0, atol = 3.0)

        # 3. Stage proportion derivation from KDE
        cw_vec = vcat(fill(12.0, 50), fill(45.0, 50), fill(100.0, 50))
        sex_vec = vcat(fill(:unknown, 50), fill(:male, 50), fill(:male, 50))
        counts, props = derive_stage_proportions_kde(cw_vec, sex_vec; stages = stages)
        @test length(counts) == 15
        @test length(props) == 15
        @test sum(props) ≈ 1.0 atol = 1e-4
        @test all(props .>= 0.0)
        # Stage 4 should dominate for 12 mm
        @test props[1] > 0.20

        # 4. Annual Stage Matrix Extraction
        synth_ann_df = DataFrame(
            sample_index = collect(1:8),
            total_numbers = [100.0, 110.0, 105.0, 115.0, 120.0, 118.0, 122.0, 125.0],
            legal_numbers = [30.0, 32.0, 31.0, 35.0, 38.0, 37.0, 39.0, 40.0],
            catch_total = fill(5.0, 8),
            catch_legal = fill(4.0, 8)
        )
        stage_mat, catch_mat, yrs = extract_annual_stage_matrix(synth_ann_df; stages = stages)
        @test size(stage_mat) == (15, 8)
        @test size(catch_mat) == (15, 8)
        @test length(yrs) == 8
        @test all(stage_mat .>= 0.0)

        # 5. Generalized Leslie Model Fitting and Eigenvalue Decomposition
        leslie_fit = fit_generalized_leslie_model(stage_mat, catch_mat; stages = stages, years = yrs)
        @test leslie_fit isa LeslieMatrixFit
        @test size(leslie_fit.A) == (15, 15)
        @test leslie_fit.dominant_eigenvalue > 0.0
        @test leslie_fit.intrinsic_growth_rate ≈ log(leslie_fit.dominant_eigenvalue)
        @test leslie_fit.damping_ratio >= 1.0
        @test length(leslie_fit.stable_stage_distribution) == 15
        @test sum(leslie_fit.stable_stage_distribution) ≈ 1.0 atol = 1e-4
        @test length(leslie_fit.reproductive_values) == 15
        # Mature females must have positive reproductive values
        @test leslie_fit.reproductive_values[15] > 0.0
        # Elasticity matrix properties
        @test size(leslie_fit.elasticity_matrix) == (15, 15)
        @test all(leslie_fit.elasticity_matrix .>= -1e-6)
        @test sum(leslie_fit.elasticity_matrix) ≈ 1.0 atol = 0.05

        # 6. Textual Summary and 4-Panel Visualization
        leslie_summary = summarize_leslie_model(leslie_fit)
        @test occursin("GENERALIZED LESLIE", leslie_summary)
        @test occursin("Dominant Eigenvalue", leslie_summary)
        @test occursin("Damping Ratio", leslie_summary)
        @test occursin("SSD (w₁)", leslie_summary)
        @test occursin("Repro Val (v₁)", leslie_summary)

        p_leslie = plot_stage_specific_dynamics(leslie_fit)
        @test p_leslie isa Figure

        # 7. Macro Model Comparison (Schaefer vs Delay-Diff vs Leslie)
        macro_comp = compare_macro_models(synth_ann_df, leslie_fit; lag = 1)
        @test macro_comp isa MacroModelComparison
        @test length(macro_comp.years) == 8
        @test length(macro_comp.biomass_obs_kg) == 8
        @test length(macro_comp.biomass_schaefer_kg) == 8
        @test length(macro_comp.biomass_delay_diff_kg) == 8
        @test length(macro_comp.biomass_leslie_kg) == 8
        @test macro_comp.cv_obs >= 0.0
        @test macro_comp.cv_leslie >= 0.0
        @test macro_comp.preferred_model in [:schaefer, :delay_difference, :generalized_leslie]

        comp_summary = summarize_macro_comparison(macro_comp)
        @test occursin("THREE-WAY MACRO-ECOLOGICAL MODEL COMPARISON", comp_summary)
        @test occursin("Mean Biomass (kg)", comp_summary)
        @test occursin("Coeff of Variation (CV)", comp_summary)
        @test occursin("Goodness of Fit R²", comp_summary)

        p_comp = plot_macro_model_comparison(macro_comp)
        @test p_comp isa Figure
    end

    @testset "Configuration File Loading and CLI Overrides" begin
        # 1. Default configuration dictionary
        cfg_def = default_ibm_configuration()
        @test cfg_def isa Dict{Symbol, Any}
        @test cfg_def[:days] == 730
        @test sum(cfg_def[:initial_female_cohort]) +
            sum(cfg_def[:initial_male_cohort]) == 0   # eggs only by default
        @test cfg_def[:grid] == (100, 100)
        @test cfg_def[:temp_base] == 5.0
        @test cfg_def[:temp_amplitude] == 4.0
        @test cfg_def[:legal_size_male] == 95.0
        @test cfg_def[:benthic_carrying_capacity] == 12000.0

        # 2. Section routing is derived from the parameter structs
        section_map = config_section_map()
        @test section_map[:use_energetics] == "toggles"
        @test section_map[:days] == "simulation"
        @test section_map[:temp_base] == "environment"
        @test section_map[:max_size_male] == "species"
        @test section_map[:maturity_l50_male] == "species"
        @test section_map[:intermolt_days_schedule] == "species"
        @test section_map[:molt_season_start] == "phenology"
        @test section_map[:soft_shell_duration] == "mortality"
        @test section_map[:legal_size_male] == "fishery"
        @test section_map[:snapshot_day] == "output"
        @test section_map[:stage_size_range_lower_male] == "species"
        @test section_map[:survival_stage1] == "leslie"
        @test issubset(Set(values(section_map)), Set(ibm_configuration_sections()))

        # 3. Loading the generic baseline config, then the species overlay
        default_path = projectdir("configs", "default.toml")
        species_path = projectdir("configs", "snowcrab.toml")
        @test isfile(default_path)
        @test isfile(species_path)

        loaded_cfg = load_ibm_configuration(default_path)
        @test loaded_cfg[:days] == 730
        @test sum(loaded_cfg[:initial_female_cohort]) +
            sum(loaded_cfg[:initial_male_cohort]) == 0   # eggs only by default
        @test loaded_cfg[:grid] == (100, 100)
        @test loaded_cfg[:temp_base] == 5.0
        @test loaded_cfg[:temp_amplitude] == 4.0
        @test loaded_cfg[:mating_search_radius] == 5
        @test loaded_cfg[:mating_energy_cost] == 8.0
        @test loaded_cfg[:aggregate_baseline_hazards] ==
              [0.002, 0.020, 0.020, 0.015, 0.0015]

        # Ecdysis schedules and maturity gates are config-driven
        @test loaded_cfg[:intermolt_days_schedule] isa Vector{Float64}
        @test loaded_cfg[:degree_days_schedule] isa Vector{Float64}
        @test loaded_cfg[:maturity_size_gate] == 35.0

        # Species overlay replaces the corresponding [species] values
        species_cfg = load_ibm_configuration(species_path; base_config = loaded_cfg)
        @test species_cfg[:maturity_l50_male] == 91.8
        @test species_cfg[:maturity_l50_female] == 51.6
        @test species_cfg[:selectivity_s50] == 95.0
        @test species_cfg[:legal_size_male] == 95.0
        @test species_cfg[:molt_season_start] == 15.0

        # It also sets the operating scale, which deliberately diverges from the
        # baseline: a 1 km x 1 km grid at one cell per metre, and a carrying
        # capacity of about 10000 organisms per square kilometre.
        @test species_cfg[:grid] == (1000, 1000)
        @test sum(species_cfg[:initial_female_cohort]) +
            sum(species_cfg[:initial_male_cohort]) == 0   # eggs only by default
        @test species_cfg[:benthic_carrying_capacity] == 10000.0
        # The seeded stock is the three initial-abundance vectors, and it must stay
        # below K or the density feedback is not a correction to the mortality
        # schedule but a replacement for it.
        sp_seed = IBMParameters(species_cfg).simulation
        seeded = sum(sp_seed.initial_aggregate_cohort) +
                 sum(sp_seed.initial_female_cohort) + sum(sp_seed.initial_male_cohort)
        @test seeded < species_cfg[:benthic_carrying_capacity]

        # 4. Constructing IBMParameters from the merged configuration
        params = IBMParameters(species_cfg)
        @test params isa IBMParameters
        @test params.temp_base == 5.0
        @test params.temp_amplitude == 4.0
        @test params.legal_size_male == 95.0
        # Carrying capacity comes from the overlay, not the struct default.
        @test params.benthic_carrying_capacity == 10000.0
        @test IBMParameters().benthic_carrying_capacity == 12000.0
        @test params.mating_search_radius == 5
        @test params.mating_energy_cost == 8.0
        @test params.aggregate_baseline_hazards ==
              [0.002, 0.020, 0.020, 0.015, 0.0015]
        @test params.maturity_l50_male == 91.8
        @test params.survival_stage1 == 0.74
@test params.intermolt_days_schedule ==
              [0.0, 0.0, 0.0, 0.0, 182.5, 182.5, 182.5, 182.5, 365.0, 365.0, 365.0, 365.0, 365.0, 365.0, 365.0, 365.0, 365.0, 365.0, 365.0, 365.0, 365.0, 365.0]
        @test params.maturity_size_gate == 35.0
        @test params.baseline_rate_by_stage[1] == 0.35
        @test params.days_per_year == 365.0

        # Configured schedules are what the ecdysis logic actually reads
        cfg_behavior = SpeciesBehaviorImpl(params.species)
        @test min_intermolt_days(cfg_behavior, 7) == params.intermolt_days_schedule[8]
        @test required_degree_days(cfg_behavior, 7) == params.degree_days_schedule[8]
        # Schedules are read positionally, so assert the tail against the vector
        # rather than a literal that a renumbering invalidates silently.
        @test min_intermolt_days(cfg_behavior, 9) == params.intermolt_days_schedule[10]

        # 5. Initialize model directly from configuration dictionary
        test_model = initialize_model(species_cfg; n_agents = 10, days = 5)
        @test nagents(test_model) == 10
        @test test_model.params.temp_base == 5.0
        @test test_model.params.maturity_l50_male == 91.8

        # 6. Serialization and round-trip verification
        tmp_cfg_file = joinpath(tempdir(), "test_ibm_config.toml")
        save_ibm_configuration(tmp_cfg_file, species_cfg)
        @test isfile(tmp_cfg_file)
        reloaded_cfg = load_ibm_configuration(tmp_cfg_file)
        @test reloaded_cfg[:days] == species_cfg[:days]
        @test reloaded_cfg[:grid] == species_cfg[:grid]
        @test reloaded_cfg[:temp_base] ≈ species_cfg[:temp_base]
        @test reloaded_cfg[:legal_size_male] ≈ species_cfg[:legal_size_male]
        @test reloaded_cfg[:maturity_l50_male] ≈ species_cfg[:maturity_l50_male]
        @test reloaded_cfg[:survival_stage1] ≈ species_cfg[:survival_stage1]
        @test reloaded_cfg[:intermolt_days_schedule] == species_cfg[:intermolt_days_schedule]
        rm(tmp_cfg_file, force = true)

        # 7. Grid dimensions stay in sync whichever exact spelling is supplied
        grid_nx_first = Dict{Symbol, Any}(:grid_nx => 50, :grid_ny => 60)
        sync_grid_dimensions!(grid_nx_first)
        @test grid_nx_first[:grid] == (50, 60)

        grid_pair_first = Dict{Symbol, Any}(:grid => (30, 40))
        sync_grid_dimensions!(grid_pair_first)
        @test grid_pair_first[:grid_nx] == 30
        @test grid_pair_first[:grid_ny] == 40

        # An inherited `grid` pair must not override what the current layer set
        layer = Dict{Symbol, Any}(:grid => (100, 100), :grid_nx => 50, :grid_ny => 60)
        sync_grid_dimensions!(layer; explicit = [:grid_nx, :grid_ny])
        @test layer[:grid] == (50, 60)

        # Parameter names are resolved from the structs, not a hand-kept list
        @test config_field_type(:maturity_l50_male) === Float64
        @test config_field_type(:max_stage_male) === Int
        @test config_field_type(:use_energetics) === Bool
        @test config_field_type(:fecundity_parameterization) === String
        @test config_field_type(:intermolt_days_schedule) <: AbstractVector
        @test config_field_type(:definitely_not_a_parameter) === nothing

        # String coercion to the declared type
        @test coerce_config_value(Float64, "2.5") == 2.5
        @test coerce_config_value(Int, "42") == 42
        @test coerce_config_value(Bool, "false") === false
        @test coerce_config_value(Bool, "true") === true
        @test coerce_config_value(String, " st.marie ") == "st.marie"
        @test coerce_config_value(Vector{Float64}, "10,20,30") == [10.0, 20.0, 30.0]
        @test coerce_config_value(Tuple{Int, Int}, "(7,9)") == (7, 9)

        # 8. CLI '=' argument syntax parsing
        include(projectdir("scripts", "organism_ibm.jl"))
        cli_parsed = parse_cli_args([
            "--config=$(default_path)",
            "--temp_base=3.5",
            "--days=100",
            "--grid=80,80",
            "--n_agents=50",
            "--fishing_probability_daily=0.03"
        ])
        @test cli_parsed[:temp_base] == 3.5
        @test cli_parsed[:days] == 100
        @test cli_parsed[:grid] == (80, 80)
        @test cli_parsed[:n_agents] == 50
        @test cli_parsed[:fishing_probability_daily] == 0.03

        # 9. Configuration layering: the explicit --config is the overlay.
        #    Passing a species file used to treat it as the baseline and load
        #    default.toml on top of it, so the baseline silently overwrote the
        #    file the user had asked for.
        species_path = projectdir("configs", "snowcrab.toml")
        b_species, o_species = resolve_configuration_files(explicit_path = species_path)
        @test basename(b_species) == "default.toml"
        @test basename(o_species) == "snowcrab.toml"

        b_base, o_base = resolve_configuration_files(explicit_path = default_path)
        @test basename(b_base) == "default.toml"
        @test basename(o_base) == "snowcrab.toml"

        b_auto, o_auto = resolve_configuration_files()
        @test basename(b_auto) == "default.toml"
        @test basename(o_auto) == "snowcrab.toml"

        # A value in the overlay must survive the baseline load and reach the
        # model, whichever file the user names.
        layered = load_ibm_configuration(b_species; base_config = default_ibm_configuration())
        for path in (b_species, o_species)
            isfile(path) || continue
            for (k, v) in load_ibm_configuration(path)
                layered[k] = v
            end
        end
        overlay_only = load_ibm_configuration(
            species_path; base_config = default_ibm_configuration()
        )
        # n_agents in the overlay must not be replaced by the baseline value
        @test layered[:n_agents] == overlay_only[:n_agents]
        @test layered[:maturity_l50_male] == 91.8

        cli_species_cfg = parse_cli_args(["--config=$(species_path)"])
        @test cli_species_cfg[:initial_male_cohort] == overlay_only[:initial_male_cohort]
        @test cli_species_cfg[:maturity_l50_male] == 91.8
        @test cli_species_cfg[:molt_season_min_stage] ==
              IBMParameters().phenology.molt_season_min_stage

        # A command-line argument still outranks both files
        cli_override = parse_cli_args(["--config=$(species_path)", "--n_agents=50"])
        @test cli_override[:n_agents] == 50

        # `--config` is consumed by the pre-scan and must not warn as unknown
        @test parse_cli_args(String[]) isa Dict{Symbol, Any}
    end

    @testset "Configuration Surface Integrity" begin
        using TOML
        types = config_field_types()
        default_raw = TOML.parsefile(projectdir("configs", "default.toml"))
        species_raw = TOML.parsefile(projectdir("configs", "snowcrab.toml"))

        # 1. The baseline must document every declared parameter. This is the
        #    check that makes an undocumented parameter impossible to add.
        documented = Set{Symbol}()
        for (_, values) in default_raw
            isa(values, AbstractDict) || continue
            for key in keys(values)
                push!(documented, Symbol(key))
            end
        end
        missing_docs = sort(collect(setdiff(Set(keys(types)), documented)))
        @test isempty(missing_docs) ||
              error("parameters declared but absent from configs/default.toml: $(missing_docs)")

        # 2. Every TOML key must name a declared parameter, in both files.
        for (label, raw) in (("default.toml", default_raw), ("snowcrab.toml", species_raw))
            unknown = Symbol[]
            for (_, values) in raw
                isa(values, AbstractDict) || continue
                for key in keys(values)
                    sym = Symbol(key)
                    haskey(types, sym) || push!(unknown, sym)
                end
            end
            @test isempty(unknown) ||
                  error("$label declares keys that are not parameters: $(unknown)")
        end

        # 3. No duplicate section across the two files (TOML forbids repeated
        #    tables, and a repeated key silently shadows).
        default_sections = Set(Symbol.(collect(keys(default_raw))))
        species_sections = Set(Symbol.(collect(keys(species_raw))))
        @test issubset(species_sections, default_sections)

        # 4. Values must match the declared types, so a typo cannot turn a
        #    numeric field into text.
        for (label, raw) in (("default.toml", default_raw), ("snowcrab.toml", species_raw))
            for (section, values) in raw
                isa(values, AbstractDict) || continue
                for (key, value) in values
                    sym = Symbol(key)
                    haskey(types, sym) || continue
                    declared = types[sym]
                    ok = if declared <: AbstractVector
                        value isa AbstractVector
                    elseif declared === Bool
                        value isa Bool
                    elseif declared <: Integer
                        value isa Integer
                    elseif declared <: AbstractFloat
                        value isa Real
                    else
                        value isa AbstractString
                    end
                    @test ok || error("$(label) [$(section)] $(key): expected $(declared), got $(typeof(value))")
                end
            end
        end

        # 5. The baseline must agree with the declared defaults field for field,
        #    proving no field is silently dropped or mistyped. The species overlay
        #    is exempt: its whole purpose is to differ. Fields are compared
        #    individually because the default `==` on a struct holding a vector
        #    compares that vector by identity, which is never true here.
        baseline_only = load_ibm_configuration(projectdir("configs", "default.toml"))
        for T in (OrganismParams, SpeciesParams, MortalityParams, PhenologyParams,
                  EnvironmentParams, FisheryParams, LeslieParams, SimulationParams,
                  OutputParams, ProcessToggles)
            rebuilt = struct_from_config(T, baseline_only)
            reference = T()
            mismatched = Symbol[]
            for field in fieldnames(T)
                a = getfield(rebuilt, field)
                b = getfield(reference, field)
                same = if a isa AbstractVector || b isa AbstractVector
                    a == b
                elseif a isa Real && b isa Real
                    isapprox(Float64(a), Float64(b); rtol = 1e-12)
                else
                    a == b
                end
                same || push!(mismatched, field)
            end
            @test isempty(mismatched) ||
                  error("$(T): default.toml disagrees with the declared default for $(mismatched)")
        end

        # 6. The overlay must still produce a usable parameter set, and the
        #    species values must win over the baseline.
        merged = default_ibm_configuration()
        baseline, overlay = resolve_configuration_files(
            explicit_path = projectdir("configs", "snowcrab.toml")
        )
        for path in (baseline, overlay)
            isfile(path) || continue
            for (k, v) in load_ibm_configuration(path)
                merged[k] = v
            end
        end
        rebuilt_species = struct_from_config(SpeciesParams, merged)
        @test rebuilt_species.maturity_l50_male == 91.8
        @test struct_from_config(SimulationParams, merged).initial_male_cohort ==
              merged[:initial_male_cohort]
    end

    @testset "The fishery has one switch" begin
        # `fishing_probability_daily` is the switch, and there is no toggle beside
        # it. The two used to express one thing: `toggles.use_fishery` was true while
        # the probability was zero, so the model reported the fishery as enabled and
        # caught nothing.
        @test !any(==(:use_fishery), fieldnames(ProcessToggles))
        @test !any(startswith(String(f), "use_fishery")
        for f in keys(config_field_types()))

        @test fishery_active(IBMParameters()) == false
        @test fishery_active(FisheryParams()) == false
        @test fishery_active(FisheryParams(fishing_probability_daily = 0.005)) == true
        @test fishery_active(IBMParameters(fishery = FisheryParams(
            fishing_probability_daily = 0.005))) == true

        # Zero really does mean nothing is harvested, and a positive probability
        # really does harvest: otherwise the switch is decorative.
        off = IBMParameters()
        on = IBMParameters(fishery = FisheryParams(fishing_probability_daily = 0.4))
        caught(p) = begin
            m = initialize_model(; n_agents = 300, grid = (40, 40), seed = 21,
                params = p)
            run_simulation(m, 200; when = 200)
            m.total_catch_numbers
        end
        @test caught(off) == 0
        @test caught(on) > 0
    end

    @testset "Species behaviour is dispatched, never defaulted" begin
        # The generic layer used to hold fallbacks that built a
        # `SpeciesBehaviorImpl()` and called it, discarding the behaviour they were
        # handed. With one species that is invisible, because every behaviour *is* a
        # `SpeciesBehaviorImpl` and dispatch already picks the right method. The cost
        # appears only when a second species is written: it would be measured against
        # snow crab coefficients and produce plausible, wrong animals. So a behaviour
        # that does not implement a species rule must now fail at the call site.
        struct DanglingBehavior <: SpeciesBehavior end

        # Each fallback takes a different arity, so call each the way production
        # code does rather than uniformly.
        @test_throws ErrorException stage_size_lower_bound(DanglingBehavior(), 6, :male)
        @test_throws ErrorException growth_increment(DanglingBehavior(), log(20.0))
        @test_throws ErrorException build_stages(DanglingBehavior())

        # And the concrete species still answers, with its own parameters, rather
        # than a fixed answer.
        b = SpeciesBehaviorImpl(SpeciesParams())
        @test stage_size_lower_bound(b, 8.0, :male) > 0.0
        altered = SpeciesBehaviorImpl(SpeciesParams(size_lb_male_intercept = -1.0))
        @test !isapprox(stage_size_lower_bound(b, 8.0, :male),
            stage_size_lower_bound(altered, 8.0, :male); rtol = 1e-6)

        # The pelagic life-stage overloads and the stage-indexed ones must agree;
        # both used to restate the egg/zoea/megalopa mapping independently.
        @test min_intermolt_days(b, egg) == min_intermolt_days(b, 0)
        @test min_intermolt_days(b, zoea) == min_intermolt_days(b, 1)
        @test min_intermolt_days(b, megalopa) == min_intermolt_days(b, 2)
        @test required_degree_days(b, egg) == required_degree_days(b, 0)
        @test required_degree_days(b, zoea) == required_degree_days(b, 1)
        @test required_degree_days(b, megalopa) == required_degree_days(b, 2)
        # `LifeStage` also names `juvenile` and `adult`, which have no fixed stage
        # number of their own: the caller's `stage` is the answer for those, and
        # collapsing them onto the megalopa index would answer a juvenile's question
        # with the megalopa's numbers.
        @test min_intermolt_days(b, juvenile, 4) == min_intermolt_days(b, 4)
        @test min_intermolt_days(b, adult, 9) == min_intermolt_days(b, 9)
        @test required_degree_days(b, juvenile, 4) == required_degree_days(b, 4)
        @test required_degree_days(b, adult, 9) == required_degree_days(b, 9)
        @test GeoIBM.life_stage_index(juvenile) === nothing
        @test GeoIBM.life_stage_index(egg) == 0
    end

    @testset "Every Declared Parameter Is Read" begin
        # A parameter that nothing reads is a silent no-op: it appears in the
        # TOML, invites tuning, and does nothing. This scans the sources for a
        # use of each declared field name.
        #
        # Two things this had to get right to work at all:
        #
        #  1. Each file is scanned separately. Concatenating them into one
        #     ~470 KB subject and running a fresh `Regex` per field over it gave
        #     order-dependent answers: `mating_search_radius` is read at
        #     `src/species.jl:737` but was reported unread whenever the scan
        #     reached it after a few dozen other patterns.
        #
        #  2. The declaration is recognised by pattern rather than by stripping
        #     the declared block. `join(readlines(f), "\n")` reconstructs an
        #     LF-separated string, which cannot match this checkout's CRLF file
        #     content, so `replace` removed nothing and every declaration satisfied
        #     its own check. That made the guard vacuous: parameters were declared,
        #     documented and unread while it passed.
        #
        # A line of the form `name::T` is the definition, not a use, so it is
        # skipped wherever it appears.
        source_lines = Pair{String, Vector{String}}[]
        for dir in ("src", "scripts")
            for file in sort(readdir(projectdir(dir); join = true))
                endswith(file, ".jl") || continue
                push!(source_lines, file => readlines(file))
            end
        end

        # The fields of `fields` that have no use site anywhere in the sources.
        function find_unread(fields)
            unread = Symbol[]
            for field in fields
                declaration = Regex(string("\\b", field, "\\s*::"))
                used = false
                for (_, lines) in source_lines
                    for line in lines
                        occursin(string(field), line) || continue
                        occursin(declaration, line) && continue
                        used = true
                        break
                    end
                    used && break
                end
                used || push!(unread, field)
            end
            return unread
        end

        # The scan must be able to fail, or passing it proves nothing.
        @test find_unread([:definitely_not_a_parameter]) == [:definitely_not_a_parameter]

        never_read = find_unread(sort!(collect(keys(config_field_types()))))
        @test isempty(never_read) ||
              error("parameters declared and exposed but never read: $(never_read)")
    end

    @testset "Carrying Capacity, Ecdysial Energy, and Hazard Bounds" begin
        # Carrying capacity must be evaluated against the *total* benthic
        # population, including the aggregate cohorts that hold the juveniles.
        # Counting only individuated agents understated density by an order of
        # magnitude and made K inert once the cohorts were exhausted.
        base_params = IBMParameters()
        @test base_params.environment.benthic_carrying_capacity_exponent == 2.0

        model = initialize_model(;
            n_agents = 40, grid = (12, 12), seed = 11,
            params = IBMParameters(benthic_carrying_capacity = 100.0)
        )
        total = total_benthic_abundance(model)
        @test total >= nagents(model)
        @test total == nagents(model) ||
              total > nagents(model)   # aggregate cohorts present at the start

        # Feedback is 1 + (N/K)^p: unity below K, 2 at K, and 5 at 2K for p = 2.
        # A linear feedback would stay at 3 at 2K and never bound the population.
        @test benthic_density_feedback(model) ≈ 1.0 + (total / 100.0)^2
        @test benthic_density_feedback(model) > 1.0

        # The ecdysial energy requirement must be reachable from the daily energy
        # surplus. At energy_molt_slope = 1.5 the requirement was ~1.5 per mm
        # (240 units at 160 mm) against a surplus of ~0.03-0.2 per day, so nothing
        # could ever moult and the population stalled at the first gated instar.
        slope = base_params.organism.energy_molt_slope
        for sz in (20.0, 60.0, 100.0, 160.0)
            required = max(2.0, slope * sz)
            dE, _ = calculate_energy_change(5.0, base_params.environment.food_max, sz;
                                            params = base_params)
            surplus = max(0.0, -dE)
            # Reachable within a plausible intermolt period, not a lifetime.
            if surplus > 0.0
                @test required / surplus < 3650.0
            end
        end
        @test max(2.0, slope * 160.0) < 10.0

        # A terminal-molt individual never moults again, so its time since ecdysis
        # grows without bound and the raw senescence hazard is unbounded. Hazards
        # must saturate instead of aborting the run with a probability above one.
        old_adult = Organism(
            9, (5, 5), adult, male, 12, 0.0, 150.0, 900.0, 100.0, 0.0, 0.0,
            false, 0, 0.0, 0, true
        )
        cap = base_params.mortality.hazard_cap
        for elapsed in (0.0, 3650.0, 36500.0, 365000.0)
            p = calculate_mortality_probability(old_adult, 9.0, 1.0, elapsed;
                                                params = base_params)
            @test p >= 0.0
            @test p <= cap
        end

        # Extremely warm water must also saturate rather than exceed one.
        hot = calculate_mortality_probability(old_adult, 60.0, 1.0, 36500.0;
                                             params = base_params)
        @test hot <= cap

        # The seasonal molt gate is configurable and defaults to a stage above
        # the individuating stage.
        @test base_params.phenology.molt_season_min_stage ==
              base_params.organism.juvenile_max_stage + 1
        @test base_params.phenology.molt_season_min_stage >
              base_params.organism.juvenile_max_stage
        all_stages = IBMParameters(
            phenology = PhenologyParams(molt_season_min_stage = 1)
        )
        @test all_stages.phenology.molt_season_min_stage == 1
    end

    @testset "Config-Driven Ecdysis Schedules and Maturity Gates" begin
        # Baseline schedule is monotone and stage-indexed.
        #
        # Stages are named by their position in the pelagic block, which is what
        # `intermolt_days_larval` is indexed by: `zoea` is stage 1 and `megalopa` is
        # the last pelagic stage. These calls took a `LifeStage` as the *behaviour*
        # argument and an integer as the stage, so they resolved against the wrong
        # schedule; they read the default one by accident rather than by design.
        b_sched = SpeciesBehaviorImpl()
        pel = larval_stage_count(b_sched)
        # The schedule is flat within each block — one year across the five pelagic
        # stages, half a year per instar to instar 4, a year per instar after — so
        # the test is the block structure, not a rising chain.
        @test min_intermolt_days(b_sched, 1) == min_intermolt_days(b_sched, pel - 1)
        @test min_intermolt_days(b_sched, pel - 1) < min_intermolt_days(b_sched, pel)
        @test min_intermolt_days(b_sched, pel) == min_intermolt_days(b_sched, pel + 3)
        @test min_intermolt_days(b_sched, pel + 3) < min_intermolt_days(b_sched, pel + 4)

        @test required_degree_days(b_sched, 1) == required_degree_days(b_sched, 2)
        @test required_degree_days(b_sched, 1) < required_degree_days(b_sched, pel - 1)
        @test required_degree_days(b_sched, pel - 1) < required_degree_days(b_sched, pel)
        @test required_degree_days(b_sched, pel) == required_degree_days(b_sched, pel + 3)
        @test required_degree_days(b_sched, pel + 3) < required_degree_days(b_sched, pel + 4)

        # Out-of-range stages clamp to the terminal schedule entry
@test min_intermolt_days(99) ==
              b_sched.params.intermolt_days_schedule[end]
        @test required_degree_days(99) ==
              b_sched.params.degree_days_schedule[end]

        # Overriding the schedule in SpeciesParams changes the reported values,
        # proving the thresholds are not hardcoded. Stages below the pelagic
        # boundary read the larval schedules and everything above reads the benthic
        # ones, so both are overridden and both are checked.
        beh = SpeciesBehaviorImpl(SpeciesParams(
            intermolt_days_schedule = [11.0, 22.0, 33.0],
            degree_days_schedule = [1.0, 2.0, 3.0],
            intermolt_days_larval = [101.0, 102.0, 103.0],
            degree_days_larval = [201.0, 202.0, 203.0],
        ))
        @test min_intermolt_days(beh, 0) == 101.0
        @test min_intermolt_days(beh, 2) == 103.0
        @test min_intermolt_days(beh, 3) == 33.0
        @test min_intermolt_days(beh, 50) == 33.0
        @test required_degree_days(beh, 0) == 201.0
        @test required_degree_days(beh, 3) == 3.0
        @test required_degree_days(beh, 50) == 3.0

        # The pelagic boundary is the length of the larval schedules, not a
        # literal: extending them moves the boundary with them.
        wider = SpeciesBehaviorImpl(SpeciesParams(
            intermolt_days_larval = [1.0, 2.0, 3.0, 4.0, 5.0],
            degree_days_larval = [1.0, 2.0, 3.0, 4.0, 5.0],
        ))
        @test min_intermolt_days(wider, 4) == 5.0
        @test min_intermolt_days(wider, 5) ==
              wider.params.intermolt_days_schedule[6]  # benthic schedule, stage 5
        @test required_degree_days(wider, 4) == 5.0
        @test required_degree_days(wider, 5) ==
              wider.params.degree_days_schedule[6]  # benthic schedule, stage 5

        # Maturity size gate is likewise configurable
        gated = SpeciesBehaviorImpl(SpeciesParams(maturity_size_gate = 60.0))
        @test maturity(gated, :male, log(30.0)) == 0.0
        @test maturity(gated, :male, log(50.0)) == 0.0
        @test maturity(gated, :male, log(70.0); deterministic = true) > 0.0

        # A non-positive gate disables the gate entirely
        ungated = SpeciesBehaviorImpl(SpeciesParams(maturity_size_gate = 0.0))
        @test maturity(ungated, :male, log(10.0); deterministic = true) < 0.01

        # Maturity temperature response is configurable
        cold = SpeciesBehaviorImpl(SpeciesParams(
            maturity_l50_temp_slope = -4.0, maturity_temp_reference = 4.0
        ))
        p_cold = maturity(cold, :male, log(92.0); temp = 1.0, deterministic = true)
        p_warm = maturity(cold, :male, log(92.0); temp = 7.0, deterministic = true)
        @test p_warm > p_cold
    end

    @testset "Enhanced Simulation Output Metrics" begin
        # 1. Weight filter functions
        # These must be genuinely mature: the old predicate treated any
        # stage-10 individual as an adult even with `is_mature == false`, which
        # is what let non-spawners into the spawning stock.
        org_f_ad = Organism(1, (1, 1), adult, female, 10, 500.0, 65.0, 150.0, 50.0, 0.0, 100.0, false, 0, 0.0, 0, true)
        org_m_ad = Organism(2, (1, 1), adult, male, 11, 600.0, 96.0, 450.0, 60.0, 0.0, 100.0, false, 0, 0.0, 0, true)
        org_m_juv = Organism(3, (1, 1), juvenile, male, 7, 200.0, 40.0, 30.0, 20.0, 0.0, 50.0, false, 0, 0.0, 0)
        # Same size as the mature male, but never matured: the `LifeStage`
        # constructor would default this to mature, so it is stated explicitly.
        # It must not appear in mature biomass while still being legal by size.
        org_m_big_immature = Organism(5, (1, 1), adult, male, 11, 600.0, 96.0, 450.0, 60.0, 0.0, 100.0, false, 0, 0.0, 0, false)

        # Spawning stock uses the *mature* classes, so a same-sized individual
        # that never matured contributes nothing to SSB.
        @test mass_if_mature_female(org_f_ad) == 150.0
        @test mass_if_mature_female(org_m_ad) == 0.0
        @test mass_if_mature_male(org_m_ad) == 450.0
        @test mass_if_mature_male(org_m_juv) == 0.0
        @test mass_if_mature_male(org_m_big_immature) == 0.0
        @test mass_if_legal_male(org_m_big_immature, 95.0, 68.0) == 450.0
        @test mass_if_legal_male(org_m_ad, 95.0, 68.0) == 450.0
        @test mass_if_legal_male(org_m_juv, 95.0, 68.0) == 0.0

        # 2. summarize_population indicators
        m = initialize_model(n_agents = 20, grid = (20, 20), seed = 42)
        summ = summarize_population(m)
        @test haskey(summ, :total_mature_females)
        @test haskey(summ, :total_mature_males)
        @test haskey(summ, :total_fishable_male_mass_kg)
        @test haskey(summ, :total_mature_male_mass_kg)
        @test haskey(summ, :total_mature_female_mass_kg)
        @test haskey(summ, :total_egg_production)
        @test summ.total_fishable_male_mass_kg >= 0.0
        @test summ.total_egg_production >= 0

        # 3. extract_annual_production column structure and snapshot_day
        agent_df, model_df = run_simulation(m, 366)
        ann = extract_annual_production(agent_df, model_df)
        @test :mature_females in propertynames(ann)
        @test :mature_males in propertynames(ann)
        @test :fishable_male_biomass_kg in propertynames(ann)
        @test :mature_male_biomass_kg in propertynames(ann)
        @test :mature_female_biomass_kg in propertynames(ann)
        @test :total_egg_production in propertynames(ann)
        @test nrow(ann) == 1
        @test ann.time[1] == 274.0

        # Custom snapshot_day
        ann_100 = extract_annual_production(agent_df, model_df; snapshot_day = 100.0)
        @test nrow(ann_100) == 1
        @test ann_100.time[1] == 100.0

        # `catch_legal` and `catch_total` are the same series, and that is a property
        # of the fishery gate rather than a shortcut: `agent_step!` removes an animal
        # only when `is_legal_male` passes, so no sublegal animal can ever be
        # harvested. Asserted so that a configuration which admits sublegal harvest
        # has to make the two columns separately.
        @test :catch_total in propertynames(ann)
        @test :catch_legal in propertynames(ann)
        @test ann.catch_total == ann.catch_legal
        @test all(iszero, ann.catch_total)   # fishing_probability_daily defaults to 0
        @test :surplus_production_legal in propertynames(ann)

        # Saving simulation results with custom snapshot_day
        tmp_csv = tempname() * ".csv"
        save_simulation_results(tmp_csv, agent_df, model_df; snapshot_day = 150.0)
        @test isfile(tmp_csv)
        loaded_df = load_simulation_results(tmp_csv)
        @test nrow(loaded_df) == 1
        @test loaded_df.time[1] == 150.0
        @test :mature_females in propertynames(loaded_df)
        @test :total_egg_production in propertynames(loaded_df)
        rm(tmp_csv, force=true)

        # CLI --snapshot_day parsing test
        cli_parsed2 = parse_cli_args(["--snapshot_day=260.0", "--snapshot_day=260.0"])
        @test cli_parsed2[:snapshot_day] == 260.0

        # Output frequency tests (frequency_days)
        ann_freq90 = extract_annual_production(agent_df, model_df; frequency_days = 90.0)
        @test nrow(ann_freq90) == 4
        @test ann_freq90.time == [90.0, 180.0, 270.0, 360.0]

        # Saving simulation results with custom frequency_days
        tmp_csv2 = tempname() * ".csv"
        save_simulation_results(tmp_csv2, agent_df, model_df; frequency_days = 120.0)
        @test isfile(tmp_csv2)
        loaded_freq_df = load_simulation_results(tmp_csv2)
        @test nrow(loaded_freq_df) == 3
        @test loaded_freq_df.time == [120.0, 240.0, 360.0]
        rm(tmp_csv2, force=true)

        # CLI sampling-interval and snapshot-day parsing, exact names only
        cli_parsed_freq = parse_cli_args(["--frequency_days=90.0"])
        @test cli_parsed_freq[:frequency_days] == 90.0
        @test parse_cli_args(["--frequency_days=45.0"])[:frequency_days] == 45.0
        @test parse_cli_args(["--snapshot_day=260.0"])[:snapshot_day] == 260.0

        # Every declared parameter is settable by its exact name, coerced to the
        # declared type, without the parser needing a per-key branch
        cli_species = parse_cli_args([
            "--maturity_l50_male=100.0",
            "--survival_stage1=0.5",
            "--legal_size_male=120.0",
            "--max_stage_male=18",
            "--use_energetics=false",
            "--intermolt_days_schedule=10,20,30",
        ])
        @test cli_species[:maturity_l50_male] === 100.0
        @test cli_species[:survival_stage1] === 0.5
        @test cli_species[:legal_size_male] === 120.0
        @test cli_species[:max_stage_male] === 18
        @test cli_species[:use_energetics] === false
        @test cli_species[:intermolt_days_schedule] == [10.0, 20.0, 30.0]

        # Those command-line values reach the parameter structs
        cli_params = IBMParameters(cli_species)
        @test cli_params.maturity_l50_male == 100.0
        @test cli_params.survival_stage1 == 0.5
        @test cli_params.legal_size_male == 120.0
        @test cli_params.max_stage_male == 18
        @test cli_params.use_energetics == false
        @test cli_params.intermolt_days_schedule == [10.0, 20.0, 30.0]

        # Abbreviated names are not accepted
        @test !haskey(parse_cli_args(["--freq=10.0"]), :freq)
        @test parse_cli_args(["--k_cap=250.0"])[:benthic_carrying_capacity] != 250.0
    end

    @testset "Multi-Attribute Criteria Evaluation & Orchestrator" begin
        # 1. Evaluate population criteria on a running model
        m_test = initialize_model(n_agents = 25, grid = (20, 20), seed = 99)
        agent_df, model_df = run_simulation(m_test, 60;
            adata = [(:stage, length), (:size, safe_mean)],
            mdata = [:total_catch_numbers]
        )
        crit = evaluate_population_criteria(m_test, agent_df; min_population = 5)
        @test crit.is_persistent == true
        @test crit.is_bounded == true
        @test crit.n_living > 0
        @test crit.maturity_rmse >= 0.0
        @test crit.growth_rmse >= 0.0
        @test crit.size_drift >= 0.0
        @test crit.composite_score >= 0.0

        crit_extinct = evaluate_population_criteria(m_test, agent_df; min_population = 1000)
        @test crit_extinct.is_persistent == false
        @test crit_extinct.passed_all == false

        # 2. Test orchestrator parsing helpers
        include(projectdir("scripts", "organism_orchestrator.jl"))
        vary_parsed = parse_vary_string("temp_base=3.5,5.0;food_max=8.0,12.0")
        @test haskey(vary_parsed, :temp_base)
        @test haskey(vary_parsed, :food_max)
        @test vary_parsed[:temp_base] == [3.5, 5.0]
        @test vary_parsed[:food_max] == [8.0, 12.0]

        range_parsed = parse_vary_string("temp_base=3.0:1.0:5.0")
        @test range_parsed[:temp_base] == [3.0, 4.0, 5.0]

        grid = build_parameter_grid(vary_parsed)
        @test length(grid) == 4

        cli_orch = parse_orchestrator_cli([
            "--vary=temp_base=4.0,5.0",
            "--days=100",
            "--max_trials=5",
            "--no-stop_on_first_pass"
        ])
        @test cli_orch[:days] == 100
        @test cli_orch[:max_trials] == 5
        @test cli_orch[:stop_on_first_pass] == false
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

    @testset "Empirical Biological Model & Trait Calibration" begin
        rng = MersenneTwister(42)

        # 1. Larval bounds and bounded carapace sampling
        @test llbm(1.0) == 0.95
        @test llbf(1.0) == 0.95
        @test llbm(2.0) == 1.45
        @test llbf(2.0) == 1.45
        @test llbm(3.0) == 2.50
        @test llbf(3.0) == 2.50

        cw_samp = get_stage_size(5, :male; type = :random, rng = rng)
        @test llbm(5) <= cw_samp <= llbm(6)

        # 2. Bivariate Hiatt growth increments and condition modulation
        cw_base = 50.0
        cw_starved = predict_postgrowth_size(
            cw_base;
            sex = :male,
            mat = 0,
            condition = 0.70,
            rng = MersenneTwister(1)
        )
        cw_fed = predict_postgrowth_size(
            cw_base;
            sex = :male,
            mat = 0,
            condition = 1.30,
            rng = MersenneTwister(1)
        )
        @test cw_fed > cw_starved
        @test cw_starved > cw_base

        # 3. Minimum size gate & temperature-size rule in maturity ogive
        @test maturity(:male, log(30.0); temp = 4.0, rng = rng) == 0.0
        @test maturity(:female, log(34.9); temp = 4.0, rng = rng) == 0.0

        p_cold = maturity(:male, log(92.0); temp = 1.0, rng = MersenneTwister(10))
        p_warm = maturity(:male, log(92.0); temp = 7.0, rng = MersenneTwister(10))
        @test p_warm > p_cold

        mock_fem = Organism(
            1, (5, 5), 10, female, default_species_behavior(), 800.0, 75.0, 120.0, 40.0, 0.0, 150.0,
            false, 50000, 365.0, 1, true
        )
        fec_agent = fecundity(mock_fem; rng = rng)
        @test fec_agent > 10000

        wt_somatic = calculate_mass(mock_fem; include_eggs = false)
        wt_total = calculate_mass(mock_fem; include_eggs = true)
        @test wt_total > wt_somatic
        @test (wt_total - wt_somatic) ≈ (50000 * 0.00007)

        # 5. Generate mock survey data
        survey_df = generate_mock_survey_data(n_samples = 300, rng = rng)
        @test isa(survey_df, DataFrame)
        @test nrow(survey_df) == 300
        @test :size in propertynames(survey_df)
        @test :mass in propertynames(survey_df)
        @test :maturity in propertynames(survey_df)
        @test :fecundity in propertynames(survey_df)

        # 6. Fit empirical biological model
        emp_model = fit_empirical_biology_model(survey_df)
        @test isa(emp_model, EmpiricalBiologicalModel)
        @test emp_model.sample_size == 300
        @test haskey(emp_model.mass_params, :male_imm)
        @test haskey(emp_model.mass_params, :male_mat)
        @test haskey(emp_model.mass_params, :female_imm)
        @test haskey(emp_model.mass_params, :female_mat)

        @test haskey(emp_model.maturity_params, :male)
        @test haskey(emp_model.maturity_params, :female)
        @test emp_model.maturity_params[:male][1] > 70.0
        @test emp_model.maturity_params[:female][1] > 40.0

        @test haskey(emp_model.fecundity_params, :primiparous)
        @test haskey(emp_model.fecundity_params, :multiparous)

        @test size(emp_model.trait_cov) == (4, 4)
        @test isposdef(emp_model.trait_cov)

        # 7. Sample individual traits from empirical model
        traits_m = sample_organism_traits(
            emp_model;
            size = 100.0,
            sex = :male,
            temp = 3.5,
            rng = rng
        )
        @test traits_m.mass > 300.0
        @test traits_m.fecundity == 0.0
        # A 100 mm male is well past individuation, so the stage inferred from
        # size must be at or above the boundary. Asserting an exact stage pair
        # here only re-encodes the stage table.
        @test traits_m.estimated_stage >=
              IBMParameters().organism.juvenile_max_stage
        
        traits_f = sample_organism_traits(
            emp_model;
            size = 70.0,
            sex = :female,
            temp = 3.5,
            shell_age = 1.0,
            rng = rng
        )
        @test traits_f.mass > 50.0
@test traits_f.estimated_stage >=
              IBMParameters().organism.juvenile_max_stage

        # 8b. Deterministic demographic prediction from the same fitted model
        demo_m = estimate_demographics(
            emp_model;
            size = 100.0,
            sex = :male,
            temp = 3.5,
            rng = rng
        )
        @test haskey(demo_m, :maturity_prob)
        @test haskey(demo_m, :mass)
        @test 0.0 <= demo_m.maturity_prob <= 1.0
        @test demo_m.mass > 0.0
        @test demo_m.mass > sample_organism_traits(
            emp_model;
            size = 60.0, sex = :male, temp = 3.5, rng = rng
        ).mass

        # A non-positive size is rejected rather than extrapolated.
        @test estimate_demographics(
            emp_model;
            size = 0.0, sex = :male, rng = rng
        ).maturity_prob == 0.0
        
        # 8. Model summary display
        io_buf = IOBuffer()
        summarize_empirical_model(emp_model; io = io_buf)
        summary_str = String(take!(io_buf))
        @test occursin("Empirical Biological Model Summary", summary_str)
        @test occursin("Allometric Mass Parameters", summary_str)
        @test occursin("Trait Covariance Matrix", summary_str)
    end

    @testset "Organism Dynamics & Bioenergetics" begin
        rng = MersenneTwister(99)

        # 1. Developmental intermolt and degree-day scaling
        # Same fix as above: a real behaviour and stage numbers, not a `LifeStage`
        # passed where the behaviour belongs.
        b_dyn = SpeciesBehaviorImpl()
        pel_dyn = larval_stage_count(b_dyn)
        @test min_intermolt_days(b_dyn, 1) == min_intermolt_days(b_dyn, pel_dyn - 1)
        @test min_intermolt_days(b_dyn, pel_dyn - 1) < min_intermolt_days(b_dyn, pel_dyn)
        @test min_intermolt_days(b_dyn, pel_dyn) == min_intermolt_days(b_dyn, pel_dyn + 3)
        @test min_intermolt_days(b_dyn, pel_dyn + 3) <
              min_intermolt_days(b_dyn, pel_dyn + 4)

        @test required_degree_days(b_dyn, 1) == required_degree_days(b_dyn, 2)
        @test required_degree_days(b_dyn, 1) == required_degree_days(b_dyn, pel_dyn - 1)
        @test required_degree_days(b_dyn, pel_dyn - 1) <
              required_degree_days(b_dyn, pel_dyn)
        @test required_degree_days(b_dyn, pel_dyn) ==
              required_degree_days(b_dyn, pel_dyn + 3)
        @test required_degree_days(b_dyn, pel_dyn + 3) <
              required_degree_days(b_dyn, pel_dyn + 4)

        # 2. Shell condition index (1 through 5)
        mock_org = Organism(
            1, (5, 5), adult, male, 11, 1000.0, 95.0, 350.0, 50.0, 1000.0, 0.0,
            false, 0, 0.0, 0
        )
        @test shell_condition(mock_org, 1010.0) == 1 # 10 days post-molt -> Soft shell
        @test shell_condition(mock_org, 1100.0) == 2 # 100 days post-molt -> New hard clean shell
        @test shell_condition(mock_org, 1500.0) == 3 # 500 days (~1.4 yr) -> Intermediate shell
        @test shell_condition(mock_org, 1900.0) == 4 # 900 days (~2.5 yr) -> Old shell
        @test shell_condition(mock_org, 2500.0) == 5 # 1500 days (~4.1 yr) -> Senescent shell

        # 3. Bioenergetics and functional response
        dE_low, intake_low = calculate_energy_change(mock_org, 3.0, 1.0)
        dE_high, intake_high = calculate_energy_change(mock_org, 3.0, 10.0)
        @test intake_high > intake_low
        @test dE_high > dE_low

        # Saturating functional response: intake at food=50 vs food=100 should be near asymptote
        _, intake_50 = calculate_energy_change(mock_org, 3.0, 50.0)
        _, intake_100 = calculate_energy_change(mock_org, 3.0, 100.0)
        @test intake_100 / intake_50 < 1.10

        # Scalar vs agent dispatch consistency
        dE_scalar, intake_scalar = calculate_energy_change(
            3.0, 5.0, 95.0; mass = 350.0
        )
        dE_agent, intake_agent = calculate_energy_change(mock_org, 3.0, 5.0)
        @test dE_scalar ≈ dE_agent
        @test intake_scalar ≈ intake_agent

        # 4. Thermal ecdysial inhibition.
        #    The onset is configurable, so drive it explicitly to prove the
        #    mechanism responds to `molt_thermal_onset` rather than to a
        #    hardcoded threshold.
        org_ready = Organism(
            2, (5, 5), juvenile, male, 5, 200.0, 18.0, 2.5, 50.0, 50.0, 400.0,
            false, 0, 0.0, 0
        )
        inhibition_params = IBMParameters(
            phenology = PhenologyParams(
                molt_thermal_onset = 5.0, molt_thermal_bandwidth = 3.0
            )
        )
        p_cold = molt_probability(org_ready, 3.0, 180.0; params = inhibition_params)
        p_warm = molt_probability(org_ready, 7.5, 180.0; params = inhibition_params)
        @test p_cold > 0.0
        @test p_warm < p_cold

        # With the default onset above the seasonal maximum, no inhibition occurs
        # within the normal thermal range, so molting is not shut off mid-season.
        default_params = IBMParameters()
        @test default_params.phenology.molt_thermal_onset >
              default_params.environment.temp_base + default_params.environment.temp_amplitude
        @test molt_probability(org_ready, 7.5, 180.0) == molt_probability(org_ready, 3.0, 180.0)

        # 5. Multi-component mortality: soft shell vulnerability & adult senescence
        org_soft = Organism(
            3, (5, 5), juvenile, male, 6, 205.0, 20.0, 3.5, 10.0, 200.0, 10.0,
            false, 0, 0.0, 0
        )
        org_hard = Organism(
            4, (5, 5), juvenile, male, 6, 260.0, 20.0, 3.5, 10.0, 150.0, 100.0,
            false, 0, 0.0, 0
        )
        m_soft = calculate_mortality_probability(org_soft, 3.0, 5.0, 205.0)
        m_hard = calculate_mortality_probability(org_hard, 3.0, 5.0, 260.0)
        @test m_soft > m_hard

        # Senescence is restricted to terminally moulted adults, so both are at the
        # male terminal stage and differ only in age. Without that the two are
        # identical: an animal that has not finished growing pays no wear-out however
        # old it is.
        #
        # Ages are set from the terminal stage's accumulated intermolt time, so shell
        # age — age since that last moult — lands either side of the threshold rather
        # than negative, which is what a raw `age_days` below the stage sum gives.
        m_st = default_params.organism.max_stage_male
        intermolt_sum = default_params.organism.shell_age_stage_offset *
                        (m_st - default_params.organism.shell_age_first_stage)
        young_adult = Organism(
            5, (5, 5), adult, male, m_st,
            intermolt_sum + 4 * default_params.simulation.days_per_year,
            105.0, 450.0, 60.0, 1000.0, 100.0, false, 0, 0.0, 0
        )
        old_adult = Organism(
            6, (5, 5), adult, male, m_st,
            intermolt_sum + 9 * default_params.simulation.days_per_year,
            105.0, 450.0, 60.0, 1000.0, 100.0, false, 0, 0.0, 0
        )
        m_young = calculate_mortality_probability(young_adult, 3.0, 5.0, 1200.0)
        m_old   = calculate_mortality_probability(old_adult, 3.0, 5.0, 2800.0)
        @test m_old > m_young

        # 6. Temperature-coupled perform_molt!
        # Stage derived: an `Organism` only exists from `juvenile_max_stage`.
        st_molt = IBMParameters().organism.juvenile_max_stage
        test_juv = Organism(
            7, (5, 5), juvenile, male, st_molt, 200.0, 18.0, 2.5, 25.0, 50.0, 400.0,
            false, 0, 0.0, 0
        )
        perform_molt!(test_juv, 200.0, 3.0; rng = rng)
        @test test_juv.stage == st_molt + 1
        @test test_juv.size > 18.0
        @test test_juv.mass > 2.5
        @test test_juv.last_growth_event_time == 200.0
        @test test_juv.degree_days == 0.0
    end

    @testset "Simulation Engine Enhancements & Dispersal" begin
        # 1. Config-based model initialization
        base_cfg = default_ibm_configuration()
        base_cfg[:n_agents] = 20
        base_cfg[:grid] = (12, 12)

        m_dict = initialize_model(base_cfg; n_agents = 15)
        @test nagents(m_dict) == 15
        @test haskey(abmproperties(m_dict), :n_benthic_organisms)
        @test haskey(abmproperties(m_dict), :n_mature_males)
        @test haskey(abmproperties(m_dict), :n_mature_females)
        @test haskey(abmproperties(m_dict), :n_legal_males)
        @test haskey(abmproperties(m_dict), :total_molts)
        @test haskey(abmproperties(m_dict), :deaths_predation)
        @test haskey(abmproperties(m_dict), :deaths_starvation)
        @test haskey(abmproperties(m_dict), :deaths_thermal)
        @test haskey(abmproperties(m_dict), :deaths_senescence)
        @test haskey(abmproperties(m_dict), :deaths_fishery)

        # TOML file-based initialization
        tmp_cfg_file = tempname() * ".config"
        save_ibm_configuration(tmp_cfg_file, base_cfg)
        @test isfile(tmp_cfg_file)
        m_toml = initialize_model(tmp_cfg_file; n_agents = 10)
        @test nagents(m_toml) == 10
        rm(tmp_cfg_file; force = true)

        # 2. Demographic caching in model_step!
        model = initialize_model(n_agents = 0, grid = (10, 10), seed = 999)
        @test model.n_benthic_organisms == 0
        @test model.n_mature_males == 0

        # Add 1 mature legal male and 1 mature female
        add_agent!(
            (3, 3), model, adult, male, 12, 1500.0, 105.0, 420.0, 60.0,
            100.0, 500.0, false, 0, 0.0, 0
        )
        add_agent!(
            (4, 4), model, adult, female, 10, 1400.0, 70.0, 150.0, 45.0,
            100.0, 500.0, true, 0, 0.0, 0
        )
        model_step!(model)
        @test model.n_benthic_organisms == 2
        @test model.n_mature_males == 1
        @test model.n_mature_females == 1
        @test model.n_legal_males == 1

        # 3. organism_disperse! behavior
        # Pelagic zoea dispersion
        zoea_org = add_agent!(
            (5, 5), model, zoea, female, 1, 10.0, 1.0, 0.01, 5.0,
            0.0, 10.0, false, 0, 0.0, 0
        )
        pos_before = zoea_org.pos
        organism_disperse!(zoea_org, model)
        # Position after randomwalk should be a valid grid coordinate
        @test 1 <= zoea_org.pos[1] <= 10
        @test 1 <= zoea_org.pos[2] <= 10

        # Starved benthic lethargy (mostly remains in place)
        starved_org = add_agent!(
            (2, 2), model, juvenile, male, 5, 200.0, 18.0, 2.5, -5.0,
            50.0, 100.0, false, 0, 0.0, 0
        )
        stationary_count = 0
        for _ in 1:20
            old_p = starved_org.pos
            organism_disperse!(starved_org, model)
            if starved_org.pos == old_p
                stationary_count += 1
            end
        end
        # With 75% stationary probability, stationary count over 20 trials should be > 5
        @test stationary_count >= 5

        # Benthic habitat taxis toward cold water (~2.5°C) and food
        model.temperature_field .= 8.0     # warm/suboptimal everywhere
        model.food_availability_field .= 0.1
        model.temperature_field[5, 6] = 2.5 # cold and rich at (5, 6)
        model.food_availability_field[5, 6] = 10.0

        forager = add_agent!(
            (5, 5), model, juvenile, male, 7, 400.0, 35.0, 15.0, 30.0,
            100.0, 200.0, false, 0, 0.0, 0
        )
        moved_to_optimal = 0
        for _ in 1:30
            # Reset to (5, 5) and disperse
            move_agent!(forager, (5, 5), model)
            organism_disperse!(forager, model)
            if forager.pos == (5, 6)
                moved_to_optimal += 1
            end
        end
        # The optimal cell should be visited significantly more than random uniform 1/9 (~3)
        @test moved_to_optimal >= 5

        # 4. Male-size scaled sperm clutches & ovigerous female mass (complex model)
        model_mate = initialize_model(
            n_agents = 0, grid = (10, 10), seed = 123, model_mode = :complex
        )
        model_mate.current_time = 45.0  # Spring mating season

        # Test sub-legal male (Size = 80 mm < 95 mm) -> confers 2 clutches
        sub_male = add_agent!(
            (5, 5), model_mate, 11, male, 1200.0, 80.0, 200.0, 50.0,
            0.0, 800.0, false, 0, 0.0, 0, true
        )
        fem_org = add_agent!(
            (5, 6), model_mate, 10, female, 1200.0, 65.0,
            calculate_mass(65.0, female; mat = 1), 50.0, 0.0, 800.0, true, 0, 0.0, 0, true
        )
        base_somatic_w = fem_org.mass
        agent_step!(fem_org, model_mate)
        # Copulated with sub-legal male -> 2 clutches, then extruded 1 -> 1 clutch left
        @test fem_org.sperm_clutches == 1
        @test fem_org.egg_count >= 1000
        @test fem_org.mass > base_somatic_w # Includes egg clutch mass

        # Test legal male (Size = 110 mm >= 95 mm) -> confers 3 clutches
        model_mate2 = initialize_model(
            n_agents = 0, grid = (10, 10), seed = 123
        )
        model_mate2.current_time = 45.0
        leg_male = add_agent!(
            (5, 5), model_mate2, 12, male, 1400.0, 110.0, 450.0, 50.0,
            0.0, 800.0, false, 0, 0.0, 0, true
        )
        fem_org2 = add_agent!(
            (5, 6), model_mate2, 10, female, 1200.0, 65.0,
            calculate_mass(65.0, female; mat = 1), 50.0, 0.0, 800.0, true, 0, 0.0, 0, true
        )
        agent_step!(fem_org2, model_mate2)
        # Copulated with legal male -> 3 clutches, then extruded 1 -> 2 clutches left
        @test fem_org2.sperm_clutches == 2
        @test fem_org2.egg_count >= 1000
        @test fem_org2.mass > base_somatic_w

        # 5. Process Toggles and Probabilistic Aggregate Dynamics
        p_toggles = IBMParameters(
            toggles = ProcessToggles(
                use_movement = false,
                use_degree_days = false,
                use_spermatheca = false
            )
        )
        m_toggled = initialize_model(params = p_toggles, n_agents = 0, grid = (10, 10), seed = 456)
        @test m_toggled.params.toggles.use_movement == false
        @test m_toggled.params.toggles.use_degree_days == false

        # Test movement toggle off: agent position does not change
        test_agent = add_agent!(
            (4, 4), m_toggled, 7, male, 300.0, 25.0, 5.0, 50.0,
            0.0, 0.0, false, 0, 0.0, 0, false
        )
        organism_disperse!(test_agent, m_toggled)
        @test test_agent.pos == (4, 4)

        # Direct mating and clutch extrusion when use_spermatheca is false
        m_toggled.current_time = 30.0
        s_male = add_agent!(
            (5, 5), m_toggled, 12, male, 1200.0, 95.0, 300.0, 50.0,
            0.0, 0.0, false, 0, 0.0, 0, true
        )
        s_fem = add_agent!(
            (5, 6), m_toggled, 10, female, 1200.0, 65.0, 85.0, 50.0,
            0.0, 0.0, true, 0, 0.0, 0, true
        )
        agent_step!(s_fem, m_toggled)
        @test s_fem.egg_count >= 1000
        @test s_fem.is_mating_ready == false
        @test s_fem.sperm_clutches == 0 # Spermathecal storage not used

        # Test Probabilistic Aggregate Groups
        # Seeded one stage below individuation, so the cohort individuates on the
        # first step. Both the stage and the expected result are derived.
        st_last_cohort = IBMParameters().organism.juvenile_max_stage - 1
        agg_model = initialize_model(n_agents = 0, grid = (10, 10), seed = 777)
        # Seed an aggregate group in the last cohort stage
        grp5 = AggregateGroup(
            id = 99,
            stage = st_last_cohort,
            abundance = 50.0,
            size = 15.0,
            mass = 1.5,
            age_days = 1400.0,
            degree_days = 5000.0, # Far past the stage requirement, so `ready` is 1
            last_growth_time = 0.0,
            pos = (5, 5)
        )
        push!(agg_model.aggregate_populations.groups, grp5)

        # Step aggregate groups. The requirement is per stage and the transition is a
        # daily hazard, so a cohort needs enough steps to accumulate the degree-days
        # *and* to draw the transition; `degree_days` is seeded well past the
        # requirement so only the hazard applies.
        initial_org_count = nagents(agg_model)
        for _ in 1:15
            step_aggregate_groups!(agg_model)
        end

        # Individuals should have individuated into the first agent stage
        @test nagents(agg_model) > initial_org_count
        st_first_agent = IBMParameters().organism.juvenile_max_stage
        for org in allagents(agg_model)
            @test org.stage == st_first_agent
            @test org.size > 0.0
            @test org.mass > 0.0
            @test org.sex in (male, female)
        end

        # Model step executes smoothly and refreshes demographic cache
        model_step!(m_toggled)
        @test m_toggled.n_benthic_organisms == nagents(m_toggled)
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

    @testset "Config rescan applies only what changed in the file" begin
        # `configs/snowcrab.toml` declares every parameter, so re-parsing it and
        # rebuilding the parameter tree would re-apply all 260 values annually and
        # discard anything changed in the parameter-steering panel since the last
        # scan. The diff is what makes the scan safe, not merely cheap.
        @testset "flatten / diff" begin
            flat = flatten_toml(Dict("environment" => Dict("temp_base" => 5.0),
                "simulation" => Dict("grid_nx" => 20)))
            @test flat["environment.temp_base"] == 5.0
            @test flat["simulation.grid_nx"] == 20

            @test isempty(config_diff(flat, flat))

            # Nested tables are accepted: `config_diff` flattens both sides, so a
            # nested `current` is not reported as one wholly-changed key. Every
            # other key is repeated so that only the one under test moves --
            # omitting one would count as a deletion and be reported separately.
            d = config_diff(flat, Dict("environment" => Dict("temp_base" => 7.5),
                "simulation" => Dict("grid_nx" => 20)))
            @test length(d) == 1
            @test d[1][1] == "environment.temp_base"
            @test d[1][2] == (5.0, 7.5)

            # A deleted key is reported rather than silently dropped.
            d2 = config_diff(flat, Dict())
            @test length(d2) == 2
            @test all(p -> last(p[2]) === nothing, d2)
            @test Set(first(p) for p in d2) ==
                Set(["environment.temp_base", "simulation.grid_nx"])
        end

        @testset "leaf parameter names are unambiguous" begin
            # `reconstruct_params(p; leaf = v)` hands the keyword to every parameter
            # struct and keeps it where it is a declared field. That is only safe
            # while a leaf name identifies exactly one parameter.
            owners = Dict{String, Vector{Symbol}}()
            for (field, T, _) in parameter_structs()
                for fn in fieldnames(T)
                    push!(get!(() -> Symbol[], owners, String(fn)), field)
                end
            end
            @test isempty([k for (k, v) in owners if length(v) > 1])
            # Not a hand-written constant: this is what makes a stale count fail
            # loudly. It broke at 260 when `stage_labels` was added and only the
            # test knew.
            @test length(owners) ==
                  sum(length(fieldnames(T)) for (_, T, _) in parameter_structs())
        end

        @testset "live GUI rescan" begin
            path = joinpath(mktempdir(), "scenario.toml")
            write(path, """
                   [environment]
                   temp_base = 5.0
                   temp_amplitude = 4.0
                   [simulation]
                   grid_nx = 20
                   """)
            fig, state = launch_organism_gui(n_agents = 20, grid = (20, 20), dt = 1.0,
                seed = 3, display_window = false, config_path = path)
            @test state.config_path == path
            # Seeded from the file, so the first scan reports only later edits rather
            # than treating every declared parameter as a change.
            @test state.config_snapshot["environment.temp_base"] == 5.0
            @test state.config_snapshot["simulation.grid_nx"] == 20

            # No file change: no-op.
            @test isempty(rescan_config!(state))

            # A GUI-side edit must survive a scan that touches something else.
            p0 = abmproperties(state.model)[:params]
            abmproperties(state.model)[:params] = reconstruct_params(p0; temp_amplitude = 2.5)

            write(path, """
                   [environment]
                   temp_base = 7.5
                   temp_amplitude = 4.0
                   [simulation]
                   grid_nx = 20
                   """)
            msg = rescan_config!(state)
            p1 = abmproperties(state.model)[:params]
            @test occursin("1 applied", msg)
            @test p1.environment.temp_base == 7.5        # the file edit landed
            @test p1.environment.temp_amplitude == 2.5   # the panel edit survived

            # Structural keys are reported, never applied: they are baked into the ABM.
            write(path, """
                   [environment]
                   temp_base = 7.5
                   temp_amplitude = 4.0
                   [simulation]
                   grid_nx = 999
                   """)
            msg = rescan_config!(state)
            p2 = abmproperties(state.model)[:params]
            @test occursin("restart-only", msg)
            @test occursin("grid_nx", msg)
            @test p2.simulation.grid_nx == 20
            @test all(f -> f in RESCAN_IMMUTABLE,
                (:grid_nx, :grid_ny, :grid_size, :dt, :seed))

            # A half-written file must not take the run down or half-apply.
            write(path, "[environment\ntemp_base = ")
            @test_throws Exception rescan_config!(state)
            @test abmproperties(state.model)[:params].environment.temp_base == 7.5

            # No path at all is a silent no-op, not an error.
            state.config_path = ""
            @test isempty(rescan_config!(state))
            @test isempty(rescan_config!(state))   # idempotent
        end
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

    @testset "snowcrab.toml is complete and self-contained" begin
        # Flatten a parsed TOML table to dotted leaf keys, so a config file can be
        # compared against the declared parameter surface key by key.
        function collect_leaves!(out, cfg, prefix = "")
            for (k, v) in cfg
                key = isempty(prefix) ? String(k) : string(prefix, ".", k)
                if v isa AbstractDict
                    collect_leaves!(out, v, key)
                else
                    push!(out, key)
                end
            end
            return out
        end
        function collect_leaves_flat!(out, cfg, prefix = "")
            for (k, v) in cfg
                key = isempty(prefix) ? String(k) : string(prefix, ".", k)
                if v isa AbstractDict
                    collect_leaves_flat!(out, v, key)
                else
                    out[key] = v
                end
            end
            return out
        end

        # It used to be a sparse overlay of 75 keys. It must now carry every
        # declared input, so it can be read and edited as the single description
        # of a snow crab run without cross-referencing the baseline.
        declared_keys = String[]
        for (field, T, _) in parameter_structs()
            base = T()
            for fn in fieldnames(T)
                push!(declared_keys, string(field, ".", fn))
            end
        end

        snow_raw = read(joinpath(@__DIR__, "..", "configs", "snowcrab.toml"), String)
        snow = TOML.parse(snow_raw)
        snow_keys = String[]
        collect_leaves!(snow_keys, snow)

        @test Set(snow_keys) == Set(declared_keys)
        @test length(snow_keys) == length(declared_keys)

        # Only the operating scale is meant to differ from the baseline. If snow
        # crab biology ever diverges from the struct defaults, this test names it.
        base_cfg = TOML.parsefile(joinpath(@__DIR__, "..", "configs", "default.toml"))
        base_keys = String[]
        collect_leaves!(base_keys, base_cfg)
        base_flat = Dict{String, Any}()
        collect_leaves_flat!(base_flat, base_cfg)
        snow_flat = Dict{String, Any}()
        collect_leaves_flat!(snow_flat, snow)

        differing = sort([k for k in base_keys if !isequal(base_flat[k], snow_flat[k])])
        # Only the operating scale now differs: the grid, the carrying capacity, and
        # the duration. The seeded clutch is deliberately identical in both files, so
        # snow crab differs from the baseline in how long it runs and over what area,
        # not in what it starts with.
        @test differing == ["environment.benthic_carrying_capacity",
            "simulation.grid_nx", "simulation.grid_ny"]

        # And the operating scale is what it claims to be.
        cfg = parse_cli_args(["--config=$(joinpath(@__DIR__, "..", "configs", "snowcrab.toml"))"])
        p = IBMParameters(cfg)
        @test p.simulation.grid_nx == 1000
        @test p.simulation.grid_ny == 1000
        @test p.simulation.initial_aggregate_cohort[2] == 2000.0
        @test p.simulation.days == 730
        @test p.environment.benthic_carrying_capacity == 10_000.0
        # Seeded stock must stay below K or the density feedback replaces the
        # mortality schedule instead of correcting it.
        seeded = sum(p.simulation.initial_aggregate_cohort) +
                 sum(p.simulation.initial_female_cohort) +
                 sum(p.simulation.initial_male_cohort)
        @test seeded < p.environment.benthic_carrying_capacity
    end

    @testset "Habitat field has spatial structure" begin
        # The HSI field was i.i.d. noise: no correlation between neighbours, so
        # nowhere better to live than anywhere else, and visual static on a large
        # grid. `habitat_peaks` Gaussian bumps give it coherent structure.
        @test IBMParameters().environment.habitat_peaks == 3

        # Zero peaks reproduces the previous pure-noise field exactly.
        p0 = EnvironmentParams(habitat_peaks = 0)
        expected = clamp.(randn(Xoshiro(4), (20, 20)) .* p0.habitat_quality_sd .+
                          p0.habitat_quality_mean, 0.0, 1.0)
        @test build_habitat_quality_field(p0, (20, 20), Xoshiro(4)) == expected

        # Peaks are reproducible under a seed and differ across seeds.
        p3 = EnvironmentParams(habitat_peaks = 3)
        a = build_habitat_quality_field(p3, (40, 40), Xoshiro(9))
        @test a == build_habitat_quality_field(p3, (40, 40), Xoshiro(9))
        @test a != build_habitat_quality_field(p3, (40, 40), Xoshiro(10))

        # Neighbour correlation is the statistic that separates structured from
        # uncorrelated. Mean neighbour gap does not: adding a smooth bump on top
        # of an unchanged noise floor leaves the noise-driven differences intact,
        # so that statistic measured clipping, not structure.
        function neighbour_correlation(f)
            return cor(vec(f[1:(end-1), :]), vec(f[2:end, :]))
        end

        # The default index must have headroom: `habitat_quality_mean = 1.0` on a
        # clipped [0, 1] field pinned half the cells at the ceiling and clipped
        # every peak away, leaving the field uncorrelated.
        @test IBMParameters().environment.habitat_quality_mean == 0.5
        @test IBMParameters().environment.habitat_peaks == 3

        pure = build_habitat_quality_field(
            EnvironmentParams(habitat_peaks = 0), (60, 60), Xoshiro(4))
        structured = build_habitat_quality_field(
            EnvironmentParams(habitat_peaks = 3), (60, 60), Xoshiro(4))
        @test abs(neighbour_correlation(pure)) < 0.15
        @test neighbour_correlation(structured) > 0.3
        @test maximum(structured) > maximum(pure)     # peaks reach higher
        @test all(0.0 .<= structured .<= 1.0)         # still clipped to the index

        # More peaks means more of the field is smooth, so correlation rises.
        cors = [neighbour_correlation(build_habitat_quality_field(
            EnvironmentParams(habitat_peaks = n), (60, 60), Xoshiro(4)))
            for n in (0, 1, 3, 6, 12)]
        @test issorted(cors)
        @test cors[1] < 0.15
        @test cors[end] > 0.6

        # The default field stays correlated across seeds and across grid sizes,
        # because the peak width is a fraction of the domain rather than a cell
        # count. A fixed cell width would vanish on a large grid.
        for n in (20, 60, 200, 600)
            f = build_habitat_quality_field(EnvironmentParams(habitat_peaks = 3),
                (n, n), Xoshiro(3))
            @test neighbour_correlation(f) > 0.3
        end
        seeds = [neighbour_correlation(build_habitat_quality_field(
            EnvironmentParams(habitat_peaks = 3), (60, 60), Xoshiro(s))) for s in 1:8]
        @test all(>(0.3), seeds)

        # The peaks are localised, not a global offset. Measured as the share of
        # cells above the midpoint between the field's mean and its maximum, which
        # scales with the peak rather than with an absolute tolerance: a fixed
        # `max - 0.05` band is a different fraction of the peak at every
        # resolution, so it measures discretisation rather than coverage.
        f = build_habitat_quality_field(EnvironmentParams(habitat_peaks = 3),
            (60, 60), Xoshiro(3))
        cutoff = (mean(f) + maximum(f)) / 2
        @test 0.0 < count(>(cutoff), f) / length(f) < 0.5

        # A non-positive width is a configuration error, not a blank field.
        @test_throws ErrorException build_habitat_quality_field(
            EnvironmentParams(habitat_peaks = 3, habitat_peak_width_fraction = 0.0),
            (10, 10), Xoshiro(1))
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
        @test !isdefined(GeoIBM, :size_frequency)
        @test !isdefined(GeoIBM, :GUI_SIZE_BIN_WIDTH_MM)
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

    @testset "Configured legal size drives every reported count" begin
        # Regression: the legal-male count, the legal-male biomass column, and
        # the GUI colour ramp each previously derived the threshold
        # independently, so a reconfigured fishery produced numbers that
        # disagreed with each other.

        adult_sz = 68.0
        # Sizes 100 / 90 / 78 mm, so the 95 mm and 80 mm legal sizes select
        # different subsets and a hardcoded threshold would be caught.
        legal_a = Organism(
            1, (1, 1), adult, male, 10, 900.0, 100.0, 400.0, 30.0, 0.0, 200.0,
            true, 0, 0.0, 0, true
        )
        legal_b = Organism(
            2, (2, 2), adult, male, 11, 900.0, 90.0, 350.0, 30.0, 0.0, 200.0,
            true, 0, 0.0, 0, true
        )
        legal_c = Organism(
            3, (3, 3), adult, male, 12, 900.0, 78.0, 250.0, 30.0, 0.0, 200.0,
            true, 0, 0.0, 0, true
        )

        tight = IBMParameters(fishery = FisheryParams(legal_size_male = 95.0))
        loose = IBMParameters(fishery = FisheryParams(legal_size_male = 80.0))

        tight_leg = count(o -> is_legal_male(o, 95.0, adult_sz), (legal_a, legal_b, legal_c))
        loose_leg = count(o -> is_legal_male(o, 80.0, adult_sz), (legal_a, legal_b, legal_c))
        @test tight_leg == 1
        @test loose_leg == 2

        # The GUI colour ramp must use the same threshold
        @test organism_color(legal_b, tight) == "#0072B2"
        @test organism_color(legal_b, loose) == "#003B73"
        # ... and so must the model's demographic cache. Seed a model with known
        # individuals and confirm the cache tracks the configured legal size.
        for (params, expected) in ((tight, 1), (loose, 2))
            m = initialize_model(n_agents = 0, grid = (10, 10), seed = 3, params = params)
            for org in (legal_a, legal_b, legal_c)
                add_agent!(org.pos, m, org.stage, org.sex, org.age_days, org.size,
                    org.mass, org.energy_reserve, org.last_growth_event_time,
                    org.degree_days, org.is_mating_ready, org.egg_count,
                    org.hatch_day, org.sperm_clutches, org.is_mature)
            end
            model_step!(m)
            @test m.n_legal_males == expected
            @test m.n_mature_males == 3
        end
    end

    @testset "Agent observation column names are config independent" begin
        # Regression: the threshold-taking predicates were wrapped in anonymous
        # closures, and `Agents.dataname` names a column after the function
        # object, so the schema came out as `count_#7` and every downstream
        # column lookup silently resolved to `nothing`.
        expected = [
            "length_stage", "safe_mean_stage", "safe_mean_size", "safe_mean_mass",
            "sum_mass", "count_is_adult_male", "count_is_mature_male",
            "count_is_mature_female", "count_is_legal_male", "count_is_ovigerous",
            "sum_mass_if_mature_female", "sum_mass_if_mature_male",
            "sum_mass_if_legal_male"
        ]
        default_p = IBMParameters()
        alt_p = IBMParameters(
            fishery = FisheryParams(legal_size_male = 80.0),
            organism = OrganismParams(adult_size_threshold = 55.0)
        )
        @test [Agents.dataname(o) for o in default_agent_observations(default_p)] == expected
        @test [Agents.dataname(o) for o in default_agent_observations(alt_p)] == expected

        # A real run must expose exactly those columns, populated.
        m = initialize_model(n_agents = 20, grid = (12, 12), seed = 11)
        agent_df, _ = run_simulation(m, 10)
        for col in expected
            @test Symbol(col) in propertynames(agent_df)
        end
    end

    @testset "Annual production schema parity" begin
        # Regression: the empty path returned 14 columns and the populated path
        # 17, so `.day` failed only on runs too short to produce a sample.
        schema_names = [String(n) for (n, _) in ANNUAL_PRODUCTION_SCHEMA]
        @test :day in first.(ANNUAL_PRODUCTION_SCHEMA)
        @test :total_biomass_kg in first.(ANNUAL_PRODUCTION_SCHEMA)
        @test :annual_egg_production in first.(ANNUAL_PRODUCTION_SCHEMA)
        @test nrow(empty_annual_production()) == 0
        @test names(empty_annual_production()) == schema_names

        m_short = initialize_model(n_agents = 15, grid = (10, 10), seed = 4)
        a_short, d_short = run_simulation(m_short, 20)
        short = extract_annual_production(
            a_short,
            d_short;
            frequency_days = 365.0,
            snapshot_day = 274.0
        )
        @test nrow(short) == 0
        @test names(short) == schema_names
        @test :year ∉ propertynames(short)

        m_long = initialize_model(n_agents = 15, grid = (10, 10), seed = 4)
        a_long, d_long = run_simulation(m_long, 400)
        long = extract_annual_production(
            a_long,
            d_long;
            frequency_days = 365.0,
            snapshot_day = 274.0
        )
        @test nrow(long) > 0
        @test names(long) == names(short)
        for col in schema_names
            @test eltype(long[!, col]) == eltype(short[!, col])
        end
    end

    @testset "Sampling regime follows frequency, not the snapshot_day value" begin
        # Regression: the first sample was chosen by
        # `snapshot_day == 274.0 && freq < 274.0`, so an explicit 274 behaved
        # differently from an explicit 100 under a sub-annual frequency.
        @test annual_production_targets(1095.0; frequency_days = 365.0, snapshot_day = 274.0) ==
              [274.0, 639.0, 1004.0]
        @test annual_production_targets(1095.0; frequency_days = 365.0, snapshot_day = 100.0) ==
              [100.0, 465.0, 830.0]

        # Sub-annual: the regime is chosen by frequency alone, so the snapshot day
        # is irrelevant and cannot silently move the sampling instants.
        @test annual_production_targets(300.0; frequency_days = 90.0, snapshot_day = 274.0) ==
              annual_production_targets(300.0; frequency_days = 90.0, snapshot_day = 100.0)
        @test annual_production_targets(300.0; frequency_days = 90.0, snapshot_day = 274.0) ==
              [90.0, 180.0, 270.0]

        # `day` is a genuine day of year in both regimes.
        m = initialize_model(n_agents = 15, grid = (10, 10), seed = 6)
        a, d = run_simulation(m, 400)
        ann = extract_annual_production(
            a,
            d;
            frequency_days = 365.0,
            snapshot_day = 100.0
        )
        @test nrow(ann) > 0
        @test all(isapprox.(ann.day, 100.0))
        @test all(isapprox.(ann.time, (ann.sample_index .- 1) .* 365.0 .+ 100.0))

        # `sample_index` counts snapshots; it must not be called `year`, and the
        # calendar position lives in `day`.
        @test :year ∉ propertynames(ann)
        @test :sample_index in propertynames(ann)
        @test ann.sample_index == collect(1:nrow(ann))
    end

    @testset "Soft-shell vulnerability window is finite" begin
        # Regression: `soft_shell_duration` was declared and configurable but the
        # hazard ignored it, so a long-lived individual accrued soft-shell risk
        # forever, set only by how long ago it last molted.
        function soft_shell_params(window, tau)
            mort = MortalityParams(
                baseline_rate_by_stage = fill(0.0, 24),
                rate_predation = 0.0,
                rate_soft_shell = 0.12,
                rate_starvation = 0.0,
                rate_thermal = 0.0,
                rate_senescence = 0.0,
                rate_cannibalism = 0.0,
                soft_shell_duration = window,
                soft_shell_decay_days = tau,
                predation_size_reference = 1.0e9,
                predation_size_decay = 1.0,
                cannibalism_size_threshold = 1.0e9,
                cannibalism_density_reference = 1.0e9,
                cannibalism_density_cap = 1.0,
                senescence_age_threshold = 1.0e9
            )
            return IBMParameters(mortality = mort)
        end

        function soft_shell_only(params, days_since_molt)
            mm = initialize_model(n_agents = 1, grid = (8, 8), seed = 2, params = params)
            org = first(allagents(mm))
            org.last_growth_event_time = -Float64(days_since_molt)
            return calculate_mortality_probability(
                org,
                3.0,
                0.0,
                0.0;
                params = params,
                density_feedback = 1.0
            )
        end

        narrow = soft_shell_params(1.0, 15.0)
        wide = soft_shell_params(400.0, 15.0)

        @test soft_shell_only(narrow, 0.0) > 0.0
        @test soft_shell_only(narrow, 0.0) ≈ soft_shell_only(wide, 0.0)
        @test soft_shell_only(narrow, 0.5) ≈ soft_shell_only(wide, 0.5)

        # Past the short window, still inside the long one: the distinction the
        # old windowless hazard could not represent.
        @test soft_shell_only(narrow, 10.0) == 0.0
        @test soft_shell_only(wide, 10.0) > 0.0
        @test soft_shell_only(narrow, 100.0) == 0.0
        @test soft_shell_only(wide, 100.0) > 0.0

        # Past both windows the risk is gone, not merely small.
        @test soft_shell_only(narrow, 500.0) == 0.0
        @test soft_shell_only(wide, 500.0) == 0.0

        # Monotone decay while inside the window.
        @test soft_shell_only(wide, 0.0) > soft_shell_only(wide, 5.0) >
              soft_shell_only(wide, 20.0) > 0.0

        # A zero decay time must not yield NaN.
        zero_tau = soft_shell_params(400.0, 0.0)
        @test isfinite(soft_shell_only(zero_tau, 0.0))
        @test isfinite(soft_shell_only(zero_tau, 5.0))
    end

    @testset "Population bounds are not derived from the carrying capacity" begin
        # Regression: `is_persistent` compared the final population against
        # `max(15, 0.10 * K)` — a floor derived from the same K the model regulates
        # around. A population that self-regulated at a fraction of K was therefore
        # reported as not persistent, and a larger K made a healthy run look extinct
        # and a failing one look healthy.

        m = initialize_model(n_agents = 40, grid = (12, 12), seed = 13)
        adf, _ = run_simulation(m, 120; adata = [(:stage, length)])

        # Default: floor is "not extinct", and does not scale with K.
        res = evaluate_population_criteria(m, adf)
        @test res.n_min == 1
        @test res.n_max == round(Int, 3.0 * m.params.benthic_carrying_capacity)
        @test res.n_living == length(allagents(m))
        @test res.peak_population >= res.n_living
        @test res.is_persistent == (res.n_living >= 1)

        # The floor must not move when K does. (Varying K on a *fresh* run would
        # also change the dynamics — density-dependent mortality scales with K — so
        # this compares the arithmetic on one fixed model rather than the verdict of
        # several different simulations.)
        for k in (30.0, 300.0, 30_000.0)
            p = reconstruct_params(m.params; benthic_carrying_capacity = k)
            mk = initialize_model(n_agents = 0, grid = (12, 12), seed = 13, params = p)
            for org in allagents(m)
                add_agent!(org.pos, mk, org.stage, org.sex, org.age_days, org.size,
                    org.mass, org.energy_reserve, org.last_growth_event_time,
                    org.degree_days, org.is_mating_ready, org.egg_count,
                    org.hatch_day, org.sperm_clutches, org.is_mature)
            end
            rk = evaluate_population_criteria(mk, adf)
            @test rk.n_min == 1
            @test rk.n_max == round(Int, 3.0 * k)
            @test rk.is_persistent == (rk.n_living >= 1)
        end

        # The floor is configuration, not a constant.
        strict = reconstruct_params(m.params; min_population = 10_000)
        mstrict = initialize_model(
            n_agents = 40,
            grid = (12, 12),
            seed = 13,
            params = strict
        )
        run_simulation(mstrict, 120; adata = [(:stage, length)])
        rstrict = evaluate_population_criteria(mstrict, adf)
        @test rstrict.n_min == 10_000
        @test rstrict.is_persistent == false

        # The ceiling is an explicit multiple of K, and is overridable in absolute
        # terms for programmatic callers.
        tight = reconstruct_params(m.params; max_population_multiple = 0.5)
        mtight = initialize_model(n_agents = 40, grid = (12, 12), seed = 13, params = tight)
        run_simulation(mtight, 120; adata = [(:stage, length)])
        rtight = evaluate_population_criteria(mtight, adf)
        @test rtight.n_max == round(Int, 0.5 * m.params.benthic_carrying_capacity)
        @test evaluate_population_criteria(m, adf; max_population = 1).n_max == 1

        # Both return paths report the same field set.
        empty_res = evaluate_population_criteria(
            initialize_model(n_agents = 0, grid = (8, 8), seed = 2),
            DataFrame(length_stage = Int[], time = Float64[])
        )
        @test Set(propertynames(empty_res)) == Set(propertynames(res))
        @test empty_res.peak_population == 0
    end

    @testset "Optimizer failures name their cause" begin
        # Regression: bare `catch; continue` blocks around the optimizer made a
        # genuine objective bug indistinguishable from a diverging trial. The
        # search proceeded, converged nowhere, and reported only that every
        # starting point failed.

        # An objective that always throws: the error must name the real cause,
        # not "did not converge".
        boom = x -> error("objective exploded at $(x)")
        err = try
            optimize_model(boom, [0.0, 0.0], [1.0, 1.0], [[0.5, 0.5]])
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("objective exploded", sprint(showerror, err))

        # A collectable hook lets a caller see the objective's own failures.
        seen = Any[]
        safe_objective(boom; label = "test", failures = seen)([1.0])
        @test length(seen) == 1
        @test occursin("objective exploded", sprint(showerror, seen[1]))

        # A finite objective still works, and NaN is mapped to a penalty.
        best_x, best_f = optimize_model(x -> (x[1] - 0.3)^2, [0.0], [1.0], [[0.5], [0.1]])
        @test best_f < 1.0
        @test 0.0 <= best_x[1] <= 1.0

        hooked = Any[]
        penalised = safe_objective(
            x -> (isnan(x[1]) ? NaN : x[1]^2);
            label = "test",
            failures = hooked
        )
        @test !isnan(penalised([2.0]))
        @test penalised([NaN]) == OBJECTIVE_PENALTY
        @test isempty(hooked)   # NaN is a value, not an exception

        # An objective that is non-finite everywhere must fail rather than
        # "converge" to the penalty and return a meaningless vector.
        nan_err = try
            optimize_model(x -> NaN, [0.0], [1.0], [[0.5]])
            nothing
        catch e
            e
        end
        @test nan_err isa ErrorException
        @test occursin("divergence penalty", sprint(showerror, nan_err))

        # Blocks report failures but still return usable parameters.
        bad_blocks = optimize_blocks(
            x -> error("block objective exploded"),
            [[1], [2]],
            [0.5, 0.5],
            [([0.0], [1.0]), ([0.0], [1.0])];
            passes = 1
        )
        @test length(bad_blocks[1]) == 2
        @test all(isfinite, bad_blocks[1])
    end

    @testset "Hazard clamping is reported" begin
        # Regression: the daily hazard was clamped to `hazard_cap` silently, so a
        # mis-specified rate could bind every step of every run with no signal.
        function clamp_events(params, org, temp, t)
            events = Tuple{Symbol, Float64, Float64}[]
            on_clamp = (source, raw, cap) -> push!(events, (source, raw, cap))
            p = calculate_mortality_probability(
                org,
                temp,
                0.0,
                t;
                params = params,
                density_feedback = 1.0,
                on_clamp = on_clamp
            )
            return p, events
        end

        mort = MortalityParams(
            baseline_rate_by_stage = fill(0.0, 24),
        rate_predation = 0.0, rate_soft_shell = 0.0,
            rate_starvation = 0.0, rate_thermal = 0.0, rate_senescence = 0.0,
            rate_cannibalism = 0.0,
            hazard_cap = 0.25,
            predation_size_reference = 1.0e9, predation_size_decay = 1.0,
            cannibalism_size_threshold = 1.0e9, cannibalism_density_reference = 1.0e9,
            cannibalism_density_cap = 1.0, senescence_age_threshold = 1e9
        )
        calm = IBMParameters(mortality = mort)
        m = initialize_model(n_agents = 1, grid = (8, 8), seed = 4, params = calm)
        org = first(allagents(m))
        org.trait_mortality_quantile = 0.5

        # A calm configuration never clamps and never reports.
        p, events = clamp_events(calm, org, 3.0, 10.0)
        @test p == 0.0
        @test isempty(events)

        # An extreme thermal stress exceeds the cap: reported, and capped.
        hot = reconstruct_params(calm; rate_thermal = 1.0e6, temp_thermal_stress = 1.0)
        p, events = clamp_events(hot, org, 40.0, 10.0)
        @test p == 0.25
        @test !isempty(events)
        @test any(e -> first(e) == :thermal_stress, events)

        # A high baseline rate drives the total past the cap.
        hotbase = reconstruct_params(calm; baseline_rate_by_stage = fill(1.0e6, 24))
        p, events = clamp_events(hotbase, org, 3.0, 10.0)
        @test p == 0.25
        @test any(e -> first(e) == :total, events)

        # A senescence clock far past the threshold does too. Senescence is keyed
        # on inferred shell age, so the organism has to actually be old. A stage
        # well past `shell_age_first_stage` keeps the age inference on the age side
        # of the subtraction.
        senescing = reconstruct_params(
            calm;
            rate_senescence = 1.0e6,
            senescence_age_threshold = 1.0
        )
        old_stage = calm.organism.shell_age_first_stage + 1
        mature = deepcopy(org)
        mature.is_mature = true
        # Senescence is restricted to terminally moulted adults, so the animal has
        # to be at its sex's terminal stage as well as old. The clock is keyed on
        # shell age, and a stage past `shell_age_first_stage` keeps the age
        # inference on the age side of the subtraction.
        mature.stage = calm.organism.max_stage_male
        mature.age_days = 100_000.0
        p, events = clamp_events(senescing, mature, 3.0, 100_000.0)
        @test p == 0.25
        @test any(e -> first(e) == :senescence, events)

        # ... and a young organism at the same stage and the same molt clock does not.
        young = deepcopy(org)
        young.is_mature = true
        young.stage = old_stage
        young.age_days = 100.0
        p_young, events_young = clamp_events(senescing, young, 3.0, 100_000.0)
        @test p_young == 0.0
        @test !any(e -> first(e) == :senescence, events_young)
    end

    @testset "Unknown configuration keys are reported" begin
        # Regression: a misspelled or renamed key in a TOML overlay was copied
        # into the configuration and then dropped without a word, so the run
        # completed and the setting simply never applied.
        path = joinpath(tempdir(), "unknown_key_config.toml")
        write(path, """
        [organism]
        adult_size_threshold = 55.0
        this_is_not_a_parameter = 1.0
        also_not_this = 2.0
        """)
        cfg = load_ibm_configuration(path)
        @test cfg[:adult_size_threshold] == 55.0
        @test cfg[:this_is_not_a_parameter] == 1.0
        @test unknown_config_keys(Set(keys(cfg))) ==
              sort([:this_is_not_a_parameter, :also_not_this])
        @test Set(keys(cfg)) != unknown_config_keys(Set(keys(cfg)))

        # Declared parameters and runtime keys are never reported.
        @test isempty(unknown_config_keys(Set(keys(config_field_types()))))
        @test isempty(unknown_config_keys(runtime_config_keys()))
        @test :grid in known_config_keys()
        @test :min_population in known_config_keys()

        # The shipped files declare nothing unknown.
        for f in ("default.toml", "snowcrab.toml")
            raw = TOML.parsefile(projectdir("configs", f))
            declared = Set{Symbol}()
            for (_, v) in raw
                isa(v, AbstractDict) || continue
                for k in keys(v)
                    push!(declared, Symbol(k))
                end
            end
            @test isempty(unknown_config_keys(declared))
        end
        rm(path; force = true)
    end

    @testset "Public surface and configuration plumbing" begin
        # Every exported name must resolve. The module once exported six
        # `FECUNDITY_*` names declared as `DEFAULT_FECUNDITY_*`, so `using
        # GeoIBM` gave you nothing to import under the exported name.
        undefined_exports = filter(n -> !isdefined(GeoIBM, n), names(GeoIBM))
        @test isempty(undefined_exports)

        # --- Configuration surface is derived, not hand-maintained ---
        structs = parameter_structs()
        @test Set(f for (f, _, _) in structs) == Set(fieldnames(IBMParameters))
        @test Set(s for (_, _, s) in structs) == Set(ibm_configuration_sections())
        smap = config_section_map()
        for (_, struct_type, section) in structs
            for field in fieldnames(struct_type)
                @test smap[field] == section
            end
        end
        @test config_field_type(:min_population) === Int
        @test config_field_type(:definitely_not_a_parameter) === nothing

        # Field names must be unique across the parameter structs. The flat
        # configuration maps a key to its section by name, so a duplicated name
        # makes one setting unreachable: `days_per_year` was declared in both
        # SimulationParams and MortalityParams, so a leap-year calendar set under
        # [simulation] was silently configuring mortality instead.
        owners = Dict{Symbol, Vector{String}}()
        for (_, struct_type, _) in structs
            for field in fieldnames(struct_type)
                push!(get!(owners, field, String[]), string(struct_type))
            end
        end
        @test sort([f for (f, o) in owners if length(o) > 1]) == Symbol[]

        defaults = parameter_struct_defaults()
        @test Set(keys(defaults)) == Set(keys(config_field_types()))
        @test default_ibm_configuration()[:grid] ==
              (defaults[:grid_nx], defaults[:grid_ny])

        # --- reconstruct_params carries every field through ---
        # Its one caller is the GUI Reset button, and the defect it had was
        # rebuilding from ten hardcoded fields and reverting the rest to struct
        # defaults.
        custom = IBMParameters(
            simulation = SimulationParams(days = 456, seed = 789),
            environment = EnvironmentParams(benthic_carrying_capacity = 1234.0),
            organism = OrganismParams(adult_size_threshold = 55.5),
            mortality = MortalityParams(rate_predation = 0.42, hazard_cap = 0.7),
            fishery = FisheryParams(legal_size_male = 80.0, fishing_probability_daily = 0.3),
            criteria = CriteriaParams(min_population = 7, max_population_multiple = 2.5)
        )
        round_tripped = reconstruct_params(custom)
        for (field, struct_type, _) in structs
            for name in fieldnames(struct_type)
                @test getfield(getfield(round_tripped, field), name) ==
                      getfield(getfield(custom, field), name)
            end
        end
        overridden = reconstruct_params(custom; rate_predation = 0.99)
        @test overridden.mortality.rate_predation == 0.99
        @test overridden.mortality.hazard_cap == 0.7

        # with_overrides applies what it is given and nothing else.
        base = MortalityParams()
        bumped = with_overrides(MortalityParams, base; rate_predation = 0.5)
        @test bumped.rate_predation == 0.5
        @test with_overrides(MortalityParams, base) == base

        # --- Coercion from TOML text ---
        @test coerce_to_declared(Float64, "3.5", :rate_predation) == 3.5
        @test coerce_to_declared(Int, "7.0", :days) === 7
        @test coerce_to_declared(Bool, "off", :use_molting) === false
        @test coerce_to_declared(String, "abc", :label) == "abc"
        @test try_coerce_to_declared(Int, "7", :days) == Some(7)
        @test try_coerce_to_declared(Int, "nonsense", :days) === nothing
        # An impossible conversion names the field and the value.
        msg = try
            coerce_to_declared(Int, "nonsense", :days)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("days", msg) && occursin("nonsense", msg)

        # --- Numeric guards ---
        # logistic_unit and logit are inverses, and both saturate instead of
        # overflowing at the ends of the range.
        for u in (-20.0, 0.0, 0.5, 20.0)
            @test logit(logistic_unit(u)) ≈ u atol = 1e-6
        end
        @test logistic_unit(1e6) == logistic_unit(60.0)
        @test logistic_unit(-1e6) == logistic_unit(-60.0)
        @test logistic_unit(0.0) == 0.5
        @test isfinite(logit(0.0)) && isfinite(logit(1.0))
        @test logit(0.5) == 0.0

        # --- Named predicates ---
        # `NamedPredicate` exists so an observation column can carry a threshold
        # and still be named after the predicate it computes.
        p = named_predicate(is_adult, :is_adult)
        @test p isa NamedPredicate && p isa Function
        @test string(p) == "is_adult" && propertynames(p) == ()
        small = Organism(
            1, (1, 1), juvenile, male, 5, 200.0, 30.0, 1.0, 5.0, 0.0, 0.0,
            false, 0, 0.0, 0, false
        )
        @test p(small) == is_adult(small)
        @test string(named_predicate(is_legal_male, :is_legal_male, 95.0, 68.0)) ==
              "is_legal_male"

        # --- CLI parsing ---
        # Flag names are the TOML key names verbatim; there are no abbreviations,
        # and an unrecognized name is reported and ignored rather than guessed at.
        args = parse_cli_args(String[
        "--days=10", "--max_stage_male=20", "--seed=7", "--grid=8,9",
        "--output_dir=results", "--temp_base=4.5"
    ])
        # `n_agents` is no longer a flag: the initial abundance is three vectors and
        # the total is derived from them, so there is no declared parameter of that
        # name for a flag to set.
        @test args[:days] == 10 && args[:max_stage_male] == 20 && args[:seed] == 7
        @test args[:grid] == (8, 9) && args[:output_dir] == "results"
        @test args[:temp_base] == 4.5
        # A flag that is not a declared parameter name is not silently accepted.
        @test !haskey(parse_cli_args(String["--species-config=x.toml"]), :species_config)
        @test parse_cli_args(String[]) isa Dict{Symbol, Any}
    end
    @testset "Seeding states the distribution, one stage per entry" begin
        # The initial age and sex structure is stated in three vectors rather than
        # derived from a count and a decay, so this seeds one cohort per pre-
        # individuation stage and one individual per benthic stage per sex, and checks
        # that exactly that came back.
        #
        # The default configuration seeds eggs only — a run begins with the age
        # structure it has to earn — so a test that wants a full range has to ask for
        # one.
        base = IBMParameters()
        juv = base.organism.juvenile_max_stage
        agg = fill(0.0, 24)
        for s in GeoIBM.FIRST_COHORT_STAGE:(juv - 1)
            agg[s+1] = 100.0
        end
        fem = fill(0.0, 24)
        mal = fill(0.0, 24)
        for s in juv:base.organism.max_stage_male
            fem[s+1] = 10.0
            mal[s+1] = 10.0
        end
        cfg = Dict{Symbol, Any}(
            :initial_aggregate_cohort => agg,
            :initial_female_cohort => fem,
            :initial_male_cohort => mal)

        m = initialize_model(cfg; grid = (24, 24), dt = 1.0, seed = 11)
        p = m.params
        orgs = collect(allagents(m))
        pop = Agents.abmproperties(m)[:aggregate_populations]
@test sort(unique([g.stage for g in pop.groups])) ==
                collect(GeoIBM.FIRST_COHORT_STAGE:(p.organism.juvenile_max_stage - 1))
        @test length(pop.groups) ==
                p.organism.juvenile_max_stage - GeoIBM.FIRST_COHORT_STAGE

        # Every benthic stage is represented, for each sex, within its ceiling.
        # Recruits are individuated into `juvenile_max_stage`, so that is where
        # the benthic range starts.
        for (sex, maxs) in (
            (male, p.organism.max_stage_male),
            (female, p.organism.max_stage_female)
        )
            present = sort(unique([o.stage for o in orgs if o.sex === sex]))
            @test present == collect(p.organism.juvenile_max_stage:maxs)
        end
        @test all(
            o.stage <=
            (o.sex === female ? p.organism.max_stage_female : p.organism.max_stage_male)
            for o in orgs
        )

        # The vectors are the distribution: what they sum to is what exists, per sex.
        # A vector is only read up to that sex's terminal stage, so a female vector
        # whose tail runs past `max_stage_female` contributes only its reachable
        # part — the same way a cohort beyond the ceiling is not seeded.
        seeded_at(vec, maxs) = sum(vec[1:min(length(vec), maxs + 1)])
        @test length(orgs) ==
              round(Int, seeded_at(fem, p.organism.max_stage_female) +
                        seeded_at(mal, p.organism.max_stage_male))
        @test count(o -> o.sex === male, orgs) ==
              round(Int, seeded_at(mal, p.organism.max_stage_male))
        @test count(o -> o.sex === female, orgs) ==
              round(Int, seeded_at(fem, p.organism.max_stage_female))

        # `n_agents` is the sum of the sex vectors when the caller does not state it,
        # and an override when they do.
        @test p.simulation.initial_female_cohort == fem
        @test p.simulation.initial_male_cohort == mal
        for n in (0, 1, 5, 19, 20, 21, 60)
            mm = initialize_model(n_agents = n, grid = (10, 10), seed = 4)
            @test nagents(mm) == n
        end

        # Maturity and the legal class are populated at t=0, and the model cache
        # agrees with a direct count.
        @test count(is_mature_male, orgs) > 0
        @test count(is_mature_female, orgs) > 0
        @test count(
            o -> is_legal_male(o, p.fishery.legal_size_male,
                               p.organism.adult_size_threshold), orgs
        ) > 0
        @test Agents.abmproperties(m)[:n_legal_males] == count(
            o -> is_legal_male(o, p.fishery.legal_size_male,
                               p.organism.adult_size_threshold), orgs
        )
        @test Agents.abmproperties(m)[:n_mature_males] == count(is_mature_male, orgs)
        @test Agents.abmproperties(m)[:n_mature_females] == count(is_mature_female, orgs)

        # Maturity follows the ogive at the individual's own size, rather than a
        # stage threshold: large males are mostly mature and medium ones are not.
        big_males = [o for o in orgs if o.sex === male && o.size >= 100.0]
        mid_males = [o for o in orgs if o.sex === male && 60.0 <= o.size < 90.0]
        if !isempty(big_males) && !isempty(mid_males)
            @test count(is_mature_male, big_males) / length(big_males) >
                  count(is_mature_male, mid_males) / length(mid_males)
        end

        # A zero-agent model seeds no individuals. The clutch is stated in its own
        # vector, so suppressing cohorts means writing zeros there — `n_agents` is the
        # sum of the two sex vectors and says nothing about cohorts.
        empty_params = IBMParameters()
        esp = empty_params.simulation
        empty_m = initialize_model(; grid = (10, 10), seed = 5,
            params = IBMParameters(
                simulation = with_overrides(SimulationParams, esp;
                    initial_aggregate_cohort = fill(0.0, 24))))
        @test isempty(collect(allagents(empty_m)))
        @test isempty(Agents.abmproperties(empty_m)[:aggregate_populations].groups)

        # Seeding is reproducible for a given seed.
        a = initialize_model(n_agents = 40, grid = (10, 10), seed = 99)
        b = initialize_model(n_agents = 40, grid = (10, 10), seed = 99)
        @test [(o.sex, o.stage, o.size) for o in allagents(a)] ==
              [(o.sex, o.stage, o.size) for o in allagents(b)]
    end

    @testset "Macro model legends carry the estimated parameters" begin
        # A Schaefer-shaped trajectory, so the optimiser converges interior rather
        # than pinning K to its upper bound — a bound-hitting fit would make every
        # assertion below meaningless.
        n = 8
        function macro_frame(scale::Float64 = 1.0)
            k_true, r_true, harvest = 400.0, 0.45, 0.12
            N = Float64[]
            cur = 300.0
            for _ in 1:n
                push!(N, cur)
                cur = cur + r_true * cur * (1 - cur / k_true) - harvest * cur
            end
            DataFrame(
                sample_index = collect(1:n),
                time = Float64.((0:n .- 1) .* 365 .+ 274),
                total_numbers = N .* scale,
                legal_numbers = (N .* 0.07) .* scale,
                mature_females = Float64.(100 .+ (1:n)),
                mature_males = Float64.(25 .+ (1:n)),
                total_biomass_kg = Float64.(500 .+ 20 .* (1:n)),
                fishable_male_biomass_kg = Float64.(15 .+ 1.5 .* (1:n)),
                mature_male_biomass_kg = Float64.(15 .+ 1.5 .* (1:n)),
                mature_female_biomass_kg = Float64.((15 .+ 1.5 .* (1:n)) .* 0.7),
                total_egg_production = Float64.(1e6 .* (1:n)),
                annual_egg_production = Float64.(1e6 .* (1:n)),
                catch_total = (harvest .* N) .* scale,
                catch_legal = (harvest .* N .* 0.07) .* scale,
                surplus_production_total = Float64.(zeros(n)),
                surplus_production_legal = Float64.(zeros(n))
            )
        end
        ann = macro_frame()
        stage_mat, catch_mat, yrs = extract_annual_stage_matrix(ann)
        fit = fit_generalized_leslie_model(stage_mat, catch_mat; years = yrs)
        comp = compare_macro_models(ann, fit; lag = 1)

        @test occursin("K=", comp.label_schaefer)
        @test occursin("r=", comp.label_schaefer)
        @test occursin("MSY=", comp.label_schaefer)
        @test occursin("s=", comp.label_delay_diff)
        @test occursin("α=", comp.label_delay_diff)
        @test occursin("λ=", comp.label_leslie)
        @test occursin("biomass", comp.label_observed)
        @test all(!isempty, (comp.label_schaefer, comp.label_delay_diff, comp.label_leslie))

        # The fits are in NUMBER, so the legend must say so: K is a carrying
        # capacity in individuals and MSY is individuals per year, and a bare
        # "K=..." beside a kg axis would be read as a biomass.
        @test occursin("ind", comp.label_schaefer)
        @test occursin("ind/yr", comp.label_schaefer)
        @test occursin("/yr", comp.label_schaefer)

        # And it must be true: scaling the number series scales K, while the
        # biomass column is held fixed. If anything in the fit chain read biomass,
        # K would not move.
        f_base = fit_population_dynamics(ann)
        f_scaled = fit_population_dynamics(macro_frame(7.0))
        @test ann.fishable_male_biomass_kg == macro_frame(7.0).fishable_male_biomass_kg
        @test isapprox(
            f_scaled.fishable_stock.schaefer.K / f_base.fishable_stock.schaefer.K,
            7.0;
            rtol = 1e-4
        )
        @test f_base.fishable_stock.schaefer.is_fitted
        @test f_scaled.fishable_stock.schaefer.is_fitted
        # The fit recovers the trajectory's own scale: K_legal = 0.07 * K_true.
        @test isapprox(f_base.fishable_stock.schaefer.K, 0.07 * 400.0; rtol = 0.02)

        # The biomass series are the number fits rescaled by one pooled mean
        # legal mass, applied identically to observed and predicted.
        leg_stages = filter(s -> s.is_legal_male, fit.stages)
        pooled = mean([s.mean_mass for s in leg_stages]) / 1000.0
        @test isapprox(comp.biomass_obs_kg, ann.legal_numbers .* pooled; rtol = 1e-9)

        # Every series shares the same x grid, so the GUI can overlay them.
        @test length(comp.years) == nrow(ann)
        @test length(comp.biomass_obs_kg) == nrow(ann)
        @test length(comp.biomass_schaefer_kg) == nrow(ann)
        @test length(comp.biomass_delay_diff_kg) == nrow(ann)
        @test length(comp.biomass_leslie_kg) == nrow(ann)

        # The model basis is a stage-mean approximation, not the measured sum, and
        # runs below it. This is why the GUI plots `fishable_male_biomass_kg` for
        # the survey markers: plotting the approximation as measured biomass put two
        # different quantities on one pair of axes.
        @test all(
            comp.biomass_obs_kg .< ann.fishable_male_biomass_kg
        )

        @test fmt_param(2000.0) == "2e+03"
        @test fmt_param(0.0004321) == "0.000432"
        @test fmt_param(95.25) == "95.2"
        @test fmt_param(1.0 / 3.0) == "0.333"
    end

    @testset "Death causes are attributed to the hazard that produced them" begin
        # Regression: the summed daily probability was computed correctly, but the
        # death was then attributed by `is_mature -> senescence, else -> predation`
        # — a rule unrelated to the sum. Every juvenile death was reported as a
        # predation death and every mature death as a senescence death, so the
        # published cause breakdown was fiction, and `deaths_starvation` and
        # `deaths_thermal` were never incremented at all.

        p = IBMParameters()
        m = initialize_model(n_agents = 60, grid = (16, 16), seed = 7, params = p)
        org = first(allagents(m))
        c = mortality_components(org, 3.0, 1.0, 10.0; params = p, density_feedback = 1.0)

        # The components sum to the raw total, and the cap applies to the total.
        parts = (:baseline, :predation, :soft_shell, :starvation,
                 :thermal, :senescence, :cannibalism)
        @test isapprox(sum(getfield(c, k) for k in parts), c.raw_total)
        @test c.total == clamp(c.raw_total, 0.0, p.mortality.hazard_cap)
        @test all(getfield(c, k) >= 0.0 for k in parts)
        @test all(isfinite, [getfield(c, k) for k in parts])

        # The convenience wrapper agrees with the breakdown.
        @test calculate_mortality_probability(
            org, 3.0, 1.0, 10.0; params = p, density_feedback = 1.0
        ) == c.total

        # Dominant cause is the largest component, and `:none` when all are zero.
        @test dominant_mortality_cause(c) == :baseline   # background dominates
        zeroed = merge(c, NamedTuple{parts}(ntuple(_ -> 0.0, length(parts))))
        @test dominant_mortality_cause(zeroed) == :none
        one_cause = merge(c, (predation = c.baseline + 1.0,))
        @test dominant_mortality_cause(one_cause) == :predation

        # Every counter is live in a real run, and they partition the deaths.
        _, mdf = run_simulation(m, 200)
        death_cols = (:deaths_baseline, :deaths_predation, :deaths_soft_shell,
            :deaths_starvation, :deaths_thermal, :deaths_senescence,
            :deaths_cannibalism, :deaths_fishery)
        cols = Set(propertynames(mdf))
        for col in death_cols
            @test col in cols
        end
        @test sum(mdf[end, col] for col in death_cols) > 0

        # A run with every mortality rate zero must record no deaths, and grow.
        immortal = IBMParameters(mortality = MortalityParams(
            baseline_rate_by_stage = fill(0.0, 24),
        rate_predation = 0.0, rate_soft_shell = 0.0,
            rate_starvation = 0.0, rate_thermal = 0.0, rate_senescence = 0.0,
            rate_cannibalism = 0.0
        ))
        m2 = initialize_model(n_agents = 60, grid = (16, 16), seed = 7, params = immortal)
        _, mdf2 = run_simulation(m2, 200)
        @test sum(mdf2[end, col] for col in death_cols) == 0
        @test mdf2[end, :n_benthic_organisms] >= 60
    end

    @testset "Shell age is one definition, and senescence follows age" begin
        # Regression: mortality senescence used *time since the last ecdysis* while
        # fecundity senescence used inferred shell age. Those are different
        # quantities, so senescence was keyed on molt recency rather than age: a
        # ten-year-old that had molted yesterday had zero mortality senescence, and
        # a one-year-old that last molted 2000 days ago matched the ten-year-old
        # exactly. Because a terminal-molt individual never molts again, its
        # "shell age" also grew without bound and the daily hazard reached 6.6e3
        # before the cap hid it.

        p = IBMParameters()
        m = initialize_model(n_agents = 1, grid = (8, 8), seed = 1, params = p)
        org = first(allagents(m))
        org.is_mature = true
        op = p.organism

        # The documented inference, and it is age-based.
        st_shell = 12
        org.stage = st_shell
        org.age_days = 4000.0
        @test shell_age_years(org, p) ≈
              (4000.0 - op.shell_age_stage_offset *
               (st_shell - op.shell_age_first_stage)) /
              p.simulation.days_per_year
        @test shell_age_years(org, p) >= 1.0     # floored
        org.age_days = 0.0
        @test shell_age_years(org, p) == 1.0     # floored, never negative

        # Fecundity and mortality now read the same value.
        fem = deepcopy(org)
        fem.sex = female
        fem.is_mature = true
        # Terminally moulted: senescence is wear-out on a grown animal, and only a
        # finished one pays it.
        fem.stage = p.organism.max_stage_female
        fem.age_days = 4000.0
        fem.last_growth_event_time = -10.0       # molted just now
        aged = deepcopy(fem)
        aged.last_growth_event_time = -4000.0    # has not molted in 11 years

        # Senescence must depend on age and NOT on molt recency.
        q = IBMParameters(
            baseline_rate_by_stage = fill(0.0, 24),
        rate_predation = 0.0, rate_soft_shell = 0.0,
            rate_starvation = 0.0, rate_thermal = 0.0, rate_cannibalism = 0.0,
            rate_senescence = 0.2, senescence_age_threshold = 3.0
        )
        s_recent = mortality_components(fem, 3.0, 0.0, 10.0; params = q).senescence
        s_stale = mortality_components(aged, 3.0, 0.0, 4000.0; params = q).senescence
        @test s_recent ≈ s_stale                # molt recency is irrelevant
        @test s_recent > 0.0                     # and it is genuinely age-based

        young = deepcopy(fem)
        young.age_days = 200.0
        @test mortality_components(young, 3.0, 0.0, 4000.0; params = q).senescence == 0.0

        # Fecundity senescence uses the same inferred age, so the two agree.
        @test fecundity(fem; params = p, deterministic = true) ==
              fecundity(aged; params = p, deterministic = true)
    end

    @testset "Individuation follows the configured juvenile boundary" begin
        # Regression: the aggregate-to-agent transition tested the literal
        # `next_stage == 6`, so configuring `juvenile_max_stage` to anything else
        # meant no individual was ever individuated and the benthic population
        # silently received no recruits.
        p = IBMParameters()
        @test individuation_stage(p) == p.organism.juvenile_max_stage

        for jms in (5, 6, 7)
            pj = IBMParameters(organism = OrganismParams(juvenile_max_stage = jms))
            @test individuation_stage(pj) == jms
        end

        # Aggregate groups must cover exactly the pre-individuation stages, so the
        # cohort that completes the last one lands on the individuation stage.
        #
        # The default seeds the clutch at the first cohort stage only; a caller that
        # wants one cohort per stage says so in `initial_aggregate_cohort`.
        for jms in (5, 6, 7)
            pj = IBMParameters(organism = OrganismParams(juvenile_max_stage = jms))
            ps = pj.simulation
            filled = fill(0.0, length(ps.initial_aggregate_cohort))
            for stg in FIRST_COHORT_STAGE:(jms - 1)
                filled[stg+1] = 50.0
            end
            seeded = IBMParameters(
                organism = pj.organism,
                simulation = with_overrides(SimulationParams, ps;
                initial_aggregate_cohort = filled))
            m = initialize_model(n_agents = 1, grid = (10, 10), seed = 2, params = seeded)
            groups = Agents.abmproperties(m)[:aggregate_populations].groups
            stages = sort(unique([g.stage for g in groups]))
            @test stages == collect(FIRST_COHORT_STAGE:(jms - 1))
            @test (jms - 1) + 1 == individuation_stage(pj)
        end
    end

    @testset "Runs survive population extinction" begin
        # Regression: `Agents._add_col_data!` calls `agg(generator)` with no `init`,
        # so a bare `sum` threw "reducing over an empty collection" the moment the
        # population reached zero — so a run that ends in extinction used to crash
        # inside data collection instead of recording the final empty state.
        #
        # Reaching extinction is set up here rather than waited for. A capacity of 1
        # alone does not reach it: the density feedback is `1 + (N/K)^exponent`, so
        # as the population thins the mortality multiplier collapses with it and the
        # last few individuals settle into a non-zero equilibrium. The baseline is
        # therefore made certain, which is the state this test is about — an empty
        # population reaching data collection.
        doomed = IBMParameters(
            environment = EnvironmentParams(benthic_carrying_capacity = 1.0),
            simulation = with_overrides(SimulationParams,
                IBMParameters().simulation;
                initial_aggregate_cohort = fill(0.0, 24),
                initial_female_cohort = fill(0.0, 24),
                initial_male_cohort = fill(0.0, 24)),
            mortality = with_overrides(MortalityParams,
                IBMParameters().mortality; baseline_rate_by_stage = fill(1.0e6, 24))
        )
        m = initialize_model(; n_agents = 60, grid = (16, 16), seed = 4,
            params = doomed)
        days = 40
        adf, mdf = run_simulation(m, days)
        nn = mdf.n_benthic_organisms
        @test length(nn) == days + 1          # inclusive of the initial frame
        @test nn[end] == 0
        # The mass column is the one that used to throw; it must be present and
        # zero (not missing) once the population is gone.
        @test :sum_mass in propertynames(adf)
        @test adf[end, :sum_mass] == 0.0
        @test adf[end, :length_stage] == 0.0

        # Directly: the aggregators must be total on an empty input.
        @test safe_sum(Float64[]) == 0.0
        @test safe_mean(Float64[]) == 0.0
        @test safe_sum(1.0:3.0) == 6.0
        @test safe_mean(1.0:3.0) == 2.0
        # A one-shot generator is what Agents actually passes.
        @test safe_mean(x for x in Float64[]) == 0.0
        @test safe_sum(x for x in Float64[]) == 0.0
        @test safe_mean(x for x in 1.0:3.0) == 2.0
    end

    @testset "Maturity ogive criterion compares like with like" begin
        # Regression: the criterion scored a cross-sectional ogive (probability of
        # having *ever* matured) against `maturity`, the probability of maturing at
        # a *single ecdysis*, so it was biased against realistic values by
        # construction.

        sizes = collect(20.0:10.0:140.0)
        cum_m, cum_f = cumulative_maturity_curves(sizes)
        @test length(cum_m) == length(sizes)
        @test length(cum_f) == length(sizes)

        # Monotone in size, and bounded.
        @test all(diff(cum_m) .>= -1e-12)
        @test all(diff(cum_f) .>= -1e-12)
        @test all(0.0 .<= cum_m .<= 1.0)
        @test all(0.0 .<= cum_f .<= 1.0)

        # Strictly above the per-ecdysis curve: an individual at a given size has
        # already ecdysed through every earlier stage.
        pm = [maturity(0, log(s); deterministic = true) for s in sizes]
        @test all(cum_m .>= pm .- 1e-12)
        @test cum_m[end] > pm[end]
        @test cum_m[end] > 0.95

        # Data drawn from the cumulative process must score ~0, and must not have
        # scored ~0 under the old per-ecdysis comparison.
        perfect = (
            bin_centers = sizes,
            prop_mature = cum_m,
            counts = fill(50, length(sizes))
        )
        @test maturity_ogive_mismatch(perfect, male) < 1e-9
        old_sq = [(cum_m[i] - pm[i])^2 for i in eachindex(sizes)]
        @test sqrt(mean(old_sq)) > 0.03

        # Data drawn from the per-ecdysis curve instead must now be penalised.
        single = (
            bin_centers = sizes,
            prop_mature = pm,
            counts = fill(50, length(sizes))
        )
        @test maturity_ogive_mismatch(single, male) > 0.01

        # An empty ogive is "no data", not "perfect fit" and not "total failure".
        empty_ogive = (bin_centers = Float64[], prop_mature = Float64[], counts = Int[])
        @test maturity_ogive_mismatch(empty_ogive, male) == 0.0
        @test isempty(maturity_ogive_residuals(empty_ogive, male))

        # Single-sex form agrees with the two-sex form, for both enum and int.
        @test cumulative_maturity_curves(sizes, male) == cum_m
        @test cumulative_maturity_curves(sizes, 0) == cum_m
        @test cumulative_maturity_curves(sizes, female) == cum_f
    end

    @testset "Habitat Suitability and GeoData Environmental Integration" begin
        # 1. Thermal suitability component
        # Cold optimal temperature
        hsi_opt = calculate_habitat_suitability(1.5, 150.0)
        @test 0.8 < hsi_opt <= 1.0

        # Half-saturation temperature (~5.0 °C)
        hsi_half = calculate_habitat_suitability(5.0, 150.0)
        @test isapprox(hsi_half, 0.5; atol = 0.05)

        # Cutoff temperature: strictly zero at or above 10 °C
        @test calculate_habitat_suitability(10.0, 150.0) == 0.0
        @test calculate_habitat_suitability(12.5, 150.0) == 0.0

        # Sub-zero cold tolerance
        hsi_subzero = calculate_habitat_suitability(-1.0, 150.0)
        @test hsi_subzero > hsi_half

        # 2. Bathymetric depth suitability component
        # Peak at 150 m
        hsi_deep_opt = calculate_habitat_suitability(2.0, 150.0)
        @test hsi_deep_opt > calculate_habitat_suitability(2.0, 75.0)
        @test hsi_deep_opt > calculate_habitat_suitability(2.0, 250.0)

        # Strictly 0 at shallow limit (<= 20 m) and deep limit (>= 400 m)
        @test calculate_habitat_suitability(2.0, 20.0) == 0.0
        @test calculate_habitat_suitability(2.0, 15.0) == 0.0
        @test calculate_habitat_suitability(2.0, 400.0) == 0.0
        @test calculate_habitat_suitability(2.0, 500.0) == 0.0

        # 3. Environmental properties in StandardABM
        m = initialize_model(grid = (20, 20), n_agents = 10, days = 2)
        @test haskey(m.properties, :depth_field)
        @test haskey(m.properties, :temperature_field)
        @test haskey(m.properties, :habitat_quality_field)
        @test size(m.depth_field) == (20, 20)
        @test size(m.temperature_field) == (20, 20)
        @test size(m.habitat_quality_field) == (20, 20)

        # 4. Step simulation and verify environmental field dynamics
        t0 = copy(m.temperature_field)
        model_step!(m)
        @test m.current_time == 1.0
        @test size(m.temperature_field) == (20, 20)
        @test all(0.0 .<= m.habitat_quality_field .<= 1.0)
    end
end
