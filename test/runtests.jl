"""
Test suite for the model_agent package.
"""

using Test
using Random
using Statistics
using DataFrames
using Agents
using Plots
using LinearAlgebra

# Activate package
using DrWatson
@quickactivate "model_agent"
using model_agent

@testset "model_agent Tests" begin

    @testset "Empirical Bounds and Allometry" begin
        # Monotonicity of lower bounds across stages
        @test llbm(4) < llbm(5) < llbm(6) < llbm(7)
        @test llbf(4) < llbf(5) < llbf(6) < llbf(7)

        # Bounds order: lower bound < upper bound
        @test get_stage_size(5, :male; type=:lb) < get_stage_size(5, :male; type=:ub)
        @test get_stage_size(5, :female; type=:lb) < get_stage_size(5, :female; type=:ub)

        mid_m = get_stage_size(6, :male; type=:mid)
        @test llbm(6) < mid_m < llbm(7)

        # Growth schedule dataframe
        sched = growth_schedule()
        @test isa(sched, DataFrame)
        @test nrow(sched) == 14
        @test all(sched.size_male_lb .< sched.size_male_ub)
        @test all(sched.size_female_lb .< sched.size_female_ub)
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

        # Large crabs should have near 1.0 maturity probability
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
        model = initialize_model(N_agents=20, dims=(20, 20), rng_seed=123)
        @test nagents(model) == 20

        # Verify all agents have valid stage and size
        for agent in allagents(model)
            @test agent.stage in 6:9
            @test agent.size > 0.0
            @test agent.age_days >= 0.0
        end

        # Test molt progression and stage increment
        test_crab = first(allagents(model))
        init_stage = test_crab.stage
        init_size = test_crab.size

        perform_molt!(test_crab, 10.0; rng=rng)
        @test test_crab.stage == init_stage + 1
        @test test_crab.size > init_size
        @test test_crab.last_growth_event_time == 10.0

        # Terminal molt check: mature organisms should not molt
        test_crab.is_mature = true
        @test check_molt_conditions(test_crab, 5.0) == false
    end

    @testset "Simulation Step and Population Summary" begin
        model = initialize_model(N_agents=30, dims=(15, 15), rng_seed=456)
        init_count = nagents(model)
        @test init_count == 30

        # Run 5 daily simulation steps
        for _ in 1:5
            agent_step!(first(allagents(model)), model)
            model_step!(model)
        end

        # Summary statistics calculation
        summary = summarize_population(model)
        @test summary.total_organisms <= 30
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
        model = initialize_model(N_agents=0, dims=(10, 10), rng_seed=789)
        model.current_time = 40.0

        # Add receptive adult female and adult male nearby
        w_f = calculate_mass(65.0, female; mat=1)
        w_m = calculate_mass(110.0, male; mat=1)
        f_crab = add_agent!(
            (5, 5), model, 10, female, 1500.0, 65.0, w_f, 50.0,
            0.0, 1000.0, true, 0, 0.0, 0, true
        )
        m_crab = add_agent!(
            (5, 6), model, 12, male, 1800.0, 110.0, w_m, 80.0,
            0.0, 1000.0, false, 0, 0.0, 0, true
        )

        # Female executes agent_step! -> copulates and extrudes clutch
        agent_step!(f_crab, model)

        @test f_crab.egg_count > 5000
        @test f_crab.hatch_day > model.current_time
        @test model.total_eggs_spawned > 5000
        @test f_crab.is_mating_ready == false

        # Advance time to hatch day
        model.current_time = f_crab.hatch_day + 1.0
        n_groups_before = length(model.aggregate_populations.groups)

        agent_step!(f_crab, model)

        @test f_crab.egg_count == 0
        @test f_crab.is_mating_ready == true
        @test length(model.aggregate_populations.groups) > n_groups_before
        @test model.aggregate_populations.groups[end].stage == 0
        @test model.aggregate_populations.groups[end].abundance > 5000
    end

    @testset "Visualization Tools & Plotting" begin
        model = initialize_model(N_agents = 40, dims = (15, 15), rng_seed = 42)
        adata = [(:stage, length), (:size, safe_mean), (is_adult, count),
                 (is_ovigerous, count)]
        mdata = [:total_eggs_spawned, :total_recruits_settled]

        agent_df, model_df = run_simulation(model, 30; adata = adata, mdata = mdata)

        # 1. Population timeseries
        p_pop = plot_population_timeseries(agent_df; model_df = model_df)
        @test p_pop isa Plots.Plot

        # 2. Stage distribution
        p_inst = plot_stage_distribution(model)
        @test plot_instar_distribution(model) isa Plots.Plot
        @test p_inst isa Plots.Plot

        # 3. Size distribution by sex
        p_size = plot_size_by_sex(model)
        @test p_size isa Plots.Plot

        # 4. Stage allometry
        p_allo = plot_stage_allometry(model)
        @test plot_instar_allometry(model) isa Plots.Plot
        @test p_allo isa Plots.Plot

        # 5. Reproduction dynamics
        p_rep = plot_reproduction_dynamics(agent_df, model_df)
        @test p_rep isa Plots.Plot

        # 6. Spatial habitat
        p_spat = plot_spatial_habitat(model)
        @test p_spat isa Plots.Plot

        # 7. Maturity ogives
        p_mat = plot_maturity_ogives(model)
        @test p_mat isa Plots.Plot

        # 8. Executive simulation dashboard
        p_dash = plot_simulation_dashboard(model, agent_df; model_df = model_df)
        @test p_dash isa Plots.Plot

        # 9. Stock-recruitment diagnostic figure
        p_sr = plot_stock_recruitment(model, agent_df; model_df = model_df)
        @test p_sr isa Plots.Plot

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
        crab_juv = Organism(
            1, (5, 5), 6, male, SnowCrabBehavior(), 400.0, 28.0,
            calculate_mass(28.0, male; mat = 0),
            50.0, 0.0, 500.0, false, 0, 0.0, 0, false
        )
        # Degree days = 500 < 550 for stage 6 -> cannot molt yet
        @test check_molt_conditions(crab_juv, 4.0, 50.0; params = p) == false

        # Accumulate degree days past threshold (550), but intermolt duration < 150 d
        crab_juv.degree_days = 600.0
        crab_juv.last_growth_event_time = 0.0
        @test check_molt_conditions(crab_juv, 4.0, 50.0; params = p) == false

        # Current time = 160 >= 150 d intermolt -> eligible to molt
        @test check_molt_conditions(crab_juv, 4.0, 160.0; params = p) == true

        # Perform molt and verify degree-day reset and mass recalculation
        init_wt = crab_juv.mass
        perform_molt!(crab_juv, 160.0; rng = MersenneTwister(42))
        @test crab_juv.degree_days == 0.0
        @test crab_juv.mass > init_wt
        @test crab_juv.stage == 7

        # 3. Molt seasonality for stages >= 7
        # Inside spring window (day 40, within [15, 140]), having molted in previous year
        crab_juv.degree_days = 1000.0
        crab_juv.energy_reserve = 100.0
        crab_juv.last_growth_event_time = -300.0
        @test check_molt_conditions(crab_juv, 4.0, 40.0; params = p) == true

        # Outside spring window (day 200, summer/autumn) -> blocked by seasonality
        @test check_molt_conditions(crab_juv, 4.0, 200.0; params = p) == false

        # 4. Stock Biomass calculation
        model = initialize_model(N_agents = 50, dims = (20, 20), rng_seed = 999)
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
        model = initialize_model(N_agents = 20, dims = (10, 10), rng_seed = 111)
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
            year = 1:length(s_vals)
        )
        p_sr = plot_stock_recruitment(sr_df; fits = fits)
        @test p_sr isa Plots.Plot
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
        c_juv = SnowCrab(
            1, (1, 1), juvenile, male, 6, 200.0, 30.0, 10.0, 60.0,
            0.0, 600.0, false, 0, 0.0, 0
        )
        p_m = molt_probability(c_juv, 4.0, 160.0; params = test_params)
        @test p_m > 0.0
        @test p_m <= 1.0

        # Deterministic check vs probabilistic trial
        @test check_molt_conditions(c_juv, 4.0, 160.0; params = test_params) == true
        molt_trials = [
            check_molt_conditions(c_juv, 4.0, 160.0; params = test_params, rng = rng)
            for _ in 1:100
        ]
        @test any(molt_trials)
        @test !all(molt_trials)

        # Ineligible when intermolt duration is insufficient
        @test molt_probability(c_juv, 4.0, 50.0; params = test_params) == 0.0

        # 3. Predicates for legal fishable crabs
        c_legal = SnowCrab(
            2, (2, 2), adult, male, 12, 1200.0, 102.0, 650.0, 30.0,
            0.0, 0.0, true, 0, 0.0, 0
        )
        c_sublegal = SnowCrab(
            3, (3, 3), adult, male, 10, 800.0, 88.0, 320.0, 25.0,
            0.0, 0.0, true, 0, 0.0, 0
        )
        c_female = SnowCrab(
            4, (4, 4), adult, female, 10, 800.0, 96.0, 280.0, 20.0,
            0.0, 0.0, true, 0, 0.0, 0
        )
        @test is_legal_male(c_legal) == true
        @test is_legal(c_legal) == true
        @test is_legal_male(c_sublegal) == false
        @test is_legal_male(c_female) == false

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
            year = collect(1:10),
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
        @test p_macro isa Plots.Plot

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
        @test p_retro isa Plots.Plot

        rm(tmp_csv, force = true)
    end

    @testset "Generalized Leslie Matrix & Macro Model Comparison" begin
        rng = MersenneTwister(999)

        # 1. Empirical Stages definition
        stages = build_stages()
        @test length(stages) == 15
        @test stages[1].name == "Stage 4 (Recruit)"
        @test stages[1].sex === undifferentiated
        @test stages[2].name == "Stage 5 (Juvenile)"
        @test stages[2].sex === undifferentiated
        @test stages[3].name == "Stage 6 Male"
        @test stages[3].sex === male
        @test stages[10].name == "Stage 6 Female"
        @test stages[10].sex === female

        # Legal males: Size >= 95 mm (Stage 11 and 12+ Male, stages 8 & 9)
        for st in stages
            if st.sex === male && st.stage >= 11
                @test st.is_legal_male == true
            else
                @test st.is_legal_male == false
            end
        end

        # Reproductive maturity flags (Mature females: Stage 9, 10, 11+, stages 13..15)
        for st in stages
            if st.sex === female && st.stage >= 9
                @test st.is_reproductive == true
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
            year = collect(1:8),
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
        @test p_leslie isa Plots.Plot

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
        @test p_comp isa Plots.Plot
    end

    @testset "Configuration File Loading and CLI Overrides" begin
        # 1. Default configuration dictionary
        cfg_def = default_ibm_configuration()
        @test cfg_def isa Dict{Symbol, Any}
        @test cfg_def[:days] == 730
        @test cfg_def[:n_agents] == 300
        @test cfg_def[:grid] == (100, 100)
        @test cfg_def[:temp_base] == 5.0
        @test cfg_def[:temp_amplitude] == 4.0
        @test cfg_def[:legal_size_male] == 95.0
        @test cfg_def[:benthic_carrying_capacity] == 300.0

        # 2. Loading snowcrab_ibm.config from file
        config_path = isfile(projectdir("snowcrab_ibm.config")) ?
            projectdir("snowcrab_ibm.config") : projectdir("scripts", "snowcrab_ibm.config")
        @test isfile(config_path)
        loaded_cfg = load_ibm_configuration(config_path)
        @test loaded_cfg[:days] == 730
        @test loaded_cfg[:n_agents] == 300
        @test loaded_cfg[:grid] == (100, 100)
        @test loaded_cfg[:temp_base] == 5.0
        @test loaded_cfg[:temp_amplitude] == 4.0
        @test loaded_cfg[:mating_search_radius] == 5
        @test loaded_cfg[:egg_survival_base] ≈ 5.0e-5

        # 3. Constructing IBMParameters from loaded configuration
        params = IBMParameters(loaded_cfg)
        @test params isa IBMParameters
        @test params.temp_base == 5.0
        @test params.temp_amplitude == 4.0
        @test params.legal_size_male == 95.0
        @test params.benthic_carrying_capacity == 300.0
        @test params.mating_search_radius == 5
        @test params.egg_survival_base ≈ 5.0e-5

        # 4. Initialize model directly from configuration dictionary
        test_model = initialize_model(loaded_cfg; N_agents = 10, days = 5)
        @test nagents(test_model) == 10
        @test test_model.params.temp_base == 5.0

        # 5. Serialization and round-trip verification
        tmp_cfg_file = joinpath(tempdir(), "test_snowcrab_ibm.config")
        save_ibm_configuration(tmp_cfg_file, loaded_cfg)
        @test isfile(tmp_cfg_file)
        reloaded_cfg = load_ibm_configuration(tmp_cfg_file)
        @test reloaded_cfg[:days] == loaded_cfg[:days]
        @test reloaded_cfg[:grid] == loaded_cfg[:grid]
        @test reloaded_cfg[:temp_base] ≈ loaded_cfg[:temp_base]
        @test reloaded_cfg[:legal_size_male] ≈ loaded_cfg[:legal_size_male]
        rm(tmp_cfg_file, force=true)

        # 6. CLI '=' argument syntax parsing
        include(projectdir("scripts", "organism_ibm.jl"))
        cli_parsed = parse_cli_args([
            "--config=snowcrab_ibm.config",
            "--temp_base=3.5",
            "--days=100",
            "--grid=80,80",
            "--n_agents=50",
            "--fishing_prob=0.03"
        ])
        @test cli_parsed[:temp_base] == 3.5
        @test cli_parsed[:days] == 100
        @test cli_parsed[:grid] == (80, 80)
        @test cli_parsed[:n_agents] == 50
        @test cli_parsed[:fishing_prob] == 0.03
        @test cli_parsed[:fishing_probability_daily] == 0.03
    end

    @testset "Enhanced Simulation Output Metrics" begin
        # 1. Weight filter functions
        crab_f_ad = SnowCrab(1, (1, 1), adult, female, 10, 500.0, 65.0, 150.0, 50.0, 0.0, 100.0, false, 0, 0.0, 0)
        crab_m_ad = SnowCrab(2, (1, 1), adult, male, 11, 600.0, 96.0, 450.0, 60.0, 0.0, 100.0, false, 0, 0.0, 0)
        crab_m_juv = SnowCrab(3, (1, 1), juvenile, male, 7, 200.0, 40.0, 30.0, 20.0, 0.0, 50.0, false, 0, 0.0, 0)

        @test weight_if_adult_female(crab_f_ad) == 150.0
        @test weight_if_adult_female(crab_m_ad) == 0.0
        @test weight_if_adult_male(crab_m_ad) == 450.0
        @test weight_if_adult_male(crab_m_juv) == 0.0
        @test weight_if_legal_male(crab_m_ad) == 450.0
        @test weight_if_legal_male(crab_m_juv) == 0.0

        # 2. summarize_population indicators
        m = initialize_model(N_agents = 20, dims = (20, 20), rng_seed = 42)
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
        cli_parsed2 = parse_cli_args(["--snapshot_day=260.0", "--survey_day=260.0"])
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

        # CLI --frequency argument parsing
        cli_parsed_freq = parse_cli_args(["--frequency=90.0"])
        @test cli_parsed_freq[:frequency_days] == 90.0
        @test cli_parsed_freq[:output_frequency] == 90.0
    end

    @testset "Multi-Attribute Criteria Evaluation & Orchestrator" begin
        # 1. Evaluate population criteria on a running model
        m_test = initialize_model(N_agents = 25, dims = (20, 20), rng_seed = 99)
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
        test_juv_m = SnowCrab(
            1, (5, 5), juvenile, male, 5, 100.0, 15.0, 1.5, 20.0, 0.0, 50.0,
            false, 0, 0.0, 0
        )
        test_juv_f = SnowCrab(
            2, (5, 5), juvenile, female, 5, 100.0, 15.0, 1.5, 20.0, 0.0, 50.0,
            false, 0, 0.0, 0
        )
        test_ovig = SnowCrab(
            3, (5, 5), adult, female, 10, 800.0, 75.0, 120.0, 40.0, 0.0, 150.0,
            false, 50000, 365.0, 1
        )
        test_legal = SnowCrab(
            4, (5, 5), adult, male, 12, 1200.0, 105.0, 450.0, 60.0, 0.0, 200.0,
            false, 0, 0.0, 0
        )
        test_subleg = SnowCrab(
            5, (5, 5), adult, male, 10, 900.0, 80.0, 180.0, 40.0, 0.0, 160.0,
            false, 0, 0.0, 0
        )
        test_larva = SnowCrab(
            6, (5, 5), zoea, undifferentiated, 1, 10.0, 1.5, 0.01, 5.0, 0.0, 10.0,
            false, 0, 0.0, 0
        )

        @test crab_color(test_juv_m) == "#56B4E9"
        @test crab_color(test_juv_f) == "#CC79A7"
        @test crab_color(test_ovig) == "#D55E00"
        @test crab_color(test_legal) == "#003B73"
        @test crab_color(test_subleg) == "#0072B2"
        @test crab_color(test_larva) == "#E69F00"

        @test crab_marker(test_juv_m) === :rect
        @test crab_marker(test_juv_f) === :circle
        @test crab_marker(test_larva) === :diamond

        @test crab_size(test_larva) >= 4.0
        @test crab_size(test_legal) > crab_size(test_juv_m)

        # 2. Environmental heatarray extraction
        model_test = initialize_model(N_agents = 10, dims = (15, 15), rng_seed = 99)
        t_mat = crab_heatarray(model_test, :temperature_field)
        @test size(t_mat) == (15, 15)
        f_mat = crab_heatarray(model_test, :food_availability_field)
        @test size(f_mat) == (15, 15)

        # 3. Spatial summary plot
        out_summary = joinpath(tempdir(), "test_abm_summary_$(rand(1000:9999)).png")
        fig = plot_abm_summary(model_test; output_path = out_summary)
        @test isfile(out_summary)
        @test filesize(out_summary) > 0
        rm(out_summary; force = true)

        # 4. Animation video generation
        out_anim = joinpath(tempdir(), "test_anim_$(rand(1000:9999)).mp4")
        anim_path = animate_snowcrab_simulation(
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
            1, (5, 5), 10, female, SnowCrabBehavior(), 800.0, 75.0, 120.0, 40.0, 0.0, 150.0,
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

        # 7. Sample crab traits from empirical model
        traits_m = sample_organism_traits(
            emp_model;
            size = 100.0,
            sex = :male,
            temp = 3.5,
            rng = rng
        )
        @test traits_m.mass > 300.0
        @test traits_m.fecundity == 0.0
        @test traits_m.estimated_stage in (11, 12)
        
        traits_f = sample_organism_traits(
            emp_model;
            size = 70.0,
            sex = :female,
            temp = 3.5,
            shell_age = 1.0,
            rng = rng
        )
        @test traits_f.mass > 50.0
        @test traits_f.estimated_stage in (9, 10, 11)
        
        # 8. Model summary display
        io_buf = IOBuffer()
        summarize_empirical_model(emp_model; io = io_buf)
        summary_str = String(take!(io_buf))
        @test occursin("Empirical Biological Model Summary", summary_str)
        @test occursin("Allometric Mass Parameters", summary_str)
        @test occursin("Trait Covariance Matrix", summary_str)
    end

    @testset "Enhanced Crab Dynamics & Bioenergetics" begin
        rng = MersenneTwister(99)

        # 1. Developmental intermolt and degree-day scaling
        @test min_intermolt_days(zoea, 1) < min_intermolt_days(megalopa, 3)
        @test min_intermolt_days(megalopa, 3) < min_intermolt_days(juvenile, 4)
        @test min_intermolt_days(juvenile, 4) < min_intermolt_days(juvenile, 6)
        @test min_intermolt_days(juvenile, 6) < min_intermolt_days(juvenile, 8)

        @test required_degree_days(zoea, 1) < required_degree_days(megalopa, 3)
        @test required_degree_days(megalopa, 3) < required_degree_days(juvenile, 4)
        @test required_degree_days(juvenile, 4) < required_degree_days(juvenile, 6)
        @test required_degree_days(juvenile, 6) < required_degree_days(juvenile, 8)

        # 2. Shell condition index (1 through 5)
        mock_c = SnowCrab(
            1, (5, 5), adult, male, 11, 1000.0, 95.0, 350.0, 50.0, 1000.0, 0.0,
            false, 0, 0.0, 0
        )
        @test shell_condition(mock_c, 1010.0) == 1 # 10 days post-molt -> Soft shell
        @test shell_condition(mock_c, 1100.0) == 2 # 100 days post-molt -> New hard clean shell
        @test shell_condition(mock_c, 1500.0) == 3 # 500 days (~1.4 yr) -> Intermediate shell
        @test shell_condition(mock_c, 1900.0) == 4 # 900 days (~2.5 yr) -> Old shell
        @test shell_condition(mock_c, 2500.0) == 5 # 1500 days (~4.1 yr) -> Senescent shell

        # 3. Bioenergetics and functional response
        dE_low, intake_low = calculate_energy_change(mock_c, 3.0, 1.0)
        dE_high, intake_high = calculate_energy_change(mock_c, 3.0, 10.0)
        @test intake_high > intake_low
        @test dE_high > dE_low

        # Saturating functional response: intake at food=50 vs food=100 should be near asymptote
        _, intake_50 = calculate_energy_change(mock_c, 3.0, 50.0)
        _, intake_100 = calculate_energy_change(mock_c, 3.0, 100.0)
        @test intake_100 / intake_50 < 1.10

        # Scalar vs SnowCrab dispatch consistency
        dE_scalar, intake_scalar = calculate_energy_change(
            3.0, 5.0, 95.0; mass = 350.0
        )
        dE_agent, intake_agent = calculate_energy_change(mock_c, 3.0, 5.0)
        @test dE_scalar ≈ dE_agent
        @test intake_scalar ≈ intake_agent

        # 4. Thermal ecdysial inhibition
        juv_ready = SnowCrab(
            2, (5, 5), juvenile, male, 5, 200.0, 18.0, 2.5, 50.0, 50.0, 400.0,
            false, 0, 0.0, 0
        )
        p_cold = molt_probability(juv_ready, 3.0, 180.0)
        p_warm = molt_probability(juv_ready, 7.5, 180.0)
        @test p_cold > 0.0
        @test p_warm < p_cold

        # 5. Multi-component mortality: soft shell vulnerability & adult senescence
        soft_crab = SnowCrab(
            3, (5, 5), juvenile, male, 6, 205.0, 20.0, 3.5, 10.0, 200.0, 10.0,
            false, 0, 0.0, 0
        )
        hard_crab = SnowCrab(
            4, (5, 5), juvenile, male, 6, 260.0, 20.0, 3.5, 10.0, 150.0, 100.0,
            false, 0, 0.0, 0
        )
        m_soft = calculate_mortality_probability(soft_crab, 3.0, 5.0, 205.0)
        m_hard = calculate_mortality_probability(hard_crab, 3.0, 5.0, 260.0)
        @test m_soft > m_hard

        young_adult = SnowCrab(
            5, (5, 5), adult, male, 12, 1200.0, 105.0, 450.0, 60.0, 1000.0, 100.0,
            false, 0, 0.0, 0
        )
        old_adult = SnowCrab(
            6, (5, 5), adult, male, 12, 2800.0, 105.0, 450.0, 60.0, 1000.0, 100.0,
            false, 0, 0.0, 0
        )
        m_young = calculate_mortality_probability(young_adult, 3.0, 5.0, 1200.0)
        m_old   = calculate_mortality_probability(old_adult, 3.0, 5.0, 2800.0)
        @test m_old > m_young

        # 6. Temperature-coupled perform_molt!
        test_juv = Organism(
            7, (5, 5), juvenile, male, 5, 200.0, 18.0, 2.5, 25.0, 50.0, 400.0,
            false, 0, 0.0, 0
        )
        perform_molt!(test_juv, 200.0, 3.0; rng = rng)
        @test test_juv.stage == 6
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

        m_dict = initialize_model(base_cfg; N_agents = 15)
        @test nagents(m_dict) == 15
        @test haskey(abmproperties(m_dict), :n_benthic_crabs)
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
        m_toml = initialize_model(tmp_cfg_file; N_agents = 10)
        @test nagents(m_toml) == 10
        rm(tmp_cfg_file; force = true)

        # 2. Demographic caching in model_step!
        model = initialize_model(N_agents = 0, dims = (10, 10), rng_seed = 999)
        @test model.n_benthic_crabs == 0
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
        @test model.n_benthic_crabs == 2
        @test model.n_mature_males == 1
        @test model.n_mature_females == 1
        @test model.n_legal_males == 1

        # 3. crab_disperse! behavior
        # Pelagic zoea dispersion
        zoea_crab = add_agent!(
            (5, 5), model, zoea, female, 1, 10.0, 1.0, 0.01, 5.0,
            0.0, 10.0, false, 0, 0.0, 0
        )
        pos_before = zoea_crab.pos
        crab_disperse!(zoea_crab, model)
        # Position after randomwalk should be a valid grid coordinate
        @test 1 <= zoea_crab.pos[1] <= 10
        @test 1 <= zoea_crab.pos[2] <= 10

        # Starved benthic crab lethargy (mostly remains in place)
        starved_crab = add_agent!(
            (2, 2), model, juvenile, male, 5, 200.0, 18.0, 2.5, -5.0,
            50.0, 100.0, false, 0, 0.0, 0
        )
        stationary_count = 0
        for _ in 1:20
            old_p = starved_crab.pos
            crab_disperse!(starved_crab, model)
            if starved_crab.pos == old_p
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
            crab_disperse!(forager, model)
            if forager.pos == (5, 6)
                moved_to_optimal += 1
            end
        end
        # The optimal cell should be visited significantly more than random uniform 1/9 (~3)
        @test moved_to_optimal >= 5

        # 4. Male-size scaled sperm clutches & ovigerous female mass (complex model)
        m_mate = initialize_model(
            N_agents = 0, dims = (10, 10), rng_seed = 123, model_mode = :complex
        )
        m_mate.current_time = 45.0  # Spring mating season

        # Test sub-legal male (Size = 80 mm < 95 mm) -> confers 2 clutches
        sub_male = add_agent!(
            (5, 5), m_mate, 11, male, 1200.0, 80.0, 200.0, 50.0,
            0.0, 800.0, false, 0, 0.0, 0, true
        )
        fem_crab = add_agent!(
            (5, 6), m_mate, 10, female, 1200.0, 65.0,
            calculate_mass(65.0, female; mat = 1), 50.0, 0.0, 800.0, true, 0, 0.0, 0, true
        )
        base_somatic_w = fem_crab.mass
        agent_step!(fem_crab, m_mate)
        # Copulated with sub-legal male -> 2 clutches, then extruded 1 -> 1 clutch left
        @test fem_crab.sperm_clutches == 1
        @test fem_crab.egg_count >= 1000
        @test fem_crab.mass > base_somatic_w # Includes egg clutch mass

        # Test legal male (Size = 110 mm >= 95 mm) -> confers 3 clutches
        m_mate2 = initialize_model(
            N_agents = 0, dims = (10, 10), rng_seed = 123
        )
        m_mate2.current_time = 45.0
        leg_male = add_agent!(
            (5, 5), m_mate2, 12, male, 1400.0, 110.0, 450.0, 50.0,
            0.0, 800.0, false, 0, 0.0, 0, true
        )
        fem_crab2 = add_agent!(
            (5, 6), m_mate2, 10, female, 1200.0, 65.0,
            calculate_mass(65.0, female; mat = 1), 50.0, 0.0, 800.0, true, 0, 0.0, 0, true
        )
        agent_step!(fem_crab2, m_mate2)
        # Copulated with legal male -> 3 clutches, then extruded 1 -> 2 clutches left
        @test fem_crab2.sperm_clutches == 2
        @test fem_crab2.egg_count >= 1000
        @test fem_crab2.mass > base_somatic_w

        # 5. Process Toggles and Probabilistic Aggregate Dynamics
        p_toggles = IBMParameters(
            toggles = ProcessToggles(
                use_movement = false,
                use_degree_days = false,
                use_spermatheca = false
            )
        )
        m_toggled = initialize_model(params = p_toggles, N_agents = 0, dims = (10, 10), rng_seed = 456)
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
        agg_model = initialize_model(N_agents = 0, dims = (10, 10), rng_seed = 777)
        # Seed an aggregate group in stage 5 (near individuation to stage 6)
        grp5 = AggregateGroup(
            id = 99,
            stage = 5,
            abundance = 50.0,
            size = 15.0,
            mass = 1.5,
            age_days = 1400.0,
            degree_days = 800.0, # Exceeds required_degree_days(5) = 350
            last_growth_time = 0.0,
            pos = (5, 5)
        )
        push!(agg_model.aggregate_populations.groups, grp5)

        # Step aggregate groups
        initial_org_count = nagents(agg_model)
        for _ in 1:15
            step_aggregate_groups!(agg_model)
        end

        # Individuals should have probabilistically individuated into stage 6 Organism agents
        @test nagents(agg_model) > initial_org_count
        for org in allagents(agg_model)
            @test org.stage == 6
            @test org.size > 0.0
            @test org.mass > 0.0
            @test org.sex in (male, female)
        end

        # Model step executes smoothly and refreshes demographic cache
        model_step!(m_toggled)
        @test m_toggled.n_benthic_organisms == nagents(m_toggled)
    end

end




