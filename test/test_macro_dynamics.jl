# Segment: Macro Population Dynamics and Leslie Matrices
include("test_fixtures.jl")

@testset "GeoIBM - Macro Dynamics" begin
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
end
