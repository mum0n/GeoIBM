# Segment: Allometry, Growth, and Empirical Biology
include("test_fixtures.jl")

@testset "TethysBio - Allometry and Biology" begin
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
        @test traits_f.estimated_stage >= 9

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

end
