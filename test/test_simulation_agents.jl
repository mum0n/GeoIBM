# Segment: Agent Simulation, Energetics, Mortality, and Dispersal
include("test_fixtures.jl")

@testset "GeoIBM - Simulation and Agents" begin
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

    @testset "Organism Dynamics & Bioenergetics" begin
        rng = MersenneTwister(99)

        # 1. Developmental intermolt and degree-day scaling
        # Same fix as above: a real behaviour and stage numbers, not a `LifeStage`
        # passed where the behaviour belongs.
        b_dyn = SpeciesBehaviorImpl()
        pel_dyn = larval_stage_count(b_dyn)
        @test min_intermolt_days(b_dyn, 1) == min_intermolt_days(b_dyn, 2)
        @test min_intermolt_days(b_dyn, 1) < min_intermolt_days(b_dyn, pel_dyn - 1)
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
            2, (5, 5), juvenile, male, 5, 200.0, 18.0, 2.5, 50.0, 0.0, 950.0,
            false, 0, 0.0, 0
        )
        inhibition_params = IBMParameters(
            phenology = PhenologyParams(
                molt_thermal_onset = 5.0, molt_thermal_bandwidth = 3.0
            )
        )
        p_cold = molt_probability(org_ready, 3.0, 190.0; params = inhibition_params)
        p_warm = molt_probability(org_ready, 7.5, 190.0; params = inhibition_params)
        @test p_cold > 0.0
        @test p_warm < p_cold

        # With the default onset above the seasonal maximum, no inhibition occurs
        # within the normal thermal range, so molting is not shut off mid-season.
        default_params = IBMParameters()
        @test default_params.phenology.molt_thermal_onset >
              default_params.environment.temp_base + default_params.environment.temp_amplitude
        @test molt_probability(org_ready, 7.5, 190.0) == molt_probability(org_ready, 3.0, 190.0)

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

    @testset "Seeding counts follow the benthic-stage mass" begin
        # Regression: the seed denominator summed both cohort vectors over every
        # stage, so vectors dense in the pelagic (aggregate) stages deflated the
        # rescaling — an explicit `n_agents` placed only a small fraction of
        # itself into the benthic cells. The denominator must count only the
        # stages that receive individuated agents; the pelagic entries state
        # where larvae are, not adults to seed.
        ps = IBMParameters()
        jms = ps.organism.juvenile_max_stage
        n_f_default = length(ps.simulation.initial_female_cohort)
        n_m_default = length(ps.simulation.initial_male_cohort)

        # Pelagic entries carry a large mass that must not dilute the seeding;
        # every benthic entry states exactly one organism.
        mixed_f = fill(1.0, n_f_default)
        mixed_m = fill(1.0, n_m_default)
        mixed_f[1:jms] .= 300.0              # stages 0 .. jms-1: pelagic
        mixed_m[1:jms] .= 300.0

        mixed_sim = with_overrides(
            SimulationParams,
            ps.simulation;
            initial_female_cohort = mixed_f,
            initial_male_cohort = mixed_m
        )

        # One organism per benthic cell: choosing `n_agents` as exactly ten times
        # the cell count makes the per-cell draw exact, so the seeded total must
        # equal `n_agents` exactly. Under the old denominator the counts would
        # round to a tiny fraction of the target instead.
        cells = (ps.organism.max_stage_female - jms + 1) +
                (ps.organism.max_stage_male - jms + 1)
        m = initialize_model(n_agents = cells * 10, grid = (10, 10), seed = 5,
            params = IBMParameters(simulation = mixed_sim))
        @test nagents(m) == cells * 10
        @test all(org.stage >= jms for org in allagents(m))

        # The even-spread branch is untouched by the change: with empty vectors
        # there is no distribution to honour and `n_agents` is spread over the
        # cells directly.
        empty_sim = with_overrides(
            SimulationParams,
            ps.simulation;
            initial_female_cohort = fill(0.0, n_f_default),
            initial_male_cohort = fill(0.0, n_m_default)
        )
        m_even = initialize_model(n_agents = 30, grid = (10, 10), seed = 8,
            params = IBMParameters(simulation = empty_sim))
        @test nagents(m_even) == 30
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
end
