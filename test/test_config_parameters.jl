# Segment: Configuration, CLI Parsing, and Parameter Validation
include("test_fixtures.jl")

@testset "TethysBio - Configuration and Parameters" begin
    @testset "Configuration File Loading and CLI Overrides" begin
        # 1. Default configuration dictionary
        cfg_def = default_ibm_configuration()
        @test cfg_def isa Dict{Symbol, Any}
        @test cfg_def[:days] == 730
        @test sum(cfg_def[:initial_female_cohort]) +
            sum(cfg_def[:initial_male_cohort]) == 0   # eggs only by default
        @test cfg_def[:grid] == (100, 100)
        @test cfg_def[:temp_offset] == 0.0
        @test cfg_def[:temp_multiplier] == 1.0
        @test cfg_def[:legal_size_male] == 95.0
        @test cfg_def[:benthic_carrying_capacity] == 12000.0

        # 2. Section routing is derived from the parameter structs
        section_map = config_section_map()
        @test section_map[:use_energetics] == "toggles"
        @test section_map[:days] == "simulation"
        @test section_map[:temp_offset] == "environment"
        @test section_map[:temp_multiplier] == "environment"
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
        @test loaded_cfg[:temp_offset] == 0.0
        @test loaded_cfg[:temp_multiplier] == 1.0
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
        @test params.temp_offset == 0.0
        @test params.temp_multiplier == 1.0
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
        @test test_model.params.temp_offset == 0.0
        @test test_model.params.temp_multiplier == 1.0
        @test test_model.params.temp_base == 0.0
        @test test_model.params.maturity_l50_male == 91.8

        # 6. Serialization and round-trip verification
        tmp_cfg_file = joinpath(tempdir(), "test_ibm_config.toml")
        save_ibm_configuration(tmp_cfg_file, species_cfg)
        @test isfile(tmp_cfg_file)
        reloaded_cfg = load_ibm_configuration(tmp_cfg_file)
        @test reloaded_cfg[:days] == species_cfg[:days]
        @test reloaded_cfg[:grid] == species_cfg[:grid]
        @test reloaded_cfg[:temp_offset] ≈ species_cfg[:temp_offset]
        @test reloaded_cfg[:temp_multiplier] ≈ species_cfg[:temp_multiplier]
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
        @test TethysBio.life_stage_index(juvenile) === nothing
        @test TethysBio.life_stage_index(egg) == 0
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
        # Stages 1 and 2 (zoea 1 and 2) have equal duration (36 days), while
        # stage 3 (megalopa, pel - 1) is 42 days, and benthic stages start at pel.
        @test min_intermolt_days(b_sched, 1) == min_intermolt_days(b_sched, 2)
        @test min_intermolt_days(b_sched, 1) < min_intermolt_days(b_sched, pel - 1)
        @test min_intermolt_days(b_sched, pel - 1) < min_intermolt_days(b_sched, pel)
        @test min_intermolt_days(b_sched, pel) == min_intermolt_days(b_sched, pel + 3)
        @test min_intermolt_days(b_sched, pel + 3) < min_intermolt_days(b_sched, pel + 4)

        @test required_degree_days(b_sched, 1) == required_degree_days(b_sched, 2)
        @test required_degree_days(b_sched, 1) == required_degree_days(b_sched, pel - 1)
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

    @testset "Config rescan applies only what changed in the file" begin
        # `configs/snowcrab.toml` declares every parameter, so re-parsing it and
        # rebuilding the parameter tree would re-apply all 260 values annually and
        # discard anything changed in the parameter-steering panel since the last
        # scan. The diff is what makes the scan safe, not merely cheap.
        @testset "flatten / diff" begin
            flat = flatten_toml(Dict("environment" => Dict("temp_offset" => 0.0),
                "simulation" => Dict("grid_nx" => 20)))
            @test flat["environment.temp_offset"] == 0.0
            @test flat["simulation.grid_nx"] == 20

            @test isempty(config_diff(flat, flat))

            # Nested tables are accepted: `config_diff` flattens both sides, so a
            # nested `current` is not reported as one wholly-changed key. Every
            # other key is repeated so that only the one under test moves --
            # omitting one would count as a deletion and be reported separately.
            d = config_diff(flat, Dict("environment" => Dict("temp_offset" => 2.5),
                "simulation" => Dict("grid_nx" => 20)))
            @test length(d) == 1
            @test d[1][1] == "environment.temp_offset"
            @test d[1][2] == (0.0, 2.5)

            # A deleted key is reported rather than silently dropped.
            d2 = config_diff(flat, Dict())
            @test length(d2) == 2
            @test all(p -> last(p[2]) === nothing, d2)
            @test Set(first(p) for p in d2) ==
                Set(["environment.temp_offset", "simulation.grid_nx"])
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
                   temp_offset = 0.0
                   temp_multiplier = 1.0
                   [simulation]
                   grid_nx = 20
                   """)
            fig, state = launch_organism_gui(n_agents = 20, grid = (20, 20), dt = 1.0,
                seed = 3, display_window = false, config_path = path)
            @test state.config_path == path
            # Seeded from the file, so the first scan reports only later edits rather
            # than treating every declared parameter as a change.
            @test state.config_snapshot["environment.temp_offset"] == 0.0
            @test state.config_snapshot["simulation.grid_nx"] == 20

            # No file change: no-op.
            @test isempty(rescan_config!(state))

            # A GUI-side edit must survive a scan that touches something else.
            p0 = abmproperties(state.model)[:params]
            abmproperties(state.model)[:params] = reconstruct_params(p0; temp_multiplier = 2.5)

            write(path, """
                   [environment]
                   temp_offset = 2.5
                   temp_multiplier = 1.0
                   [simulation]
                   grid_nx = 20
                   """)
            msg = rescan_config!(state)
            p1 = abmproperties(state.model)[:params]
            @test occursin("1 applied", msg)
            @test p1.environment.temp_offset == 2.5        # the file edit landed
            @test p1.environment.temp_multiplier == 2.5    # the panel edit survived

            # Structural keys are reported, never applied: they are baked into the ABM.
            write(path, """
                   [environment]
                   temp_offset = 2.5
                   temp_multiplier = 1.0
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
            write(path, "[environment\ntemp_offset = ")
            @test_throws Exception rescan_config!(state)
            @test abmproperties(state.model)[:params].environment.temp_offset == 2.5

            # No path at all is a silent no-op, not an error.
            state.config_path = ""
            @test isempty(rescan_config!(state))
            @test isempty(rescan_config!(state))   # idempotent
        end
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
        @test differing == ["environment.bathymetry_source",
            "environment.benthic_carrying_capacity",
            "environment.use_geodata",
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
            "sum_mass_if_legal_male",
            "count_stage_0", "count_stage_1", "count_stage_2", "count_stage_3", "count_stage_4", "count_stage_5", "count_stage_6", "count_stage_7", "count_stage_8", "count_stage_9", "count_stage_10", "count_stage_11", "count_stage_12", "count_stage_13", "count_stage_14", "count_stage_15", "count_stage_16", "count_stage_17", "count_stage_18", "count_stage_19"
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
        # TethysBio` gave you nothing to import under the exported name.
        undefined_exports = filter(n -> !isdefined(TethysBio, n), names(TethysBio))
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
        for s in TethysBio.FIRST_COHORT_STAGE:(juv - 1)
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
                collect(TethysBio.FIRST_COHORT_STAGE:(p.organism.juvenile_max_stage - 1))
        @test length(pop.groups) ==
                p.organism.juvenile_max_stage - TethysBio.FIRST_COHORT_STAGE

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
        # stage threshold: large males (>= 120 mm) have higher maturity than sub-105 mm males.
        big_males = [o for o in orgs if o.sex === male && o.size >= 120.0]
        mid_males = [o for o in orgs if o.sex === male && o.size < 105.0]
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

end
