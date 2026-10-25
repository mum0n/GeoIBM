# Segment: Habitat Suitability and Tethys Environmental Integration
include("test_fixtures.jl")

@testset "TethysBio - Habitat and Environment" begin
    @testset "Habitat suitability offset and multiplier" begin
        # In synthetic mode (use_geodata == false), habitat_offset acts as mean
        # and habitat_multiplier acts as amplitude scale on noise.
        p_def = EnvironmentParams()
        @test p_def.habitat_offset == 0.0
        @test p_def.habitat_multiplier == 1.0

        p0 = EnvironmentParams(habitat_offset = 0.0, habitat_multiplier = 1.0)
        expected = clamp.(randn(Xoshiro(4), (20, 20)) .* p0.habitat_quality_sd .+
                          p0.habitat_quality_mean, 0.0, 1.0)
        @test build_habitat_quality_field(p0, (20, 20), Xoshiro(4)) == expected

        # Non-zero offset acts as mean
        p_off = EnvironmentParams(habitat_offset = 0.8, habitat_multiplier = 0.0)
        f_off = build_habitat_quality_field(p_off, (20, 20), Xoshiro(4))
        @test all(f_off .≈ 0.8)

        # Multiplier scales noise amplitude
        p_m2 = EnvironmentParams(habitat_offset = 0.5, habitat_multiplier = 2.0)
        f_m2 = build_habitat_quality_field(p_m2, (50, 50), Xoshiro(4))
        @test isapprox(std(f_m2), 2.0 * p_m2.habitat_quality_sd, atol = 0.05)
    end

    @testset "Habitat Suitability and Tethys Environmental Integration" begin
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
        props = Agents.abmproperties(m)
        @test haskey(props, :depth_field)
        @test haskey(props, :temperature_field)
        @test haskey(props, :habitat_quality_field)
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
