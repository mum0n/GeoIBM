# Shared imports and test utilities for GeoIBM test suite
using Test
using Random
using Statistics
using DataFrames
using Agents
using LinearAlgebra
using TOML

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
projectdir(args...) = normpath(joinpath(PROJECT_ROOT, args...))
using GeoIBM
