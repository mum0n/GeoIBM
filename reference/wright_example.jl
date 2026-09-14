
using Agents
using Statistics: mean

numagents = 100
mutable struct Haploid <: AbstractAgent
    id::Int
    trait::Float64
end

model = ABM(Haploid)  # no spatial structures
for i in 1:numagents
    add_agent!(model, rand(model.rng))
end

# sample!(model, nagents(model))  # new generation .. startup

modelstep_neutral!(model::ABM) = sample!(model, nagents(model))

# modelstep_selection!(model::ABM) = sample!(model, nagents(model), :trait)  # trait increases pr selection (weight)

# data, _ = run!(model, dummystep, modelstep_neutral!, 20; adata = [(:trait, mean)])
# data

# data, _ = run!(model, dummystep, modelstep_selection!, 20; adata = [(:trait, mean)])
# data

# using JuliaCall from R
# install.packages("JuliaCall")

if  (0  < 0 )
  
  # to do a call from within R ...

  library(JuliaCall)
  # install_julia()

  julia_install_package_if_needed("Optim")
  julia_installed_package("Optim")
  #> [1] "0.22.0"

  # julia_library("Optim")

  julia_source("/home/jae/models/abm/wright_example.jl")
 
  o = julia_eval( "
      data, _ = run!(model, dummystep, modelstep_neutral!, 20; adata = [(:trait, mean)])
  ")

  o = julia_eval( "
      data, _ = run!(model, dummystep, modelstep_selection!, 20; adata = [(:trait, mean)])
  ")

end
