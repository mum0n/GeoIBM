
  # Individual-based-model of snow crab

   combine: loc.x, loc.y into 1 index (lookup-table)
   combine: instar, age, 
   combine: cw  ... 
          
   determine growth : empirically using approxfun (linear) or an actual growth functyion + error
   

  # life cycle 
# moult stage (
#   soft: slow movement
#         higher mortality
#   hard: faster movement
#         
#   growth increments (sex and stage dependent) -- cw
#   age increment 
#   maturity (?rule)
#   egg.age increment
#   s
# 
          
transition matrices: 

  mortality (age, moult, sex)
  
  stages: instar


  steps:
  
  1. determine RMS annual movement .. which constrains the spatial scale of a "local" population
  2. for each local population:
      a. estimate rates of mortality (F(fishery data), N(trawl data))
      b. growth increments
  3. combine mean fields of mortality etc for an initial simple model
  4. create area-specific fields of mortality, growth, etc...
  
  


      



