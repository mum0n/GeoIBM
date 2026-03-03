
  # snow crab simulation model
  
	loadfunctions( "snowcrab", functionname="initialise.local.environment.r")
	loadfunctions( "model.abm.crab" )

  ppp = get.population.projection.parameters( assessment.year=assessment.year )

  # state-space: N = numbers
  
  nyears = 7
  nvars = length(ppp$nodes) # currently male categories only
  nvars0 = nvars-1

  nx = length(p$plons)
  ny = length(p$plats)
  
  ix = c(1:nx)
  iy = c(1:ny)
  it = c(1:nyears) 
  iv = c(1:nvars)
  iv0 = c(1:nvars0)

  N = array(data=NA, dim=c(nyears, nvars, nx, ny))    # numbers
  B = array(data=NA, dim=c(nyears, nvars, nx, ny))    # biomass
  S = array(data=NA, dim=c(nyears, nvars, nx, ny))    # mean size

  # ----------------
  # Mortality probabilities (annual)
    Z = array(data=NA, dim=c(nyears, nvars, nx, ny ))  # n-1 because the smallest category is a composite "-" group

  # ------------------
  # Fishing (total biomass extraction)
    F = array(data=NA, dim=c(nyears, nvars, nx, ny) )

  # ----------------
  # Transition probabilities (i.e., growth probability) (annual)
    T = array(data=NA, dim=c(nyears, nvars, nx, ny) ) 
      
  # ------------------
  # Movement fields
    angle = 1
    angle.sd = 2
    length = 3
    length.sd =4
    Movement = array(data=NA, dim=c(nx, ny, 4))
   

  # Time loop
    for (y in 2:nyears) {
      y0 = y-1  #the previous year

      # fishing (as number)
      Fishing = Fm[p,,,]/Sm[p,,,] # use the average size of the commerical catch to estimate the number of crab removed

      # estimate natural mortality
      Mm[y0,,,] = Nm[y0,iv0,,] - Nm[y,iv0,,] + X + Y - Fishing
      Mf[y0,,,] = Nf[y0,iv0,,] - Nf[y,iv0,,] + X + Y

      # movement
      Nm[y,,,q]

      # fishing
      Nm[y,,,] = Nm[y0,,,] - Fm[y0,,,]/Sm[y0,,,]  # use the average size of the commerical catch to estimate the number of crab removed
      
      # estimate growth
      Tm[y0,,,] = Nm[y,iv0,,] / (Nm[y0,iv0,,] * Mm[y0,,,])
      Tf[y0,,,] = Nf[y,iv0,,] / (Nf[y0,iv0,,] * Mf[y0,,,])
  
    }






    
