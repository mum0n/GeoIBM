
  instar.estimate = function( age, sex ) {
    if (sex==male) {
      instar = age - 2
      instar [ which(age == 0) ] = 0
      instar [ which(age == 1) ] = rbinom( length(age[age==1] ), 1, 0.5) + 1
      instar [ which(age == 2) ] = rbinom( length(age[age==2] ), 1, 0.5) + 3
      instar [ which(age > 15) ] = 13
    }
    if (sex==female) {
      instar = age - 2
      instar [ which(age == 0) ] = 0
     uinstar [ which(age == 1) ] = rbinom( length(age[age==1] ), 1, 0.5) + 1
      instar [ which(age == 2) ] = rbinom( length(age[age==2] ), 1, 0.5) + 3
      instar [ which(age >  9) ] = 10
    }

    return (instar)
  }



