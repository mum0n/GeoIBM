
  mortality = function( O, stime, sex=NULL, G ) {
    
    if ( sex==female ) {
      maxage = G$age.max.f
      rate = G$mortality.f
    }
    if ( sex==male ) {
      maxage = G$age.max.f
      rate = G$mortality.m
    }

    SD = G$mortality.error * rate

    R = switch( G$mortality.type,
      with.SD.normal.error = rnorm( 1, mean=rate, sd=SD ),  # error is a fraction of mean 
      default = rate
    )

    n.O = nrow(O)
    pr.dead = exp(R)* G$time.step  
    n.survival = round( n.O * (1-pr.dead),0 ) 
    O = O[ order( sample( n.O, n.survival )) ,]
    O = subset( O, age > maxage )  # remove the very old
    
    return(O)
   
  }


