
  move = function( O, sex=NULL, G) {

    if (sex==male) {
      ninds = G$ninds.m
    }

    if (sex==female) {
      ninds = G$ninds.f
    }

    velocity.per.year = switch(G$move.type ,
      .uniform = G$move.velocity * runif( ninds ) ,
      .poisson = rpois( n=ninds, lambda=G$move.velocity ) ,
      .lognormal = rlnorm( n=ninds, meanlog=log(G$move.velocity), sdlog=log(G$move.sd) ) ,
      .normal = rnorm( n=ninds, mean=G$move.velocity, sd=G$sd )
    )
    
    distance = velocity.per.year * G$time.step  # in km
    distance[ which(distance<0) ] = 0

    angle = runif( ninds, min=0, max=2*pi )
    O$xpos = O$xpos + distance * cos(angle)
    O$ypos = O$ypos + distance * sin(angle)
    
    O$xpos [ which( O$xpos< p$corners$plon[1] )] = p$corners$plon[1]
    O$xpos [ which( O$xpos> p$corners$plon[2] )] = p$corners$plon[2]
    O$ypos [ which( O$ypos< p$corners$plat[1] )] = p$corners$plat[1]
    O$ypos [ which( O$ypos> p$corners$plat[2] )] = p$corners$plat[2]
    
    return( O )
  }


