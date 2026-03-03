
  fecundity.estimate = function( O, G, type ) {

    loadfunctions( "snowcrab", functionname="fecundity.allometric" )

    # wrapper to determine fecunity from cw
    if (type == "allometric") {
      fecundity.predicted = floor( fecundity.allometric(O$cw) )
    }
    if (type == "interpolated" ) {
      # inverse sampling from an empiric distribution
      # must be defined after cw.m is defined
      # this does not have to be used if allometric representation is faster
    
      fecundity.approx = approxfun( x=G$cw.m, y=fecundity.allometric( G$cw.m ), method="linear", rule=2 )
      fecundity.predicted = fecundity.approx( O$cw )
    }

    if (type == "detailed.biological" ) {

      # <<<<<<<<<<<<<<<<<< ------------------ MUST ADD more Biology : carapace condition, age, etc

      # inverse sampling from an empirical distribution or modelled ... check speeds

    }


    fecundity.predicted[ which( O$mat==immature ) ] = 0
    fecundity.predicted = floor( fecundity.predicted  )

    return ( fecundity.predicted )

  }



