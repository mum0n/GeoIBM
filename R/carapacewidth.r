 
  carapacewidth = function(instar, sex, type="lb") {

    # lower bound or upper bound in cw (mm)
    # based upon Scotian Shelf data visual assessment / kernel density estimates
    # prior to instar 5 moulting can be variable and so should not extrapolated carelessly
    if (type=="lb") {
      out = switch( sex,
        male = exp(1.917564 + 0.298914*(instar-3) ),
        female = exp(2.198848 + 0.315026*(instar-4 ) )
      )
    }
    if (type=="ub") {
      out = switch( sex,
        male = exp(1.917564 + 0.298914*(instar-2) ),
        female = exp(2.198848 + 0.315026*(instar-3 ) )
      )
    }
    if (type=="mid") {
      out = ( carapacewidth(instar, sex, "lb") + carapacewidth(instar, sex, "ub") ) / 2
    }

    if (type=="randomuniform") {
      out = sapply( instar, FUN=function(x,...){ runif( 1, min=carapacewidth(x, sex, "lb"), max=carapacewidth(x, sex, "ub") ) }, sex )
    }

    if (type=="randomnormal.1") { # range ~ +/- 1 SD
      out = sapply( instar, FUN=function(x,...){ rnorm( 1, mean=carapacewidth(x, sex, "mid"), sd=(carapacewidth(x, sex, "ub") - carapacewidth(x, sex, "lb")  ) / 2 ) }, sex )
    }

    if (type=="randomnormal.2") { # range ~ +/- 2 SD
      out = sapply( instar, FUN=function(x,...){ rnorm( 1, mean=carapacewidth(x, sex, "mid"), sd=(carapacewidth(x, sex, "ub") - carapacewidth(x, sex, "lb")  ) / 4 ) }, sex )
    }

    if (type=="randomnormal.3") { # range ~ +/- 3 SD
      out = sapply( instar, FUN=function(x,...){ rnorm( 1, mean=carapacewidth(x, sex, "mid"), sd=(carapacewidth(x, sex, "ub") - carapacewidth(x, sex, "lb")  ) / 6 ) }, sex )
    }

    return(out)
  }



