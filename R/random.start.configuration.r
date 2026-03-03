 
  random.start.configuration = function( sex, ninds=100, p, G=NULL ) {

    O = NULL

    if (sex==male) {
      O = data.frame( age = random.integer( n=ninds, min=0, max=G$age.max.m) )
      O$instar = instar.estimate ( O$age, sex )
      O$cw = carapacewidth( O$instar, sex="male", type="randomnormal.2" )
      O$mat = maturity.estimate( O$cw, sex, type="model.solutions", subtype=".with.error" )
      O$cc = cc.estimate ( O$age, O$instar, O$mat, sex )
    }
    if (sex==female){
      O = data.frame( age = random.integer( n=ninds, min=0, max=G$age.max.f) )
      O$instar = instar.estimate ( O$age, sex )
      O$cw = carapacewidth( O$instar, sex="female", type="randomnormal.2" )
      O$mat = maturity.estimate( O$cw, sex, type="model.solutions", subtype=".with.error" )
      O$cc = cc.estimate ( O$age, O$instar, O$mat, sex )
      O$fecundity= fecundity.estimate( O, G, type="interpolated" )
      O$eggage = eggage.estimate( n=ninds, method="random" )
    }

    O$xpos = p$plons[ random.integer( n=ninds, min=1, max=p$nplons, type="sample")  ]  # in km
    O$ypos = p$plats[ random.integer( n=ninds, min=1, max=p$nplats, type="sample")  ]

    return (O)
  }


