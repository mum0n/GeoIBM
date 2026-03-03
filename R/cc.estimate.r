  
  cc.estimate = function( age, instar, mat, sex ) {

    cc = rep(1, length( age ) ) # default to CC 1

    if (sex==male) {
      # cc3 represents 3 and 4:
      # there is a 2 yr difference between instar and age for a newly moulted individual (cc12)
      # assuming 1 yr period for cc1/2, then those that have an age greater than this would be considered in stage cc34
      # the upper limit being another ~ 3 to 4 yrs?  .. need more data here    <<<--------------------------------------
      cc34 = which( mat==mature & instar %in% c(9,10,11,12,13) &  (age-instar) %in% c(3,4,5) )
      cc5  = which( mat==mature & instar %in% c(9,10,11,12,13) &  (age-instar) > 5 )
    }

    if (sex==female) {
      # cc3 represents 3 and 4:
      # there is a 2 yr difference between instar and age for a newly moulted individual (cc12)
      # assuming 1 yr period for cc1/2, then those that have an age greater than this would be considered in stage cc34
      # the upper limit being another ~ 3 to 4 yrs?  .. need more data here    <<<--------------------------------------
      cc34 = which( mat==mature && instar %in% c(8,9,10,11) &&  (age-instar) %in% c(3,4,5) )
      cc5  = which( mat==mature && instar %in% c(8,9,10,11) &&  (age-instar) > 5 )
    }
    cc[ cc34 ] = 3
    cc[ cc5  ] = 5

    return (cc)
  }


