
  # crab ibm prototype

  #################
  # to do: 
  #
  # - move individual state data to storage: direct binary saves or consider a RDBMS (sqlite?, postgres?)
  # - develop fast mechanism to probabilistically apply logistic regression  
  #
  #################
  # parameters

	loadfunctions( "snowcrab", functionname="initialise.local.environment.r")
	loadfunctions( "model.abm.crab" )

  # this is run once to start the generator as it does not use the set.seed() mechanism
  # it is apparently 2X faster than the Mersenne
  # from library SuppDists
    runif.fast( 1, new.start=TRUE, seed=1 )

    set.seed(1)  # seed the random number generator


  ppp = get.population.projection.parameters( assessment.year=assessment.year )


  # Pr( moult )
  #  growth = growth schedule

  # Pr( mortality ) :: size, sex, CC, temp ...
  #  need a mortlaity curve/ survivorship-curve


  # various parameters

    ndays.per.year = 365 # days

    male = 1
    female =2
    immature = 0
    mature =1

    G = NULL
    G$move.velocity = 20  # 65 km / yr 
    G$move.sd = 10
    G$move.type = ".lognormal"  # see function move for more options
    G$cw.l = seq( 0, 180, 10 )  # lower bounds of size categories .. must be in sync
    G$cw.m = ( G$cw.l[ - length(G$cw.l) ] + G$cw.l [-1] ) / 2 # midpoints of size

    G$mortality.f = 0.2  # instataneous mortality, female
    G$mortality.m = 0.2  # instataneous mortality, male
    G$mortality.type = "with.SD.normal.error"
    G$mortality.error = 0.1  # error in instantaneous mortality as a fraction of the mean
    G$age.max.f = 16
    G$age.max.m = 19

    G$time.start = 0  # starting time in fractional years
    G$time.end   = 10 # years or fractions of years
    G$time.step = 1 / ndays.per.year # fraction of year

    G$period.moult = c( 60, 120 ) / ndays.per.year # fraction of year
    G$period.egg.release = c( 60, 160 ) / ndays.per.year # fraction of year
    G$period.recruitment = c( 60, 120 ) / ndays.per.year # fraction of year  .. entry of new recruits
    G$period.fishing = c( 60, 120 ) / ndays.per.year # fraction of year  .. entry of new recruits


  # template for one individual
    newM <- data.frame(age=1, instar=1, cw=1, mat=immature, sex=male, cc=1, xpos=NA, ypos=NA )
    newF <- data.frame(age=1, instar=1, cw=1, mat=immature, sex=female, cc=1, eggs=NA, eggage=1, xpos=NA, ypos=NA)

  # start individuals M=males, F=females
    random.start=T
    if (random.start) {
      oM = random.start.configuration(male, n=10^3, p=p, G=G )
      oF = random.start.configuration(female, n=10^3, p=p, G=G)
    } else {
      # load historical data and parameterise as close as possible to reality
      oM = ..
      oF = ..
    }



# life loop

  sample.n <- NULL
  sample.cw <- NULL
  sample.eggs <- NULL
  sample.agedist <- NULL


  time.start = proc.time()

  .time.step = G$time.step # fraction of a year
  .time = G$time.start  # starting time in fractional years
  .time.end   = G$time.end # in fractional years

  while ( .time < .time.end ) {

    .time = .time + .time.step
    .julian = ( .time - floor(.time) ) * ndays.per.year
    
    print(paste("time step", round(.time )))

    # time increment
    oM$age = oM$age + .time.step
    oF$age = oF$age + .time.step
    oF$eggage <- ifelse( oF$mat, oF$eggage + .time.step, 0)
    
 
    # death -- ~ cw, sex, julian, condition
    oM = mortality( oM, stime, sex=male, G)
    oF = mortality( oF, stime, sex=female, G)

    # reset counts once, so that this does not happen again
    G$ninds.m =  nrow(oM)
    G$ninds.f =  nrow(oF)
   

    # determine movement
    oM = move( oM, sex=male,   G )
    oF = move( oF, sex=female, G )
    
    
    # increment carapace condition 
#    oM = carapace.condition (oM, sex=male, G)
#    oF = carapace.condition (oF, sex=female, G)

    if ( .julian >= G$period.moult[1] & .julian <= G$period.moult[2] ) {
      # growth /moult and maturity as a function of sex, cw, age, julian, condition, etc ... use instar growth function? 
  #    oM = growth( oM. G )  # need the transition matrix as a look-up table ??
  #    oF = growth( oF, G )  
    }


    if ( .julian >= G$period.egg.release[1] & .julian <= G$period.egg.release[2] ) {
      # egg maturation/ production based on eggage and temp/time
#      id.eggsrelease = fecundity.estimate( oF, G )
#      total.egg.production = sum( oF$eggs[id.eggrelease] )  # total number of eggs released
#      oF$eggs[id.eggsrelease] = fecundity.est( oF[ id.egg.release ,] )  #
#      oF$eggage[id.eggsrelease] = 0
    }


    if ( .julian >= G$period.recruitment[1] & .julian <= G$period.recruitment[2] ) {
      # recruitment ... apply mortliaty to total egg production

#      newM = replicate new recruites with  "newcrab" and add some noise to age of organisms , male
#      oM = rbind(oM, newM)
#      rm (newM); gc()

#      newF = replicate new recruites with  "newcrab" and add some noise to age of organisms , female
#      oF = rbind(oF, newF)
#      rm (newF); gc()
    }
   

    if ( .julian >= G$period.fishing[1] & .julian <= G$period.fishing[2] ) {
#      fish the crab
    }


    # timeseries outputs
#    every.ten.days: do
    sample.n <- c(sample.n, nrow(oM))
    sample.cw <- c(sample.cw, mean(oM$cw))

#    every month: do
 #   sample.fb <- c(sample.fb, fishable.biomass( oM ) )
    plot (oM$xpos, oM$ypos, xlim=p$corners$plons, ylim=p$corners$plats )

  }

  time.end = proc.time()
  (time.delta = time.end - time.start)


# results and graphics

  par(mfrow=c(2,2))
  plot(sample.n, xlab="time (d)", ylab="abundance", type="l")
  plot(sample.cw, xlab="time (d)", ylab="mean body length (tm)", type="l")
  hist(sample.eggs, freq=FALSE, breaks=0:10, right=FALSE, ylab="rel. freq.", xlab="egg number", main="")
  time <- seq(1,.time.ends,2)
  boxplot(sample.agedist[time], names=as.character(time), xlab="time (d)", ylab="age distribution (d)")






