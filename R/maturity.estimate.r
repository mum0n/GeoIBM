
  maturity.estimate = function( cw, sex, type, subtype, p=NULL ) {

    if ( type == "interpolated" ) {
      # Pr (maturity) :: currently only by sex and cw, should also add CC, temp, etc?
      # created in 9.analysis.maturity.r
      maturity.schedule = read.table( file.path(R.sc, "maturity.predictions.csv"), sep=";", header=T)

      if (sex ==male) {
        var.name = "se.m"
        maturity.interpolate = approxfun( x=maturity.schedule[, "cw"], y=maturity.schedule[, "maturity.m"], method="linear", rule=2)
      }
      if (sex ==female) {
        var.name = "se.f"
        maturity.interpolate = approxfun( x=maturity.schedule[, "cw"], y=maturity.schedule[, "maturity.f"], method="linear", rule=2)
      }

      out = switch( subtype,
       .no.error = maturity.interpolate(cw) ,
       .with.error = maturity.interpolate(cw) + rnorm( n=length(cw), mean=0, sd=maturity.schedule[, var.name]) * sqrt( length(cw) )
      )
      out = rbinom( length(out), 1, out )

    }

    if (type == "redo.relationships" ) {

        loadfunctions( "snowcrab", functionname="initialise.local.environment.r")
        
				det = snowcrab.db("det.georef")
         
        # recode maturity to be (0,1) for logistic regression (0 = imm, 1=mat)
        det$maturity = NA
        det$maturity[ which(det$mat==2) ] = 0
        det$maturity[ which(det$mat==1) ] = 1

        # females
        x = det[ which(det$sex==2) ,]
        model.glm = glm( maturity ~ cw, data=x, family=binomial(link="logit") )   
        xpred = data.frame( cw=G$cw.l )
        pred = predict(model.glm, xpred, type="response", se.fit=T)
        xpred$maturity = pred$fit
        xpred$se = pred$se.fit
        mat.f = xpred
        
        # males
        x = det[which(det$sex==1) ,]
        model.glm = glm( maturity ~ cw, data=x, family=binomial(link="logit") )   
        xpred = data.frame( cw=G$cw.l )
        pred = predict(model.glm, xpred, type="response", se.fit=T)
        xpred$maturity = pred$fit
        xpred$se = pred$se.fit
        mat.m = xpred

        maturity = merge( mat.f, mat.m, by="cw", suffixes=c(".f", ".m")) 

        write.table(maturity, file=file.path(R.sc, "maturity.predictions.csv"), sep=";", row.names=F)
        out = maturity.estimate( cw, sex, type="interpolated", subtype=subtype)
      }
      if (type == "model.solutions.redo" ) {
        loadfunctions( "snowcrab", functionname="initialise.local.environment.r")
        det = snowcrab.db("det.georef")
         
        # recode maturity to be (0,1) for logistic regression (0 = imm, 1=mat)
        det$maturity = NA
        det$maturity[ which(det$mat==2) ] = 0
        det$maturity[ which(det$mat==1) ] = 1

        # females
        x = det[ which(det$sex==2) ,]
        females = glm( maturity ~ cw, data=x, family=binomial(link="logit") )   
        f = summary(females)
        f1 = NULL
        f1$coeff = f$coefficients
        f1$n = f$df.residual + 2


        # males
        x = det[ which(det$sex==1) ,]
        males = glm( maturity ~ cw, data=x, family=binomial(link="logit") )   
        m = summary(males)
        m1 = NULL
        m1$coeff = m$coefficients
        m1$n = m$df.residual + 2

        out = list(males=males, females=females )
        read_write_fast( data=out, fn=file.path(R.sc, "maturity.results.rdz") )

        out = list(males=m1, females=f1)
        read_write_fast( data=out, fn=file.path(R.sc, "maturity.parameters.rdz") )


     }
      if (type == "model.solutions" ) {
        out = read_write_fast(file.path(R.sc, "maturity.parameters.rdz"))   
        coeff = NULL
        if (sex==male) dat = out$males
        if (sex==female) dat = out$females

        coeff = dat$coeff
        ndata = dat$n
        nval = length(cw) 

        out = switch( subtype,
          .no.error =  1 / ( 1 + exp( - (  coeff[1,1] + coeff[2,1]* cw ) ) )  ,
          .with.error =  1 / ( 1 + exp( - (  rnorm(n=nval, mean=coeff[1,1], sd=coeff[1,2]*sqrt(ndata-1)) 
            +  cw * rnorm( n=nval, mean=coeff[2,1], sd=coeff[2,2]*sqrt(ndata-1) ) ) ) )
        )
        out = rbinom( length(out), 1, out )

      }

    return( out )
  }


