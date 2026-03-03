


showall(x) = show(stdout, "text/plain", x)

function fecundity( cw; method="moncton.primiparous" ) 
	
	# https://stats.stackexchange.com/questions/522454/how-to-construct-simulate-data-that-will-have-a-given-coefficient-of-determina

	if method == "st.marie" 
		# this is from Saint-Marie(1993, unpublished) ... look for a more recent summary
		# est = cw .^2.725 .* 10^0.7311  # looks wrong --check
	end

	if method == "moncton.primiparous"
		# R^2 = 0.88; fec = cw.^2.94 .* 10^(-0.7229)  
		# crude estimate of variability spanning the size range
		fec = log10.(cw)*2.94 .- 0.7229
		
		n = length(cw)
		R2 = 0.88
		var_y = var( log10(35:75) .* 2.94 .- 0.7229 )  # 35:75 is range of sizes ..  in reality, obs distribution will alter this 
		err = Normal.( zeros(n), sqrt( var_y*(1.0-R2)/R2 ) )
		return( 10.0 .^( fec .+ err) ) 
	end

	if method == "moncton.multiparous" 
		# est = cw.^2.5811 .* 10^0.0158   ## R^2 = 0.607
		fec = log10.(cw)*2.5811 .+ 0.0158
		
		n = length(cw)
		R2 = 0.607
		var_y = var( log10(35:75) .* 2.5811 .+ 0.0158 )  # 35:75 is range of sizes ..  in reality, obs distribution will alter this 
		err = Normal.( zeros(n), sqrt( var_y*(1.0-R2)/R2 ) )
		return( 10.0 .^( fec .+ err) ) 
	end

	return( est )

 end


function selectivity( s; s50=75, s50_sigma=s50 )
	# sigmoid; sr = steepness; s50 = s at 50% ; 
	# sbw/bw0 to rescale such that bandwidth is:
	# bw0=10.5865998 for 99.5% interval
	# bw0=7.327120 for 95% interval
	( 1 / (1+exp( -(s - s50)/ s50_sigma ) ) )   
end



function llbm(instar) 
	# mb = male bound in cw where i=1 when cw = ~ 9.2 mm,
	# instar 5 is bounded by mb(2) and mb(3)
	# prior to instar 5 moulting can be variable and so should not be
	# relied upon to heavily
	i = instar .- 3 
	exp.(1.917564 .+ 0.298914 .* i)  # Need SD's
end

function llbf(instar)
	# fb = female bound in cw where i=1 when cw = ~ 12.5 mm,
	# instar 5 is bounded by fb(1) and fb(2)
	# prior to instar 5 moulting can be variable and so should not be
	# relied upon to heavily
	i = instar .- 4 
	exp.(2.198848 .+ 0.315026 .* i) # Need SD's
end


function growth_increment( logcw; sex=0, mat=1  )

	# sex codes
	# male = 0
	# female = 1
	# sex.unknown = 2

	# # maturity codes
	# immature = 0
	# mature = 1
	# mat.unknown = 2

	# Coefficients:
	# Estimate Std. Error t value Pr(>|t|)
	# (Intercept)                 0.1931     0.0545    3.54    0.038
	# females$logcw[1:(nf - 1)]   1.0372     0.0181   57.43  1.2e-05
	# females$cat[2:nf]fmat      -0.1310     0.0252   -5.21    0.014

	# Residual standard error: 0.017 on 3 degrees of freedom
	# Multiple R-squared:  0.999,	Adjusted R-squared:  0.999 
	# F-statistic: 2.66e+03 on 2 and 3 DF,  p-value: 1.34e-05


	# # Male growth of modes  (t vs t-1)
	
	# Call:
	# lm(formula = males$logcw[2:nf] ~ males$logcw[1:(nf - 1)] + males$cat[2:nf])

	# Residuals:
	# 1         2         3         4         5         6 
	# -2.39e-04  5.68e-03 -5.40e-03 -5.48e-03  5.44e-03 -1.14e-18 

	# Coefficients:
	# Estimate Std. Error t value Pr(>|t|)
	# (Intercept)              0.30153    0.02198    13.7  0.00084
	# males$logcw[1:(nf - 1)]  1.00027    0.00667   150.0  6.5e-07
	# males$cat[2:nf]mmat     -0.13564    0.00922   -14.7  0.00068

	# Residual standard error: 0.00635 on 3 degrees of freedom
	# Multiple R-squared:     1,	Adjusted R-squared:     1 
	# F-statistic: 1.74e+04 on 2 and 3 DF,  p-value: 8e-07

	if (sex == 0) 
		# male 
		intercept = Normal(0.30153, 0.02198)
		slope =  Normal(1.00027, 0.00667)
		if mat ==0
			cw_next =  rand(intercept) + rand(slope) * logcw     
		elseif mat ==1
			maturity = Normal(-0.13564, 0.00922)
			cw_next =  rand(intercept) + rand(slope) * logcw + rand(maturity)   
		end
	end
	
	if sex == 1 
		# female
		intercept = Normal(0.1931, 0.0545)
		slope =  Normal(1.0372, 0.0181)
		if mat ==0
			cw_next =  rand(intercept) + rand(slope) * logcw     
		elseif mat==1
			maturity = Normal(-0.1310, 0.0252)
			cw_next =  rand(intercept) + rand(slope) * logcw + rand(maturity)   
		end
	end
 
	return cw_next
end



function growth_schedule()

	# create growth schedule to be used as priors
	
	instars = 1:14 
	G = DataFrame( 
		instar =instars,  
		cwm_lb = llbm(instars), 
		cwf_lb = llbf(instars),  # cm
		cwm_ub = llbm(instars.+1), 
		cwf_ub = llbf(instars.+1)  # cm
	)

	G[:, :cwm_med] = (G[:, :cwm_lb] + G[:, :cwm_ub] ) / 2
	G[:, :cwf_med] = (G[:, :cwf_lb] + G[:, :cwf_ub] ) / 2

	G[:, :cwm_range] = G[:, :cwm_ub] - G[:, :cwm_lb]  
	G[:, :cwf_range] = G[:, :cwf_ub] - G[:, :cwf_lb]  

	G.cwm_inc .= 0.0
	G.cwf_inc .= 0.0

	G[1:13, :cwm_inc] = (G[2:14, :cwm_med] - G[1:13, :cwm_med] ) / 2
	G[1:13, :cwf_inc] = (G[2:14, :cwf_med] - G[1:13, :cwf_med] ) / 2

#  last increment does not seem allometric 
#  .. computed from empirical data of CC13-CC12
#  G[13, :wm_lb] = G[12, :wm_lb] + 0.450  
#  G[13, :wm_ub] = G[12, :wm_ub] + 0.450  

	return(G)  
end


function growth_simulate(modes; gcv = 0.4,  n_max=10000, s50=50, s50_sigma=s50 )    
	
    ninstars = size(modes)[1]
    
    incr = modes[2:ninstars] - modes[1:(ninstars - 1)] 
 
	W = zeros(n_max, ninstars)

	i=1
	W[:,i] = rand( Normal( modes[i], modes[i]/20), n_max)

	for i in 2:ninstars
	inc = rand(Normal(incr[i-1], incr[i-1] * gcv), n_max)
	W[:,i] = W[:,i-1] .+ inc 
	end

	Smodes = selectivity.(modes, s50=s50, s50_sigma=s50_sigma ) 

	O = deepcopy( W )
	oi = collect(1:n_max)
	for i in 1:ninstars
		drop = sample(oi, Integer(round( (1.0- Smodes[i])*n_max)); replace=false)
		O[drop,i] .= NaN 
	end

	return(W, O)
end


function maturity(sex, logcw)
	# probability of becoming mature, given size and sex
   
# 	summary(of)
# 	cor(Mf$predicted,  Mf$mat)  # 0.687
# 	plot(Mf$predicted,  Mf$mat)  
	
# Coefficients:
# 					Estimate Std. Error z value Pr(>|z|)
# (Intercept) -51.1947     0.2551    -201   <2e-16
# logcw        12.9812     0.0641     203   <2e-16

# (Dispersion parameter for binomial family taken to be 1)

# 	Null deviance: 269910  on 194875  degrees of freedom
# Residual deviance: 120750  on 194874  degrees of freedom
# AIC: 120754

# 	summary(om)
# 	cor(Mm$predicted,  Mm$mat)  # 0.55
# 	plot(Mm$predicted,  Mm$mat)  
	
# 						 Estimate Std. Error z value Pr(>|z|)
# (Intercept)  -31.620      0.121    -261   <2e-16
# logcw          6.996      0.027     259   <2e-16

# (Dispersion parameter for binomial family taken to be 1)

# 	Null deviance: 413040  on 335294  degrees of freedom
# Residual deviance: 243377  on 335293  degrees of freedom
# AIC: 243381
 
	if sex==0 
		intercept  = Normal(-31.620, 0.121) 
		slope      = Normal( 6.9957, 0.0270) 
	elseif sex==1
		intercept  = Normal(-51.1947, 0.2551) 
		slope      = Normal(12.9812, 0.0641) 
	end
 
	a =  rand(intercept)  + rand(slope) * logcw
	prob = 1.0 / (1.0 + exp(-a))

	return(prob)
end



function sample_population( year )
	if year == -1 
			# this indicates a purely random choice for instar 5 ... annual thereafter
			degdays = rand(Uniform()) * 0.1 # start with little energy budget 
			size = rand(Normal(2.75, 0.25/3))  # approximately instar 5most of distribution is +/- 0.25 .. assume that is 99%
			sex = rand(Binomial())   
			mat = 0 # rand(Binomial())   
	else
			# sample marginal to various constraints

	end
	return degdays, size, sex, mat
end 
