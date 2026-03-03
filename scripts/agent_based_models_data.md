# An agent-based representation of snow crab life stages and dynamics

Prepare the data. Mostly manipulation in R.

Most of the results (parameter estimates) are entered manually into the corresponding Julia functions.
 

## Size structure

Base data for size and maturity and growth is created already in bio.snowcrab::50_size_distributions.md.  

Here we bring in the saved normalised distribution densities to sample from them. 

This is an example of bringing in the data.

```R
# save distributions as rdata format to permit lad into Julia:
# from bio.snowcrab/inst/scripts/50_size_distributions.md

M = size_distributions(p=p, toget="kernel_density_modes", strata="yasm", bw=bw, np=512, 
    Y=years, pg=pg, sigdigits=3, n_min=1,
    lowpassfilter=lowpassfilter, lowpassfilter2=lowpassfilter2, outdir=outdir,
    redo=FALSE )
read_write_fast( data=M, fn=file.path(outdir, "year_sex_mat_region_julia.rdz") )

```

Sampling from it is to be completed.

```julia
    bio_data_directory = joinpath( homedir(), "bio.data" )  
    fndat = joinpath( bio_data_directory, "bio.snowcrab", "output", "size_structure", "year_sex_mat_region_julia.rdz"  )
    o = load( fndat, convert=true)
    o["K"][!,"variable"]
```
  
 
## Maturity schedules by sex and cw



## Modal groups and growth increments.
 
Using the kernel density approach, we can compute on the normalized densities to identify size modes. 

The algorithm is simply to go to every polygon and identify samples from it and potentially surrounding areas and times (potentially time window) and compute modes and troughs from first and second order differentials of the smoothed kernel densities. For production: we use a small local spatial group with a time window...

However, we can use other factors to stratify, such as depth (zlevels are left bounds in meters) and temperature (tlevels are left bounds of temperature in Celcius). Here simple, low/med/high are implemented, but not used as the above small-area-based methods work well. 

 

 
   #######  Collecting estimates of modes at finer time and space 
   #######  resolution to build a statistical model with error estimates:

    fn = file.path( p$annual.results, "size_distributions_summary.rdz" )
    res = aegis::read_write_fast(fn)



```



