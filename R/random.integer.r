 
  random.integer = function(n, min, max, type="sample") {
    out = switch( type,
      sample = sample( seq.int( from=min, to=max ), size=n, replace=T ), # about 1.5 X faster
      default = floor( runif(n, min=min, max=(max+1)) )
    )
  }

