 
  eggage.estimate = function( n, method="random" ) {
    out = NULL
    if (method=="random" ) {
      out = as.vector( random.integer( n=n, min=0, max=2, type="sample" ) )
    }
    if (method=="biological") {
      
      # add  Penny's results here


    }
    return(out)
  }



