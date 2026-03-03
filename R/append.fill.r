 
  # cumulative egg frequency distribution
  append.fill = function(x, y, len)  x = c( x, rep.int(y, len - length(x) ) ) # assume y is an integer ... faster


