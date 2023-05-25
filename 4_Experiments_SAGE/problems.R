# dgp for fooling SAGE algorithm 

dgp = function(n = 200, p = 4, cor_strength = 0.5, signal_to_noise = 2, mean_range = 0){
  
  # define correlation matrix
  sigma = matrix(data = rep(cor_strength, p^2), nrow = p) + diag(x = 1-cor_strength, nrow = p)
  
  df = mvtnorm::rmvnorm(n, mean = seq(from = -mean_range, to = mean_range, length.out = p), sigma = sigma) # think of varying mean values
  y = rowSums(df)
  
  # add error term according to SNR = Var(Y) / Var(error)
  y = y + rnorm(n, mean = 0, sd = sqrt(var(y)/signal_to_noise))
  
  # define variable names, pick one variable that is sensitive 
  colnames(df)[2:p] = lapply(2:p, function(p){paste0("x", p)})
  colnames(df)[1] <- "sensitive_feature"
  #colnames(df)[sample(1:p,1)] <- "sensitive_feature"
  
  return(data.frame(cbind(df, y)))
}

