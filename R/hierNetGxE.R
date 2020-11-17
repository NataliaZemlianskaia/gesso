
hierNetGxE.fit = function(G, E, Y, standardize=FALSE, grid=NULL, grid_size=10, grid_min_ratio=1e-4, family="gaussian",
                      tolerance=1e-4, max_iterations=10000, min_working_set_size=100) {
  stopifnot(!standardize)
  if (is.null(grid)) {
    grid = 10^seq(-4, log10(0.1), length.out=grid_size)
  }
  n = dim(G)[1]
  weights = rep(1, n) / n
  return(fitModelRcpp(G, E, Y, weights, standardize, grid, grid_size, grid_min_ratio, family, tolerance, max_iterations, min_working_set_size))
}

hierNetGxE.cv = function(G, E, Y, standardize=FALSE, grid=NULL, grid_size=10, grid_min_ratio=1e-4, family="gaussian",
                         nfolds=5, parallel=TRUE, seed=42,
                         tolerance=1e-4, max_iterations=10000, min_working_set_size=100) {
  
  set.seed(seed)
  stopifnot(!standardize)
  if (is.null(grid)) {
    grid = 10^seq(-4, log10(0.1), length.out=grid_size)
  }
  n = dim(G)[1]  

  if (nfolds < 2) {
    stop("number of folds (nfolds) must be at least 2")
  }
  foldid = sample(rep(seq(nfolds), length=n))

  if (parallel) {
    print("Parallel")
    start_parallel = Sys.time()
    result = fitModelCVRcpp(G, E, Y, standardize, grid, grid_size,
                            grid_min_ratio, family, tolerance, max_iterations, min_working_set_size, nfolds, seed, nfolds)
    print(Sys.time() - start_parallel)
  } else {
    print("Non-parallel")
    result = fitModelCVRcpp(G, E, Y, standardize, grid, grid_size,
                            grid_min_ratio, family, tolerance, max_iterations, min_working_set_size, nfolds, seed, 1)
  }
  result_ = colMeans(result)
  
  weights = rep(1, n)
  weights = weights / sum(weights)
  print("fit on full data")
  start_all = Sys.time()
  fit_all_data = fitModelRcpp(G, E, Y, weights, standardize, grid, grid_size, grid_min_ratio,
                              family, tolerance, max_iterations, min_working_set_size)
  print(Sys.time() - start_all)
  
  #result_ = rowMeans(result)
  #result_ = colMeans(result)
  lambda.min.index = which.min(result_)
  loss.min = result_[lambda.min.index]
  loss.se = sd(result_)/sqrt(length(result_))
  
  #result = cbind(fit_all_data$lambda_1, fit_all_data$lambda_2, rowMeans(result)) 
  result = tibble(lambda_1=fit_all_data$lambda_1, lambda_2=fit_all_data$lambda_2, 
                  mean_loss=result_) 
  
  #lambda.se = (result %>%
  #  filter(mean_loss <= loss.min + loss.se) %>%
  #  arrange(desc(lambda_1), desc(lambda_2)) %>%
  #  slice(1))[1:2]
  
  lambda.min = result[lambda.min.index, 1:2]
  return(list(result=result, lambda.min=lambda.min, 
              #lambda.se=lambda.se, 
              fit=fit_all_data, grid=grid))
}

hierNet.coef = function(fit, lambda){
 lambda_idx = which(fit$lambda_1 == lambda$lambda_1 & fit$lambda_2 == lambda$lambda_2)
 beta_0 = fit$beta_0[lambda_idx]
 beta_e = fit$beta_e[lambda_idx]
 beta_g = fit$beta_g[lambda_idx,]
 beta_gxe = fit$beta_gxe[lambda_idx,]
 return(list(beta_0=beta_0, beta_e=beta_e, beta_g=beta_g, beta_gxe=beta_gxe))
}
  
  

  
  
