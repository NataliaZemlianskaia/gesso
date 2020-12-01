compute.grid = function(G, E, Y, normalize, grid_size, grid_min_ratio) {
  std = function(x) {
    mx = mean(x)
    return(sqrt(mean((x - mx)^2)))
  }
  
  colStd = function(x) {
    return(apply(x, 2, std))
  }
  
  n = dim(G)[1]
  G_by_Yn_abs = abs(Y %*% G)[1,] / n
  GxE_by_Yn_abs = abs((Y * E) %*% G)[1,] / n  
  if (normalize) {
    std_G = colStd(G)
    G_by_Yn_abs = G_by_Yn_abs / std_G
    GxE_by_Yn_abs = GxE_by_Yn_abs / (std_G * std(E))
  }
  
  lambda_max = max(c(G_by_Yn_abs, GxE_by_Yn_abs))
  lambda_min = grid_min_ratio * lambda_max
  grid = 10^seq(log10(lambda_min), log10(lambda_max), length.out=grid_size)
  return(grid)
}


hierNetGxE.fit = function(G, E, Y, normalize=FALSE, grid=NULL, grid_size=20, 
                          grid_min_ratio=1e-4, family="gaussian",
                          tolerance=1e-4, max_iterations=10000, min_working_set_size=100) {
  if (is(G, "matrix")) {
    if (typeof(G) != "double")
      stop("G must be of type double")
    is_sparse_g = FALSE
  } else if ("dgCMatrix" %in% class(G)) {
    if (typeof(G@x) != "double")
      stop("G must be of type double")
    is_sparse_g = TRUE
  } else {
    stop("G must be a standard R matrix, big.matrix, filebacked.big.matrix, or dgCMatrix")
  }  
  if (is.null(grid)) {
    grid = compute.grid(G, E, Y, normalize, grid_size, grid_min_ratio)
  }
  
  n = dim(G)[1]
  weights = rep(1, n) / n
  fit = fitModel(G, E, Y, weights, normalize, grid, family, 
                 tolerance, max_iterations, min_working_set_size, is_sparse_g)
  fit$beta_g_nonzero = rowSums(fit$beta_g != 0)
  fit$beta_gxe_nonzero = rowSums(fit$beta_gxe != 0) 
  return(fit)
}

hierNetGxE.cv = function(G, E, Y, normalize=FALSE, grid=NULL, grid_size=20, grid_min_ratio=1e-4, 
                         family="gaussian",
                         nfolds=5, parallel=TRUE, seed=42,
                         tolerance=1e-4, max_iterations=10000, min_working_set_size=100) {
  
  set.seed(seed)
  if (is(G, "matrix")) {
    if (typeof(G) != "double")
      stop("G must be of type double")
    is_sparse_g = FALSE
  } else if ("dgCMatrix" %in% class(G)) {
    if (typeof(G@x) != "double")
      stop("G must be of type double")
    is_sparse_g = TRUE
  } else {
    stop("G must be a standard R matrix, big.matrix, filebacked.big.matrix, or dgCMatrix")
  }  
  if (is.null(grid)) {
    grid = compute.grid(G, E, Y, normalize, grid_size, grid_min_ratio)
  }
  n = dim(G)[1]  

  if (nfolds < 2) {
    stop("number of folds (nfolds) must be at least 2")
  }
  foldid = sample(rep(seq(nfolds), length=n))

  if (parallel) {
    print("Parallel")
    start_parallel = Sys.time()
    result = fitModelCV(G, E, Y, normalize, grid, family, tolerance,
                        max_iterations, min_working_set_size, nfolds, seed, nfolds, is_sparse_g)
    print(Sys.time() - start_parallel)
  } else {
    print("Non-parallel")
    result = fitModelCV(G, E, Y, normalize, grid, family, tolerance,
                        max_iterations, min_working_set_size, nfolds, seed, 1, is_sparse_g)
  }
  result_ = colMeans(result)
  
  weights = rep(1, n)
  weights = weights / sum(weights)
  print("fit on full data")
  start_all = Sys.time()
  fit_all_data = fitModel(G, E, Y, weights, normalize, grid, family,
                          tolerance, max_iterations, min_working_set_size, is_sparse_g)
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

hierNetGxE.coef = function(fit, lambda){
 lambda_idx = which(fit$lambda_1 == lambda$lambda_1 & fit$lambda_2 == lambda$lambda_2)
 beta_0 = fit$beta_0[lambda_idx]
 beta_e = fit$beta_e[lambda_idx]
 beta_g = fit$beta_g[lambda_idx,]
 beta_gxe = fit$beta_gxe[lambda_idx,]
 return(list(beta_0=beta_0, beta_e=beta_e, beta_g=beta_g, beta_gxe=beta_gxe))
}
  
  

  
  
