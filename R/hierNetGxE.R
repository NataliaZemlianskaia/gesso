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


hierNetGxE.fit = function(G, E, Y, normalize=TRUE, grid=NULL, grid_size=20, 
                          grid_min_ratio=1e-4, family="gaussian", weights=NULL,
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
  if (is.null(weights)) {
    weights = rep(1, n)
    #weights = rep(1, n) / n
  }
  fit = fitModel(G, E, Y, weights, normalize, grid, family, 
                 tolerance, max_iterations, min_working_set_size, is_sparse_g)
  fit$beta_g_nonzero = rowSums(fit$beta_g != 0)
  fit$beta_gxe_nonzero = rowSums(fit$beta_gxe != 0) 
  return(fit)
}

hierNetGxE.cv = function(G, E, Y, normalize=TRUE, grid=NULL, grid_size=20, grid_min_ratio=1e-4, 
                         family="gaussian",
                         nfolds=4, parallel=TRUE, seed=42,
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
    print("Parallel cv")
    start_parallel = Sys.time()
    result = fitModelCV(G, E, Y, normalize, grid, family, tolerance,
                        max_iterations, min_working_set_size, nfolds, seed, nfolds, is_sparse_g)
    print(Sys.time() - start_parallel)
  } else {
    print("Non-parallel cv")
    result = fitModelCV(G, E, Y, normalize, grid, family, tolerance,
                        max_iterations, min_working_set_size, nfolds, seed, 1, is_sparse_g)
  }
  result_ = colMeans(result$test_loss)
  mean_beta_g_nonzero = colMeans(result$beta_g_nonzero)
  mean_beta_gxe_nonzero = colMeans(result$beta_gxe_nonzero)
  
  weights = rep(1, n)
  weights = weights / sum(weights)
  print("fit on full data")
  start_all = Sys.time()
  fit_all_data = fitModel(G, E, Y, weights, normalize, grid, family,
                          tolerance, max_iterations, min_working_set_size, is_sparse_g)
  print(Sys.time() - start_all)
  
  lambda_min_index = which.min(result_)
  loss_min = result_[lambda_min_index]
  loss_se = sd(result_)/sqrt(length(result_))
  
  result = tibble(lambda_1=fit_all_data$lambda_1,
                  lambda_2=fit_all_data$lambda_2, 
                  mean_loss=result_,
                  mean_beta_g_nonzero=mean_beta_g_nonzero,
                  mean_beta_gxe_nonzero=mean_beta_gxe_nonzero) 
  
  lambda_se = (result %>%
    filter(mean_loss <= loss_min + loss_se) %>%
    arrange(desc(lambda_1), desc(lambda_2)) %>%
    slice(1))[1:2]
  
  lambda_min = result[lambda_min_index, 1:2]
  
  return(list(cv_result=result, lambda_min=lambda_min, 
              lambda_se=lambda_se, 
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
  
hierNetGxE.coefnum = function(fit, cv_result, target_b_gxe_non_zero){
  best_lambdas = cv_result %>%
    filter(mean_beta_gxe_nonzero <= target_b_gxe_non_zero) %>%
    filter(mean_loss == min(mean_loss)) %>%
    select(lambda_1, lambda_2)
  
  return(hierNetGxE.coef(fit, best_lambdas))
}

  
  
