compute.grid = function(G, E, Y, normalize, grid_size, grid_min_ratio=1e-4) {
  std = function(x) {
    mx = mean(x)
    return(sqrt(mean((x - mx)^2)))
  }
  
  p = dim(G)[2]
  n = dim(G)[1]
  
  std_E = std(E)
  
  max_G_by_Yn_abs = 0 
  max_GxE_by_Yn_abs = 0
  
  for (i in 1:p) {
    cur_G_by_Yn_abs = abs(Y %*% G[,i])[1,1] / n
    cur_GxE_by_Yn_abs = abs((Y * E) %*% G[,i])[1,1] / n
    
    if (normalize) {
      std_G = std(G[,i])
      cur_G_by_Yn_abs = cur_G_by_Yn_abs / std_G
      cur_GxE_by_Yn_abs = cur_GxE_by_Yn_abs / (std_G * std_E)
    }
    
    if (cur_G_by_Yn_abs > max_G_by_Yn_abs) {
      max_G_by_Yn_abs = cur_G_by_Yn_abs
    }
    if (cur_GxE_by_Yn_abs > max_GxE_by_Yn_abs) {
      max_GxE_by_Yn_abs = cur_GxE_by_Yn_abs
    }
  }
  
  lambda_max = max(c(max_G_by_Yn_abs, max_GxE_by_Yn_abs))
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
    mattype_g = 0
  } else if ("dgCMatrix" %in% class(G)) {
    if (typeof(G@x) != "double")
      stop("G must be of type double")
    mattype_g = 1
  } else if (is.big.matrix(G)) {
    if (bigmemory::describe(G)@description$type != "double")
      stop("G must be of type double")
    mattype_g = 2
  } else {
    stop("G must be a standard R matrix, big.matrix, filebacked.big.matrix, or dgCMatrix")
  }  
  if (is.null(grid)) {
    grid = compute.grid(G, E, Y, normalize, grid_size, grid_min_ratio)
  }
  
  n = dim(G)[1]
  if (is.null(weights)) {
    weights = rep(1, n) / n
  }
  fit = fitModel(G, E, Y, weights, normalize, grid, family, 
                 tolerance, max_iterations, min_working_set_size, mattype_g)
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
    mattype_g = 0
  } else if ("dgCMatrix" %in% class(G)) {
    if (typeof(G@x) != "double")
      stop("G must be of type double")
    mattype_g = 1
  } else if (is.big.matrix(G)) {
    if (bigmemory::describe(G)@description$type != "double")
      stop("G must be of type double")
    mattype_g = 2
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

  if (parallel) {
    print("Parallel cv")
    start_parallel = Sys.time()
    result = fitModelCV(G, E, Y, normalize, grid, family, tolerance,
                        max_iterations, min_working_set_size, nfolds, seed, nfolds, mattype_g)
    print(Sys.time() - start_parallel)
  } else {
    print("Non-parallel cv")
    result = fitModelCV(G, E, Y, normalize, grid, family, tolerance,
                        max_iterations, min_working_set_size, nfolds, seed, 1, mattype_g)
  }
  result_ = colMeans(result$test_loss)
  mean_beta_g_nonzero = colMeans(result$beta_g_nonzero)
  mean_beta_gxe_nonzero = colMeans(result$beta_gxe_nonzero)
  
  weights = rep(1, n)
  weights = weights / sum(weights)
  print("fit on full data")
  start_all = Sys.time()
  fit_all_data = fitModel(G, E, Y, weights, normalize, grid, family,
                          tolerance, max_iterations, min_working_set_size, mattype_g)
  print(Sys.time() - start_all)
  
  lambda_min_index = which.min(result_)
  loss_min = result_[lambda_min_index]
  
  result_table = tibble(lambda_1=fit_all_data$lambda_1,
                        lambda_2=fit_all_data$lambda_2, 
                        mean_loss=result_,
                        mean_beta_g_nonzero=mean_beta_g_nonzero,
                        mean_beta_gxe_nonzero=mean_beta_gxe_nonzero) 
  

  lambda_min = result_table[lambda_min_index, 1:2]
  
  return(list(cv_result=result_table,
              lambda_min=lambda_min, 
              fit=fit_all_data,
              grid=grid,
              full_cv_result=result))
}

hierNetGxE.coef = function(fit, lambda){
 lambda_idx = which(fit$lambda_1 == lambda$lambda_1 & fit$lambda_2 == lambda$lambda_2)
 beta_0 = fit$beta_0[lambda_idx]
 beta_e = fit$beta_e[lambda_idx]
 beta_g = fit$beta_g[lambda_idx,]
 beta_gxe = fit$beta_gxe[lambda_idx,]
 
 return(list(beta_0=beta_0, beta_e=beta_e, beta_g=beta_g, beta_gxe=beta_gxe))
}
  
hierNetGxE.coefnum = function(cv_model, target_b_gxe_non_zero, less_than=TRUE){
  cv_result = cv_model$cv_result; fit = cv_model$fit
  if (less_than){
  # best_lambdas = cv_result %>%
  #   filter(mean_beta_gxe_nonzero <= target_b_gxe_non_zero) %>%
  #   filter(mean_loss == min(mean_loss)) %>%
  #   select(lambda_1, lambda_2)
    best_lambdas = cv_result %>%
      filter(mean_beta_gxe_nonzero <= target_b_gxe_non_zero) %>%
      arrange(mean_loss) %>%
      slice(1) %>%
      select(lambda_1, lambda_2)
  } else {
    # best_lambdas = cv_result %>%
    #   filter(mean_beta_gxe_nonzero >= target_b_gxe_non_zero) %>%
    #   filter(mean_loss == min(mean_loss)) %>%
    #   select(lambda_1, lambda_2)
    
    best_lambdas = cv_result %>%
      filter(mean_beta_gxe_nonzero >= target_b_gxe_non_zero) %>%
      arrange(mean_loss) %>%
      slice(1) %>%
      select(lambda_1, lambda_2)
  }
  
  return(hierNetGxE.coef(fit, best_lambdas))
}

hierNetGxE.predict = function(beta_0, beta_e, beta_g, beta_gxe, new_G, new_E, 
                              family="gaussian"){
  new_GxE = new_G * new_E
  lp = (beta_0 + beta_e * new_E +  new_G %*% beta_g + new_GxE %*% beta_gxe)[,1]
  
  if (family == "gaussian"){
    return(lp)
  } else if (family == "binomial"){
    prob = 1/(1+exp(-lp))
    return(prob)
  }
}








