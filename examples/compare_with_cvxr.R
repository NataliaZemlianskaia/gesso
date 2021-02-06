library("CVXR")

linear.predictor = function(G, E, b_0, b_e, b_g, b_gxe) {
  return(b_0 + E * b_e + G %*% b_g + (G * E) %*% b_gxe)
}

hiernet.loss = function(G, E, Y, b_0, b_e, b_g, b_gxe, lambda_1, lambda_2, family="gaussian") {
  n = dim(G)[1]
  xbeta = linear.predictor(G, E, b_0, b_e, b_g, b_gxe)
  abs_b_g = abs(b_g)
  abs_b_gxe = abs(b_gxe)
  if (family == "gaussian"){
    loss = sum((Y - xbeta)^2) / (2*n) + lambda_1 * sum(pmax(abs_b_g, abs_b_gxe)) + lambda_2 * sum(abs_b_gxe)
  }
  if (family == "binomial"){
    logistic_loss = sum(log(1 + exp(xbeta)) - Y * xbeta) / n
    loss = logistic_loss + lambda_1 * sum(pmax(abs_b_g, abs_b_gxe)) + lambda_2 * sum(abs_b_gxe)
  }
  return(loss)
}

hierNetGxE.cvxr.fit = function(p, q=1, xx, y, lambda_1, lambda_2, penalty, 
                              tol=1e-5, max_iterations=10000, verbose=FALSE, family="gaussian"){
  n = dim(xx)[1]
  beta_0 = Variable(1)
  beta_x = Variable(p)
  beta_e = Variable(q)
  theta = Variable(p, q)
  
  linear.predictor.cvxr = function(xx, beta_0, beta_e, beta_x, theta){
    f = beta_0 + xx[,1:p] %*% beta_x + xx[,(p + 1):(p + q)] %*% beta_e + xx[,(p + q + 1):(p*q + p + q)] %*% reshape_expr(t(theta), c(p*q, 1))
    return(f)
  }
  
  if (family == "gaussian"){
    loss = (sum((y - linear.predictor.cvxr(xx, beta_0, beta_e, beta_x, theta))^2)) / (2 * n)
  }
  if (family == "binomial"){
    loss = (sum(logistic(linear.predictor.cvxr(xx[y == 0,], beta_0, beta_e, beta_x, theta))) + sum(logistic(-linear.predictor.cvxr(xx[y == 1,], beta_0, beta_e, beta_x, theta)))) / n
  }
  lasso_reg = function(theta, lambda) {
    lasso =  p_norm(theta, 1)
    lambda * (lasso)
  }
  if (penalty == "reg"){
    obj = loss
  }
  if (penalty == "group_lasso"){
    group_reg = function(beta_x, beta_e, theta, lambda) {
      group_x = sum(p_norm(hstack(beta_x, theta), p=2, axis=1))
      group_e = sum(p_norm(vstack(t(beta_e), theta), p=2, axis=2))
      lambda * (group_x + group_e)
    }
    obj = loss + lasso_reg(theta, lambda_1) + group_reg(beta_x, beta_e, theta, lambda_2)
  }
  if (penalty == "l_inf"){
    l_inf_reg = function(beta_x, beta_e, theta, lambda) {
      group_x = sum(max_entries(abs(hstack(beta_x, theta)), axis=1))
      group_e = sum(max_entries(abs(vstack(t(beta_e), theta)), axis=2))
      lambda * (group_x + group_e)
    }
    obj = loss + lasso_reg(theta, lambda_1) + l_inf_reg(beta_x, beta_e, theta, lambda_2)
  }
  max_reg = function(beta_x, beta_e, theta, lambda) {
    max_x = sum(max_elemwise(sum_entries(abs(theta), axis=1), abs(beta_x)))
    lambda * (max_x)
  }
  if (penalty == "hierNet"){
    obj = loss + lasso_reg(theta, lambda_1) + max_reg(beta_x, beta_e, theta, lambda_1)
  }
  if (penalty == "hierNet2"){
    obj = loss + lasso_reg(theta, lambda_2) + max_reg(beta_x, beta_e, theta, lambda_1) 
  }
  if (penalty == "all_pairs_lasso"){
    obj = loss + lasso_reg(theta, lambda_2) + lasso_reg(beta_x, lambda_2) 
  }
  if (penalty == "l_1/l_2"){
    l12_reg = function(beta_x, beta_e, theta, lambda){
      l12_x = sum_entries(p_norm(vstack(beta_x, sum_entries(abs(theta), axis=1)), p=2, axis=2), axis=1)
      lambda * (l12_x)
    }
    obj = loss + lasso_reg(theta, lambda_2) + l12_reg(beta_x, beta_e, theta, lambda_1)
  }
  if (penalty == "l_1/l_2_1"){
    l12_reg = function(beta_x, beta_e, theta, lambda){
      l12_x = sum_entries(p_norm(vstack(beta_x, sum_entries(abs(theta), axis=1)), p=2, axis=2), axis=1)
      lambda * (l12_x)
    }
    obj = loss + lasso_reg(theta, lambda_1) + l12_reg(beta_x, beta_e, theta, lambda_1)
  }
  
  prob = Problem(Minimize(obj))
  result = solve(prob, ignore_dcp=TRUE, verbose=verbose, abstol=tol, reltol=tol, 
                 feastol=tol, max_iter=max_iterations)
  beta_x = result$getValue(beta_x)
  beta_e = result$getValue(beta_e)
  beta_0 = result$getValue(beta_0)
  beta_interaction = result$getValue(theta)

  return(list(beta_0=beta_0, beta_x=beta_x, beta_e=beta_e, beta_interaction=beta_interaction, 
              value=result$value))
}

hierNetGxE.cvxr = function(G, E, GxE, Y, grid, tol=1e-5, max_iterations=10000, family="gaussian"){
  n = dim(G)[1]
  p = dim(G)[2]
  X = cbind(G, E, GxE)
  grid = sort(grid, decreasing=TRUE)
  original_objective_value = rep(0, length(grid) * length(grid))
  objective_value = rep(0, length(grid) * length(grid))
  grid_lambda_2 = grid
  index = 0
  for (lambda_1 in grid) {
    for (lambda_2 in grid_lambda_2) {
      cat(lambda_1, " ", lambda_2, "\n")
      fit = hierNetGxE.cvxr.fit(p, 1, X, Y, lambda_1, lambda_2, penalty="hierNet2", 
                                     tol=tol, max_iterations=max_iterations, family=family)
      index = index + 1
      original_objective_value[index] = fit$value
      objective_value[index] = hiernet.loss(G, E, Y, fit$beta_0, fit$beta_e, fit$beta_x, fit$beta_interaction, 
                                            lambda_1, lambda_2, family=family)
    }
    grid_lambda_2 = rev(grid_lambda_2)
  }
  return(list(objective_value=objective_value, original_objective_value=original_objective_value))
}

tol = 1e-4
family = "binomial"
data = data.gen(seed=123, family=family, normalize=TRUE)
# data = data.gen(sample_size=500, p=1000, n_g_non_zero=20, n_gxe_non_zero=10, seed=31415)

grid = 10^seq(-4, log10(1), length.out=10) 

start = Sys.time()
fit = hierNetGxE.fit(data$G_train, data$E_train, data$Y_train,
                     tolerance=tol, grid=grid, family=family, normalize=FALSE)
stop_cd = Sys.time() - start; stop_cd

cvxr_fit = hierNetGxE.cvxr(data$G_train, data$E_train, data$GxE_train, data$Y_train,
                           grid=grid, tol=tol, family=family)

summary(fit$objective_value - cvxr_fit$objective_value)

(fit$objective_value - cvxr_fit$objective_value) < tol
(fit$objective_value - cvxr_fit$original_objective_value) < tol


cvxr_fit$objective_value - cvxr_fit$original_objective_value
