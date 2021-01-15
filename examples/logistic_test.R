library(hierNetGxE)

sigmoid = function(x) {
  return(1 / (1 + exp(-x)))
}

linear.predictor = function(G, E, b_0, b_e, b_g, b_gxe) {
  return(b_0 + E * b_e + G %*% b_g + (G * E) %*% b_gxe)
}

hiernet.loss = function(data, b_0, b_e, b_g, b_gxe, lambda_1, lambda_2) {
  xbeta = linear.predictor(data$G_train, data$E_train, b_0, b_e, b_g, b_gxe)
  logistic_loss = sum(log(1 + exp(xbeta)) - data$Y_train * xbeta)
  abs_b_g = abs(b_g)
  abs_b_gxe = abs(b_gxe)
  loss = logistic_loss + lambda_1 * sum(pmax(abs_b_g, abs_b_gxe)) + lambda_2 * sum(abs_b_gxe)
  return(loss)
}

hiernet.loss.cvxr = function(data, fit, lambda_1, lambda_2) {
  return(hiernet.loss(data, fit$beta_0, fit$beta_e, fit$beta_x, fit$beta_interaction, lambda_1, lambda_2))
}

hiernet.loss.cd = function(data, fit, lambda_1, lambda_2) {
  return(hiernet.loss(data, fit$beta_0[1], fit$beta_e[1], fit$beta_g[1,], fit$beta_gxe[1,], lambda_1, lambda_2))
}

dual.link = function(data, b_0, b_e, b_g, b_gxe) {
  xbeta = linear.predictor(data$G_train, data$E_train, b_0, b_e, b_g, b_gxe)
  xbeta_s = sigmoid(xbeta)
  return((data$Y_train - xbeta_s)[,1])
}

hiernet.dual.loss = function(data, b_0, b_e, b_g, b_gxe) {
  xbeta = linear.predictor(data$G_train, data$E_train, b_0, b_e, b_g, b_gxe)
  xbeta_s = data$Y_train - dual.link(data, b_0, b_e, b_g, b_gxe)
  result = 0
  for (index in 1:(data$sample_size)) {
    if (xbeta_s[index] > 0 && xbeta_s[index] < 1) {
      result = result + xbeta_s[index] * log(xbeta_s[index])
      result = result + (1 - xbeta_s[index]) * log(1 - xbeta_s[index])
    } else {
      cat("hiernet.dual.loss: ", xbeta_s[index], "\n")
    }
  }
  return(-result)
}

tol = 1e-4
max_iter = 10000
family = "binomial"
grid_size = 10
#family = "gaussian"

data = data.gen(family=family)
#data = data.gen(family=family, sample_size=100, p=100)
#data = data.gen(family=family, sample_size=200, p=500)

fit = hierNetGxE.fit(data$G_train, data$E_train, data$Y_train,
                     normalize=FALSE, family=family, tol=tol,
                     grid_size=grid_size, max_iterations=max_iter)

fit_cvxr = hierNetGxE.cvxr(data$G_train, data$E_train, data$G_train * data$E_train, data$Y_train,
                           fit$grid, tol=1e-5, max_iterations=10000)

summary(fit$objective_value - fit_cvxr$objective_value)

lambda_1 = fit$lambda_1
lambda_2 = fit$lambda_2
hiernet.loss.cd(data, fit, lambda_1, lambda_2)
hiernet.dual.loss(data, fit$beta_0[1], fit$beta_e[1], fit$beta_g[1,], fit$beta_gxe[1,])
nu = dual.link(data, fit$beta_0[1], fit$beta_e[1], fit$beta_g[1,], fit$beta_gxe[1,])
delta_upperbound = lambda_1 - abs(nu %*% data$G_train)[1,]
delta_lowerbound = abs(nu %*% (data$G_train * data$E_train))[1,] - lambda_2
delta_lowerbound = pmax(delta_lowerbound, 0)
delta_lowerbound <= delta_upperbound
sum(nu)
sum(nu * data$E_train)

X = cbind(data$G_train, data$E_train, data$G_train * data$E_train)
fit_cvxr = hier_linear_regression_e(data$p, 1, X, data$Y_train, fit$lambda_2, fit$lambda_1, penalty="hierNet2", hierarchy="strong", tol=1e-7, max_iterations=10000)


hiernet.loss.cvxr(data, fit_cvxr, lambda_1, lambda_2)
hiernet.dual.loss(data, fit_cvxr$beta_0, fit_cvxr$beta_e, fit_cvxr$beta_x, fit_cvxr$beta_interaction)
nu = dual.link(data, fit_cvxr$beta_0, fit_cvxr$beta_e, fit_cvxr$beta_x, fit_cvxr$beta_interaction)
delta_upperbound = lambda_1 - abs(nu %*% data$G_train)[1,]
delta_lowerbound = abs(nu %*% (data$G_train * data$E_train))[1,] - lambda_2
delta_lowerbound = pmax(delta_lowerbound, 0)
delta_lowerbound <= delta_upperbound
sum(nu)
sum(nu * data$E_train)

cbind(round(abs(fit_cvxr$beta_x) - abs(fit_cvxr$beta_interaction), 4)[,1], round(abs(fit$beta_g[1,]) - abs(fit$beta_gxe[1,]), 4))


weights = c(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
Z = c(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2)

fit = hierNetGxE.fit(data$G_train, data$E_train, Z,
                     normalize=FALSE, family="gaussian", tol=tol,
                     grid=fit$grid,
                     weights=weights, max_iterations=max_iter)
